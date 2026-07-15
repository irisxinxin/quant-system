#!/usr/bin/env python3
"""
zhaoge_backtest.py — 跟单模拟回测 #股票赵哥-日内 (正股/杠杆ETF, 真实5分K, fill-realistic)。

与"他的自报账本"的本质区别: 镜像跟单者的真实处境 —
  他喊买 → 我们按他喊的价挂限价单(信号当日~次一交易日有效; 价格摸不到=空手, 不追)
  他喊卖 → 下一根5m bar开盘价市价跟卖 ("一半"→卖持仓一半, 否则全卖)
  他不卖 → 我们只能拿着 (他的摊平/装死在这里现形), 期末按最后收盘mark
仓位: 每个买入信号=1批($10,000), 每票最多3批(防无限补仓); 成本0.1%/边。
K线: history_candlesticks_by_date 分段抓, 缓存 data/zhaoge_bars/<TK>.csv (增量, 兼归档)。
诚实边界: 批次归属做了近似(他说'出掉X的那批'我们按比例卖); 夜盘/盘前信号只能在RTH成交;
          K线拉不到的票如实排除。
"""
import csv, json, re, sys, time
from datetime import datetime, date, time as dtime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent
BARS_DIR = ROOT / "data" / "zhaoge_bars"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
LOT_USD, MAX_LOTS, COST = 10000, 3, 0.001
import os
ALL_SESS = os.environ.get("ALL", "") == "1"   # 全时段(盘前+盘后; 夜盘20-04长桥不提供)
START, END = date(2026, 4, 15), date(2026, 7, 15)

BUY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:附近)?\s*(买了|买入|买|加了|加仓|加回了|加回|加|接了|接)")
SELL_RE = re.compile(r"(出掉|出一半|平出|卖出|[^买]出|^出|平|清)")
SELL_PX_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:附近)?\s*(?:出掉|出一半|平出|出|卖出|卖|平|清)")
TK_RE = re.compile(r"[A-Za-z]{2,6}")
STOP = {"spx", "spy", "qqq", "ndx", "vix", "cpi", "ppi", "fomc", "gdp", "ipo", "etf", "ath",
        "ceo", "gtc", "ai", "cpu", "gpu", "nbsp", "amp", "http", "https", "png", "jpg", "com",
        "cdn", "www", "sk", "pce", "adp", "pmi", "sec", "fda", "usd", "krw", "jpy", "a", "us",
        "hk", "app", "api", "max", "min", "el", "ev", "opec"}


def strip_prefix(t):
    t = re.sub(r"\*\*\[.+?\]\*\*", " ", t)
    t = re.sub(r"\*\*.+?\*\*:?", " ", t)
    return t.strip()


def parse_stream():
    """→ [(ts_utc, 'buy'/'sell', ticker, px_or_None, frac)]; 同日同文本去重(他爱重发)。"""
    msgs = json.load(open(ROOT / "output" / "zhaoge_history.json"))
    out, seen = [], set()
    n_skip_multi = n_cn = 0
    for m in msgs:
        raw = strip_prefix(m["text"])
        if len(raw) > 200 or "http" in raw:
            continue
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        if not (START <= ts.date() <= END + timedelta(days=1)):
            continue
        key = (ts.date().isoformat(), " ".join(raw.split()))
        if key in seen:
            continue
        seen.add(key)
        tks = [w.lower() for w in TK_RE.findall(raw) if w.lower() not in STOP]
        tks = list(dict.fromkeys(tks))
        line = " ".join(raw.split())
        mb = BUY_RE.search(line)
        is_sell = bool(SELL_PX_RE.search(line)) or bool(re.search(r"出掉|出一半|平出|清", line))
        if len(tks) != 1:
            if (mb or is_sell) and tks:
                n_skip_multi += 1
            elif (mb or is_sell) and re.search(r"双倍|三倍", line):
                n_cn += 1
            continue
        tk = tks[0]
        if is_sell and (not mb or line.index(SELL_PX_RE.search(line).group(0) if SELL_PX_RE.search(line) else "出") < (mb.start() if mb else 9999)):
            frac = 0.5 if "一半" in line else 1.0
            ms = SELL_PX_RE.search(line)
            px = float(ms.group(1)) if ms else None
            out.append((ts, "sell", tk, px, frac))
        elif mb:
            out.append((ts, "buy", tk, float(mb.group(1)), 1.0))
    out.sort(key=lambda x: x[0])
    print(f"信号流: {len(out)} 条 ({sum(1 for x in out if x[1]=='buy')}买/{sum(1 for x in out if x[1]=='sell')}卖) | "
          f"多票跳过{n_skip_multi} 中文名跳过{n_cn}")
    return out


def fetch_bars(q, tk):
    """5m bars (UTC), 增量缓存 CSV。"""
    BARS_DIR.mkdir(parents=True, exist_ok=True)
    f = BARS_DIR / f"{tk.upper()}{'_ALL' if ALL_SESS else ''}.csv"
    rows = {}
    if f.exists():
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows[r["ts"]] = r
    have_days = {r[:10] for r in rows}
    need = []
    d = START
    while d <= END:
        if d.weekday() < 5 and d.isoformat() not in have_days:
            need.append(d)
        d += timedelta(days=1)
    if need:
        from longport.openapi import Period, AdjustType
        chunk_start = need[0]
        while chunk_start <= need[-1]:
            chunk_end = min(chunk_start + timedelta(days=3 if ALL_SESS else 11), need[-1])
            try:
                from longport.openapi import TradeSessions
                kw = dict(trade_sessions=TradeSessions.All) if ALL_SESS else {}
                b = q.history_candlesticks_by_date(f"{tk.upper()}.US", Period.Min_5,
                                                   AdjustType.ForwardAdjust, chunk_start, chunk_end, **kw)
                for x in b:
                    t = x.timestamp.astimezone(UTC).isoformat()
                    rows[t] = dict(ts=t, o=float(x.open), h=float(x.high),
                                   l=float(x.low), c=float(x.close), v=int(x.volume))
            except Exception:
                pass
            time.sleep(0.25)
            chunk_start = chunk_end + timedelta(days=1)
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["ts", "o", "h", "l", "c", "v"])
            w.writeheader()
            for t in sorted(rows):
                w.writerow(rows[t])
    out = [dict(ts=datetime.fromisoformat(r["ts"]), o=float(r["o"]), h=float(r["h"]),
                l=float(r["l"]), c=float(r["c"])) for r in rows.values()]
    return sorted(out, key=lambda x: x["ts"])


def next_trading_day_end(ts_utc):
    """信号时刻起, 次一交易日 16:00 ET 的 UTC 时刻 (买单有效期)。"""
    d = ts_utc.astimezone(ET).date()
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:
        nd += timedelta(days=1)
    return datetime.combine(nd, dtime(20, 0) if ALL_SESS else dtime(16, 0), tzinfo=ET).astimezone(UTC)


def simulate(stream):
    from longport.openapi import Config, QuoteContext
    q = QuoteContext(Config.from_env())
    tickers = sorted({tk for _, _, tk, _, _ in stream})
    bars = {}
    for tk in tickers:
        b = fetch_bars(q, tk)
        if b:
            bars[tk] = b
        else:
            print(f"  ⚠️ {tk}: 无K线, 该票全部信号排除")
    pos = {}          # tk -> list[dict(shares, cost_px)]
    trades = []       # realized
    n_nofill = 0
    for ts, side, tk, px, frac in stream:
        B = bars.get(tk)
        if not B:
            continue
        if side == "buy":
            if px is None or len(pos.get(tk, [])) >= MAX_LOTS:
                continue
            deadline = next_trading_day_end(ts)
            fill = None
            for b in B:
                if b["ts"] <= ts or b["ts"] > deadline:
                    continue
                if b["o"] <= px:
                    fill = b["o"]; break
                if b["l"] <= px:
                    fill = px; break
            if fill is None:
                n_nofill += 1
                continue
            fill *= (1 + COST)
            pos.setdefault(tk, []).append(dict(shares=LOT_USD / fill, cost=fill, ts=ts))
        else:
            lots = pos.get(tk) or []
            if not lots:
                continue
            nxt = next((b for b in B if b["ts"] > ts), None)
            if nxt is None:
                continue
            sell_px = nxt["o"] * (1 - COST)
            tot_sh = sum(l["shares"] for l in lots)
            sh = tot_sh * frac
            avg_cost = sum(l["shares"] * l["cost"] for l in lots) / tot_sh
            pnl = (sell_px - avg_cost) * sh
            trades.append(dict(date=str(ts.astimezone(ET).date()), ticker=tk,
                               sell=round(sell_px, 3), cost=round(avg_cost, 3),
                               shares=round(sh), pnl=round(pnl),
                               pct=round((sell_px / avg_cost - 1) * 100, 2)))
            remain = tot_sh - sh
            if remain < 1e-6:
                pos[tk] = []
            else:
                pos[tk] = [dict(shares=remain, cost=avg_cost, ts=lots[0]["ts"])]
    # 期末mark
    inv = []
    for tk, lots in pos.items():
        if not lots:
            continue
        B = bars.get(tk)
        last = B[-1]["c"] if B else None
        for l in lots:
            upnl = (last - l["cost"]) * l["shares"] if last else None
            inv.append(dict(ticker=tk, cost=round(l["cost"], 3), last=last,
                            usd=round(l["shares"] * l["cost"]),
                            upnl=round(upnl) if upnl is not None else None,
                            upct=round((last / l["cost"] - 1) * 100, 1) if last else None))
    return trades, inv, n_nofill


def main():
    stream = parse_stream()
    trades, inv, n_nofill = simulate(stream)
    import statistics as st
    print(f"\n═══ 跟单模拟 {START} ~ {END} (每批$1万, 每票≤3批, 成本0.1%/边) ═══")
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    tot = sum(t["pnl"] for t in trades)
    print(f"已实现 {len(trades)} 笔: 赢{len(wins)}/输{len(losses)} 胜率{len(wins)/len(trades)*100 if trades else 0:.0f}%")
    if trades:
        print(f"  合计 ${tot:+,.0f} | 平均每笔 ${tot/len(trades):+,.0f} ({st.mean(t['pct'] for t in trades):+.2f}%)")
        print(f"  最大赢 ${max(t['pnl'] for t in trades):+,.0f} / 最大亏 ${min(t['pnl'] for t in trades):+,.0f}")
        by_m = {}
        for t in trades:
            by_m.setdefault(t["date"][:7], []).append(t["pnl"])
        print("  月度: " + " | ".join(f"{k}: ${sum(v):+,.0f}({len(v)}笔)" for k, v in sorted(by_m.items())))
    print(f"买入未成交(不追价) {n_nofill} 笔")
    upnl_tot = sum(i["upnl"] for i in inv if i["upnl"] is not None)
    print(f"\n期末库存 {len(inv)} 批 (他没喊卖/我们跟着拿着的): 浮动盈亏合计 ${upnl_tot:+,.0f}")
    for i in sorted(inv, key=lambda x: x["upnl"] or 0):
        print(f"  {i['ticker']:6} 成本{i['cost']:<8} 现价{i['last']:<8} ${i['upnl']:+7,.0f} ({i['upct']:+.1f}%)")
    print(f"\n总账 = 已实现 ${tot:+,.0f} + 浮动 ${upnl_tot:+,.0f} = ${tot+upnl_tot:+,.0f}"
          f"  (总动用资金上限 ≈ ${LOT_USD*MAX_LOTS*8:,})")
    out = dict(generated=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds"),
               window=f"{START}~{END}", trades=trades, inventory=inv,
               realized=tot, unrealized=upnl_tot, n_nofill=n_nofill)
    (ROOT / "output" / "zhaoge_backtest.json").write_text(json.dumps(out, ensure_ascii=False))
    print("\n存 → output/zhaoge_backtest.json")


if __name__ == "__main__":
    main()
