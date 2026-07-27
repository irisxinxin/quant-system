#!/usr/bin/env python3
"""
shishi_10f_backtest.py — 诗诗 CONL "10F" 机械规则回测 (真实5分K合成10分K, 全时段)。

规则(他的原话, #诗诗 频道):
  "今天開盤工作大概先看10F K線…之上持股 之下離場" (2026-07-09)
  "收完的K 後面走勢決定方向 往上突破K線最高點才能持有 不行就先離場"
  "保守等10F均線上去" (2026-05-15)
→ 骨架: 10分钟K收盘 > 均线 → 做多; 收盘 < 均线 → 离场。均线周期/类型他没说 → 扫参。
两个变体:
  A simple : 收盘上穿MA → 次bar开盘买; 收盘下穿MA → 次bar开盘卖
  B confirm: 上穿后还需突破【信号K最高点】才进 (stop-buy挂前K高, bar.High≥level成交,
             跳空高开按Open); 出场同A (收盘跌破MA→次bar开盘卖)
成交/成本: $10k满进满出, 0.1%/边; 全时段(04:00-20:00 ET)信号即时执行, 无隔夜豁免。
对照: 买入持有CONL同窗口。诚实边界: 盘前盘后流动性差, 实际滑点会比0.1%大;
      参数高原若不存在则判定曲线拟合不可用。
"""
import csv, json, sys, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
START, END = date(2026, 3, 2), date(2026, 7, 25)
COST = 0.001
CACHE = ROOT / "data" / "shishi_bars" / "CONL_5m_ALL.csv"


def fetch_5m():
    """CONL 5分全时段K, 增量缓存。"""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if CACHE.exists():
        with open(CACHE) as fh:
            for r in csv.DictReader(fh):
                rows[r["ts"]] = r
    have = {r[:10] for r in rows}
    need = []
    d = START
    while d <= END:
        if d.weekday() < 5 and d.isoformat() not in have:
            need.append(d)
        d += timedelta(days=1)
    if need:
        from longport.openapi import Config, QuoteContext, Period, AdjustType, TradeSessions
        q = QuoteContext(Config.from_env())
        cs = need[0]
        while cs <= need[-1]:
            ce = min(cs + timedelta(days=3), need[-1])
            try:
                b = q.history_candlesticks_by_date("CONL.US", Period.Min_5,
                                                   AdjustType.ForwardAdjust, cs, ce,
                                                   trade_sessions=TradeSessions.All)
                for x in b:
                    t = x.timestamp.astimezone(UTC).isoformat()
                    rows[t] = dict(ts=t, o=float(x.open), h=float(x.high),
                                   l=float(x.low), c=float(x.close), v=int(x.volume))
            except Exception as e:
                print(f"  ⚠️ {cs}~{ce}: {e}")
            time.sleep(0.25)
            cs = ce + timedelta(days=1)
        with open(CACHE, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ts", "o", "h", "l", "c", "v"])
            w.writeheader()
            for t in sorted(rows):
                w.writerow(rows[t])
    out = [dict(ts=datetime.fromisoformat(r["ts"]), o=float(r["o"]), h=float(r["h"]),
                l=float(r["l"]), c=float(r["c"])) for r in rows.values()]
    return sorted(out, key=lambda x: x["ts"])


def to_10m(b5):
    """5m→10m: 按ET时钟10分钟槽合并 (跨日不并)。"""
    out = []
    cur = None
    for b in b5:
        t = b["ts"].astimezone(ET)
        slot = (t.date(), t.hour, t.minute // 10)
        if cur and cur["slot"] == slot:
            cur["h"] = max(cur["h"], b["h"]); cur["l"] = min(cur["l"], b["l"])
            cur["c"] = b["c"]
        else:
            if cur:
                out.append(cur)
            cur = dict(slot=slot, ts=b["ts"], o=b["o"], h=b["h"], l=b["l"], c=b["c"])
    if cur:
        out.append(cur)
    return out


def ma_series(closes, n, kind):
    out = [None] * len(closes)
    if kind == "SMA":
        s = 0.0
        for i, c in enumerate(closes):
            s += c
            if i >= n:
                s -= closes[i - n]
            if i >= n - 1:
                out[i] = s / n
    else:  # EMA
        k = 2 / (n + 1)
        e = None
        for i, c in enumerate(closes):
            e = c if e is None else c * k + e * (1 - k)
            if i >= n - 1:
                out[i] = e
    return out


def run(bars, n, kind, confirm):
    closes = [b["c"] for b in bars]
    ma = ma_series(closes, n, kind)
    pos = None      # dict(entry, ts)
    pend = None     # confirm变体: 待突破的前K高
    trades = []
    for i in range(1, len(bars) - 1):
        b, nxt = bars[i], bars[i + 1]
        if ma[i] is None:
            continue
        above = b["c"] > ma[i]
        if pos is None:
            if above:
                if not confirm:
                    pos = dict(entry=nxt["o"] * (1 + COST), ts=nxt["ts"])
                elif pend is None:
                    pend = b["h"]           # 挂突破单: 信号K最高点
            if confirm and pend is not None:
                # 次bar起, High≥pend → 成交 (跳空按Open)
                if nxt["h"] >= pend:
                    fill = max(pend, nxt["o"])
                    pos = dict(entry=fill * (1 + COST), ts=nxt["ts"])
                    pend = None
                elif b["c"] < ma[i]:        # 跌回均线下, 撤单
                    pend = None
        else:
            if not above:
                ex = nxt["o"] * (1 - COST)
                trades.append(dict(ein=pos["ts"], eout=nxt["ts"],
                                   entry=pos["entry"], exit=ex,
                                   pct=(ex / pos["entry"] - 1) * 100))
                pos = None
                pend = None
    if pos is not None:   # 期末强平
        ex = bars[-1]["c"] * (1 - COST)
        trades.append(dict(ein=pos["ts"], eout=bars[-1]["ts"], entry=pos["entry"],
                           exit=ex, pct=(ex / pos["entry"] - 1) * 100))
    return trades


def metrics(trades):
    if not trades:
        return dict(n=0)
    eq = 1.0
    peakv, mdd = 1.0, 0.0
    for t in trades:
        eq *= (1 + t["pct"] / 100)
        peakv = max(peakv, eq)
        mdd = max(mdd, 1 - eq / peakv)
    w = sum(1 for t in trades if t["pct"] > 0)
    gw = sum(t["pct"] for t in trades if t["pct"] > 0)
    gl = -sum(t["pct"] for t in trades if t["pct"] <= 0)
    monthly = {}
    for t in trades:
        m = t["eout"].astimezone(ET).strftime("%Y-%m")
        monthly[m] = monthly.get(m, 0.0) + t["pct"]
    return dict(n=len(trades), wr=round(w / len(trades) * 100),
                ret=round((eq - 1) * 100, 1), mdd=round(mdd * 100, 1),
                pf=round(gw / gl, 2) if gl else 99.0,
                monthly={k: round(v, 1) for k, v in sorted(monthly.items())})


def load_cached_fallback():
    """token不可用时: 退回 zhaoge 缓存的 CONL 5分ALL K线 (4/15~7/17)。"""
    f = ROOT / "data" / "zhaoge_bars" / "CONL_ALL.csv"
    out = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            out.append(dict(ts=datetime.fromisoformat(r["ts"]), o=float(r["o"]),
                            h=float(r["h"]), l=float(r["l"]), c=float(r["c"])))
    return sorted(out, key=lambda x: x["ts"])


def main():
    try:
        b5 = fetch_5m()
    except Exception as e:
        print(f"⚠️ API不可用({str(e)[:60]}) → 退回缓存K线(窗口4/15~7/17)")
        b5 = load_cached_fallback()
    bars = to_10m(b5)
    print(f"CONL 5分K {len(b5)}根 → 10分K {len(bars)}根 "
          f"({bars[0]['ts'].astimezone(ET):%m-%d} ~ {bars[-1]['ts'].astimezone(ET):%m-%d})")
    bh = (bars[-1]["c"] / bars[0]["o"] - 1) * 100
    print(f"对照·买入持有: {bh:+.1f}%\n")

    grid = []
    for kind in ("SMA", "EMA"):
        for n in (10, 20, 30, 48, 60):
            for confirm in (False, True):
                m = metrics(run(bars, n, kind, confirm))
                grid.append((kind, n, confirm, m))
    print(f"{'均线':10}{'确认闸':7}{'笔数':>6}{'胜率':>6}{'累计':>9}{'最大回撤':>9}{'PF':>6}")
    print("─" * 56)
    for kind, n, confirm, m in grid:
        tag = f"{kind}{n}"
        cf = "突破" if confirm else "简单"
        if m["n"] == 0:
            print(f"{tag:10}{cf:7}{'—':>6}")
            continue
        print(f"{tag:10}{cf:7}{m['n']:>6}{m['wr']:>5}%{m['ret']:>+8.1f}%{m['mdd']:>8.1f}%{m['pf']:>6}")

    # 最优参数月度分解
    best = max(grid, key=lambda g: g[3].get("ret", -999))
    kind, n, confirm, m = best
    print(f"\n最优: {kind}{n} {'突破确认' if confirm else '简单'} → 月度%: {m['monthly']}")
    out = dict(generated=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds"),
               window=f"{START}~{END}", buyhold=round(bh, 1),
               grid=[dict(ma=f"{k}{nn}", confirm=c, **mm) for k, nn, c, mm in grid])
    (ROOT / "output" / "shishi_10f.json").write_text(json.dumps(out, ensure_ascii=False))
    print("存 → output/shishi_10f.json")


if __name__ == "__main__":
    main()
