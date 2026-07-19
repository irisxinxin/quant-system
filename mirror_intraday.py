#!/usr/bin/env python3
"""
mirror_intraday.py — 把镜像跟单盈亏的估值精确到【他喊出场那一刻的盘中5分钟股价】,
收窄 mirror_follow_pnl 的 [收盘估~强势估] 宽区间为一个可操作的点估计。

对每条出场腿, 取该股当日5分K里覆盖出场时间戳的那根bar的收盘价 (=跟单者收到alert时的大致成交价),
按BS(entry-IV, 衰减T)给期权估值。剩余扛到期→到期日收盘内在价值。
用规则+LLM出场集(实盘系统真正执行的)。加可选滑点。
跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 mirror_intraday.py
"""
import json, statistics as st
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone, date
from pathlib import Path
from enrich_parser import parse_signal
from signal_history import _resolve_ambig_hist
import backtest_enrich as E
from mirror_follow_pnl import bs, solve_iv, build_exits, applies
from longport.openapi import Config, QuoteContext, Period, AdjustType

UTC = timezone.utc
LO, HI = date(2026, 5, 1), date(2026, 7, 17)
OUT = Path(__file__).parent / "output"
SLIP = 0.0   # 可设滑点(如0.01=晚1%), 保守起见先0


def main():
    q = QuoteContext(Config.from_env())
    msgs = json.load(open(OUT / "enrich_history.json"))
    cache = json.loads((OUT / "llm_ab_cache.json").read_text())
    parsed = []
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"]); ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        parsed.append((m["id"], ts, m["text"], parse_signal(m["text"], ts.date())))

    buys, seen = [], set()
    for mid, ts, text, s in parsed:
        if s.kind in ("BUY", "BUY_NOEXPIRY") and s.limit_price <= 8.0:
            pass
        elif s.kind == "BUY_AMBIG" and s.limit_price <= 8.0:
            side = _resolve_ambig_hist(q, E, s, ts)
            if side is None: continue
            s.kind, s.right = "BUY", side
        else:
            continue
        if not (LO <= s.expiry <= HI): continue
        key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
        if key in seen: continue
        seen.add(key); buys.append(dict(ts=ts, sig=s))
    universe = sorted({b["sig"].ticker for b in buys})
    _, aug_ex = build_exits(parsed, cache, universe)

    daily, intr = {}, {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 300, AdjustType.ForwardAdjust)
                daily[tk] = {x.timestamp.date(): (float(x.open), float(x.high), float(x.low), float(x.close)) for x in b}
            except Exception: daily[tk] = None
        return daily[tk]
    def gi(tk, d):
        k = (tk, d)
        if k not in intr:
            try:
                b = q.history_candlesticks_by_date(f"{tk}.US", Period.Min_5, AdjustType.ForwardAdjust, d, d)
                intr[k] = sorted([(x.timestamp.astimezone(UTC), float(x.close)) for x in b])
            except Exception: intr[k] = []
        return intr[k]
    def stock_at(tk, ts):
        """他喊出场时间戳对应的盘中股价; 无盘中数据→当日收盘。"""
        bars = gi(tk, ts.date())
        if not bars:
            db = gd(tk); return db[ts.date()][3] if db and ts.date() in db else None
        px = bars[0][1]
        for bt, c in bars:
            if bt <= ts: px = c
            else: break
        return px

    rows = []
    for b in buys:
        s = b["sig"]; d0 = b["ts"].date(); db = gd(s.ticker)
        if not db: continue
        days = [dd for dd in sorted(db) if d0 <= dd <= s.expiry]
        if not days: continue
        S0 = db[days[0]][3]
        intrinsic0 = max(0.0, (S0 - s.strike) if s.right == "C" else (s.strike - S0))
        if s.limit_price <= intrinsic0 + 0.05:   # 自相矛盾信号剔除(同mirror_follow)
            continue
        T0 = max(0.5, (s.expiry - d0).days + 0.5) / 365
        try: iv = solve_iv(s.limit_price, S0, s.strike, T0, s.right)
        except Exception: continue
        if iv >= 5.9: continue
        evs = sorted([e for e in aug_ex if applies(e, s.ticker)
                      and e["ts"] > b["ts"] and e["ts"].date() <= s.expiry], key=lambda e: e["ts"])
        pos, reduced, legs = 1.0, False, []
        for e in evs:
            if pos <= 1e-9: break
            if e["level"] == "full":
                legs.append((e["ts"], pos)); pos = 0.0
            else:
                if not reduced: legs.append((e["ts"], pos * 0.5)); pos *= 0.5; reduced = True
                else: legs.append((e["ts"], pos)); pos = 0.0
        bag = pos > 1e-9
        # 估值
        val = 0.0
        for ets, frac in legs:
            Sx = stock_at(s.ticker, ets)
            if Sx is None: Sx = db[days[-1]][3]
            ed = min([dd for dd in days if dd >= ets.date()], default=days[-1])
            T_rem = max(0.001, (s.expiry - ed).days + 0.4) / 365
            theo = bs(Sx * (1 - SLIP if s.right == "C" else 1 + SLIP), s.strike, T_rem, iv, s.right)
            val += frac * theo
        if bag:
            Sx = db[days[-1]][3]
            val += pos * max(0.0, (Sx - s.strike) if s.right == "C" else (s.strike - Sx))
        ret = val / s.limit_price - 1
        lotto = ("lotto" in (s.size_tag or "").lower()
                 or "scalp" in " ".join(s.raw.split()).lower() or s.expiry == d0)
        rows.append(dict(m=str(d0)[:7], tk=s.ticker, ret=ret, bag=bag, lotto=lotto, prem=s.limit_price))

    n = len(rows)
    def wt(r): return 0.3333 if r["lotto"] else 0.5
    def wavg(sub):
        W = sum(wt(r) for r in sub); return sum(wt(r) * r["ret"] for r in sub) / W * 100 if W else 0
    print(f"盘中精确估值 (规则+LLM出场, 他喊单时刻5分K股价) | n={n}")
    print("=" * 72)
    print(f"{'月份':10}{'笔':>4}{'仓位加权收益':>12}{'中位':>9}{'胜率':>8}{'归零%':>7}{'扛到期%':>9}")
    for m in sorted({r['m'] for r in rows}):
        g = [r for r in rows if r['m'] == m]
        c = [r['ret'] for r in g]
        print(f"{m:10}{len(g):>4}{wavg(g):>+11.0f}%{st.median(c)*100:>+8.0f}%"
              f"{sum(1 for x in c if x>0)/len(g)*100:>7.0f}%"
              f"{sum(1 for r in g if r['ret']<=-0.99)/len(g)*100:>6.0f}%"
              f"{sum(r['bag'] for r in g)/len(g)*100:>8.0f}%")
    allc = [r['ret'] for r in rows]
    print("-" * 72)
    print(f"{'合计':10}{n:>4}{wavg(rows):>+11.0f}%{st.median(allc)*100:>+8.0f}%"
          f"{sum(1 for x in allc if x>0)/n*100:>7.0f}%"
          f"{sum(1 for r in rows if r['ret']<=-0.99)/n*100:>6.0f}%{sum(r['bag'] for r in rows)/n*100:>8.0f}%")
    for lab, g in (("波段", [r for r in rows if not r["lotto"]]), ("lotto", [r for r in rows if r["lotto"]])):
        if g:
            c = [r['ret'] for r in g]
            print(f"  {lab}: {len(g)}笔 加权{wavg(g):+.0f}% 中位{st.median(c)*100:+.0f}% "
                  f"胜率{sum(1 for x in c if x>0)/len(g)*100:.0f}%")
    c2 = sorted(allc, reverse=True)[1:]
    print(f"\n等额均值 {sum(allc)/n*100:+.0f}% | 剔最大赢家后 {sum(c2)/len(c2)*100:+.0f}% "
          f"| 前3赢家占正收益 {sum(sorted([x for x in allc if x>0],reverse=True)[:3])/sum(x for x in allc if x>0)*100 if any(x>0 for x in allc) else 0:.0f}%")
    json.dump(rows, open(OUT / "mirror_intraday.json", "w"), ensure_ascii=False)
    print("存 → output/mirror_intraday.json")


if __name__ == "__main__":
    main()
