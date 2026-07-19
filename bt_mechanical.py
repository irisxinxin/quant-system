#!/usr/bin/env python3
"""
bt_mechanical.py — "跟他进场 + 按他手册的技术规则自己出场"回测 (无AI)。

依据 Enrich 手册 (Community Manual p13/19/43/49/50):
  · 9 EMA 是所有时间框架的首要趋势线, 画在【标的股票】上 (非期权权利金, theta会扭曲EMA)
  · 出场: "9 EMA lost on the timeframe you are trading" = 出场; runner "as long as price holds above 9 EMA"
  · 止盈: 阶梯 trim into strength; "scale out to cover initial investment"; 减完首档→止损移保本(never let green go red)
  · 止损: 期权 -25%~-30% on contract value (初始); 或标的丢9ema
时间框架: 波段(expiry>入场日)=日线9ema; 0DTE/scalp=5分9ema(单独处理)。

期权估值: 真实期权5分K(7月)优先聚合到日OHLC; 清库的(5-6月)用BS(entry-IV反推)从标的OHLC重建。
可配出场规则(cfg), 供网格对比: 用户提案 vs 手册忠实版 vs 纯9ema 等。
跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 bt_mechanical.py
"""
import json, math, statistics as st
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone, date
from pathlib import Path
from enrich_parser import parse_signal
from signal_history import _resolve_ambig_hist
import backtest_enrich as E
from longport.openapi import Config, QuoteContext, Period, AdjustType

UTC = timezone.utc
LO, HI = date(2026, 5, 1), date(2026, 7, 17)
OUT = Path(__file__).parent / "output"


def Nf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def bs(S, K, T, v, r):
    if T <= 0 or v <= 0:
        return max(0.0, S - K) if r == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + 0.5 * v * v * T) / (v * math.sqrt(T)); d2 = d1 - v * math.sqrt(T)
    c = S * Nf(d1) - K * Nf(d2); return c if r == "C" else c - S + K
def solve_iv(px, S, K, T, r):
    lo, hi = 0.05, 6.0
    for _ in range(60):
        m = (lo + hi) / 2
        if bs(S, K, T, m, r) < px: lo = m
        else: hi = m
    return (lo + hi) / 2
def ema(vals, period=9):
    k = 2 / (period + 1); out = []; e = vals[0]
    for v in vals:
        e = v * k + e * (1 - k); out.append(e)
    return out


# ── 出场规则配置 ──
# ladder: [(涨幅阈值, 卖出比例)]; init_stop: 初始止损(0.7=-30%, 0=无); be_after_trim: 首档后止损移保本;
# ema_n: 标的连续几日收盘<9ema出runner(0=不用9ema); trail_after_ladder_only: 9ema只管ladder后的runner
CONFIGS = {
    "用户提案(20/40/60+9ema×2+-60硬止损)":
        dict(ladder=[(.2,.2),(.4,.2),(.6,.2)], init_stop=0.4, be_after_trim=False, ema_n=2),
    "手册忠实(先回本+首档保本+9ema×2, 初始-30)":
        dict(ladder=[(.3,.5),(.6,.25)], init_stop=0.7, be_after_trim=True, ema_n=2),
    "手册忠实-9ema×1(更快出)":
        dict(ladder=[(.3,.5),(.6,.25)], init_stop=0.7, be_after_trim=True, ema_n=1),
    "阶梯20/40/60+首档保本+9ema×2(初始-30)":
        dict(ladder=[(.2,.2),(.4,.2),(.6,.2)], init_stop=0.7, be_after_trim=True, ema_n=2),
    "纯9ema拖尾(不阶梯, 初始-30)":
        dict(ladder=[], init_stop=0.7, be_after_trim=False, ema_n=2),
    "阶梯+无9ema(纯TP+保本, 对照)":
        dict(ladder=[(.3,.5),(.6,.25)], init_stop=0.7, be_after_trim=True, ema_n=0),
}


def main():
    q = QuoteContext(Config.from_env())
    msgs = json.load(open(OUT / "enrich_history.json"))
    parsed = []
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"]); ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        parsed.append((m["id"], ts, m["text"], parse_signal(m["text"], ts.date())))
    buys, seen = [], set()
    for mid, ts, text, s in parsed:
        if s.kind in ("BUY", "BUY_NOEXPIRY") and s.limit_price <= 8.0: pass
        elif s.kind == "BUY_AMBIG" and s.limit_price <= 8.0:
            side = _resolve_ambig_hist(q, E, s, ts)
            if side is None: continue
            s.kind, s.right = "BUY", side
        else: continue
        if not (LO <= s.expiry <= HI): continue
        key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
        if key in seen: continue
        seen.add(key); buys.append(dict(ts=ts, sig=s))

    daily = {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 300, AdjustType.ForwardAdjust)
                daily[tk] = [(x.timestamp.date(), float(x.open), float(x.high), float(x.low), float(x.close)) for x in b]
            except Exception: daily[tk] = None
        return daily[tk]

    def opt_daily(osi, iv, K, right, ub):
        """返回 {date: (o,h,l,c)} 期权日OHLC。真实K优先, 否则BS从标的重建。"""
        B = E.bars(q, osi)
        if B:
            agg = {}
            for x in B:
                d = x["ts"].astimezone(UTC).date()
                if d not in agg: agg[d] = [x["o"], x["h"], x["l"], x["c"]]
                else:
                    agg[d][1] = max(agg[d][1], x["h"]); agg[d][2] = min(agg[d][2], x["l"]); agg[d][3] = x["c"]
            return {d: tuple(v) for d, v in agg.items()}, True
        # BS重建
        out = {}
        for (d, o, h, l, c) in ub:
            Tr = max(0.001, (osi_expiry(osi) - d).days + 0.4) / 365
            out[d] = (bs(o, K, Tr, iv, right), bs(h if right == "C" else l, K, Tr, iv, right),
                      bs(l if right == "C" else h, K, Tr, iv, right), bs(c, K, Tr, iv, right))
        return out, False

    def osi_expiry(osi):
        # OSI: TICKER + YYMMDD + C/P + strike
        import re
        m = re.search(r"(\d{6})[CP]", osi); return datetime.strptime("20" + m.group(1), "%Y%m%d").date()

    def simulate(s, entry_ts, cfg):
        ub = gd(s.ticker)
        if not ub: return None
        entry_d = entry_ts.date()
        idx = next((i for i, r in enumerate(ub) if r[0] >= entry_d), None)
        if idx is None: return None
        entry_d = ub[idx][0]
        closes = [r[4] for r in ub]
        ema9 = ema(closes, 9)
        iv = None
        if not E.bars(q, osi_of(s)):
            S0 = ub[idx][4]; T0 = max(0.5, (s.expiry - entry_d).days + 0.5) / 365
            intr = max(0.0, (S0 - s.strike) if s.right == "C" else (s.strike - S0))
            if s.limit_price <= intr + 0.05: return None
            try: iv = solve_iv(s.limit_price, S0, s.strike, T0, s.right)
            except Exception: return None
            if iv >= 5.9: return None
        od, real = opt_daily(osi_of(s), iv, s.strike, s.right, ub)
        entry = s.limit_price
        pos, first_trim, val = 1.0, False, 0.0
        ladder = list(cfg["ladder"]); done = [False] * len(ladder)
        ema_break = 0
        exit_days = [i for i in range(idx + 1, len(ub)) if ub[i][0] <= s.expiry]
        for i in exit_days:
            d = ub[i][0]
            if d not in od: continue
            oo, oh, ol, oc = od[d]
            # 1) 止损 (期权)
            if not first_trim and cfg["init_stop"] > 0 and ol <= entry * cfg["init_stop"]:
                val += pos * entry * cfg["init_stop"]; pos = 0; break
            if first_trim and cfg["be_after_trim"] and ol <= entry:
                val += pos * entry; pos = 0; break
            # 2) 阶梯止盈 (期权高点)
            for j, (thr, frac) in enumerate(ladder):
                if done[j] or pos <= 1e-9: continue
                if oh >= entry * (1 + thr):
                    f = min(frac, pos); val += f * entry * (1 + thr); pos -= f; done[j] = True; first_trim = True
            if pos <= 1e-9: break
            # 3) 9ema拖尾 (标的收盘)
            if cfg["ema_n"] > 0:
                if ub[i][4] < ema9[i]:
                    ema_break += 1
                    if ema_break >= cfg["ema_n"]:
                        val += pos * oc; pos = 0; break
                else:
                    ema_break = 0
        if pos > 1e-9:  # 到期
            last = od.get(ub[exit_days[-1]][0]) if exit_days else None
            val += pos * (last[3] if last else 0.0)
        return dict(ret=val / entry - 1, real=real)

    def osi_of(s): return f"{s.ticker}{s.expiry:%y%m%d}{s.right}{int(round(s.strike*1000)):06d}.US"

    # 预取 bars
    testable = []
    for b in buys:
        r = gd(b["sig"].ticker)
        if r: testable.append(b)
    print(f"可回测 {len(testable)} 笔 (5-7月, 标的日线9ema+期权真实K/BS)")
    print("=" * 96)
    print(f"{'出场配置':44}{'仓位加权':>9}{'等额':>8}{'中位':>8}{'胜率':>7}{'归零%':>7}{'最差':>7}")
    print("-" * 96)
    def wt(lotto): return 0.3333 if lotto else 0.5
    results = {}
    for name, cfg in CONFIGS.items():
        rows = []
        for b in testable:
            s = b["sig"]
            lotto = ("lotto" in (s.size_tag or "").lower() or "scalp" in " ".join(s.raw.split()).lower() or s.expiry == b["ts"].date())
            r = simulate(s, b["ts"], cfg)
            if r: rows.append(dict(ret=r["ret"], lotto=lotto))
        n = len(rows)
        W = sum(wt(r["lotto"]) for r in rows)
        wavg = sum(wt(r["lotto"]) * r["ret"] for r in rows) / W * 100
        eq = sum(r["ret"] for r in rows) / n * 100
        med = st.median([r["ret"] for r in rows]) * 100
        win = sum(1 for r in rows if r["ret"] > 0) / n * 100
        zero = sum(1 for r in rows if r["ret"] <= -0.99) / n * 100
        worst = min(r["ret"] for r in rows) * 100
        results[name] = dict(wavg=wavg, eq=eq, med=med, win=win, zero=zero, n=n)
        print(f"{name:44}{wavg:>+8.0f}%{eq:>+7.0f}%{med:>+7.0f}%{win:>6.0f}%{zero:>6.0f}%{worst:>+6.0f}%")
    print("-" * 96)
    best = max(results.items(), key=lambda kv: kv[1]["wavg"])
    print(f"\n最优(仓位加权): 【{best[0]}】 {best[1]['wavg']:+.0f}% 胜率{best[1]['win']:.0f}% 归零{best[1]['zero']:.0f}%")
    print("对照 上轮镜像跟单(跟他出场): 七月真实+33%加权 / BS 5-7月约打平")
    json.dump(results, open(OUT / "bt_mechanical.json", "w"), ensure_ascii=False, indent=1)
    print("存 → output/bt_mechanical.json")


if __name__ == "__main__":
    main()
