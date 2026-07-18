#!/usr/bin/env python3
"""
proxy_tp_analysis.py — 无期权K线时, 用股价路径 + Black-Scholes 重建期权理论价,
测 enrich 信号的 +60% 止盈命中率 (短时视角, 回应"期权不会一直拿着"的质疑)。

方法:
  1. 用信号喊的权利金反推隐含波动率 (S0=信号日收盘, T0=(到期-d0+0.5)/365, r=0, 二分求解)
  2. 沿每日股价路径: call 取日内 High / put 取日内 Low, T 逐日衰减, 算 BS 理论价
  3. 理论价 ≥ 1.6×权利金 → 记 TP 命中 (与实盘 TP 挂单 bar.High≥tp 的 fill 语义一致)
  4. 校准: 7月有真实期权5分K的子集, 代理判定 vs 真实 any(bar.h ≥ prem×1.6) 对照
     (2026-07-18 跑: 10/10 一致)

局限 (诚实声明):
  - IV 固定为入场时反推值, 财报后 IV crush 会让部分"命中"偏乐观
  - call 用日 High 假设期权峰值与股价日内极值同时出现 (近似)
  - "摸到+60%"≠最终盈利: 卖半后剩仓仍有止损/归零风险, 完整经济学看 bt_style_grid

输出: output/enrich_tp_proxy.json
跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 proxy_tp_analysis.py
"""
import json, math, statistics as st
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone, date

from enrich_parser import parse_signal
from longport.openapi import Config, QuoteContext, Period, AdjustType

UTC = timezone.utc
TP = 1.6
CUTOFF = date(2026, 7, 17)   # 只看已到期信号 (结局已定)


def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, vol, right):
    if T <= 0 or vol <= 0:
        return max(0.0, S - K) if right == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + 0.5 * vol * vol * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    c = S * N(d1) - K * N(d2)
    return c if right == "C" else c - S + K   # put-call parity, r=0


def solve_iv(price, S, K, T, right):
    lo, hi = 0.05, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs(S, K, T, mid, right) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def osi_of(s):
    # 不走 to_longport_symbol (它 assert kind=="BUY", 这里含 BUY_NOEXPIRY)
    return f"{s.ticker}{s.expiry:%y%m%d}{s.right}{int(round(s.strike * 1000)):06d}.US"


def main():
    q = QuoteContext(Config.from_env())
    msgs = json.load(open("output/enrich_history.json"))
    buys, seen = [], set()
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        s = parse_signal(m["text"], ts.date())
        if s.kind in ("BUY", "BUY_NOEXPIRY") and s.limit_price <= 8.0 and s.expiry <= CUTOFF:
            key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
            if key in seen:
                continue
            seen.add(key)
            buys.append(dict(ts=ts, sig=s))

    daily = {}
    def get_daily(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 300, AdjustType.ForwardAdjust)
                daily[tk] = {x.timestamp.date(): (float(x.open), float(x.high),
                                                  float(x.low), float(x.close)) for x in b}
            except Exception:
                daily[tk] = None
        return daily[tk]

    rows = []
    for b in buys:
        s = b["sig"]; d0 = b["ts"].date()
        db = get_daily(s.ticker)
        if not db:
            continue
        win = [dd for dd in sorted(db) if d0 <= dd <= s.expiry]
        if not win:
            continue
        S0 = db[win[0]][3]
        T0 = max(0.5, (s.expiry - d0).days + 0.5) / 365
        try:
            iv = solve_iv(s.limit_price, S0, s.strike, T0, s.right)
        except Exception:
            continue
        if iv >= 5.9:      # 二分打到上界 = 权利金与股价不自洽, 弃
            continue
        tp_day = None; peak = 0.0
        for i, dd in enumerate(win):
            _, h, l, _ = db[dd]
            S_best = h if s.right == "C" else l
            T_rem = max(0.25, (s.expiry - dd).days + 0.4) / 365
            theo = bs(S_best, s.strike, T_rem, iv, s.right)
            peak = max(peak, theo / s.limit_price)
            if tp_day is None and theo >= s.limit_price * TP:
                tp_day = i
        d2 = win[min(1, len(win) - 1)]
        dir2 = (db[d2][3] > S0) if s.right == "C" else (db[d2][3] < S0)
        lotto = ("lotto" in (s.size_tag or "").lower()
                 or "scalp" in " ".join(s.raw.split()).lower() or s.expiry == d0)
        rows.append(dict(m=str(d0)[:7], tk=s.ticker, right=s.right,
                         tp_hit=tp_day is not None, tp_day=tp_day,
                         tp_fast=(tp_day is not None and tp_day <= 1),
                         dir2=dir2, peak=peak, lotto=lotto, iv=round(iv, 2),
                         osi=osi_of(s), prem=s.limit_price, ts=b["ts"]))

    # ── 校准: 真实期权K线子集 ──
    import backtest_enrich as E
    agree = tot = 0; mism = []
    for r in rows:
        B = E.bars(q, r["osi"])
        if not B:
            continue
        post = [x for x in B if x["ts"] > r["ts"]]
        if len(post) < 5:
            continue
        real_hit = any(x["h"] >= r["prem"] * TP for x in post)
        tot += 1
        if real_hit == r["tp_hit"]:
            agree += 1
        else:
            mism.append(f"{r['tk']}(真{real_hit}/代理{r['tp_hit']})")
    print(f"① 代理校准 ({tot}笔有真实期权K线): 一致 {agree}/{tot}"
          + (f"  分歧: {mism}" if mism else ""))

    n = len(rows)
    print(f"\n② 短时视角 全部{n}笔 (BS重建):")
    print(f"{'月份':8}{'笔数':>4}{'次日方向':>8}{'+60%命中':>9}{'2日内命中':>9}{'峰值中位x':>9}")
    for mth in sorted({r['m'] for r in rows}):
        g = [r for r in rows if r['m'] == mth]
        print(f"{mth:8}{len(g):>4}{sum(r['dir2'] for r in g)/len(g)*100:>7.0f}%"
              f"{sum(r['tp_hit'] for r in g)/len(g)*100:>8.0f}%"
              f"{sum(r['tp_fast'] for r in g)/len(g)*100:>8.0f}%"
              f"{st.median(r['peak'] for r in g):>9.1f}")
    print(f"{'合计':8}{n:>4}{sum(r['dir2'] for r in rows)/n*100:>7.0f}%"
          f"{sum(r['tp_hit'] for r in rows)/n*100:>8.0f}%"
          f"{sum(r['tp_fast'] for r in rows)/n*100:>8.0f}%"
          f"{st.median(r['peak'] for r in rows):>9.1f}")
    for lab, g in (("lotto/scalp", [r for r in rows if r["lotto"]]),
                   ("波段单", [r for r in rows if not r["lotto"]])):
        if g:
            print(f"{lab}: {len(g)}笔 次日方向{sum(r['dir2'] for r in g)/len(g)*100:.0f}% "
                  f"+60%命中{sum(r['tp_hit'] for r in g)/len(g)*100:.0f}% "
                  f"2日内{sum(r['tp_fast'] for r in g)/len(g)*100:.0f}% "
                  f"峰值中位{st.median(r['peak'] for r in g):.1f}x")
    json.dump([{k: (str(v) if isinstance(v, datetime) else v) for k, v in r.items()}
               for r in rows],
              open("output/enrich_tp_proxy.json", "w"), ensure_ascii=False)
    print("\n存 → output/enrich_tp_proxy.json")


if __name__ == "__main__":
    main()
