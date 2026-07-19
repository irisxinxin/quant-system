#!/usr/bin/env python3
"""
proxy_ladder.py — 阶梯止盈 vs 跟单 的大样本代理验证 (135单, BS重建路径)。
补 bt_ladder.py(真实K仅n=9)的统计力不足。方法同 proxy_tp_analysis: 反推IV+沿股价路径重建理论价。

每笔归一为3等份, 按日粒度走理论价路径, 每份独立结算:
  机械止盈档: 某日 best_theo(call取日High/put取日Low) ≥ prem×倍数 → 该份按(倍数-1)结算
  runner止损: 某日 worst_theo ≤ prem×止损 → 剩余份按(止损-1)结算 (保守: 日内先于止盈)
  到期: 未平仓份按 内在价值/prem-1 结算 (T→0)
  组合收益 = 三份收益均值

⚠️代理局限 (诚实声明, 会系统性影响结论方向):
  · runner用"持有到期"结算 = 悲观 (站长真实出场通常优于到期归零) → 低估纯跟单/runner档
  · IV固定入场值 → 财报IV crush后偏乐观 (高估止盈命中)
  · 日High/Low近似期权日内极值
  两个偏差方向相反, 但都指向"机械止盈>持有"; 结论只看跨配置相对排序, 不看绝对值。
跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 proxy_ladder.py
"""
import json, math, statistics as st
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone, date
from enrich_parser import parse_signal
from longport.openapi import Config, QuoteContext, Period, AdjustType

UTC = timezone.utc
CUTOFF = date(2026, 7, 17)


def N(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def bs(S, K, T, v, r):
    if T <= 0 or v <= 0:
        return max(0.0, S - K) if r == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + 0.5 * v * v * T) / (v * math.sqrt(T)); d2 = d1 - v * math.sqrt(T)
    c = S * N(d1) - K * N(d2); return c if r == "C" else c - S + K
def solve_iv(px, S, K, T, r):
    lo, hi = 0.05, 6.0
    for _ in range(60):
        m = (lo + hi) / 2
        if bs(S, K, T, m, r) < px: lo = m
        else: hi = m
    return (lo + hi) / 2


CONFIGS = [
    ("纯跟单(持到期)", [],            None),
    ("纯跟单+50兜底",  [],            0.5),
    ("单档+60",        [1.6],         0.5),
    ("阶梯40/60",      [1.4, 1.6],    0.5),
    ("阶梯40/60/80",   [1.4, 1.6, 1.8], 0.5),
    ("阶梯40/60/80+紧30", [1.4, 1.6, 1.8], 0.7),
]


def eval_cfg(path_theo, prem, peels, rstop):
    """path_theo: [(best, worst, expiry_intrinsic_flag)] 逐日; 末日 worst=内在价值。返回组合收益率(小数)。"""
    thirds = 3
    peel_ret = []              # 已止盈份的收益
    peels_left = list(peels)
    remain = thirds
    stopped = None
    for i, (best, worst) in enumerate(path_theo):
        if remain <= 0:
            break
        # 止损先 (保守)
        if rstop and worst <= prem * rstop:
            for _ in range(remain):
                peel_ret.append(rstop - 1)
            remain = 0; break
        # 止盈档 (可连破多档, 每档1份)
        newp = []
        for p in peels_left:
            if remain <= 0:
                newp.append(p); continue
            if best >= prem * p:
                peel_ret.append(p - 1); remain -= 1
            else:
                newp.append(p)
        peels_left = newp
    # 到期剩余份: 末日内在价值
    if remain > 0:
        exp_intrinsic = path_theo[-1][1] if path_theo else 0.0
        for _ in range(remain):
            peel_ret.append(exp_intrinsic / prem - 1)
    return sum(peel_ret) / thirds


def main():
    q = QuoteContext(Config.from_env())
    msgs = json.load(open("output/enrich_history.json"))
    buys, seen = [], set()
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"]); ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        s = parse_signal(m["text"], ts.date())
        if s.kind in ("BUY", "BUY_NOEXPIRY") and s.limit_price <= 8.0 and s.expiry <= CUTOFF:
            key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
            if key in seen: continue
            seen.add(key); buys.append(dict(ts=ts, sig=s))

    daily = {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 300, AdjustType.ForwardAdjust)
                daily[tk] = {x.timestamp.date(): (float(x.open), float(x.high), float(x.low), float(x.close)) for x in b}
            except Exception: daily[tk] = None
        return daily[tk]

    rows = []
    for b in buys:
        s = b["sig"]; d0 = b["ts"].date(); db = gd(s.ticker)
        if not db: continue
        win = [dd for dd in sorted(db) if d0 <= dd <= s.expiry]
        if not win: continue
        S0 = db[win[0]][3]; T0 = max(0.5, (s.expiry - d0).days + 0.5) / 365
        try: iv = solve_iv(s.limit_price, S0, s.strike, T0, s.right)
        except Exception: continue
        if iv >= 5.9: continue
        path = []
        for i, dd in enumerate(win):
            _, h, l, c = db[dd]
            T_rem = max(0.25, (s.expiry - dd).days + 0.4) / 365
            is_last = (dd == win[-1])
            if is_last:
                # 末日: best用高点理论价, worst用到期内在价值
                Sb = h if s.right == "C" else l
                best = bs(Sb, s.strike, 1e-6, iv, s.right) if False else bs(Sb, s.strike, T_rem, iv, s.right)
                worst = max(0.0, (c - s.strike) if s.right == "C" else (s.strike - c))
            else:
                Sb = h if s.right == "C" else l
                Sw = l if s.right == "C" else h
                best = bs(Sb, s.strike, T_rem, iv, s.right)
                worst = bs(Sw, s.strike, T_rem, iv, s.right)
            path.append((best, worst))
        lotto = ("lotto" in (s.size_tag or "").lower()
                 or "scalp" in " ".join(s.raw.split()).lower() or s.expiry == d0)
        rows.append(dict(path=path, prem=s.limit_price, m=str(d0)[:7], lotto=lotto))

    print(f"大样本代理: {len(rows)} 笔已到期单 (BS重建, 归一3份)")
    print("=" * 72)
    print(f"{'配置':20}{'平均收益率':>12}{'胜率':>8}{'中位收益':>10}{'最差':>9}")
    print("-" * 72)
    out = {}
    for name, peels, rstop in CONFIGS:
        rets = [eval_cfg(r["path"], r["prem"], peels, rstop) for r in rows]
        avg = sum(rets) / len(rets); win = sum(1 for x in rets if x > 0) / len(rets)
        out[name] = dict(avg=avg, win=win, median=st.median(rets), worst=min(rets))
        print(f"{name:20}{avg*100:>+11.1f}%{win*100:>7.0f}%{st.median(rets)*100:>+9.1f}%{min(rets)*100:>+8.0f}%")
    print("-" * 72)
    # lotto/波段分档看阶梯40/60/80+紧30
    for lab, sub in (("lotto/scalp", [r for r in rows if r["lotto"]]),
                     ("波段", [r for r in rows if not r["lotto"]])):
        rets = [eval_cfg(r["path"], r["prem"], [1.4, 1.6, 1.8], 0.7) for r in sub]
        print(f"  [阶梯40/60/80+紧30] {lab}: {len(sub)}笔 均{sum(rets)/len(rets)*100:+.1f}% 胜率{sum(1 for x in rets if x>0)/len(rets)*100:.0f}%")
    json.dump(out, open("output/proxy_ladder.json", "w"), ensure_ascii=False, indent=1)
    print("存 → output/proxy_ladder.json")


if __name__ == "__main__":
    main()
