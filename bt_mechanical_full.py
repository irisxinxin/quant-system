#!/usr/bin/env python3
"""
bt_mechanical_full.py — 机械出场回测(全5-7月), 盘中真实时序。
期权5分K: 真实(7月)优先; 清库的(5-6月)用【5分标的K + BS(entry-IV)】重建盘中期权OHLC, 保持止盈/止损真实时序。
标的9ema日线拖尾runner。⚠️BS对5-6月系统性低估~28pp(固定入场IV), 绝对值偏悲观, 看跨配置相对排序。
跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 bt_mechanical_full.py
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
    k = 2 / (period + 1); e = vals[0]; out = []
    for v in vals:
        e = v * k + e * (1 - k); out.append(e)
    return out


CONFIGS = {
    "用户提案(20/40/60+9ema×2+-60硬止损)": dict(ladder=[(.2,.2),(.4,.2),(.6,.2)], init_stop=0.4, be=False, ema_n=2),
    "用户提案改-30止损":                   dict(ladder=[(.2,.2),(.4,.2),(.6,.2)], init_stop=0.7, be=False, ema_n=2),
    "阶梯20/40/60+首档保本+9ema×2":        dict(ladder=[(.2,.2),(.4,.2),(.6,.2)], init_stop=0.7, be=True, ema_n=2),
    "手册忠实(50%@30+25%@60+首档保本+9ema×2)": dict(ladder=[(.3,.5),(.6,.25)], init_stop=0.7, be=True, ema_n=2),
    "手册忠实-9ema×1":                     dict(ladder=[(.3,.5),(.6,.25)], init_stop=0.7, be=True, ema_n=1),
    "末期+60半仓+9ema×2runner":            dict(ladder=[(.6,.5)], init_stop=0.7, be=True, ema_n=2),
    "纯9ema拖尾(-30)":                     dict(ladder=[], init_stop=0.7, be=False, ema_n=2),
    "纯TP+保本无9ema(对照)":               dict(ladder=[(.3,.5),(.6,.25)], init_stop=0.7, be=True, ema_n=0),
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
        if not (date(2026, 5, 1) <= s.expiry <= date(2026, 7, 17)): continue
        key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
        if key in seen: continue
        seen.add(key); buys.append(dict(ts=ts, sig=s))

    daily, intr5 = {}, {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 400, AdjustType.ForwardAdjust)
                daily[tk] = [(x.timestamp.date(), float(x.close)) for x in b]
            except Exception: daily[tk] = None
        return daily[tk]
    def gi5(tk, d0, d1):
        """标的5分K, 覆盖[d0,d1]。分块拉(1000上限≈13天)。"""
        key = (tk, d0, d1)
        if key in intr5: return intr5[key]
        out = []
        cur = d0
        from datetime import timedelta
        while cur <= d1:
            end = min(cur + timedelta(days=12), d1)
            try:
                b = q.history_candlesticks_by_date(f"{tk}.US", Period.Min_5, AdjustType.ForwardAdjust, cur, end)
                out += [(x.timestamp.astimezone(UTC), float(x.open), float(x.high), float(x.low), float(x.close)) for x in b]
            except Exception: pass
            cur = end + timedelta(days=1)
        out = sorted(set(out))
        intr5[key] = out
        return out

    def osi_of(s): return f"{s.ticker}{s.expiry:%y%m%d}{s.right}{int(round(s.strike*1000)):06d}.US"

    def opt5(s, entry_ts):
        """返回期权5分bar列表 [dict(ts,o,h,l,c)], 真实K优先, 否则5分标的+BS。"""
        B = E.bars(q, osi_of(s))
        if B:
            return [x for x in B if x["ts"] > entry_ts], True
        ud = gd(s.ticker)
        if not ud: return None, False
        entry_d = entry_ts.date()
        S0 = next((c for d, c in ud if d >= entry_d), None)
        if S0 is None: return None, False
        intr = max(0.0, (S0 - s.strike) if s.right == "C" else (s.strike - S0))
        if s.limit_price <= intr + 0.05: return None, False
        T0 = max(0.5, (s.expiry - entry_d).days + 0.5) / 365
        try: iv = solve_iv(s.limit_price, S0, s.strike, T0, s.right)
        except Exception: return None, False
        if iv >= 5.9: return None, False
        us = gi5(s.ticker, entry_d, s.expiry)
        if not us: return None, False
        out = []
        for (ts_, o, h, l, c) in us:
            if ts_ <= entry_ts: continue
            Tr = max(0.0007, (s.expiry - ts_.date()).days + 0.4) / 365
            hi = bs(h if s.right == "C" else l, s.strike, Tr, iv, s.right)
            lo = bs(l if s.right == "C" else h, s.strike, Tr, iv, s.right)
            out.append(dict(ts=ts_, o=bs(o, s.strike, Tr, iv, s.right), h=hi, l=lo, c=bs(c, s.strike, Tr, iv, s.right)))
        return (out, False) if out else (None, False)

    def simulate(s, entry_ts, cfg, obars):
        ud = gd(s.ticker)
        if not ud: return None
        closes = [c for _, c in ud]; e9 = ema(closes, 9)
        ema_by_day = {ud[i][0]: (closes[i], e9[i]) for i in range(len(ud))}
        entry = s.limit_price
        pos, first_trim, val = 1.0, False, 0.0
        ladder = list(cfg["ladder"]); done = [False] * len(ladder)
        ema_break = 0; last_day = None
        for x in obars:
            if pos <= 1e-9: break
            d = x["ts"].astimezone(UTC).date()
            if not first_trim and cfg["init_stop"] > 0 and x["l"] <= entry * cfg["init_stop"]:
                val += pos * entry * cfg["init_stop"]; pos = 0; break
            if first_trim and cfg["be"] and x["l"] <= entry:
                val += pos * entry; pos = 0; break
            for j, (thr, frac) in enumerate(ladder):
                if done[j] or pos <= 1e-9: continue
                if x["h"] >= entry * (1 + thr):
                    f = min(frac, pos); val += f * entry * (1 + thr); pos -= f; done[j] = True; first_trim = True
            if pos <= 1e-9: break
            if cfg["ema_n"] > 0 and d != last_day and last_day is not None and last_day in ema_by_day:
                c, e = ema_by_day[last_day]
                if c < e:
                    ema_break += 1
                    if ema_break >= cfg["ema_n"]:
                        val += pos * x["o"]; pos = 0; break
                else: ema_break = 0
            last_day = d
        if pos > 1e-9: val += pos * obars[-1]["c"]
        return val / entry - 1

    # 预取每笔的期权bar(缓存)
    prepped = []
    for b in buys:
        ob, real = opt5(b["sig"], b["ts"])
        if ob and len(ob) >= 3:
            lotto = ("lotto" in (b["sig"].size_tag or "").lower() or "scalp" in " ".join(b["sig"].raw.split()).lower() or b["sig"].expiry == b["ts"].date())
            prepped.append((b, ob, real, lotto))
    nreal = sum(1 for *_ , real, _ in [(p[0],p[1],p[2],p[3]) for p in prepped] if _)
    nreal = sum(1 for p in prepped if p[2])
    print(f"可测 {len(prepped)} 笔 (真实期权K {nreal} / BS盘中重建 {len(prepped)-nreal}) | 盘中真实时序")
    print("=" * 96)
    print(f"{'出场配置':44}{'加权':>7}{'加权(仅真实K)':>13}{'等额':>7}{'中位':>7}{'胜率':>6}{'归零':>6}")
    print("-" * 96)
    def wt(l): return 0.3333 if l else 0.5
    results = {}
    for name, cfg in CONFIGS.items():
        rows = []
        for (b, ob, real, lotto) in prepped:
            r = simulate(b["sig"], b["ts"], cfg, ob)
            if r is not None: rows.append((r, lotto, real))
        n = len(rows); W = sum(wt(l) for _, l, _ in rows)
        wavg = sum(wt(l) * r for r, l, _ in rows) / W * 100
        rl = [(r, l) for r, l, real in rows if real]
        Wr = sum(wt(l) for _, l in rl) or 1
        wreal = sum(wt(l) * r for r, l in rl) / Wr * 100 if rl else 0
        eq = sum(r for r, _, _ in rows) / n * 100
        med = st.median([r for r, _, _ in rows]) * 100
        win = sum(1 for r, _, _ in rows if r > 0) / n * 100
        zero = sum(1 for r, _, _ in rows if r <= -0.99) / n * 100
        results[name] = dict(wavg=round(wavg), wreal=round(wreal), eq=round(eq), med=round(med), win=round(win), zero=round(zero), n=n)
        print(f"{name:44}{wavg:>+6.0f}%{wreal:>+12.0f}%{eq:>+6.0f}%{med:>+6.0f}%{win:>5.0f}%{zero:>5.0f}%")
    print("-" * 96)
    print("对照 镜像跟他出场: 七月真实K 纯镜像+23% / 镜像+止损兜底+33%")
    print("⚠️ 全周期'加权'列含5-6月BS(系统性低估~28pp), 偏悲观; '加权(仅真实K)'=七月可信锚")
    json.dump(results, open(OUT / "bt_mechanical_full.json", "w"), ensure_ascii=False, indent=1)
    print("存 → output/bt_mechanical_full.json")


if __name__ == "__main__":
    main()
