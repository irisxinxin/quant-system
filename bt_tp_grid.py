#!/usr/bin/env python3
"""
bt_tp_grid.py — ①enrich首次减仓时的期权涨幅分布(他的风格指纹) ②TP1×TP2网格(结构不变)。
结构固定: TTL20分成交规则 + 卖⅓@TP1 → 卖⅓@TP2(=武装线) → runner⅓ 武装后15m9ema×2 → -60%止损 → 到期强平。
跑法: source ~/.longport_creds.env && /usr/local/bin/python3 bt_tp_grid.py
"""
import json, math, re, statistics as stt
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone, date, time as dtime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from enrich_parser import parse_signal
from signal_history import _resolve_ambig_hist
import backtest_enrich as E
from longport.openapi import Config, QuoteContext, Period, AdjustType

UTC = timezone.utc; ET = ZoneInfo("America/New_York")
OUT = Path(__file__).parent / "output"


def Nf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
def bs(S, K, T, v, r):
    if T <= 0 or v <= 0: return max(0.0, S - K) if r == "C" else max(0.0, K - S)
    d1 = (math.log(S / K) + 0.5 * v * v * T) / (v * math.sqrt(T)); d2 = d1 - v * math.sqrt(T)
    c = S * Nf(d1) - K * Nf(d2); return c if r == "C" else c - S + K
def solve_iv(px, S, K, T, r):
    lo, hi = 0.05, 6.0
    for _ in range(60):
        m = (lo + hi) / 2
        if bs(S, K, T, m, r) < px: lo = m
        else: hi = m
    return (lo + hi) / 2
def ema_seq(vals, period=9):
    k = 2 / (period + 1); e = vals[0]; out = []
    for v in vals: e = v * k + e * (1 - k); out.append(e)
    return out


def main():
    q = QuoteContext(Config.from_env())
    msgs = json.load(open(OUT / "enrich_history.json"))
    parsed = []
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"]); ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        parsed.append((ts, m["text"], parse_signal(m["text"], ts.date())))
    exits = [dict(ts=ts, tk=s.ticker, lv=s.exit_level) for ts, _, s in parsed
             if s.kind == "EXIT" and s.exit_level in ("full", "partial", "vague")]

    buys, seen = [], set()
    for ts, text, s in parsed:
        if s.kind in ("BUY", "BUY_NOEXPIRY") and s.limit_price <= 8.0: pass
        elif s.kind == "BUY_AMBIG" and s.limit_price <= 8.0:
            side = _resolve_ambig_hist(q, E, s, ts)
            if side is None: continue
            s.kind, s.right = "BUY", side
        else: continue
        if not (date(2026, 2, 1) <= s.expiry <= date(2026, 7, 17)): continue
        key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
        if key in seen: continue
        seen.add(key)
        one = " ".join(text.split())
        hedge = bool(re.search(r"\bhedge\b", one, re.I))
        lotto = hedge or "lotto" in (s.size_tag or "").lower() or "scalp" in one.lower() or s.expiry == ts.date()
        buys.append(dict(ts=ts, sig=s, lotto=lotto, zdte=s.expiry == ts.date()))

    daily, i5c = {}, {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 500, AdjustType.ForwardAdjust)
                daily[tk] = [(x.timestamp.date(), float(x.close)) for x in b]
            except Exception: daily[tk] = None
        return daily[tk]
    def gi5(tk, d0, d1):
        key = (tk, d0, d1)
        if key in i5c: return i5c[key]
        out = []; cur = d0
        while cur <= d1:
            end = min(cur + timedelta(days=12), d1)
            try:
                b = q.history_candlesticks_by_date(f"{tk}.US", Period.Min_5, AdjustType.ForwardAdjust, cur, end)
                out += [(x.timestamp.astimezone(UTC), float(x.open), float(x.high), float(x.low), float(x.close)) for x in b]
            except Exception: pass
            cur = end + timedelta(days=1)
        out = sorted(set(out)); i5c[key] = out; return out
    def osi_of(s): return f"{s.ticker}{s.expiry:%y%m%d}{s.right}{int(round(s.strike*1000)):06d}.US"

    def opt5(s, ts):
        B = E.bars(q, osi_of(s))
        if B: return B, True
        ud = gd(s.ticker)
        if not ud: return None, False
        ed = ts.date(); S0 = next((c for d, c in ud if d >= ed), None)
        if S0 is None: return None, False
        intr = max(0.0, (S0 - s.strike) if s.right == "C" else (s.strike - S0))
        if s.limit_price <= intr + 0.05: return None, False
        T0 = max(0.5, (s.expiry - ed).days + 0.5) / 365
        try: iv = solve_iv(s.limit_price, S0, s.strike, T0, s.right)
        except Exception: return None, False
        if iv >= 5.9: return None, False
        us = gi5(s.ticker, ed, s.expiry)
        if not us: return None, False
        out = []
        for (t_, o, h, l, c) in us:
            Tr = max(0.0007, (s.expiry - t_.date()).days + 0.4) / 365
            out.append(dict(ts=t_, o=bs(o, s.strike, Tr, iv, s.right),
                            h=bs(h if s.right == "C" else l, s.strike, Tr, iv, s.right),
                            l=bs(l if s.right == "C" else h, s.strike, Tr, iv, s.right),
                            c=bs(c, s.strike, Tr, iv, s.right)))
        return (out, False) if out else (None, False)

    _e15 = {}
    def ema15(tk, d0, d1):
        key = (tk, d0, d1)
        if key in _e15: return _e15[key]
        u5 = gi5(tk, d0, d1); bk = {}
        for (t_, o, h, l, c) in u5:
            b0 = t_.replace(minute=(t_.minute // 15) * 15, second=0, microsecond=0)
            bk[b0] = (t_, c) if b0 not in bk or t_ > bk[b0][0] else bk[b0]
        rows = [(bk[b0][0] + timedelta(minutes=5), bk[b0][1]) for b0 in sorted(bk)]
        if not rows: _e15[key] = []; return []
        es = ema_seq([c for _, c in rows])
        res = [(rows[i][0], rows[i][1] < es[i]) for i in range(len(rows))]
        _e15[key] = res; return res

    # 预处理: TTL20分成交
    prepped = []
    for b in buys:
        s = b["sig"]
        B, real = opt5(s, b["ts"])
        if not B or len(B) < 5: continue
        sig_ts = b["ts"]; sig_bar = None
        for x in B:
            if x["ts"] <= sig_ts: sig_bar = x
            else: break
        base = fill_t = None
        if sig_bar and sig_bar["c"] <= s.limit_price * 1.02:
            base = min(s.limit_price, sig_bar["c"]); fill_t = sig_ts
        else:
            ttl_end = sig_ts + timedelta(seconds=1200)
            for x in B:
                if x["ts"] <= sig_ts: continue
                if x["ts"] > ttl_end: break
                if x["l"] <= s.limit_price:
                    base = min(s.limit_price, x["o"]); fill_t = x["ts"]; break
        if base is None: continue
        prepped.append(dict(b=b, B=B, real=real, base=base, fill_t=fill_t))
    print(f"可测 {len(prepped)} 笔 (TTL20分成交, 真实K {sum(1 for p in prepped if p['real'])})")

    # ── ① enrich首次减仓时的涨幅分布 ──
    mults = []
    for p in prepped:
        s = p["b"]["sig"]
        evs = [e for e in exits if (e["tk"] == s.ticker or e["tk"] == "*")
               and e["ts"] > p["fill_t"] and e["ts"].date() <= s.expiry]
        if not evs: continue
        first = min(evs, key=lambda e: e["ts"])
        px = None
        for x in p["B"]:
            if x["ts"] <= first["ts"]: px = x["c"]
            else: break
        if px is None: continue
        mults.append(px / p["base"])
    if mults:
        mm = sorted(mults)
        print(f"\n① enrich首次减仓(partial/vague/full)时的期权涨幅 (n={len(mults)}):")
        print(f"   分位: 25%={mm[len(mm)//4]*100-100:+.0f}%  中位={stt.median(mm)*100-100:+.0f}%  75%={mm[3*len(mm)//4]*100-100:+.0f}%")
        for lab, lo, hi in [("亏损中减", 0, 1.0), ("+0~30%", 1.0, 1.3), ("+30~60%", 1.3, 1.6),
                            ("+60~100%", 1.6, 2.0), ("+100%以上", 2.0, 99)]:
            g = [m for m in mults if lo <= m < hi]
            print(f"   {lab:10} {len(g):>3}笔 ({len(g)/len(mults)*100:.0f}%)")

    # ── ② TP1×TP2 网格 ──
    def sim(p, tp1m, tp2m):
        s = p["b"]["sig"]; base = p["base"]
        path = [x for x in p["B"] if x["ts"] > p["fill_t"]]
        if not path: return None
        ES = ema15(s.ticker, p["b"]["ts"].date(), min(s.expiry, date(2026, 7, 18)))
        tp1, tp2, stop = base * tp1m, base * tp2m, base * 0.4
        pos, val, d1, d2, armed, eb, ep = 1.0, 0.0, False, False, False, 0, 0
        force = datetime.combine(s.expiry, dtime(15, 40), tzinfo=ET).astimezone(UTC)
        for x in path:
            if pos <= 1e-9: break
            if x["ts"] >= force: val += x["o"] * pos; pos = 0; break
            if x["l"] <= stop: val += stop * pos; pos = 0; break
            if not d1 and x["h"] >= tp1: val += tp1 / 3; pos -= 1/3; d1 = True
            if not d2 and pos > 1e-9 and x["h"] >= tp2: val += tp2 / 3; pos -= 1/3; d2 = True; armed = True
            if pos <= 1e-9: break
            if not armed and x["h"] >= tp2: armed = True
            while ep < len(ES) and ES[ep][0] <= x["ts"]:
                eb = eb + 1 if ES[ep][1] else 0; ep += 1
            if armed and eb >= 2: val += x["o"] * pos; pos = 0; break
        if pos > 1e-9: val += path[-1]["c"] * pos
        return val / base - 1

    def wt(p): return 0.10 if p["b"]["zdte"] else (1/3 if p["b"]["lotto"] else 0.5)
    print(f"\n② TP1×TP2 网格 (结构/止损/9ema/TTL全部不变, 只动两档价位):")
    print(f"{'TP1/TP2':12}{'加权':>7}{'真实K':>7}{'胜率':>6}{'中位':>7}{'最差月':>8}")
    results = []
    for tp1 in (1.2, 1.25, 1.3, 1.4, 1.5):
        for tp2 in (1.5, 1.6, 1.8, 2.0, 2.5):
            if tp2 <= tp1 + 0.15: continue
            rows = [(sim(p, tp1, tp2), p) for p in prepped]
            rows = [(r, p) for r, p in rows if r is not None]
            n = len(rows); W = sum(wt(p) for _, p in rows)
            wavg = sum(wt(p) * r for r, p in rows) / W * 100
            rl = [(r, p) for r, p in rows if p["real"]]
            Wr = sum(wt(p) for _, p in rl) or 1
            wreal = sum(wt(p) * r for r, p in rl) / Wr * 100 if rl else 0
            win = sum(1 for r, _ in rows if r > 0) / n * 100
            med = stt.median([r for r, _ in rows]) * 100
            mmin = 999
            for mth in {str(p["b"]["ts"])[:7] for _, p in rows}:
                g = [(r, p) for r, p in rows if str(p["b"]["ts"])[:7] == mth]
                Wm = sum(wt(p) for _, p in g)
                mmin = min(mmin, sum(wt(p) * r for r, p in g) / Wm * 100)
            results.append(dict(tp1=tp1, tp2=tp2, wavg=wavg, wreal=wreal, win=win, med=med, mmin=mmin))
            cur = " ←现行" if (tp1, tp2) == (1.3, 1.6) else ""
            print(f"+{(tp1-1)*100:.0f}/+{(tp2-1)*100:.0f}%    {wavg:>+6.0f}%{wreal:>+6.0f}%{win:>5.0f}%{med:>+6.0f}%{mmin:>+7.0f}%{cur}")
    best = max(results, key=lambda r: r["wavg"])
    print(f"\n加权最优: +{(best['tp1']-1)*100:.0f}/+{(best['tp2']-1)*100:.0f}% (加权{best['wavg']:+.0f}% 真实K{best['wreal']:+.0f}%)")
    json.dump(results, open(OUT / "bt_tp_grid.json", "w"), ensure_ascii=False)
    print("存 → output/bt_tp_grid.json")


if __name__ == "__main__":
    main()
