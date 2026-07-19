#!/usr/bin/env python3
"""
bt_ttl_sweep.py — 入场TTL扫描: enrich喊单后价格回到限价的延迟分布 × 成交质量。
问题(用户): TTL 5分钟 vs 10 vs 20, 哪个稳妥? 晚成交是"回踩好价"还是"接刀"?

方法: 全部BUY信号(2-7月), 期权路径=真实5分K(7月)或标的5分K+BS重建。
  成交规则: 信号bar收盘≤限价×1.02→秒级成交(延迟0); 否则首个low≤限价的bar成交, 记延迟。
  出场=最终定稿: +30%卖⅓ → +60%卖⅓(武装) → runner⅓ -60%止损 → 武装后15m9ema×2 → 到期强平。
输出: ①延迟分桶(0/≤5m/5-10/10-20/20-40/>40)的胜率与均值 ②TTL四档对比。
跑法: source ~/.longport_creds.env && /usr/local/bin/python3 bt_ttl_sweep.py
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
    buys, seen = [], set()
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"]); ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        s = parse_signal(m["text"], ts.date())
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
        one = " ".join(m["text"].split())
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

    # 预处理: 每笔的 (成交延迟, base, path起点)
    prepped = []
    for b in buys:
        s = b["sig"]
        B, real = opt5(s, b["ts"])
        if not B or len(B) < 5: continue
        sig_ts = b["ts"]
        sig_bar = None
        for x in B:
            if x["ts"] <= sig_ts: sig_bar = x
            else: break
        fill_t = base = None; delay = None
        if sig_bar and sig_bar["c"] <= s.limit_price * 1.02:
            base = min(s.limit_price, sig_bar["c"]); fill_t = sig_ts; delay = 0.0
        else:
            for x in B:
                if x["ts"] <= sig_ts: continue
                if x["l"] <= s.limit_price:
                    base = min(s.limit_price, x["o"]); fill_t = x["ts"]
                    delay = (x["ts"] - sig_ts).total_seconds() / 60
                    break
        if base is None: continue     # 永不回落 = 任何TTL都不成交
        prepped.append(dict(b=b, B=B, real=real, base=base, fill_t=fill_t, delay=delay))

    def sim(p):
        s = p["b"]["sig"]; base = p["base"]
        path = [x for x in p["B"] if x["ts"] > p["fill_t"]]
        if not path: return None
        ES = ema15(s.ticker, p["b"]["ts"].date(), min(s.expiry, date(2026, 7, 18)))
        tp1, tp2, stop = base * 1.3, base * 1.6, base * 0.4
        pos, val, d1, d2, armed, eb, ep = 1.0, 0.0, False, False, False, 0, 0
        force = datetime.combine(s.expiry, dtime(15, 40), tzinfo=ET).astimezone(UTC)
        for x in path:
            if pos <= 1e-9: break
            if x["ts"] >= force: val += x["o"] * pos; pos = 0; break
            if x["l"] <= stop: val += stop * pos; pos = 0; break
            if not d1 and x["h"] >= tp1: val += tp1 / 3; pos -= 1/3; d1 = True
            if not d2 and pos > 1e-9 and x["h"] >= tp2: val += tp2 / 3; pos -= 1/3; d2 = True; armed = True
            if pos <= 1e-9: break
            if not armed and x["h"] >= base * 1.6: armed = True
            while ep < len(ES) and ES[ep][0] <= x["ts"]:
                eb = eb + 1 if ES[ep][1] else 0; ep += 1
            if armed and eb >= 2: val += x["o"] * pos; pos = 0; break
        if pos > 1e-9: val += path[-1]["c"] * pos
        return val / base - 1

    for p in prepped:
        p["ret"] = sim(p)
    prepped = [p for p in prepped if p["ret"] is not None]
    def wt(p): return 0.10 if p["b"]["zdte"] else (1/3 if p["b"]["lotto"] else 0.5)

    n = len(prepped)
    print(f"可测 {n} 笔 (真实K {sum(1 for p in prepped if p['real'])}, 含任意延迟成交)")
    # ── ① 延迟分桶 ──
    BUCKETS = [("秒级(喊价即市价)", 0, 0.01), ("≤5分", 0.01, 5), ("5-10分", 5, 10),
               ("10-20分", 10, 20), ("20-40分", 20, 40), (">40分", 40, 1e9)]
    print(f"\n① 首次触及喊价的延迟 × 成交质量 (最终定稿出场):")
    print(f"{'延迟桶':16}{'笔数':>5}{'胜率':>7}{'等额均值':>9}{'中位':>8}")
    for lab, lo, hi in BUCKETS:
        g = [p for p in prepped if lo <= p["delay"] < hi]
        if not g: print(f"{lab:16}{'0':>5}"); continue
        rets = [p["ret"] for p in g]
        print(f"{lab:16}{len(g):>5}{sum(1 for r in rets if r>0)/len(g)*100:>6.0f}%"
              f"{sum(rets)/len(g)*100:>+8.0f}%{stt.median(rets)*100:>+7.0f}%")
    # ── ② TTL 扫描 ──
    print(f"\n② TTL档位对比 (超时=跳过该单):")
    print(f"{'TTL':10}{'成交':>5}{'跳过':>5}{'胜率':>7}{'等额':>8}{'加权':>8}{'中位':>8}")
    for ttl in (5, 10, 20, 40, 99999):
        g = [p for p in prepped if p["delay"] <= ttl]
        skip = n - len(g)
        rets = [p["ret"] for p in g]
        W = sum(wt(p) for p in g)
        wavg = sum(wt(p) * p["ret"] for p in g) / W * 100 if W else 0
        lab = f"{ttl}分" if ttl < 99999 else "无限"
        print(f"{lab:10}{len(g):>5}{skip:>5}{sum(1 for r in rets if r>0)/len(g)*100:>6.0f}%"
              f"{sum(rets)/len(g)*100:>+7.0f}%{wavg:>+7.0f}%{stt.median(rets)*100:>+7.0f}%")
    print("\n注: '跳过'=喊价在TTL内没回来(不含永不回落的单, 那些任何TTL都不进)。")


if __name__ == "__main__":
    main()
