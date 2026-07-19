#!/usr/bin/env python3
"""
bt_mechanical_grid.py — 机械出场全网格穷举 (二~七月), 找甜点策略。

网格维度 (~180配置):
  TP档位: (20,50)(20,60)(20,80)(30,50)(30,60)(30,80)(30,100)(40,60)(40,80)(40,100)
  减仓比例: 50%@TP1+25%@TP2(runner25%) | 33%@TP1+33%@TP2(runner34%)
  9ema拖尾: 15分×2 | 5分×2 | 5分×3(防whipsaw)
  初始止损: -40 | -45 | -50   (全程有效, 首档止盈后一律移保本)
甜点选择: 不取裸最大wavg(过拟合陷阱), 在wavg头部25%里选【最差月份wavg最高】的配置。
引擎同 bt_mechanical_v2 (盘中真实时序/硬止损全程/入场需真触及limit/%相对实际成交价):
  7月真实期权5分K; 2~6月用5分标的K+BS(entry-IV)重建 (幅度低估~20pp偏保守, 胜率较稳健)。
跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 bt_mechanical_grid.py
"""
import json, math, statistics as st
from datetime import datetime, timezone, date, timedelta
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from enrich_parser import parse_signal
from signal_history import _resolve_ambig_hist
import backtest_enrich as E
from longport.openapi import Config, QuoteContext, Period, AdjustType

UTC = timezone.utc
OUT = Path(__file__).parent / "output"
LO, HI = date(2026, 2, 1), date(2026, 7, 17)


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

    daily, i5cache = {}, {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 500, AdjustType.ForwardAdjust)
                daily[tk] = [(x.timestamp.date(), float(x.close)) for x in b]
            except Exception: daily[tk] = None
        return daily[tk]
    def gi5(tk, d0, d1):
        key = (tk, d0, d1)
        if key in i5cache: return i5cache[key]
        out = []; cur = d0
        while cur <= d1:
            end = min(cur + timedelta(days=12), d1)
            try:
                b = q.history_candlesticks_by_date(f"{tk}.US", Period.Min_5, AdjustType.ForwardAdjust, cur, end)
                out += [(x.timestamp.astimezone(UTC), float(x.open), float(x.high), float(x.low), float(x.close)) for x in b]
            except Exception: pass
            cur = end + timedelta(days=1)
        out = sorted(set(out)); i5cache[key] = out; return out

    def osi_of(s): return f"{s.ticker}{s.expiry:%y%m%d}{s.right}{int(round(s.strike*1000)):06d}.US"

    def opt5(s, entry_ts):
        B = E.bars(q, osi_of(s))
        if B: return [x for x in B if x["ts"] > entry_ts], True
        ud = gd(s.ticker)
        if not ud: return None, False
        ed = entry_ts.date(); S0 = next((c for d, c in ud if d >= ed), None)
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
        for (ts_, o, h, l, c) in us:
            if ts_ <= entry_ts: continue
            Tr = max(0.0007, (s.expiry - ts_.date()).days + 0.4) / 365
            out.append(dict(ts=ts_, o=bs(o, s.strike, Tr, iv, s.right),
                            h=bs(h if s.right == "C" else l, s.strike, Tr, iv, s.right),
                            l=bs(l if s.right == "C" else h, s.strike, Tr, iv, s.right),
                            c=bs(c, s.strike, Tr, iv, s.right)))
        return (out, False) if out else (None, False)

    def make_ema_series(u5, minutes):
        """5分标的K → 按minutes聚合的 (可用时刻, close<ema?) 序列。可用时刻=该bar结束。"""
        if minutes == 5:
            rows = [(ts_ + timedelta(minutes=5), c) for (ts_, o, h, l, c) in u5]
        else:
            buckets = {}
            for (ts_, o, h, l, c) in u5:
                b0 = ts_.replace(minute=(ts_.minute // minutes) * minutes, second=0, microsecond=0)
                buckets[b0] = (ts_, c) if b0 not in buckets or ts_ > buckets[b0][0] else buckets[b0]
            rows = [(buckets[b0][0] + timedelta(minutes=5), buckets[b0][1]) for b0 in sorted(buckets)]
        if not rows: return []
        es = ema_seq([c for _, c in rows], 9)
        return [(rows[i][0], rows[i][1] < es[i]) for i in range(len(rows))]

    # ── 预处理每笔 (与配置无关的部分全部预计算) ──
    prepped = []
    for b in buys:
        s = b["sig"]
        ob, real = opt5(s, b["ts"])
        if not ob or len(ob) < 3: continue
        entry = s.limit_price
        fi = next((i for i, x in enumerate(ob) if x["l"] <= entry), None)
        if fi is None: continue                     # 不追高, 未成交
        base = min(entry, ob[fi]["o"])
        path = ob[fi:]
        u5 = gi5(s.ticker, b["ts"].date(), s.expiry)
        if not u5: continue
        lotto = ("lotto" in (s.size_tag or "").lower() or "scalp" in " ".join(s.raw.split()).lower()
                 or s.expiry == b["ts"].date())
        prepped.append(dict(base=base, path=path, real=real, lotto=lotto, m=str(b["ts"])[:7],
                            e5=make_ema_series(u5, 5), e15=make_ema_series(u5, 15)))
    nreal = sum(1 for p in prepped if p["real"])
    bym = {}
    for p in prepped: bym[p["m"]] = bym.get(p["m"], 0) + 1
    print(f"可测 {len(prepped)} 笔 (真实K {nreal} / BS {len(prepped)-nreal}) 按月: {dict(sorted(bym.items()))}")

    def simulate(p, ladder, fracs, stop, ema_series, ema_n):
        base, path = p["base"], p["path"]
        pos, first_trim, val = 1.0, False, 0.0
        done = [False] * len(ladder)
        ebreak, ep = 0, 0
        es = p[ema_series] if ema_series else []
        for x in path:
            if pos <= 1e-9: break
            sp = base if first_trim else base * stop
            if x["o"] <= sp: val += pos * x["o"]; pos = 0; break
            if x["l"] <= sp: val += pos * sp; pos = 0; break
            for j, thr in enumerate(ladder):
                if done[j] or pos <= 1e-9: continue
                if x["h"] >= base * (1 + thr):
                    f = min(fracs[j], pos); val += f * base * (1 + thr); pos -= f
                    done[j] = True; first_trim = True
            if pos <= 1e-9: break
            if es:
                fired = False
                while ep < len(es) and es[ep][0] <= x["ts"]:
                    ebreak = ebreak + 1 if es[ep][1] else 0
                    ep += 1
                    if ebreak >= ema_n: fired = True
                if fired: val += pos * x["o"]; pos = 0; break
        if pos > 1e-9: val += pos * path[-1]["c"]
        return val / base - 1

    # ── 网格 ──
    TP_PAIRS = [(.2,.5),(.2,.6),(.2,.8),(.3,.5),(.3,.6),(.3,.8),(.3,1.0),(.4,.6),(.4,.8),(.4,1.0)]
    FRACS = [((.5,.25), "50/25"), ((.334,.333), "33/33")]
    EMAS = [("e15", 2, "15m×2"), ("e5", 2, "5m×2"), ("e5", 3, "5m×3")]
    STOPS = [(0.6, "-40"), (0.55, "-45"), (0.5, "-50")]
    def wt(l): return 0.3333 if l else 0.5

    grid = []
    def eval_cfg(name, tag, ladder, fracs, esk, en, sp):
        rows = [(simulate(p, ladder, fracs, sp, esk, en), p["lotto"], p["m"]) for p in prepped]
        n = len(rows); W = sum(wt(l) for _, l, _ in rows)
        wavg = sum(wt(l) * r for r, l, _ in rows) / W * 100
        win = sum(1 for r, _, _ in rows if r > 0) / n * 100
        med = st.median([r for r, _, _ in rows]) * 100
        monthly = {}
        for mth in sorted({m for *_, m in rows}):
            g = [(r, l) for r, l, m in rows if m == mth]
            Wm = sum(wt(l) for _, l in g)
            monthly[mth] = sum(wt(l) * r for r, l in g) / Wm * 100
        grid.append(dict(name=name, tag=tag, wavg=wavg, win=win, med=med,
                         min_m=min(monthly.values()), monthly=monthly))

    for (t1, t2) in TP_PAIRS:
        for (fr, frlab) in FRACS:
            for (esk, en, elab) in EMAS:
                for (sp, slab) in STOPS:
                    eval_cfg(f"{int(t1*100)}/{int(t2*100)} {frlab} {elab} {slab}", "2档",
                             [t1, t2], fr, esk, en, sp)
    # 单档: 只止盈一次, 其余全跑runner (用户问)
    for t in (.3, .4, .6):
        for (f, flab) in ((1/3, "⅓"), (.5, "½"), (2/3, "⅔")):
            for (sp, slab) in STOPS:
                eval_cfg(f"单档{int(t*100)}卖{flab} 15m×2 {slab}", "1档", [t], [f], "e15", 2, sp)
    # 纯runner对照 (无止盈无保本)
    for (sp, slab) in STOPS:
        eval_cfg(f"纯runner 15m×2 {slab}", "0档", [], [], "e15", 2, sp)

    grid.sort(key=lambda g: -g["wavg"])
    print(f"\n网格 {len(grid)} 配置 | 前15 (按加权收益):")
    print(f"{'配置':30}{'类':>4}{'加权':>7}{'胜率':>7}{'中位':>7}{'最差月':>8}")
    for g in grid[:15]:
        print(f"{g['name']:30}{g['tag']:>4}{g['wavg']:>+6.0f}%{g['win']:>6.0f}%{g['med']:>+6.0f}%{g['min_m']:>+7.0f}%")

    # ── 单档 vs 2档冠军 ──
    print(f"\n【单档/纯runner 全列, 按加权】(对照 2档冠军 30/60 33/33 15m×2 -50)")
    print(f"{'配置':30}{'排名':>5}{'加权':>7}{'胜率':>7}{'中位':>7}{'最差月':>8}")
    for g in [x for x in grid if x["tag"] in ("1档", "0档")]:
        print(f"{g['name']:30}#{grid.index(g)+1:>4}{g['wavg']:>+6.0f}%{g['win']:>6.0f}%{g['med']:>+6.0f}%{g['min_m']:>+7.0f}%")
    ref = next(x for x in grid if x["name"] == "30/60 33/33 15m×2 -50")
    print(f"{'30/60 33/33 15m×2 -50(冠军)':30}#{grid.index(ref)+1:>4}{ref['wavg']:>+6.0f}%{ref['win']:>6.0f}%{ref['med']:>+6.0f}%{ref['min_m']:>+7.0f}%")

    # ── 甜点: wavg头部25%里选最差月最高 ──
    top = grid[:max(8, len(grid) // 4)]
    sweet = max(top, key=lambda g: g["min_m"])
    print(f"\n{'='*74}\n🎯 甜点(头部25%里最差月最高): 【{sweet['name']}】")
    print(f"   加权{sweet['wavg']:+.0f}% 胜率{sweet['win']:.0f}% 中位{sweet['med']:+.0f}% 最差月{sweet['min_m']:+.0f}%")
    print(f"   按月加权: " + "  ".join(f"{m[-2:]}月{v:+.0f}%" for m, v in sorted(sweet["monthly"].items())))
    json.dump([{k: v for k, v in g.items() if k != 'monthly'} | {"monthly": g["monthly"]} for g in grid],
              open(OUT / "bt_mechanical_grid.json", "w"), ensure_ascii=False)
    print(f"\n⚠️ {len(grid)}配置×n={len(prepped)}同池=过拟合风险高; 甜点按稳健性(最差月)选, 前向验证为准。")
    print("存 → output/bt_mechanical_grid.json")


if __name__ == "__main__":
    main()
