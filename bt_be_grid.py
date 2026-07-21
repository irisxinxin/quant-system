#!/usr/bin/env python3
"""
bt_be_grid.py — 聚焦回测: 保本(止损移入场价) vs 不保本, × 止盈结构 × 止损宽度。

回答用户 2026-07-21 的问题: 穷举止盈/保本/runner组合找最优。现有 bt_mechanical_grid.py
已扫 210 配置但【所有配置首档止盈后都强制保本】, 没有"不保本"对照 —— 而当前线上正好是
不保本(-60%全程)。本脚本补上这个对照, 让保本/不保本在相同止盈结构下同台竞争。

引擎与 bt_mechanical_grid 完全一致(同一 simulate 逻辑, 只把保本改成可开关的 be 参数):
  · 期权价值: 7月真实5分K优先; 2-6月用【标的5分K + 入场时反解IV固定 → 逐根BS】模拟。
  · 入场: 信号后标的对应期权真跌到≤limit才成交(不追高); 止损跳空按bar开盘价(滑点)。
  · runner: 标的15分9ema连破2根平。
⚠ BS固定IV忽略vega, 脚本自身标注"低估~28pp偏悲观" → 只信【相对排序】不信绝对幅度。
⚠ n≈125同池穿多配置 = 过拟合; 看稳健性(最差月/胜率), 不追加权收益最高的那个。

首次跑拉行情存 prepped 缓存(scratchpad), 之后调网格秒级重跑。
跑法: source ~/.longport_creds.env && /usr/local/bin/python3 bt_be_grid.py
"""
import json, math, pickle, sys, os
import statistics as st
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from enrich_parser import parse_signal
from signal_history import _resolve_ambig_hist
import backtest_enrich as E

UTC = timezone.utc
OUT = Path(__file__).parent / "output"
CACHE = Path("/private/tmp/claude-501/-Users-xin-Documents-Claude-Projects-money-quant-system/"
             "aaa59d92-9f71-4554-b343-debe0d7d9278/scratchpad/prepped_be.pkl")
LO, HI = date(2026, 2, 1), date(2026, 7, 31)


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


def build_prepped():
    """拉行情, 生成每笔的 base/path/real/lotto/月份/9ema序列。与 bt_mechanical_grid 同口径。"""
    from longport.openapi import Config, QuoteContext, Period, AdjustType
    q = QuoteContext(Config.from_env())
    msgs = json.load(open(OUT / "enrich_history.json"))

    # 解析口径与 bt_mechanical_grid.py 完全一致(用 m["ts"]/m["text"], limit≤8, 歧义单同法消歧)
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

    prepped = []
    for b in buys:
        s = b["sig"]
        ob, real = opt5(s, b["ts"])
        if not ob or len(ob) < 3: continue
        entry = s.limit_price
        fi = next((i for i, x in enumerate(ob) if x["l"] <= entry), None)
        if fi is None: continue
        base = min(entry, ob[fi]["o"])
        path = ob[fi:]
        u5 = gi5(s.ticker, b["ts"].date(), s.expiry)
        if not u5: continue
        lotto = ("lotto" in (s.size_tag or "").lower()
                 or "scalp" in " ".join(s.raw.split()).lower() or s.expiry == b["ts"].date())
        prepped.append(dict(base=base, path=path, real=real, lotto=lotto,
                            m=str(b["ts"])[:7], e15=make_ema_series(u5, 15)))
    return prepped


def simulate(p, ladder, fracs, stop, ema_n, be):
    """单笔收益率。be=True: 首档止盈后止损移入场价(保本); be=False: 止损始终 base*stop。"""
    base, path = p["base"], p["path"]
    pos, first_trim, val = 1.0, False, 0.0
    done = [False] * len(ladder)
    ebreak, ep = 0, 0
    es = p["e15"]
    for x in path:
        if pos <= 1e-9: break
        sp = base if (be and first_trim) else base * stop
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
                ebreak = ebreak + 1 if es[ep][1] else 0; ep += 1
                if ebreak >= ema_n: fired = True
            if fired: val += pos * x["o"]; pos = 0; break
    if pos > 1e-9: val += pos * path[-1]["c"]
    return val / base - 1


def main():
    if CACHE.exists():
        prepped = pickle.load(open(CACHE, "rb"))
        print(f"[缓存] 载入 {len(prepped)} 笔 prepped")
    else:
        print("[拉行情] 首次运行, 生成 prepped ...")
        prepped = build_prepped()
        pickle.dump(prepped, open(CACHE, "wb"))
    nreal = sum(1 for p in prepped if p["real"])
    bym = {}
    for p in prepped: bym[p["m"]] = bym.get(p["m"], 0) + 1
    print(f"可测 {len(prepped)} 笔 (真实K {nreal} / BS {len(prepped)-nreal}) 按月 {dict(sorted(bym.items()))}\n")

    def wt(l): return 0.3333 if l else 0.5

    # 止盈结构(含用户举的4个例子: 单档30=例3, 30/50=例2, 30/60=例4, 单档30=例1同结构)
    STRUCTS = [
        ("纯runner",   [],        []),
        ("单档30卖⅓",  [.3],      [1/3]),
        ("单档30卖½",  [.3],      [.5]),
        ("单档40卖⅓",  [.4],      [1/3]),
        ("单档60卖⅓",  [.6],      [1/3]),
        ("30/50 各⅓",  [.3, .5],  [1/3, 1/3]),
        ("30/60 各⅓",  [.3, .6],  [1/3, 1/3]),
        ("30/100 各⅓", [.3, 1.0], [1/3, 1/3]),
    ]
    STOPS = [(.6, "-40"), (.5, "-50"), (.4, "-60")]
    BE = [(True, "保本"), (False, "不保本")]

    rows = []
    for sname, ladder, fracs in STRUCTS:
        for sp, slab in STOPS:
            for be, belab in BE:
                if not ladder and be:   # 纯runner无首档止盈, 保本永不触发 → 跳过重复
                    continue
                res = [(simulate(p, ladder, fracs, sp, 2, be), p["lotto"], p["m"]) for p in prepped]
                n = len(res); W = sum(wt(l) for _, l, _ in res)
                wavg = sum(wt(l) * r for r, l, _ in res) / W * 100
                win = sum(1 for r, _, _ in res if r > 0) / n * 100
                med = st.median([r for r, _, _ in res]) * 100
                monthly = {}
                for mth in sorted({m for *_, m in res}):
                    g = [(r, l) for r, l, m in res if m == mth]
                    Wm = sum(wt(l) for _, l in g)
                    monthly[mth] = sum(wt(l) * r for r, l in g) / Wm * 100
                rows.append(dict(struct=sname, stop=slab, be=belab,
                                 name=f"{sname} {slab} {belab}", wavg=wavg, win=win,
                                 med=med, min_m=min(monthly.values()), monthly=monthly))

    rows.sort(key=lambda r: -r["wavg"])
    print(f"全表 {len(rows)} 配置, 按加权收益:")
    print(f"{'配置':26}{'加权':>7}{'胜率':>7}{'中位':>7}{'最差月':>8}")
    for r in rows:
        print(f"{r['name']:26}{r['wavg']:>+6.0f}%{r['win']:>6.0f}%{r['med']:>+6.0f}%{r['min_m']:>+7.0f}%")

    # ── 保本 vs 不保本 同结构对照(止损固定-50) ──
    print(f"\n{'='*66}\n保本 vs 不保本 (止损-50, 同结构直接对比):")
    print(f"{'结构':14}{'保本加权':>9}{'不保本加权':>11}{'保本胜率':>9}{'不保本胜率':>11}{'保本最差月':>11}{'不保本最差月':>13}")
    for sname, ladder, _ in STRUCTS:
        if not ladder: continue
        b = next((r for r in rows if r["struct"] == sname and r["stop"] == "-50" and r["be"] == "保本"), None)
        nb = next((r for r in rows if r["struct"] == sname and r["stop"] == "-50" and r["be"] == "不保本"), None)
        if b and nb:
            print(f"{sname:14}{b['wavg']:>+8.0f}%{nb['wavg']:>+10.0f}%{b['win']:>8.0f}%"
                  f"{nb['win']:>10.0f}%{b['min_m']:>+10.0f}%{nb['min_m']:>+12.0f}%")

    # ── 甜点(头部1/3里最差月最高) + 线上定位 ──
    top = rows[:max(6, len(rows) // 3)]
    sweet = max(top, key=lambda r: r["min_m"])
    print(f"\n{'='*66}\n🎯 甜点(头部1/3里最差月最高): 【{sweet['name']}】")
    print(f"   加权{sweet['wavg']:+.0f}% 胜率{sweet['win']:.0f}% 中位{sweet['med']:+.0f}% 最差月{sweet['min_m']:+.0f}%")
    print(f"   按月: " + "  ".join(f"{m[-2:]}月{v:+.0f}%" for m, v in sorted(sweet['monthly'].items())))

    live = next((r for r in rows if r["struct"] == "30/60 各⅓" and r["stop"] == "-60" and r["be"] == "不保本"), None)
    if live:
        print(f"\n📍 当前线上(30/60各⅓ -60 不保本, 无保本移动): 排名#{rows.index(live)+1}/{len(rows)}")
        print(f"   加权{live['wavg']:+.0f}% 胜率{live['win']:.0f}% 中位{live['med']:+.0f}% 最差月{live['min_m']:+.0f}%")

    json.dump([{k: v for k, v in r.items()} for r in rows],
              open(OUT / "bt_be_grid.json", "w"), ensure_ascii=False, indent=0)
    print(f"\n⚠️ {len(rows)}配置×n={len(prepped)}同池=过拟合; BS偏悲观只信相对排序; 前向验证为准。")
    print("存 → output/bt_be_grid.json")


if __name__ == "__main__":
    main()
