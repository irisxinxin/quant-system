#!/usr/bin/env python3
"""
bt_mechanical_v2.py — 机械出场回测(修3处审查发现的问题), 盘中真实时序。
修复 vs v1(bt_mechanical_full):
  ① 硬止损全程有效: 无保本(be=False)配置首档止盈后止损不再消失(v1 bug: first_trim后裸奔)
  ② 9ema时间框架可选: 日线 vs 【15分钟】——周期权2-4天到期, 日线9ema几乎不触发(审查坐实runner变裸持到期彩票),
     他手册"9 EMA on the timeframe you are trading", 短线该用盘中9ema
  ③ 入场成交现实化: 期权需在信号后真跌到≤limit才成交(镜像基线同口径), 止损跳空按bar开盘价(滑点)
期权5分K真实(7月)优先, 5-6月用5分标的+BS(低估~28pp偏悲观)。⚠️n=14真实, 只信相对排序不信绝对幅度。
跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 bt_mechanical_v2.py
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
def ema(vals, period=9):
    k = 2 / (period + 1); e = vals[0]; out = []
    for v in vals: e = v * k + e * (1 - k); out.append(e)
    return out


# 核心对比: 保护runner(保本/9ema, 高胜率但砍肥尾) vs 扛runner到期(吃GOOGL型大鱼但亏损单多)
# 卖⅓@30% 后剩⅔ runner, 变量=保本开关×9ema开关×止损宽度
# 用户定稿方向: 卖⅓@30 → runner自由跑(不保本不早9ema) → 涨到+60%/+100%后【加保护】(enrich从不扛到期)
# 保护三种: 15m9ema×2 / 追踪止损(峰值回撤X%) / 利润地板(止损抬到+X%锁死)。全配 -60%初始止损。
CONFIGS = {
    "只30卖⅓, runner⅔ arm60+9ema (我推荐)":   dict(ladder=[(.3,.334)], stop=0.4, be=False, ema_n=2, ema_tf="i15", arm_after=0.6),
    "30/60各卖⅓, runner⅓ arm60+9ema (用户)":  dict(ladder=[(.3,.334),(.6,.333)], stop=0.4, be=False, ema_n=2, ema_tf="i15", arm_after=0.6),
    "30/100各卖⅓, runner⅓ arm60+9ema":       dict(ladder=[(.3,.334),(1.0,.333)], stop=0.4, be=False, ema_n=2, ema_tf="i15", arm_after=0.6),
    "30卖⅓+60卖¼, runner约42% arm60+9ema":    dict(ladder=[(.3,.334),(.6,.25)], stop=0.4, be=False, ema_n=2, ema_tf="i15", arm_after=0.6),
}
HEADLINE = "30/60各卖⅓, runner⅓ arm60+9ema (用户)"


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
        if not (date(2026, 2, 1) <= s.expiry <= date(2026, 7, 17)): continue
        key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
        if key in seen: continue
        seen.add(key); buys.append(dict(ts=ts, sig=s))

    daily, i5 = {}, {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 400, AdjustType.ForwardAdjust)
                daily[tk] = [(x.timestamp.date(), float(x.close)) for x in b]
            except Exception: daily[tk] = None
        return daily[tk]
    def gi5(tk, d0, d1):
        key = (tk, d0, d1)
        if key in i5: return i5[key]
        out = []; cur = d0
        while cur <= d1:
            end = min(cur + timedelta(days=12), d1)
            try:
                b = q.history_candlesticks_by_date(f"{tk}.US", Period.Min_5, AdjustType.ForwardAdjust, cur, end)
                out += [(x.timestamp.astimezone(UTC), float(x.open), float(x.high), float(x.low), float(x.close)) for x in b]
            except Exception: pass
            cur = end + timedelta(days=1)
        out = sorted(set(out)); i5[key] = out; return out
    def i15_ema(u5):
        """5分→15分聚合, 返回 [(ts_close, close, ema9)] 按15分bar收盘时刻。"""
        buckets = {}
        for (ts_, o, h, l, c) in u5:
            b0 = ts_.replace(minute=(ts_.minute // 15) * 15, second=0, microsecond=0)
            buckets.setdefault(b0, []).append((ts_, c))
        rows = []
        for b0 in sorted(buckets):
            last_ts = max(buckets[b0])[0]; last_c = [c for t, c in buckets[b0] if t == max(buckets[b0])[0]][0]
            rows.append((last_ts + timedelta(minutes=5), last_c))   # 该15分bar收盘可用时刻≈末5分bar结束
        if not rows: return []
        es = ema([c for _, c in rows], 9)
        return [(rows[i][0], rows[i][1], es[i]) for i in range(len(rows))]

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

    def simulate(s, entry_ts, cfg, obars):
        entry = s.limit_price
        # ③ 入场成交现实化: 需真跌到≤limit; 首个 l<=limit 的bar成交(gap则按open)
        fi = next((i for i, x in enumerate(obars) if x["l"] <= entry), None)
        if fi is None: return ("no_fill", None)
        base = min(entry, obars[fi]["o"])   # 实际成交价=成本基准, 所有%规则相对它
        path = obars[fi:]
        # 9ema序列
        ud = gd(s.ticker)
        ema_daily = {ud[i][0]: (ud[i][1], ema([c for _, c in ud], 9)[i]) for i in range(len(ud))} if ud else {}
        i15 = i15_ema(gi5(s.ticker, entry_ts.date(), s.expiry)) if cfg["ema_tf"] == "i15" else []
        pos, first_trim, val = 1.0, False, 0.0
        ladder = list(cfg["ladder"]); done = [False] * len(ladder)
        ebreak, last_day, i15p = 0, None, 0
        arm_after = cfg.get("arm_after", 0.0)   # runner涨够 base*(1+arm_after) 才启动保护(9ema/trail/floor)
        armed = arm_after <= 0
        trail = cfg.get("trail", 0.0)           # armed后: 从峰值回撤trail比例→出 (0=off)
        floor_ = cfg.get("floor", None)         # armed后: 止损抬到 base*(1+floor) 锁利润 (None=off)
        peak_arm = 0.0
        for x in path:
            if pos <= 1e-9: break
            d = x["ts"].astimezone(UTC).date()
            # ① 止损: 未减仓=stop; 已减仓 be→保本; armed+floor→利润地板 (取最高线)
            sp = (base if (first_trim and cfg["be"]) else base * cfg["stop"]) if cfg["stop"] > 0 else 0
            if armed and floor_ is not None:
                sp = max(sp, base * (1 + floor_))
            if sp > 0:
                if x["o"] <= sp: val += pos * x["o"]; pos = 0; break     # 跳空按开盘(滑点)
                if x["l"] <= sp: val += pos * sp; pos = 0; break
            # ①b 追踪止损 (armed后, 从armed期峰值回撤trail→出)
            if armed and trail > 0 and peak_arm > 0:
                tl = peak_arm * (1 - trail)
                if tl > sp:
                    if x["o"] <= tl: val += pos * x["o"]; pos = 0; break
                    if x["l"] <= tl: val += pos * tl; pos = 0; break
            # ② 阶梯止盈 (相对实际成交价base)
            for j, (thr, frac) in enumerate(ladder):
                if done[j] or pos <= 1e-9: continue
                if x["h"] >= base * (1 + thr):
                    f = min(frac, pos); val += f * base * (1 + thr); pos -= f; done[j] = True; first_trim = True
            if pos <= 1e-9: break
            # ③ 9ema拖尾 (arm_after: runner涨够阈值才启动, 防早盘回踩把肥尾洗掉)
            if not armed and x["h"] >= base * (1 + arm_after):
                armed = True
            if armed:
                peak_arm = max(peak_arm, x["h"])   # armed期峰值(追踪止损锚)
            if cfg["ema_n"] > 0:
                if cfg["ema_tf"] == "i15":              # 每根bar推进指针+更新连破态(不论armed)
                    while i15p < len(i15) and i15[i15p][0] <= x["ts"]:
                        _, c, e = i15[i15p]; i15p += 1
                        ebreak = ebreak + 1 if c < e else 0
                    if armed and ebreak >= cfg["ema_n"]: val += pos * x["o"]; pos = 0; break
                elif cfg["ema_tf"] == "daily" and d != last_day and last_day is not None and last_day in ema_daily:
                    c, e = ema_daily[last_day]
                    ebreak = ebreak + 1 if c < e else 0
                    if armed and ebreak >= cfg["ema_n"]: val += pos * x["o"]; pos = 0; break
            last_day = d
        if pos > 1e-9: val += pos * path[-1]["c"]
        return ("traded", val / base - 1)   # 收益相对实际成交价

    prepped = []
    for b in buys:
        ob, real = opt5(b["sig"], b["ts"])
        if ob and len(ob) >= 3:
            lotto = ("lotto" in (b["sig"].size_tag or "").lower() or "scalp" in " ".join(b["sig"].raw.split()).lower() or b["sig"].expiry == b["ts"].date())
            prepped.append((b, ob, real, lotto))
    nreal = sum(1 for p in prepped if p[2])
    print(f"可测 {len(prepped)} 笔 (真实K {nreal} / BS {len(prepped)-nreal}) | 修: 硬止损全程+15分9ema+入场现实化")
    print("=" * 100)
    print(f"{'出场配置':40}{'加权':>7}{'真实K':>7}{'中位':>7}{'胜率':>6}{'最差月':>7}{'归零':>6}")
    print("-" * 100)
    def wt(l): return 0.3333 if l else 0.5
    results = {}; rows_by_cfg = {}
    for name, cfg in CONFIGS.items():
        rows, nf = [], 0
        for (b, ob, real, lotto) in prepped:
            st_, r = simulate(b["sig"], b["ts"], cfg, ob)
            if st_ == "no_fill": nf += 1; continue
            rows.append((r, lotto, real, str(b["ts"])[:7]))
        rows_by_cfg[name] = rows
        n = len(rows); W = sum(wt(l) for _, l, _, _ in rows) or 1
        wavg = sum(wt(l) * r for r, l, _, _ in rows) / W * 100
        rl = [(r, l) for r, l, real, _ in rows if real]; Wr = sum(wt(l) for _, l in rl) or 1
        wreal = sum(wt(l) * r for r, l in rl) / Wr * 100 if rl else 0
        med = st.median([r for r, _, _, _ in rows]) * 100
        win = sum(1 for r, _, _, _ in rows if r > 0) / n * 100
        zero = sum(1 for r, _, _, _ in rows if r <= -0.99) / n * 100
        mmin = 999
        for mth in {m for *_, m in rows}:
            g = [(r, l) for r, l, _, m in rows if m == mth]
            Wm = sum(wt(l) for _, l in g) or 1
            mmin = min(mmin, sum(wt(l) * r for r, l in g) / Wm * 100)
        results[name] = dict(wavg=round(wavg), wreal=round(wreal), med=round(med), win=round(win), zero=round(zero), worst_m=round(mmin), n=n, nf=nf)
        print(f"{name:40}{wavg:>+6.0f}%{wreal:>+6.0f}%{med:>+6.0f}%{win:>5.0f}%{mmin:>+6.0f}%{zero:>5.0f}%")
    print("-" * 100)
    # ── 四五六月逐月 胜率+加权收益 (用户问) ──
    print(f"\n【四/五/六月 逐月: 胜率 | 加权收益】")
    print(f"{'配置':40}" + "".join(f"{m:>18}" for m in ["4月", "5月", "6月"]))
    for name in CONFIGS:
        hr = rows_by_cfg[name]
        cells = []
        for mm in ("2026-04", "2026-05", "2026-06"):
            g = [(r, l) for r, l, _, m in hr if m == mm]
            if g:
                n = len(g); W = sum(wt(l) for _, l in g) or 1
                cells.append(f"{sum(1 for r,_ in g if r>0)/n*100:>3.0f}%|{sum(wt(l)*r for r,l in g)/W*100:>+6.0f}%")
            else:
                cells.append("—")
        print(f"{name:40}" + "".join(f"{c:>18}" for c in cells))
    print("\n对照 镜像跟他出场(真实K七月): 纯镜像+23% / 镜像+止损兜底+33%")
    print("⚠️ 5-6月用盘中股价BS重建(样本量足但对幅度低估~20pp); 胜率对BS较稳健。7月含真实期权K。")
    json.dump(results, open(OUT / "bt_mechanical_v2.json", "w"), ensure_ascii=False, indent=1)
    print("存 → output/bt_mechanical_v2.json")


if __name__ == "__main__":
    main()
