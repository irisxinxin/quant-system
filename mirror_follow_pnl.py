#!/usr/bin/env python3
"""
mirror_follow_pnl.py — 纯镜像跟单("他买我买, 他卖我卖")的真实盈亏重建 (5-7月)。

回应用户: "还是跟enrich的单吧, 他出的时候咱们就出"。这是唯一尊重他edge(出场手速)的估法:
不用机械止盈, 用他的【实际出场时点】给期权估值。

两套出场解读并列 (核心对比):
  · 纯规则: 只认 enrich_parser 判出的可执行 EXIT(full/partial/vague)
  · 规则+LLM: 加上 LLM 从 EXIT/alert(多票) 和 NOISE(持仓更新/"manage 35%"/"holding 1/4") 里
              捞回的出场 (conf>=0.75) —— 这才是实盘系统真正会执行的。二者之差 = LLM层价值 + 扛到期风险的缩小。

方法:
  入场: BUY当日股票收盘=S0, 喊的权利金=entry_prem, BS反推IV (r=0)。持仓归一为1单位。
  出场: 该票(+全局scope=all, 尊重except豁免)在入场后的出场, 按bot镜像语义减仓:
        首个partial/vague→卖50%; 之后→卖剩余; full→清仓。每次按【出场当日】股价BS估值。
  未出场剩余→到期内在价值 (扛单风险敞口)。
  估值区间(诚实括号): 保守=出场日收盘; 乐观=出场日高点(call)/低点(put)(他"卖进强势")。

指标: 方向胜率(model-free) / 收益率区间(中位+胜率, 均值肥尾故弃) / 扛到期% / 归零% / 最差单。
⚠️局限: IV固定入场值(财报IV crush→乐观); 日粒度; BS近似。7月真实期权K子集另校准。
跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 mirror_follow_pnl.py
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


def build_exits(parsed, cache, universe):
    """返回 (rule_only, augmented) 两套出场列表。每项 dict(ts,ticker,level,exc=set)。"""
    rule = []
    for _, ts, _, s in parsed:
        if s.kind == "EXIT" and s.exit_level in ("full", "partial", "vague"):
            rule.append(dict(ts=ts, ticker=s.ticker, level=s.exit_level, exc=set()))
    llm = []
    for mid, ts, text, s in parsed:
        v = cache.get(str(mid))
        if not v or v.get("action") not in ("exit_full", "exit_partial") or v.get("confidence", 0) < 0.75:
            continue
        if not (s.kind == "NOISE" or (s.kind == "EXIT" and s.exit_level == "alert")):
            continue
        level = "full" if v["action"] == "exit_full" else "partial"
        tks = v.get("tickers") or ([v["ticker"]] if v.get("ticker") else [])
        exc = set(v.get("except") or [])
        if v.get("scope") == "all" or not tks:
            llm.append(dict(ts=ts, ticker="*", level=level, exc=exc))
        else:
            for tk in tks:
                if tk in universe:
                    llm.append(dict(ts=ts, ticker=tk, level=level, exc=set()))
    return rule, sorted(rule + llm, key=lambda e: e["ts"])


def applies(e, tk):
    return e["ticker"] == tk or (e["ticker"] == "*" and tk not in e["exc"])


def eval_trade(sig, d0, entry_ts, db, days, iv, all_exits):
    """按镜像语义走一遍, 返回 (ret_close, ret_high, dir_ok, bag)。"""
    # 严格入场后 (精确时间戳, 防前视: 不算买入前同日的出场)
    evs = sorted([e for e in all_exits if applies(e, sig.ticker)
                  and e["ts"] > entry_ts and e["ts"].date() <= sig.expiry],
                 key=lambda e: e["ts"])
    S0 = db[days[0]][3]
    pos, reduced, legs = 1.0, False, []
    for e in evs:
        if pos <= 1e-9:
            break
        ed = min([dd for dd in days if dd >= e["ts"].date()], default=days[-1])
        if e["level"] == "full":
            legs.append((ed, pos)); pos = 0.0
        else:  # partial/vague
            if not reduced:
                legs.append((ed, pos * 0.5)); pos *= 0.5; reduced = True
            else:
                legs.append((ed, pos)); pos = 0.0
    bag = pos > 1e-9
    if bag:
        legs.append((days[-1], pos))

    def val_at(ed, strength):
        _, h, l, c = db[ed]
        Sx = (h if sig.right == "C" else l) if strength else c
        if ed == days[-1]:
            return max(0.0, (Sx - sig.strike) if sig.right == "C" else (sig.strike - Sx))
        T_rem = max(0.001, (sig.expiry - ed).days + 0.4) / 365
        return bs(Sx, sig.strike, T_rem, iv, sig.right)

    ret_c = sum(f * val_at(ed, False) for ed, f in legs) / sig.limit_price - 1
    ret_h = sum(f * val_at(ed, True) for ed, f in legs) / sig.limit_price - 1
    exit_day = legs[0][0] if legs else days[-1]
    S_exit = db[exit_day][3]
    dir_ok = (S_exit > S0) if sig.right == "C" else (S_exit < S0)
    return ret_c, ret_h, dir_ok, bag


def report(rows, title):
    n = len(rows)
    if not n:
        print(f"\n== {title} == (无数据)"); return
    print(f"\n{'='*86}\n== {title} ==  (n={n})")
    print(f"{'月份':8}{'笔数':>4}{'方向胜率':>9}{'胜率(收>0)':>11}{'中位收益':>10}{'收益区间[收盘~强势]':>20}{'扛到期%':>9}")
    for mth in sorted({r['m'] for r in rows}):
        g = [r for r in rows if r['m'] == mth]
        gc = [r['ret_c'] for r in g]; gh = [r['ret_h'] for r in g]
        print(f"{mth:8}{len(g):>4}{sum(r['dir_ok'] for r in g)/len(g)*100:>7.0f}%"
              f"{sum(1 for x in gc if x>0)/len(g)*100:>9.0f}%"
              f"{st.median(gc)*100:>+9.0f}%"
              f"{('['+format(st.median(gc)*100,'+.0f')+'%~'+format(st.median(gh)*100,'+.0f')+'%]'):>20}"
              f"{sum(r['bag'] for r in g)/len(g)*100:>8.0f}%")
    allc = [r['ret_c'] for r in rows]; allh = [r['ret_h'] for r in rows]
    print(f"{'合计':8}{n:>4}{sum(r['dir_ok'] for r in rows)/n*100:>7.0f}%"
          f"{sum(1 for x in allc if x>0)/n*100:>9.0f}%{st.median(allc)*100:>+9.0f}%"
          f"{('['+format(st.median(allc)*100,'+.0f')+'%~'+format(st.median(allh)*100,'+.0f')+'%]'):>20}"
          f"{sum(r['bag'] for r in rows)/n*100:>8.0f}%")
    for lab, g in (("lotto/scalp", [r for r in rows if r["lotto"]]),
                   ("波段", [r for r in rows if not r["lotto"]])):
        if g:
            gc = [r['ret_c'] for r in g]
            print(f"  {lab}: {len(g)}笔 方向{sum(r['dir_ok'] for r in g)/len(g)*100:.0f}% "
                  f"胜率{sum(1 for x in gc if x>0)/len(g)*100:.0f}% 中位{st.median(gc)*100:+.0f}% "
                  f"扛到期{sum(r['bag'] for r in g)/len(g)*100:.0f}%")
    zeros = sum(1 for r in rows if r["ret_c"] <= -0.99)
    print(f"  风险: 归零(-100%) {zeros}/{n}={zeros/n*100:.0f}% | 扛到期(无出场) {sum(r['bag'] for r in rows)}/{n}")


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
            if side is None:
                continue
            s.kind, s.right = "BUY", side
        else:
            continue
        if not (LO <= s.expiry <= HI):
            continue
        key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
        if key in seen:
            continue
        seen.add(key)
        buys.append(dict(ts=ts, sig=s))
    universe = sorted({b["sig"].ticker for b in buys})
    rule_ex, aug_ex = build_exits(parsed, cache, universe)
    print(f"买入类 {len(buys)} 笔 | 出场: 纯规则 {len(rule_ex)} 条, 规则+LLM {len(aug_ex)} 条 (LLM捞回 {len(aug_ex)-len(rule_ex)})")

    daily = {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 300, AdjustType.ForwardAdjust)
                daily[tk] = {x.timestamp.date(): (float(x.open), float(x.high), float(x.low), float(x.close)) for x in b}
            except Exception: daily[tk] = None
        return daily[tk]

    rows_rule, rows_aug = [], []
    skipped_bad = []
    for b in buys:
        s = b["sig"]; d0 = b["ts"].date(); db = gd(s.ticker)
        if not db:
            continue
        days = [dd for dd in sorted(db) if d0 <= dd <= s.expiry]
        if not days:
            continue
        S0 = db[days[0]][3]; T0 = max(0.5, (s.expiry - d0).days + 0.5) / 365
        intrinsic0 = max(0.0, (S0 - s.strike) if s.right == "C" else (s.strike - S0))
        if s.limit_price <= intrinsic0 + 0.05:
            # 权利金≤内在价值 = 信号自相矛盾(误解析/他笔误, 如ARM"$73"应为$273), BS无解 → 诚实剔除
            skipped_bad.append(f"{s.ticker} ${s.strike:g}{s.right}@{s.limit_price}(内在${intrinsic0:.0f})")
            continue
        try: iv = solve_iv(s.limit_price, S0, s.strike, T0, s.right)
        except Exception: continue
        if iv >= 5.9:
            continue
        lotto = ("lotto" in (s.size_tag or "").lower()
                 or "scalp" in " ".join(s.raw.split()).lower() or s.expiry == d0)
        base = dict(m=str(d0)[:7], tk=s.ticker, lotto=lotto, prem=s.limit_price,
                    osi=f"{s.ticker}{s.expiry:%y%m%d}{s.right}{int(round(s.strike*1000)):06d}.US",
                    ts=str(b["ts"]))
        for exset, dst in ((rule_ex, rows_rule), (aug_ex, rows_aug)):
            rc, rh, dok, bag = eval_trade(s, d0, b["ts"], db, days, iv, exset)
            dst.append(dict(base, ret_c=rc, ret_h=rh, dir_ok=dok, bag=bag))

    if skipped_bad:
        print(f"⚠️ 剔除自相矛盾信号 {len(skipped_bad)} 笔(权利金≤内在价值, 误解析/笔误): {skipped_bad}")
    report(rows_rule, "纯规则出场 (没有LLM时系统只能这样)")
    report(rows_aug, "规则+LLM出场 (实盘系统真正执行的)")
    json.dump(dict(rule=rows_rule, aug=rows_aug), open(OUT / "mirror_follow_pnl.json", "w"), ensure_ascii=False)
    print("\n存 → output/mirror_follow_pnl.json")


if __name__ == "__main__":
    main()
