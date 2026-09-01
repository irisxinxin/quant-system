#!/usr/bin/env python3
"""诗诗10F逆向 v3: 实体确认状态机 × 单线/双线网格, 对5约束锚点链打分。

锚点依据(全部出自他本人消息, ET时间):
  08-25 09:58 "又回到10F空頭"            → 此刻状态=空
  08-28 12:22 "10f在6.58賣出後 一路都賣出" → 死叉@6.58附近(实际发生8/27午后)
  08-28 12:24 "上次金叉從5.8長到6.5 然後6.5死叉到現在" → 金5.8→死6.5是一个干净周期
  08-28 12:22 "沒有金叉"                 → 死叉后至8/28 12:22无金叉
  08-31 09:37 "剛剛5.56免強要買看看 要5.56之上的實體" → 金叉刚触发@5.56, 待实体确认
约束:
  A1 金叉 08-25 09:58 ~ 08-26 14:00, px 5.70-6.15
  N1 A1后至 08-27 11:00 无死叉
  A2 死叉 08-27 11:30-14:30, px 6.40-6.70
  N2 08-27 14:30 ~ 08-28 12:20 无金叉
  A3 金叉 08-31 09:00-09:45, px 5.45-5.75
"""
import sys, csv
from datetime import datetime, date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from zoneinfo import ZoneInfo
import shishi_10f_backtest as S
from shishi_10f_v2 import ema, forcast, hongli_line

ET = ZoneInfo("America/New_York")


def load_bars():
    rows = []
    with open(S.CACHE) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(ts=datetime.fromisoformat(r["ts"]), o=float(r["o"]), h=float(r["h"]),
                             l=float(r["l"]), c=float(r["c"])))
    return S.to_10m(sorted(rows, key=lambda x: x["ts"]))


def sma(v, n):
    out, s = [], 0.0
    for i, x in enumerate(v):
        s += x
        if i >= n:
            s -= v[i - n]
        out.append(s / min(i + 1, n))
    return out


def wma(v, n):
    out = []
    for i in range(len(v)):
        k = min(i + 1, n)
        ys = v[i - k + 1:i + 1]
        den = k * (k + 1) / 2
        out.append(sum((j + 1) * y for j, y in enumerate(ys)) / den)
    return out


def line_of(bars, kind, n):
    c = [b["c"] for b in bars]
    tp = [(b["o"] + b["h"] + b["l"] + b["c"]) / 4 for b in bars]
    if kind == "sma":
        return sma(c, n)
    if kind == "ema":
        return ema(c, n)
    if kind == "wma":
        return wma(c, n)
    if kind == "ema_tp":
        return ema(tp, n)
    if kind == "hl3":
        return [(x + y + z) / 3 for x, y, z in zip(ema(tp, n), ema(tp, 2 * n), ema(tp, 4 * n))]
    if kind == "hl3fc":
        return hongli_line(bars, n)
    raise ValueError(kind)


def signals_body_state(bars, L, warm=200):
    """实体确认状态机: 实体(min(O,C),max(O,C))完全在线上方→金, 完全下方→死。只在状态翻转时出信号。"""
    out, state = [], None
    for i in range(warm, len(bars)):
        b = bars[i]
        lo, hi = min(b["o"], b["c"]), max(b["o"], b["c"])
        if lo > L[i]:
            ns = "L"
        elif hi < L[i]:
            ns = "S"
        else:
            continue
        if ns != state:
            out.append(("金" if ns == "L" else "死", i))
            state = ns
    return out


def signals_close_cross(bars, F, SL, warm=200):
    """双线交叉(经典金死叉)。"""
    out, state = [], None
    for i in range(warm, len(bars)):
        ns = "L" if F[i] > SL[i] else ("S" if F[i] < SL[i] else state)
        if ns is not None and ns != state:
            if state is not None:
                out.append(("金" if ns == "L" else "死", i))
            state = ns
    return out


def signals_two_line_body(bars, F, SL, warm=200):
    """双线交叉+实体确认: 金叉且实体在快线上方才算金; 死叉且实体在快线下方才算死。未确认的交叉不改变状态。"""
    out, state = [], None
    for i in range(warm, len(bars)):
        b = bars[i]
        lo, hi = min(b["o"], b["c"]), max(b["o"], b["c"])
        ns = None
        if F[i] > SL[i] and lo > F[i]:
            ns = "L"
        elif F[i] < SL[i] and hi < F[i]:
            ns = "S"
        if ns is not None and ns != state:
            if state is not None:
                out.append(("金" if ns == "L" else "死", i))
            state = ns
    return out


def dt(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=ET)


CONSTRAINTS = dict(
    A1=(("金",), dt(2026, 8, 25, 9, 58), dt(2026, 8, 26, 14, 0), 5.70, 6.15),
    A2=(("死",), dt(2026, 8, 27, 11, 30), dt(2026, 8, 27, 14, 30), 6.40, 6.70),
    A3=(("金",), dt(2026, 8, 31, 9, 0), dt(2026, 8, 31, 9, 45), 5.45, 5.75),
)


def score(bars, sigs):
    hits = {}
    for key, (typs, t0, t1, p0, p1) in CONSTRAINTS.items():
        hits[key] = False
        for typ, i in sigs:
            t = bars[i]["ts"].astimezone(ET)
            px = bars[i]["c"]
            if typ in typs and t0 <= t <= t1 and p0 <= px <= p1:
                hits[key] = True
    # N1: A1命中时点之后 ~ 8/27 11:00 无死叉
    n1 = True
    a1_t = None
    for typ, i in sigs:
        t = bars[i]["ts"].astimezone(ET)
        if typ == "金" and CONSTRAINTS["A1"][1] <= t <= CONSTRAINTS["A1"][2] and 5.70 <= bars[i]["c"] <= 6.15:
            a1_t = t
            break
    if a1_t:
        for typ, i in sigs:
            t = bars[i]["ts"].astimezone(ET)
            if typ == "死" and a1_t < t < dt(2026, 8, 27, 11, 0):
                n1 = False
    else:
        n1 = False
    # N2: 8/27 14:30 ~ 8/28 12:20 无金叉
    n2 = True
    for typ, i in sigs:
        t = bars[i]["ts"].astimezone(ET)
        if typ == "金" and dt(2026, 8, 27, 14, 30) <= t <= dt(2026, 8, 28, 12, 20):
            n2 = False
    hits["N1"], hits["N2"] = n1, n2
    return sum(hits.values()), hits


def rth_only(bars):
    out = []
    for b in bars:
        t = b["ts"].astimezone(ET)
        hm = t.hour * 60 + t.minute
        if 9 * 60 + 30 <= hm < 16 * 60:
            out.append(b)
    return out


def main():
    bars_full = load_bars()
    results = []
    kinds = ("sma", "ema", "wma", "ema_tp", "hl3", "hl3fc")
    for sess, bars in (("RTH", rth_only(bars_full)), ("ALL", bars_full)):
        # 单线 × 实体状态机 / 收盘穿越状态机
        for kind in kinds:
            for n in (8, 10, 12, 14, 16, 20, 24, 30):
                L = line_of(bars, kind, n)
                sigs = signals_body_state(bars, L)
                sc, hits = score(bars, sigs)
                results.append((sc, f"{sess} 单线实体 {kind}({n})", hits, sigs, bars))
        # 双线 × 经典交叉 与 × 实体确认
        pairs = ((5, 10), (5, 20), (8, 20), (8, 24), (10, 20), (10, 24), (10, 30), (12, 26))
        for kind in kinds:
            for f, s in pairs:
                F, SL = line_of(bars, kind, f), line_of(bars, kind, s)
                for nm, fn in (("经典", signals_close_cross), ("实体", signals_two_line_body)):
                    sigs = fn(bars, F, SL)
                    sc, hits = score(bars, sigs)
                    results.append((sc, f"{sess} 双线{nm} {kind}({f},{s})", hits, sigs, bars))
    results.sort(key=lambda x: (-x[0], len(x[3])))
    print(f"共 {len(results)} 个候选, Top 12:")
    for sc, name, hits, sigs, bb in results[:12]:
        d = " ".join(f"{k}{'✓' if v else '✗'}" for k, v in hits.items())
        n_aug = sum(1 for typ, i in sigs
                    if dt(2026, 8, 20, 0, 0) <= bb[i]["ts"].astimezone(ET) <= dt(2026, 9, 1, 0, 0))
        print(f"  {sc}/5  {name:32s} {d}  8月下旬信号数={n_aug}")
    print("\n== 满分/最高分构造在 8/20-8/31 的信号序列 ==")
    best_sc = results[0][0]
    shown = 0
    for sc, name, hits, sigs, bb in results:
        if sc < best_sc or shown >= 4:
            break
        print(f"\n{name} ({sc}/5):")
        for typ, i in sigs:
            t = bb[i]["ts"].astimezone(ET)
            if dt(2026, 8, 20, 0, 0) <= t <= dt(2026, 9, 1, 0, 0):
                print(f"  {typ} {t:%m-%d %a %H:%M} @{bb[i]['c']:.2f}")
        shown += 1


if __name__ == "__main__":
    main()
