#!/usr/bin/env python3
"""
backtest_resonance.py — 「多源共振是否真的更好」的控制变量回测。

设计思路(为什么这么做):
  直接比较"共振票 vs 非共振票"会混入档位质量差异 —— 不同源的档位精细度天差地别。
  所以这里**控制档位来源**: 全部用蛋挞 8/10-8/14 点位表(四件套最完整、已人工核验),
  只按"该票在发布日前后是否被其他源同期提及"分组, 测共振的**增量**价值。

口径(fill-realistic, 遵循 feedback_realistic_fill):
  · 入场 = 发布日起挂限价在第一入场区上沿, **必须 Low<=限价** 才算成交(不是 High>=)
  · 成交价 = min(限价, 当日开盘) —— 跳空低开按开盘价给, 不给不可能的价
  · 出场 = 止损/TP1 日线赛跑; 同一根bar同时触及按**止损优先**(保守)
  · 未触发则持有到最后交易日按收盘 mark
  · 基准 = 同期 QQQ(从成交日算起)

共振判定(避免事后偏差):
  只用**发布日 T 之前 10 天 ~ T 之后 2 天**窗口内的归档消息判定,
  即"当时就能知道有几个源在谈这只票", 不用后来才出现的信息。
"""
import json, os, re, glob, sys, statistics as st
from datetime import datetime, timedelta, timezone
from longport.openapi import Config, QuoteContext, Period, AdjustType

# ── 蛋挞点位表: 发布日 → 票 (取自 output/danta_vip_history.json 原始时间戳) ──
TABLE_DATE = {
    "2026-08-10": ["PLTR", "NOW", "CRWD", "PANW", "SNOW", "PATH", "NET", "DDOG", "HPQ",
                   "RZLV", "PDYN", "BBAI", "GRRR", "SOUN", "AUR", "MITK",
                   "ASTS", "RKLB", "RDW", "LUNR", "OPTX"],
    "2026-08-11": ["FCX", "STLD", "CF", "MOS", "NUE"],
    "2026-08-12": ["INTC", "SOFI", "HOOD", "NFLX"],
    "2026-08-13": ["AMAT", "LRCX", "TER", "ENTG", "KLAC", "ICHR", "COHR", "CELH"],
    "2026-08-14": ["ON", "PENG", "OUST", "CRWV", "VELO", "OSS", "TEM", "AAOI", "CIFR"],
}
# 票: (第一入场区上沿, 下沿, 止损, TP1) —— 全部来自图片点位表转录
ZONES = {
    "PLTR": (162, 158, 145, 185), "NOW": (94, 91, None, 108), "CRWD": (202, 198, 188, 225),
    "PANW": (308, 300, None, 335), "SNOW": (252, 246, None, 280), "PATH": (13.3, 12.8, 11.8, 16),
    "NET": (288, 280, 265, 320), "DDOG": (247, 240, 228, 280), "HPQ": (27.2, 26.4, None, 29.1),
    "RZLV": (2.60, 2.45, 2.20, 3.00), "PDYN": (5.1, 4.8, 4.4, 6.2), "BBAI": (3.15, 3.05, 2.75, 3.80),
    "GRRR": (11.0, 10.5, None, 13), "SOUN": (7.9, 7.5, 6.9, 9.5), "AUR": (5.8, 5.5, 5.1, 6.8),
    "MITK": (18.2, 17.5, 16.5, 20.5),
    "ASTS": (69, 67, 59, 75), "RKLB": (81, 78, 70, 88), "RDW": (13.0, 12.5, 10.8, 15),
    "LUNR": (16.0, 15.5, 13.6, 18.5), "OPTX": (7.55, 7.25, 7.10, 7.75),
    "FCX": (66, 64, 54, 72), "STLD": (258, 252, 222, 285), "CF": (115, 112, 94, 130),
    "MOS": (22.3, 21.8, 18, 25.5), "NUE": (267, 263, 235, 285),
    "INTC": (95, 90, None, 100), "SOFI": (17.5, 16, None, 19), "HOOD": (92, 85, None, 108),
    "NFLX": (72, 69, None, 78),
    "AMAT": (530, 510, 450, 575), "LRCX": (315, 305, 268, 350), "TER": (328, 320, 282.3, 372),
    "ENTG": (152, 145, 125, 175), "KLAC": (198, 190, 168, 220), "ICHR": (67, 63, 51, 78),
    "COHR": (330, 325, None, 380), "CELH": (25, 24, None, 27.8),
    "ON": (89.2, 87.5, 79.8, 97.8), "PENG": (63, 60, 51.6, 74), "OUST": (38.0, 36.5, 30.8, 45.3),
    "CRWV": (77, 74.5, 64.2, 91), "VELO": (11.6, 11.0, 9.1, 14.1), "OSS": (12.6, 12.0, 9.9, 15.2),
    "TEM": (46.3, 45.1, 39.6, 53.4), "AAOI": (111, 107, 105, 124), "CIFR": (16.4, 16.0, None, 17.0),
}
# 归档文件 → 源名 (蛋挞自己的两个频道排除, 避免自己给自己算共振)
SRC = {"xiaoyu_vip_history.json": "小鱼", "seek_vip_history.json": "Seeker",
       "kova_signal_history.json": "Kova", "qianli_duo_history.json": "形态多",
       "samlam_history.json": "Sam", "suoya_history.json": "索亚",
       "biancheng_history.json": "边城", "tangzhuren_history.json": "唐主任",
       "tiange_trades_history.json": "天哥", "tiange_radar_history.json": "天哥",
       "abtrades_history.json": "AbTrades", "wallst_history.json": "华尔街"}

ctx = QuoteContext(Config.from_env())
_cache = {}


def bars(sym):
    """长桥日线, 返回 [(date, open, high, low, close)]"""
    if sym in _cache:
        return _cache[sym]
    out = []
    try:
        for b in ctx.candlesticks(sym + ".US", Period.Day, 40, AdjustType.NoAdjust):
            out.append((str(b.timestamp.date()), float(b.open), float(b.high),
                        float(b.low), float(b.close)))
    except Exception as e:
        print(f"  ⚠️ {sym}: {type(e).__name__}", file=sys.stderr)
    _cache[sym] = out
    return out


def resonance(tk, pub):
    """发布日 pub 时, 有几个**其他**源在窗口(T-10 ~ T+2)内谈过这只票 —— 无事后偏差"""
    lo = (datetime.fromisoformat(pub) - timedelta(days=10)).isoformat()
    hi = (datetime.fromisoformat(pub) + timedelta(days=2)).isoformat()
    hit = set()
    pat = re.compile(rf"\b{tk}\b", re.I)
    for fn, name in SRC.items():
        p = "output/" + fn
        if not os.path.exists(p):
            continue
        try:
            msgs = json.load(open(p))
        except Exception:
            continue
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if lo <= m["ts"] <= hi and pat.search(m.get("text", "")):
                hit.add(name)
                break
    return hit


def run(tk, pub):
    hi_, lo_, stop, tp = ZONES[tk]
    B = bars(tk)
    if not B:
        return None
    idx = next((i for i, b in enumerate(B) if b[0] >= pub), None)
    if idx is None:
        return None
    last = B[-1]
    # 入场: Low<=限价才成交, 成交价=min(限价, 开盘)
    ent = edate = ei = None
    for j in range(idx, len(B)):
        if B[j][3] <= hi_:
            ent, edate, ei = min(hi_, B[j][1]), B[j][0], j
            break
    if ent is None:
        return dict(tk=tk, pub=pub, st="未成交", cur=last[4])
    # 出场: 止损优先(同bar同时触及按最坏算)
    ex, how = None, "持有mark"
    for b in B[ei + 1:]:
        if stop and b[3] <= stop:
            ex, how = min(b[1], stop), "止损"
            break
        if tp and b[2] >= tp:
            ex, how = max(tp, b[1]) if b[1] > tp else tp, "止盈TP1"
            break
    if ex is None:
        ex = last[4]
    # 基准 QQQ
    Q = bars("QQQ")
    qi = next((i for i, b in enumerate(Q) if b[0] >= edate), None)
    qq = (Q[-1][4] / Q[qi][4] - 1) * 100 if qi is not None else 0.0
    return dict(tk=tk, pub=pub, st=how, ent=round(ent, 2), exit=round(ex, 2),
                pct=round((ex / ent - 1) * 100, 2), qqq=round(qq, 2), hold=len(B) - 1 - ei)


rows = []
print("逐票判定共振数(窗口 T-10~T+2, 仅用当时可知信息)...\n")
for pub, tks in sorted(TABLE_DATE.items()):
    for tk in tks:
        if tk not in ZONES:
            continue
        r = run(tk, pub)
        if not r:
            continue
        src = resonance(tk, pub)
        r["nsrc"] = len(src)
        r["srcs"] = ",".join(sorted(src)) or "—"
        rows.append(r)

fill = [r for r in rows if r["st"] != "未成交"]
print(f"{'票':6} {'发布':6} {'源数':>3} {'状态':8} {'成交':>8} {'出场':>8} {'收益':>7} {'QQQ':>6}  同期源")
print("─" * 100)
for r in sorted(rows, key=lambda x: (-x["nsrc"], x["tk"])):
    if r["st"] == "未成交":
        print(f"{r['tk']:6} {r['pub'][5:]:6} {r['nsrc']:>3} {'未成交':8} {'—':>8} {'—':>8} {'—':>7} {'—':>6}  {r['srcs'][:38]}")
    else:
        print(f"{r['tk']:6} {r['pub'][5:]:6} {r['nsrc']:>3} {r['st']:8} {r['ent']:>8.2f} {r['exit']:>8.2f} "
              f"{r['pct']:>+6.1f}% {r['qqq']:>+5.1f}%  {r['srcs'][:38]}")


def stat(name, rs):
    if not rs:
        print(f"  {name:22} 无样本")
        return
    v = [r["pct"] for r in rs]
    q = [r["qqq"] for r in rs]
    win = sum(1 for x in v if x > 0)
    print(f"  {name:22} {len(v):>2}笔 | 胜率 {win/len(v)*100:>5.1f}% | 均笔 {st.mean(v):>+6.2f}% | "
          f"中位 {st.median(v):>+6.2f}% | QQQ {st.mean(q):>+5.2f}% | **alpha {st.mean(v)-st.mean(q):>+6.2f}%**")


print("\n" + "═" * 100)
print("【分组结果】同一批蛋挞档位, 按发布时的同期源数分组")
print("═" * 100)
stat("全部成交", fill)
print()
stat("0 个其他源(单源)", [r for r in fill if r["nsrc"] == 0])
stat("1 个其他源", [r for r in fill if r["nsrc"] == 1])
stat("2 个其他源", [r for r in fill if r["nsrc"] == 2])
stat("≥3 个其他源(真共振)", [r for r in fill if r["nsrc"] >= 3])
print()
stat("低共振组 (0-1源)", [r for r in fill if r["nsrc"] <= 1])
stat("高共振组 (≥2源)", [r for r in fill if r["nsrc"] >= 2])

nofill = [r for r in rows if r["st"] == "未成交"]
print(f"\n未成交 {len(nofill)} 只(档位没等到): " + " ".join(f"{r['tk']}({r['nsrc']}源)" for r in nofill))
print(f"成交率: {len(fill)}/{len(rows)} = {len(fill)/len(rows)*100:.0f}%")
json.dump(rows, open("output/resonance_backtest.json", "w"), ensure_ascii=False, indent=1)
print("\n→ output/resonance_backtest.json")
