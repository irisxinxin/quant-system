#!/usr/bin/env python3
"""
backtest_danta_vs_xiaoyu.py — 蛋挞 vs 小鱼 同窗口同口径对比。

上一版对比不成立的原因:
  · 窗口不同: 蛋挞样本只有 8/04~8/14, 小鱼是 6/08~8/14 (含一整段下跌)
  · 口径不同: 蛋挞用「限价入场 + 止损/TP」(她给了完整四件套), 小鱼用「收盘买入持有」
    → 等于拿蛋挞的最优方法对小鱼的最差方法

这里做两件事:
  A. 把双方都放进同一个窗口 (8/04 起), 同一个口径, 2x2 交叉
  B. 用蛋挞图片点位表的真实发布日期逐张回测 (Low<=入场区上沿才成交, 止损/TP1 日线赛跑)
"""
import json, os, re, sys, time, urllib.request, statistics as st
from datetime import datetime, timezone

PROXY = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7897")
CACHE = {}

# 蛋挞点位表 → 真实发布日 (取自 output/danta_vip_history.json 原始消息时间戳)
TABLE_DATE = {
    "2026-08-10": ["PLTR", "NOW", "CRWD", "PANW", "SNOW", "PATH", "NET", "DDOG", "HPQ",
                   "RZLV", "PDYN", "BBAI", "GRRR", "SOUN", "AUR", "MITK",
                   "ASTS", "RKLB", "RDW", "LUNR", "OPTX"],
    "2026-08-11": ["FCX", "STLD", "CF", "MOS", "NUE"],
    "2026-08-12": ["INTC", "SOFI", "HOOD", "NFLX"],
    "2026-08-13": ["AMAT", "LRCX", "TER", "ENTG", "KLAC", "ICHR", "COHR", "CELH"],
    "2026-08-14": ["ON", "PENG", "OUST", "CRWV", "VELO", "OSS", "TEM", "AAOI", "CIFR"],
}
# 她的档位: 第一入场区(上沿,下沿), 止损, TP1  —— 全部来自图片表转录(见 Notion 蛋挞看板)
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


def bars(sym):
    if sym in CACHE:
        return CACHE[sym]
    out = []
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=6mo"
        op = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY}))
        d = json.loads(op.open(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read())
        r = d["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        out = [(str(datetime.fromtimestamp(t, timezone.utc).date()), q["open"][i], q["high"][i], q["low"][i], q["close"][i])
               for i, t in enumerate(r["timestamp"]) if q["close"][i]]
        time.sleep(0.05)
    except Exception as e:
        print(f"  ⚠️ {sym}: {type(e).__name__}", file=sys.stderr)
    CACHE[sym] = out
    return out


def i_on_after(B, d):
    for i, b in enumerate(B):
        if b[0] >= d:
            return i
    return None


Q = bars("QQQ")
LAST = Q[-1][0]
print("═" * 100)
print(f"蛋挞 vs 小鱼 — 同窗口同口径对比    最后交易日 {LAST}")
print("═" * 100)

# ───────── B. 蛋挞图片点位表, 按真实发布日回测 ─────────
print("\n【蛋挞】图片点位表逐张回测 (限价挂第一入场区上沿, Low<=才成交; 之后止损/TP1 日线赛跑)")
print("-" * 100)
drows = []
for pub, tks in sorted(TABLE_DATE.items()):
    for tk in tks:
        if tk not in ZONES:
            continue
        hi, lo, stop, tp = ZONES[tk]
        B = bars(tk)
        i = i_on_after(B, pub)
        if i is None:
            continue
        ndays = len(B) - 1 - i          # 发布后还剩几个交易日
        qi = i_on_after(Q, pub)
        ent = edate = None
        for j in range(i, len(B)):
            if B[j][3] <= hi:
                ent, edate, ei = min(hi, B[j][1]), B[j][0], j
                break
        if ent is None:
            drows.append(dict(tk=tk, pub=pub, st="未成交", nd=ndays, hi=hi, cur=B[-1][4]))
            continue
        exitp, how = None, "持有mark"
        for b in B[ei + 1:]:
            if stop and b[3] <= stop:
                exitp, how = min(b[1], stop), "止损"; break
            if tp and b[2] >= tp:
                exitp, how = max(b[1], tp) if b[1] > tp else tp, "止盈TP1"; break
        if exitp is None:
            exitp = B[-1][4]
        qj = i_on_after(Q, edate)
        drows.append(dict(tk=tk, pub=pub, st=how, nd=ndays, hi=hi, ent=round(ent, 2), exit=round(exitp, 2),
                          pct=round((exitp / ent - 1) * 100, 1),
                          qqq=round((Q[-1][4] / Q[qj][4] - 1) * 100, 1),
                          hold=len(B) - 1 - ei))
fill = [r for r in drows if r["st"] != "未成交"]
nofill = [r for r in drows if r["st"] == "未成交"]
print(f"  {len(drows)} 只票: 成交 {len(fill)} / 未成交 {len(nofill)}")
print(f"  ⚠️ 发布日后剩余交易日数分布: " +
      ", ".join(f"{d[5:]}发的→{len(bars('QQQ'))-1-i_on_after(Q,d)}天" for d in sorted(TABLE_DATE)))
if fill:
    v = [r["pct"] for r in fill]
    q = [r["qqq"] for r in fill]
    hold = [r["hold"] for r in fill]
    print(f"\n  成交 {len(v)} 笔: 胜率 {sum(1 for x in v if x>0)/len(v)*100:.0f}% | 均笔 {st.mean(v):+.1f}% | "
          f"中位 {st.median(v):+.1f}% | 同期QQQ {st.mean(q):+.1f}% | alpha {st.mean(v)-st.mean(q):+.1f}%")
    print(f"  平均持仓 {st.mean(hold):.1f} 个交易日 (最长 {max(hold)} 天) ← 这就是全部证据长度")
    print(f"  出场: " + str({s: sum(1 for r in fill if r['st'] == s) for s in {r['st'] for r in fill}}))
    for r in sorted(fill, key=lambda x: -x["pct"]):
        print(f"    {r['tk']:6} {r['pub'][5:]}发 挂{r['hi']:>7} 成交{r['ent']:>8} → {r['exit']:>8} "
              f"{r['pct']:>+6.1f}% ({r['st']}, 持{r['hold']}天, QQQ{r['qqq']:+.1f}%)")
print(f"\n  未成交(她的点位没等到): " + " ".join(f"{r['tk']}(挂{r['hi']},现{r['cur']:.2f})" for r in nofill))

# ───────── A. 2x2 同窗口交叉 ─────────
print("\n" + "═" * 100)
print("【2×2 交叉】同窗口 8/04~%s, 两种口径各跑一次" % LAST)
print("═" * 100)
XY = json.load(open("output/xiaoyu_vip_backtest.json"))
xa = [r for r in XY["A"] if r.get("st") == "ok" and r["d0"] >= "2026-08-04"]
xc = [r for r in XY["C"] if r.get("fill", "") >= "2026-08-04" and r["st"] == "成交"]
xc_all = [r for r in XY["C"] if r["ts"] >= "2026-08-03"]


def line(who, method, rs, key, n_total=None):
    if not rs:
        print(f"  {who:6} {method:26} 无样本"); return
    v = [r[key] for r in rs]
    q = [r["qqq"] for r in rs]
    extra = f" (挂单{n_total}个, 成交{len(rs)})" if n_total else ""
    print(f"  {who:6} {method:26} {len(v):3d}笔 | 胜率 {sum(1 for x in v if x>0)/len(v)*100:5.1f}% | "
          f"均笔 {st.mean(v):+6.1f}% | QQQ {st.mean(q):+5.1f}% | alpha {st.mean(v)-st.mean(q):+6.1f}%{extra}")


print()
line("小鱼", "① 喊单日收盘买入持有", xa, "A")
line("小鱼", "② 他给的点位挂限价", xc, "pct", n_total=len(xc_all))
line("蛋挞", "② 她的点位挂限价+止损/TP", fill, "pct", n_total=len(drows))
# 蛋挞用小鱼口径: 发布日收盘买入, 持有到最后
dclose = []
for pub, tks in sorted(TABLE_DATE.items()):
    for tk in tks:
        B, i = bars(tk), None
        i = i_on_after(bars(tk), pub)
        if i is None or i >= len(B) - 1:
            continue
        qi = i_on_after(Q, pub)
        dclose.append(dict(tk=tk, pct=(B[-1][4] / B[i][4] - 1) * 100,
                           qqq=(Q[-1][4] / Q[qi][4] - 1) * 100))
line("蛋挞", "① 发表日收盘买入持有", dclose, "pct")

json.dump(dict(danta=drows, danta_close=dclose), open("output/danta_vs_xiaoyu.json", "w"), ensure_ascii=False, indent=1)
print("\n→ output/danta_vs_xiaoyu.json")
