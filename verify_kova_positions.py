#!/usr/bin/env python3
"""
verify_kova_positions.py — Kova 7/2清仓后的每笔喊单, 用日线核验止损是否已被打掉。

背景(用户明确要求): Kova 止损极紧, 喊单常当天就被打; 不能把喊过的单都当持仓。
两套止损口径:
  A 原始止损: 他喊单时给的 stop
  B 保本止损: 8/3 他公开指令"raise your stop to buy price, if trigger then sell"
    → 8/3 前入场且当时还活着的单, 从 8/3 起止损上移到入场价
判定: 从入场**次日**起首个 Low<=stop 的交易日 = 止损打掉 (日线无法排序当日先后,
当日 Low<=stop 只能标"⚠️当日触及不可判", 除非他原话确认)。
他原话确认的出局: BMO(7/20自认打掉), TWST口径存疑(8/5"It hit my stop"上下文不明),
AXTI 8/7加仓单自认打掉(8/4主仓他明言保留观察)。
代码存疑: CLBK喊23.8实价~10, BMO喊255.65实价~181 → 量纲对不上, 结果单独标注。
"""
import json, os, time, urllib.request
from datetime import datetime, timezone

PROXY = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7897")
CACHE = {}

# (ticker, 入场日, 入场价, 原止损, 备注)  — 全部摘自 kova_signal_history.json 原话
SIGNALS = [
    ("PENG", "2026-07-09", 84.5, 77,    "buy peng at 84.5 stop 77"),
    ("ON",   "2026-07-09", 101,  97,    "BUY ON AT 101 STOP 97"),
    ("DELL", "2026-07-09", 455,  432,   "buy dell at 455 stop 432"),
    ("CLBK", "2026-07-17", 23.8, 22.9,  "CLBK 23.8 stop 22.9"),
    ("BMO",  "2026-07-17", 255.65, 250, "add BMO 255.65 stop 250 (7/20他自己确认止损打掉)"),
    ("CIFR", "2026-07-20", 20.18, 18.5, "buying CIFR at 20.18 stop 18.5"),
    ("DOCN", "2026-07-21", 129,  125,   "BUY DOCN AT 129 STOP 125"),
    ("NBIS", "2026-07-21", 200,  190,   "nbis buy at 200 stop 190"),
    ("MXL",  "2026-07-22", 91,   85,    "add mxl here at 91 stop 85"),
    ("MEDP", "2026-07-23", 671,  630,   "add MEDP at 671 stop 630"),
    ("ILMN", "2026-07-23", 195,  192,   "add ILMN at 195 stop 192"),
    ("FIX",  "2026-07-23", 1848, 1788,  "buy fix at 1848 stop 1788"),
    ("MU",   "2026-07-23", None, 961,   "buying mu here stop 961 (入场价'here'未报→按当日收盘近似)"),
    ("SNDK", "2026-07-30", 1205, 1134,  "buy SNDK AT 1205 STOP 1134"),
    ("AMD",  "2026-07-30", 478,  460,   "buy amd here at 478 stop 460"),
    ("BE",   "2026-07-30", 209,  190,   "buy be here at 209 stop 190"),
    ("HPE",  "2026-08-03", 49.2, 46,    "buy hpe here at 49.2 stop 46"),
    ("NBIS", "2026-08-03", 212,  203,   "Buy back NBIS at 212 stop 203"),
    ("ELVN", "2026-08-03", 58.4, 55.5,  "BUY ELVN at 58.4 stop 55.5"),
    ("MRVL", "2026-08-04", 216,  208,   "add mrvl at 216 stop 208"),
    ("AXTI", "2026-08-04", 64.5, 55,    "add axti at 64.5 stop 55 (8/7的加仓单他自认已被打)"),
    ("DOCN", "2026-08-04", 133,  122,   "Add docn at 133 stop 122"),
    ("SIVE", "2026-08-04", 37.4, None,  "buy sive at 37.4 (止损消息为空→无损核验)"),
    ("TWST", "2026-08-05", 108,  104,   "BUY TWST 108 STOP 104 (当天'It hit my stop')"),
    ("MSFT", "2026-08-06", 496,  487,   "buy MSFT 496 stop 487 (8/13上移到496保本)"),
    ("ONTO", "2026-08-07", 309,  294,   "buying onto at 309 stop 294"),
    ("MU",   "2026-08-12", 907,  903,   "Buy mu 907 stop 903 (0.4%极紧损)"),
    ("LUNR", "2026-08-14", 19.7, 19,    "BUY LUNR AT 19.7 STOP 19"),
    ("AEHR", "2026-08-14", 142,  131,   "BUY AEHR HERE 142 STOP 131"),
]
BREAKEVEN_DAY = "2026-08-03"


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
    except Exception:
        pass
    CACHE[sym] = out
    return out


print(f"{'票':6} {'入场日':10} {'入场':>8} {'原损':>7} | {'A:原始止损口径':28} | {'B:8/3保本口径':24} | 现价/浮动")
print("─" * 130)
res = []
for tk, d0, ent, stop, note in SIGNALS:
    B = bars(tk)
    fut = [b for b in B if b[0] >= d0]
    if not fut:
        print(f"{tk:6} {d0} 无数据 ({note})")
        res.append(dict(tk=tk, d0=d0, st="无数据", note=note))
        continue
    if ent is None:
        ent = fut[0][4]
    cur = B[-1][4]
    # A: 原始止损 — 次日起判; 当日触及仅标注
    hitA, sameday = None, False
    if stop:
        if fut and fut[0][3] <= stop:
            sameday = True
        for b in fut[1:]:
            if b[3] <= stop:
                hitA = b[0]; break
    # B: 保本(仅8/3前入场且A口径当时还活着)
    hitB, be_note = None, ""
    if d0 < BREAKEVEN_DAY and stop:
        aliveA_at_be = hitA is None or hitA >= BREAKEVEN_DAY
        if aliveA_at_be:
            for b in [x for x in fut if x[0] >= BREAKEVEN_DAY]:
                eff = max(stop, ent)   # 8/3起损上移到成本
                if b[3] <= eff:
                    hitB = b[0]; break
            be_note = "保本损生效"
    sA = f"❌止损 {hitA} @{stop}" if hitA else ("⚠️当日触及不可判" if sameday else "✅活着")
    if hitB and (hitA is None or hitB < hitA):
        sB = f"⚠️保本出局 {hitB} @{ent}"
    elif be_note:
        sB = "✅保本损未触发"
    else:
        sB = "—(8/3后入场)"
    pnl = (cur / ent - 1) * 100
    live = "❌" if hitA else ("⚠️" if hitB else "✅")
    print(f"{tk:6} {d0} {ent:>8.2f} {str(stop):>7} | {sA:28} | {sB:24} | 现{cur:>8.2f} ({pnl:+.1f}%)")
    res.append(dict(tk=tk, d0=d0, ent=round(ent, 2), stop=stop, hitA=hitA, hitB=hitB, sameday=sameday,
                    cur=round(cur, 2), pnl=round(pnl, 1), note=note))

nA = sum(1 for r in res if r.get("hitA"))
nSD = sum(1 for r in res if not r.get("hitA") and r.get("sameday"))
alive = [r for r in res if r.get("st") != "无数据" and not r.get("hitA") and not r.get("sameday")]
print("─" * 130)
print(f"共 {len(SIGNALS)} 笔喊单: 次日起止损确认打掉 {nA} 笔 | ⚠️仅当日触及(先后不可判) {nSD} 笔 | 明确活着 {len(alive)} 笔:")
for r in alive:
    print(f"   ✅ {r['tk']:6} {r['d0']} 入{r['ent']} 损{r['stop']} 现{r['cur']} ({r['pnl']:+.1f}%)")
json.dump(res, open("output/kova_positions_verified.json", "w"), ensure_ascii=False, indent=1)
print("\n→ output/kova_positions_verified.json")
