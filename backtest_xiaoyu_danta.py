#!/usr/bin/env python3
"""
backtest_xiaoyu_danta.py — 小鱼vip / 蛋挞vip 推荐质量回测 (Yahoo日线, fill-realistic近似)。

蛋挞: 每单有买点/止损/目标 → 入场区成交(日Low<=区上沿)后, 止损/TP1按日线赛跑
      (同日双触按止损算=保守口径, 计数标注)。
小鱼: 无止损 → 两类: ①数字档位单=限价成交后持有至今/至他离场信号
      ②现价分批单=喊单日收盘买入, 持有至今mark。
入场日=原话引用中最早日期(近似); 全量含已过时票, 无幸存者剔除。
"""
import json, re, sys, time, urllib.request, os
from datetime import datetime, date

RAW = json.load(open('output/vip_buypoints_raw.json'))
PROXY = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7897")
_cache = {}

def bars(sym):
    if sym in _cache: return _cache[sym]
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=6mo"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        op = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY}))
        d = json.loads(op.open(req, timeout=15).read())
        r = d["chart"]["result"][0]; q = r["indicators"]["quote"][0]
        out = [(datetime.fromtimestamp(t).date(), q["open"][i], q["high"][i], q["low"][i], q["close"][i])
               for i, t in enumerate(r["timestamp"]) if q["close"][i]]
        _cache[sym] = out
        time.sleep(0.12)
        return out
    except Exception:
        _cache[sym] = []
        return []

def first_date(s):
    """从文本抽最早的 MM-DD 日期 → 2026年date"""
    ds = re.findall(r"(?<![\d.])(0[5-8])[-/](\d{1,2})(?![\d.])", s or "")
    if not ds: return None
    return min(date(2026, int(m), int(dd)) for m, dd in ds)

def num_zone(s):
    """从买点文本抽第一个价格区间/价格: '78-81'→(78,81), '600'→(594,606)±1%"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)", s or "")
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if 0 < a < b and b/a < 1.6: return (a, b)
    m = re.search(r"(\d+(?:\.\d+)?)", s or "")
    if m:
        v = float(m.group(1))
        if v > 0.3: return (v*0.99, v*1.01)
    return None

def simulate(src, rows, use_stops):
    res, skipped = [], 0
    for r in rows:
        tk = r["ticker"].upper().split("/")[0].split(" ")[0].strip()
        if not re.fullmatch(r"[A-Z]{1,5}", tk): skipped += 1; continue
        d0 = first_date((r.get("buy_points") or "") + " " + (r.get("evidence") or "")) \
             or first_date(r.get("latest") or "")
        if d0 is None: skipped += 1; continue
        B = bars(tk)
        fut = [b for b in B if b[0] >= d0]
        if len(fut) < 2: skipped += 1; continue
        zone = num_zone(r.get("buy_points"))
        # 入场
        entry = None; edate = None
        if zone and src == "danta":
            for b in fut:
                if b[3] <= zone[1]:      # Low触区上沿
                    entry = min(zone[1], b[1]) if b[1] <= zone[1] else zone[1]
                    edate = b[0]; break
        else:                              # 现价分批/无区 → 喊单日收盘
            entry = fut[0][4]; edate = fut[0][0]
        if not entry:
            res.append(dict(tk=tk, st="未成交", pct=None)); continue
        # 出场
        stop = tp = None
        if use_stops:
            ms = re.search(r"(\d+(?:\.\d+)?)", (r.get("stop") or ""))
            stop = float(ms.group(1)) if ms else None
            mt = re.search(r"(\d+(?:\.\d+)?)", (r.get("targets") or ""))
            tp = float(mt.group(1)) if mt else None
            if stop and stop >= entry: stop = None
            if tp and tp <= entry: tp = None
        exitp = None; how = "持有"
        amb = 0
        for b in [x for x in B if x[0] > edate]:
            hit_s = stop and b[3] <= stop
            hit_t = tp and b[2] >= tp
            if hit_s and hit_t: amb = 1
            if hit_s: exitp, how = stop if b[1] > stop else b[1], "止损"; break
            if hit_t: exitp, how = tp if b[1] < tp else b[1], "止盈"; break
        if exitp is None:
            exitp = B[-1][4]; how = "持有mark"
        pct = (exitp/entry - 1) * 100
        res.append(dict(tk=tk, st=how, pct=round(pct,1), entry=round(entry,2),
                        exit=round(exitp,2), d0=str(d0), amb=amb, lv=r.get("level","")))
    return res, skipped

print("═══ 蛋挞vip (8/4起, 带止损/目标, fill-realistic) ═══")
dres, dskip = simulate("danta", RAW["danta"]["stocks"], use_stops=True)
print("═══ 小鱼vip (6/8起, 喊单日买入持有/跟离场) ═══")
xres, xskip = simulate("xiaoyu", RAW["xiaoyu"]["stocks"], use_stops=False)

for name, res, skip in (("蛋挞", dres, dskip), ("小鱼", xres, xskip)):
    tr = [x for x in res if x["pct"] is not None]
    nofill = sum(1 for x in res if x["st"] == "未成交")
    w = [x for x in tr if x["pct"] > 0]
    amb = sum(x.get("amb",0) for x in tr)
    print(f"\n【{name}】可测{len(tr)}笔 (未成交{nofill}, 无法解析跳过{skip})")
    if tr:
        import statistics as st
        print(f"  胜率 {len(w)}/{len(tr)} = {len(w)/len(tr)*100:.0f}% | 均笔 {st.mean(x['pct'] for x in tr):+.1f}% | 中位 {st.median(x['pct'] for x in tr):+.1f}% | 同日双触(按止损计) {amb}")
        for x in sorted(tr, key=lambda z: z["pct"]):
            print(f"    {x['tk']:6} {x['d0']} {x['entry']:>8}→{x['exit']:>8} {x['pct']:>+7.1f}% {x['st']} [{x['lv'][:4]}]")
json.dump(dict(danta=dres, xiaoyu=xres), open("output/xiaoyu_danta_backtest.json","w"), ensure_ascii=False)
