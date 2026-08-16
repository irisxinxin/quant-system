#!/usr/bin/env python3
"""
analyze_xiaoyu_radar.py — 小鱼 "DAILY RADAR + FLOW" 图表拆解 + 可用性回测。

输入: output/xiaoyu_radar_snapshots.json (18张截图人工转录, 2026-08-03~08-14)
问题:
  1. PRICE 下面那个 % 到底是什么? (拿真实日线穷举比对)
  2. PREMIUM/CNT/BIAS/LAST SEEN 是当天新数据还是陈年旧账? (跨快照追踪同一行)
  3. 上榜之后能不能赚钱? 09:25那张次日开盘买 / 15:50那张当日收盘买, T+1/T+3/T+5 对比QQQ
  4. 什么条件下的上榜才有效? (按 premium 大小 / flow新鲜度 / 上榜当日涨幅 分组)
"""
import json, os, time, urllib.request, statistics as st
from datetime import datetime, timezone, date

PROXY = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7897")
SNAP = json.load(open("output/xiaoyu_radar_snapshots.json"))["snapshots"]
CACHE = {}


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


def idx_of(B, d):
    for i, b in enumerate(B):
        if b[0] == d:
            return i
    return None


tickers = sorted({r["tk"] for s in SNAP for r in s["rows"]})
print(f"18张快照, {sum(len(s['rows']) for s in SNAP)} 行, {len(tickers)} 只票: {' '.join(tickers)}\n")
for t in tickers + ["QQQ"]:
    bars(t)
missing = [t for t in tickers if not CACHE[t]]
if missing:
    print(f"⚠️ 取不到日线: {missing}\n")

# ───────── 1. PRICE 下面的 % 是什么 ─────────
print("═" * 96)
print("【1】PRICE 下面那个百分比 = ?  (每种假设看有多少行能对上, 容差 ±0.35pp)")
print("═" * 96)
hyps = {"当日涨跌(vs前一日收盘)": 1, "vs前两日收盘(即滞后一天)": 2, "vs前三日收盘": 3, "5日涨跌": 5}
score = {k: [0, 0] for k in hyps}
for s in SNAP:
    for r in s["rows"]:
        B = CACHE.get(r["tk"])
        if not B:
            continue
        i = idx_of(B, s["date"])
        if i is None:
            continue
        for name, lag in hyps.items():
            if i - lag < 0:
                continue
            calc = (r["px"] / B[i - lag][4] - 1) * 100
            score[name][1] += 1
            if abs(calc - r["pct"]) <= 0.35:
                score[name][0] += 1
for name, (hit, tot) in sorted(score.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1)):
    print(f"  {name:26s} 命中 {hit:3d}/{tot:3d} = {hit/max(tot,1)*100:5.1f}%")

print("\n  分开看盘前(09:25)和收盘前(15:50)两张:")
for tm in ("09:25", "15:50"):
    sc = {k: [0, 0] for k in hyps}
    for s in [x for x in SNAP if x["time"] == tm]:
        for r in s["rows"]:
            B = CACHE.get(r["tk"])
            i = idx_of(B, s["date"]) if B else None
            if i is None:
                continue
            for name, lag in hyps.items():
                if i - lag < 0:
                    continue
                sc[name][1] += 1
                if abs((r["px"] / B[i - lag][4] - 1) * 100 - r["pct"]) <= 0.35:
                    sc[name][0] += 1
    best = max(sc.items(), key=lambda kv: kv[1][0] / max(kv[1][1], 1))
    print(f"    {tm} → 最匹配: {best[0]}  ({best[1][0]}/{best[1][1]} = {best[1][0]/max(best[1][1],1)*100:.0f}%)")

# PRICE 本身准不准
err = []
for s in [x for x in SNAP if x["time"] == "15:50"]:
    for r in s["rows"]:
        B = CACHE.get(r["tk"])
        i = idx_of(B, s["date"]) if B else None
        if i is not None:
            err.append(abs(r["px"] / B[i][4] - 1) * 100)
print(f"\n  15:50那张的PRICE vs 当日真实收盘: 平均偏差 {st.mean(err):.2f}% (中位 {st.median(err):.2f}%) → 就是实时价")

# ───────── 2. flow 数据的新鲜度 ─────────
print("\n" + "═" * 96)
print("【2】PREMIUM/CNT/BIAS/LAST SEEN 是新数据还是旧账?")
print("═" * 96)
seen_life = {}
for s in SNAP:
    for r in s["rows"]:
        key = (r["tk"], r["seen"], r["prem"])
        seen_life.setdefault(key, []).append(f"{s['date'][5:]} {s['time']}")
multi = {k: v for k, v in seen_life.items() if len(v) > 1}
print(f"  同一条flow记录(票+LAST SEEN+premium完全相同)在多张快照重复出现: {len(multi)}/{len(seen_life)} 条")
for k, v in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:8]:
    print(f"    {k[0]:5} flow@{k[1]}  ${k[2]/1e6:.2f}M  连续挂榜 {len(v)} 张: {' → '.join(v)}")

ages = []
for s in SNAP:
    sd = date(*map(int, s["date"].split("-")))
    for r in s["rows"]:
        m, dd = r["seen"].split()[0].split("/")
        age = (sd - date(2026, int(m), int(dd))).days
        ages.append(age)
        r["_age"] = age
print(f"\n  flow距离快照当天的自然日: 当天 {ages.count(0)}行 / 隔1天 {ages.count(1)}行 / 2天 {ages.count(2)}行 / ≥3天 {sum(1 for a in ages if a>=3)}行")
print(f"  → 只有 {ages.count(0)/len(ages)*100:.0f}% 的行是当天的新单, {sum(1 for a in ages if a>=1)/len(ages)*100:.0f}% 是隔夜或更旧的")

bias = [r["bias"] for s in SNAP for r in s["rows"]]
print(f"\n  BIAS 分布: Bull {bias.count('Bull')} / Bear {bias.count('Bear')} / Mixed {bias.count('Mixed')} "
      f"→ {bias.count('Bull')/len(bias)*100:.0f}% 恒为Bull, 几乎没有区分度")

notes = {s.get("macro") for s in SNAP}
print(f"  RADAR NOTES 正文: 18张全部一模一样(固定模板); 大盘打法文案: {len(notes)}种不同")

# ───────── 3. 上榜后表现 ─────────
print("\n" + "═" * 96)
print("【3】上榜之后能赚钱吗? — 可执行口径")
print("      09:25那张 → 当日开盘买入(盘前发布, 开盘可执行)")
print("      15:50那张 → 当日收盘买入(收盘前10分发布, 来得及)")
print("═" * 96)
Q = CACHE["QQQ"]
trades = []
for s in SNAP:
    qi = idx_of(Q, s["date"])
    for r in s["rows"]:
        B = CACHE.get(r["tk"])
        i = idx_of(B, s["date"]) if B else None
        if i is None or qi is None:
            continue
        ent = B[i][1] if s["time"] == "09:25" else B[i][4]
        qent = Q[qi][1] if s["time"] == "09:25" else Q[qi][4]
        row = dict(tk=r["tk"], d=s["date"], tm=s["time"], prem=r["prem"], age=r["_age"],
                   bias=r["bias"], daypct=r["pct"], ent=ent)
        for h in (1, 3, 5):
            if i + h < len(B) and qi + h < len(Q):
                row[f"t{h}"] = (B[i + h][4] / ent - 1) * 100 - (Q[qi + h][4] / qent - 1) * 100
        trades.append(row)


def rep(label, rs):
    line = f"  {label:30s} n={len(rs):3d}"
    for h in (1, 3, 5):
        v = [r[f"t{h}"] for r in rs if f"t{h}" in r]
        if v:
            line += f" | T+{h} {st.mean(v):+6.2f}% (赢{sum(1 for x in v if x>0)/len(v)*100:3.0f}%)"
        else:
            line += f" | T+{h}    n/a    "
    print(line)


print("\n  ▸ 全样本 (超额收益 = 个股 - QQQ)")
rep("全部", trades)
rep("盘前那张(09:25)", [r for r in trades if r["tm"] == "09:25"])
rep("收盘那张(15:50)", [r for r in trades if r["tm"] == "15:50"])

print("\n  ▸ 按 flow 新鲜度 (LAST SEEN 距快照几天)")
rep("当天新单 (age=0)", [r for r in trades if r["age"] == 0])
rep("隔夜 (age=1)", [r for r in trades if r["age"] == 1])
rep("陈旧 (age>=2)", [r for r in trades if r["age"] >= 2])

print("\n  ▸ 按权利金大小")
rep("PREMIUM >= $3M", [r for r in trades if r["prem"] >= 3e6])
rep("$1M ~ $3M", [r for r in trades if 1e6 <= r["prem"] < 3e6])
rep("< $1M", [r for r in trades if r["prem"] < 1e6])

print("\n  ▸ 按上榜时那个百分比(追高程度)")
rep("已涨 >= +10% (追高)", [r for r in trades if r["daypct"] >= 10])
rep("+3% ~ +10%", [r for r in trades if 3 <= r["daypct"] < 10])
rep("-3% ~ +3% (横盘)", [r for r in trades if -3 <= r["daypct"] < 3])
rep("下跌 < -3% (回踩)", [r for r in trades if r["daypct"] < -3])

print("\n  ▸ 组合过滤: 当天新单 + 权利金>=1M + 未追高(<+3%)  ← 最像他文案说的用法")
rep("组合", [r for r in trades if r["age"] == 0 and r["prem"] >= 1e6 and r["daypct"] < 3])

print("\n  ▸ 首次上榜 vs 连续挂榜(同票已出现过)")
first, again = [], []
seen_tk = set()
for r in sorted(trades, key=lambda x: (x["d"], x["tm"])):
    (first if r["tk"] not in seen_tk else again).append(r)
    seen_tk.add(r["tk"])
rep("首次上榜", first)
rep("重复上榜", again)

json.dump(dict(trades=trades), open("output/xiaoyu_radar_backtest.json", "w"), ensure_ascii=False, indent=1)
print("\n→ output/xiaoyu_radar_backtest.json")
