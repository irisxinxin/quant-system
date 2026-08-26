#!/usr/bin/env python3
"""
backtest_xiaoyu_radar.py — 小鱼「DAILY RADAR + FLOW」期权资金流雷达的准确度回测。

数据来源(2026-08-26 提取):
  归档 output/xiaoyu_vip_history.json 里每日 19:50 UTC(=15:50 ET) 那条带图消息,
  图片存于 data/xiaoyu_img/<msgid>_*.png, 逐张人工读表转录成下方 RADAR。
  雷达本身不给点位, 只给「标的 + 期权premium + CNT + BIAS(Bull/Bear/Mixed)」,
  所以能测的是**方向准确度**, 不是入场质量。

口径(遵循 feedback_realistic_fill):
  · 雷达 15:50 ET 发布 = 收盘前 10 分钟, 跟单者最现实的成交点是**次日开盘**
    → 主口径 entry = 次日 open。另给「当日收盘价入场」作对比上界。
  · Bull 做多, Bear 做空(收益取反), Mixed 不计入方向统计(单列)。
  · 无止损无目标 —— 雷达没给, 不能替它编。按固定持有期 mark。
  · 基准 = QQQ 同持有期(同样次日开盘进), 报 alpha。
"""
import json
import statistics as st
import sys

from longport.openapi import AdjustType, Config, Period, QuoteContext

# 日期 → [(ticker, bias)] ; bias: B=Bull, S=Bear, M=Mixed
RADAR = {
    "2026-08-03": [("SPY","B"),("AMD","B"),("AAOI","B"),("RBLX","B"),("DJT","B"),
                   ("AXTI","B"),("NVTS","B"),("SPCE","B"),("NBIS","B")],
    "2026-08-04": [("SPY","B"),("MU","B"),("NFLX","B"),("DELL","B"),("GLW","B"),
                   ("AAPL","B"),("SKHY","B"),("ORCL","B")],
    "2026-08-05": [("SPY","B"),("TSLA","B"),("DELL","B"),("HUT","B"),("AFRM","B"),("UNH","B")],
    "2026-08-06": [("NVDA","B"),("TSLA","B"),("BE","B"),("ASTS","B"),("HUT","B"),
                   ("AFRM","B"),("UNH","B"),("FLNC","S"),("POET","B")],
    "2026-08-07": [("MU","B"),("AXTI","M"),("NVDA","B"),("BE","B"),("ASTS","B"),
                   ("SPCX","B"),("FLNC","S"),("AAOI","B"),("POET","B")],
    "2026-08-10": [("NVDA","B"),("META","B"),("IBIT","S"),("VLO","B"),("CBRS","B"),
                   ("ZTS","B"),("PENG","B"),("IREN","B"),("SHAK","B")],
    "2026-08-11": [("SPCX","B"),("INTC","B"),("IREN","B"),("META","B"),("IBIT","S"),
                   ("DINO","B"),("VLO","B"),("SHAK","B")],
    "2026-08-12": [("ORCL","B"),("SPCX","B"),("NBIS","B"),("AAPL","B"),("NVDA","B"),
                   ("INDI","B"),("HTZ","B")],
    "2026-08-13": [("AAPL","B"),("MU","B"),("INTC","B"),("SPCX","B"),("IREN","B"),
                   ("USO","B"),("PANW","B"),("MRVL","B"),("INDI","B")],
    "2026-08-14": [("MU","B"),("NOW","B"),("INTC","B"),("AAOI","B"),("INDI","B"),
                   ("INFQ","B"),("SPCE","B")],
    "2026-08-17": [("AXTI","B"),("SPCX","B"),("SKHY","B"),("MU","B"),("INTC","B"),
                   ("NBIS","B"),("NVTS","B"),("AAOI","B"),("INFQ","B")],
    "2026-08-18": [("AXTI","B"),("SPCX","B"),("MU","B"),("INTC","B"),("NBIS","B"),
                   ("META","B"),("QBTS","B")],          # VIX 那行无价格, 已剔除
    "2026-08-19": [("IBIT","B"),("PFE","B"),("TSLA","B"),("OUST","S"),("QBTS","B"),
                   ("TER","B"),("MNST","B"),("ASTL","B"),("WULF","B")],
    "2026-08-20": [("IONQ","B"),("AAPL","B"),("PFE","B"),("TSLA","B"),("MSTR","B"),
                   ("BMNR","B"),("PURR","B"),("OUST","S"),("BTDR","B"),("MNST","B")],
    "2026-08-21": [("IONQ","B"),("AAPL","B"),("MSTR","B"),("ALMS","B"),("BMNR","B"),
                   ("DK","B"),("PURR","B"),("BTDR","B"),("MNST","B"),("PGEN","B")],
    "2026-08-24": [("GEO","B"),("MRNA","B"),("AAPL","B"),("MU","B"),("ALMS","B"),
                   ("GDX","B"),("BMNR","B"),("DK","B"),("PGEN","B")],
}

HOLDS = [1, 3, 5]          # 持有 N 个交易日
ctx = QuoteContext(Config.from_env())
_cache = {}


def bars(sym):
    if sym in _cache:
        return _cache[sym]
    try:
        b = ctx.candlesticks(sym + ".US", Period.Day, 40, AdjustType.NoAdjust)
        out = [(str(x.timestamp.date()), float(x.open), float(x.close)) for x in b]
    except Exception:
        out = []
    _cache[sym] = out
    return out


def ret(sym, pub, hold, entry_mode):
    """返回 (收益%, 实际持有天数) ; entry_mode: 'nextopen' | 'sameclose'"""
    B = bars(sym)
    if not B:
        return None
    i = next((k for k, b in enumerate(B) if b[0] == pub), None)
    if i is None:
        return None
    if entry_mode == "sameclose":
        ent, j0 = B[i][2], i
    else:
        if i + 1 >= len(B):
            return None
        ent, j0 = B[i + 1][1], i + 1
    j = min(j0 + hold, len(B) - 1)
    if j <= j0 and entry_mode == "sameclose":
        return None
    return (B[j][2] / ent - 1) * 100, j - j0


def stat(name, vals):
    if not vals:
        print(f"  {name:26} 无样本")
        return None
    win = sum(1 for v in vals if v > 0)
    print(f"  {name:26} {len(vals):>3}笔 | 胜率 {win/len(vals)*100:>5.1f}% | "
          f"均笔 {st.mean(vals):>+6.2f}% | 中位 {st.median(vals):>+6.2f}%")
    return st.mean(vals)


for MODE in ("nextopen", "sameclose"):
    label = "次日开盘入场(可跟单口径)" if MODE == "nextopen" else "当日收盘入场(理论上界)"
    print("\n" + "=" * 92)
    print(f"【{label}】")
    print("=" * 92)
    for hold in HOLDS:
        bull, bear, mixed, bench, miss = [], [], [], [], set()
        for pub, rows in RADAR.items():
            q = ret("QQQ", pub, hold, MODE)
            for tk, bias in rows:
                r = ret(tk, pub, hold, MODE)
                if r is None:
                    miss.add(tk)
                    continue
                v = r[0]
                if bias == "B":
                    bull.append(v)
                elif bias == "S":
                    bear.append(-v)          # 看空 → 收益取反
                else:
                    mixed.append(v)
                if q:
                    bench.append(q[0])
        print(f"\n─── 持有 {hold} 个交易日 " + "─" * 60)
        mb = stat("Bull 信号", bull)
        stat("Bear 信号(已取反)", bear)
        stat("Mixed 信号", mixed)
        mq = stat("同期 QQQ 基准", bench)
        if mb is not None and mq is not None:
            print(f"  {'▶ Bull alpha':26} {mb - mq:>+6.2f}%")
        allv = bull + bear
        if allv:
            wa = sum(1 for v in allv if v > 0)
            print(f"  {'▶ 全部方向性信号':26} {len(allv):>3}笔 | "
                  f"方向正确率 {wa/len(allv)*100:>5.1f}% | 均笔 {st.mean(allv):>+6.2f}%")
        if miss and hold == HOLDS[0]:
            print(f"  ⚠️ 取不到日线({len(miss)}): {' '.join(sorted(miss))}", file=sys.stderr)

# ── 逐票汇总: 哪些标的被反复上榜, 表现如何(次日开盘/持有5日) ──
print("\n" + "=" * 92)
print("【上榜次数 ≥3 的标的 — 次日开盘入场 / 持有5日】")
print("=" * 92)
per = {}
for pub, rows in RADAR.items():
    for tk, bias in rows:
        if bias != "B":
            continue
        r = ret(tk, pub, 5, "nextopen")
        if r:
            per.setdefault(tk, []).append(r[0])
rank = sorted(((len(v), st.mean(v), k) for k, v in per.items()), key=lambda x: (-x[0], -x[1]))
print(f"{'票':8}{'上榜':>5}{'均笔':>9}  明细")
for n, m, k in rank:
    if n < 3:
        continue
    print(f"{k:8}{n:>5}{m:>+8.2f}%  " + " ".join(f"{v:+.1f}" for v in per[k]))

json.dump({"radar": {k: v for k, v in RADAR.items()}}, open("output/radar_backtest_input.json", "w"))
print("\n→ 原始雷达转录已存 output/radar_backtest_input.json")
