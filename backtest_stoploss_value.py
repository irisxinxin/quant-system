#!/usr/bin/env python3
"""
backtest_stoploss_value.py — 「收盘跌破博主止损后, 砍掉 vs 扛着」哪个更好。

起因(2026-08-26): CRWD 8/25 收盘 185.38 跌破蛋挞止损 188, 按纪律止损;
8/26 财报盘后 +10.53% 回到 209。用户问这个规则还要不要守。
单次结果不能判断规则 —— 这里用体系内所有带明确止损的票, 统计每一次
「收盘破损」事件之后的真实走势。

口径:
  · 破损事件 = 某日收盘价首次跌破止损(用收盘, 不用日内插针 —— 这是看板既有规则)
  · 砍掉 = 次日开盘卖出(现实可执行价, 不是破损当天收盘)
  · 扛着 = 从同一个次日开盘继续持有 N 日
  · 两者用同一个基准点比较, 差值就是「止损这个动作」的净贡献
  · 同一只票只取窗口内第一次破损, 避免反复计数
"""
import statistics as st
import sys

from longport.openapi import AdjustType, Config, Period, QuoteContext

# (票, 止损价, 来源) —— 全部是各源公开给过的明确止损位
STOPS = [
    ("CRWD", 188, "蛋挞"), ("RKLB", 70, "蛋挞"), ("DDOG", 228, "蛋挞"),
    ("NET", 265, "蛋挞"), ("OSS", 9.9, "蛋挞"), ("ASTS", 59, "蛋挞"),
    ("PLTR", 145, "蛋挞"), ("AMAT", 450, "蛋挞"), ("LRCX", 268, "蛋挞"),
    ("TER", 282.3, "蛋挞"), ("ENTG", 125, "蛋挞"), ("KLAC", 168, "蛋挞"),
    ("ICHR", 51, "蛋挞"), ("ON", 79.8, "蛋挞"), ("PENG", 51.6, "蛋挞"),
    ("OUST", 30.8, "蛋挞"), ("CRWV", 64.2, "蛋挞"), ("VELO", 9.1, "蛋挞"),
    ("TEM", 39.6, "蛋挞"), ("AAOI", 105, "蛋挞"), ("FCX", 54, "蛋挞"),
    ("STLD", 222, "蛋挞"), ("CF", 94, "蛋挞"), ("MOS", 18, "蛋挞"),
    ("NUE", 235, "蛋挞"), ("LUNR", 13.6, "蛋挞"), ("RDW", 10.8, "蛋挞"),
    ("SOUN", 6.9, "蛋挞"), ("AUR", 5.1, "蛋挞"), ("MITK", 16.5, "蛋挞"),
    ("PATH", 11.8, "蛋挞"), ("RZLV", 2.20, "蛋挞"), ("PDYN", 4.4, "蛋挞"),
    ("BBAI", 2.75, "蛋挞"), ("WING", 105, "蛋挞"), ("LYFT", 15.8, "蛋挞"),
    ("BA", 202, "蛋挞"), ("MRLN", 3.0, "蛋挞"), ("AEHR", 131, "Kova"),
    ("PLSE", 46.9, "Kova"),
]
HOLDS = [1, 3, 5, 10]
ctx = QuoteContext(Config.from_env())


def bars(s):
    try:
        b = ctx.candlesticks(s + ".US", Period.Day, 120, AdjustType.NoAdjust)
        return [(str(x.timestamp.date()), float(x.open), float(x.high),
                 float(x.low), float(x.close)) for x in b]
    except Exception:
        return []


events = []
for tk, stop, src in STOPS:
    B = bars(tk)
    if len(B) < 20:
        continue
    # 找第一次收盘跌破止损(且之前至少有一天收在止损上方 —— 确保是"跌破"不是一直在下面)
    for i in range(1, len(B) - 1):
        if B[i][4] < stop <= B[i - 1][4]:
            nxt_open = B[i + 1][1]          # 次日开盘 = 砍掉的成交价
            row = {"tk": tk, "src": src, "stop": stop, "date": B[i][0],
                   "close": B[i][4], "exit": nxt_open, "fwd": {}}
            for h in HOLDS:
                j = min(i + 1 + h, len(B) - 1)
                if j > i + 1:
                    row["fwd"][h] = (B[j][4] / nxt_open - 1) * 100
            if row["fwd"]:
                events.append(row)
            break

print("=" * 100)
print("【收盘跌破止损事件】砍掉 = 次日开盘卖; 扛着 = 从同一价继续持有")
print("=" * 100)
print(f"{'票':6}{'源':5}{'破损日':12}{'止损':>8}{'破损收盘':>10}{'次开(卖价)':>11}"
      + "".join(f"{'+'+str(h)+'日':>9}" for h in HOLDS))
print("-" * 100)
for e in sorted(events, key=lambda x: x["date"]):
    print(f"{e['tk']:6}{e['src']:5}{e['date']:12}{e['stop']:>8}{e['close']:>10.2f}{e['exit']:>11.2f}"
          + "".join(f"{e['fwd'].get(h, float('nan')):>+8.1f}%" for h in HOLDS))

print("-" * 100)
print(f"\n共 {len(events)} 次破损事件\n")
print("【扛着不砍的话, 之后会怎样】(相对次日开盘价)")
for h in HOLDS:
    v = [e["fwd"][h] for e in events if h in e["fwd"]]
    if not v:
        continue
    up = sum(1 for x in v if x > 0)
    print(f"  持有 {h:>2} 日: {len(v):>2}笔 | 反弹概率 {up/len(v)*100:>5.1f}% | "
          f"均值 {st.mean(v):>+6.2f}% | 中位 {st.median(v):>+6.2f}% | "
          f"最好 {max(v):>+6.1f}% | 最差 {min(v):>+6.1f}%")

print("\n【结论口径】")
print("  均值 > 0 → 破损后平均会反弹, 止损是负贡献(砍在坑里)")
print("  均值 < 0 → 破损后平均继续跌, 止损是正贡献(躲过下跌)")

# 尾部风险: 止损真正的价值在于砍掉最差的那一批
print("\n【尾部: 如果不砍, 最差的5次会亏多少】(持有10日)")
v10 = sorted([(e["fwd"].get(10, 0), e["tk"], e["date"]) for e in events if 10 in e["fwd"]])
for x, tk, d in v10[:5]:
    print(f"  {tk:6} {d}  {x:+.1f}%")
print("\n【尾部: 如果不砍, 最好的5次能赚多少】(持有10日)")
for x, tk, d in v10[-5:][::-1]:
    print(f"  {tk:6} {d}  {x:+.1f}%")
if v10:
    vals = [x[0] for x in v10]
    print(f"\n  分布: 标准差 {st.pstdev(vals):.1f}%  |  "
          f"亏损>10%的次数 {sum(1 for x in vals if x < -10)}/{len(vals)}  |  "
          f"盈利>10%的次数 {sum(1 for x in vals if x > 10)}/{len(vals)}")
