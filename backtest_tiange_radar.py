#!/usr/bin/env python3
"""
backtest_tiange_radar.py — 天哥 #机会雷达 频道的准确度回测。

频道: 美股聚合群 / #机会雷达 (id 1478932700361392139, 转发号"站长转发1")
归档: output/kova_radar_history.json (= 旧文件名 tiange_radar_history.json, 同一批149条)

为什么这个频道能用正股数据回测:
  他的主打信号是「每周分享5只适合 sell put 的公司 + 建议行权价, 到期日以30日左右为标准」。
  sell put 的胜负判定**不需要期权报价** —— 到期日正股收盘价 vs 行权价就能定:
    · 收盘 > K  → 期权归零, 全额吃权利金 = 胜
    · 收盘 < K  → 被行权接股, 账面亏 (收盘-K)/K (未计权利金, 所以是保守下界)
  另外单独统计「期间最低价是否击穿K」—— 即使到期收回, 过程中被击穿也意味着
  保证金压力和心理压力, 对跟单者是真实成本。

口径:
  · 到期日 = 发布日 + 30 自然日, 取该日或之前最近的交易日收盘
  · 同一天同一标的重复出现只取一次(他常连续几天重复同一批名单)
  · 正股加仓点位单独回测: fill-realistic, Low<=点位才算成交, 成交价=min(点位, 当日开盘)
"""
import json
import re
import statistics as st
from datetime import datetime, timedelta

from longport.openapi import AdjustType, Config, Period, QuoteContext

HIST = "output/tiange_radar_history.json"   # = #机会雷达 频道
DTE = 30

# 段落切分: SELL PUT 段 vs 正股段 vs LEAPS 段
SP_START = re.compile(r"(SELL\s*PUT|sell\s*put\s*[:：])", re.I)
STOCK_START = re.compile(r"(正股\s*值得考虑|值得考虑的正股|正股加仓)")
LEAPS_START = re.compile(r"(LEAPS?\s*的机会|LEAPS?\s*[:：])", re.I)
# TICKER: $123.4  /  TICKER：123
PAIR = re.compile(r"^\s*([A-Z]{2,5})\s*[:：]\s*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.M)
BAD = {"SELL", "PUT", "CALL", "LEAPS", "DTE", "QQQ", "SPY", "ET"}

ctx = QuoteContext(Config.from_env())
_c = {}


def bars(s):
    if s in _c:
        return _c[s]
    try:
        b = ctx.candlesticks(s + ".US", Period.Day, 300, AdjustType.NoAdjust)
        _c[s] = [(str(x.timestamp.date()), float(x.open), float(x.high),
                  float(x.low), float(x.close)) for x in b]
    except Exception:
        _c[s] = []
    return _c[s]


def seg(text, start_re, *end_res):
    m = start_re.search(text)
    if not m:
        return ""
    body = text[m.end():]
    ends = [r.search(body) for r in end_res]
    cut = min([e.start() for e in ends if e], default=len(body))
    return body[:cut]


msgs = json.load(open(HIST))
sp_sig, stk_sig = [], []
for m in msgs:
    t = m["text"]
    d = m["ts"][:10]
    if not SP_START.search(t) and not re.search(r"sell\s*put", t, re.I):
        continue
    sp_body = seg(t, SP_START, STOCK_START, LEAPS_START)
    for tk, px in PAIR.findall(sp_body):
        if tk in BAD:
            continue
        sp_sig.append((d, tk, float(px)))
    stk_body = seg(t, STOCK_START, LEAPS_START)
    for tk, px in PAIR.findall(stk_body):
        if tk in BAD:
            continue
        stk_sig.append((d, tk, float(px)))

# ── 6/15 起他改成行内格式(每条一个标的+明确到期日), 正则抓不准, 逐条人工转录 ──
#   (发布日, 票, 行权价, 到期日, 触发条件or None)
#   条件单: 只有正股先满足条件才算这张单成立 —— 未触发的单列, 不混进胜率
MANUAL = [
    ("2026-06-15", "DRAM",  60,  None,        None),          # 无到期日 → 按+30天
    ("2026-06-26", "SOXL",  180, "2026-07-31", None),
    ("2026-06-26", "DRAM",  62,  "2026-07-31", None),
    ("2026-06-26", "COHR",  350, "2026-07-31", ("<=", 364)),   # "掉到360-364之间可以"
    ("2026-06-26", "MRVL",  265, "2026-07-31", None),          # 35DTE → 7/31
    ("2026-06-29", "SOXL",  170, "2026-07-31", ("<=", 209)),   # 32DTE → 7/31
    ("2026-06-29", "DRAM",  65,  "2026-07-31", ("<=", 70)),    # 32DTE → 7/31
    ("2026-07-01", "MU",    980, "2026-07-31", None),          # 30天 → 7/31
    ("2026-07-02", "SOXL",  180, "2026-07-31", None),
    ("2026-07-02", "DRAM",  60,  "2026-07-31", None),
    ("2026-07-02", "MU",    980, "2026-07-31", None),
    ("2026-07-02", "AMD",   500, "2026-07-31", None),
    ("2026-07-06", "DRAM",  60,  "2026-07-31", None),
    ("2026-07-07", "PLTR",  120, "2026-08-07", None),
    ("2026-07-08", "AVGO",  370, "2026-08-07", None),
    ("2026-07-09", "SPCX",  125, "2026-08-07", None),
    ("2026-07-10", "COHR",  300, "2026-08-07", None),
    ("2026-07-14", "COHR",  270, "2026-08-14", None),
    ("2026-07-14", "NBIS",  170, "2026-08-14", None),
    ("2026-07-14", "MSFT",  350, "2026-08-14", None),
    ("2026-07-14", "AMZN",  225, "2026-08-14", None),
    ("2026-07-15", "DRAM",  50,  "2026-08-14", None),
    ("2026-07-15", "TQQQ",  68,  "2026-08-14", None),
    ("2026-07-16", "NVDA",  190, "2026-08-14", None),
    ("2026-07-16", "MSFT",  370, "2026-08-14", None),
    ("2026-07-16", "APP",   400, "2026-08-14", None),
    ("2026-07-20", "SPCX",  95,  "2026-08-21", None),
    ("2026-07-20", "AMZN",  225, "2026-08-21", None),
    ("2026-07-20", "MU",    650, "2026-08-21", None),
    ("2026-07-20", "NBIS",  120, "2026-08-21", ("<=", 180)),   # "如果能再次跌到$180以下"
    ("2026-07-21", "GOOGL", 330, "2026-08-21", ("<=", 350)),   # "$350以下, 可以$330 sell put"
    ("2026-07-21", "CRWD",  165, "2026-08-21", None),
    ("2026-07-22", "PLTR",  115, "2026-08-21", None),
    ("2026-07-22", "APP",   350, "2026-08-21", None),
    ("2026-07-23", "TQQQ",  60,  "2026-08-21", None),
    # 8/24 TQQQ $60 9/25 → 尚未到期, 不计入
]
EXPLICIT_EXP = {}
for d, tk, K, exp, cond in MANUAL:
    sp_sig.append((d, tk, float(K)))
    if exp:
        EXPLICIT_EXP[(d, tk, float(K))] = exp
COND = {(d, tk, float(K)): c for d, tk, K, e, c in MANUAL if c}

# 去重: 同一天同一标的只留一条
sp_sig = sorted(set(sp_sig))
stk_sig = sorted(set(stk_sig))
print(f"解析出 sell put 信号 {len(sp_sig)} 条 (含手工转录 {len(MANUAL)} 条) / "
      f"正股点位 {len(stk_sig)} 条\n")


def on_or_before(B, target):
    """返回 <=target 的最后一根bar索引"""
    idx = None
    for i, b in enumerate(B):
        if b[0] <= target:
            idx = i
        else:
            break
    return idx


print("=" * 100)
print(f"【SELL PUT 回测】到期 = 发布日 +{DTE} 天, 判定 = 到期收盘 vs 行权价")
print("=" * 100)
print(f"{'发布':11}{'票':6}{'行权价':>9}{'发布日价':>10}{'到期价':>9}{'虚值度':>8}  {'结果':10}{'期间最低':>9}")
print("-" * 100)
res, breached, assigned_loss = [], 0, []
skipped, untriggered, bymonth = [], [], {}
for d, tk, K in sp_sig:
    B = bars(tk)
    if not B:
        skipped.append(tk)
        continue
    i = on_or_before(B, d)
    exp = EXPLICIT_EXP.get((d, tk, K)) or \
        (datetime.fromisoformat(d) + timedelta(days=DTE)).date().isoformat()
    j = on_or_before(B, exp)
    if i is None or j is None or j <= i:
        skipped.append(f"{tk}@{d}")
        continue
    # 条件单: 到期前正股必须先触及条件价, 否则这张单根本没开
    c = COND.get((d, tk, K))
    if c:
        lo_win = min(b[3] for b in B[i + 1:j + 1])
        if not (lo_win <= c[1]):
            untriggered.append((d, tk, K, c[1], lo_win))
            continue
    spot = B[i][4]
    end = B[j][4]
    lo = min(b[3] for b in B[i + 1:j + 1])
    otm = (spot / K - 1) * 100          # 发布时行权价在现价下方多少
    win = end > K
    res.append(win)
    bymonth.setdefault(d[:7], []).append(win)
    if lo <= K:
        breached += 1
    if not win:
        assigned_loss.append((end / K - 1) * 100)
    print(f"{d:11}{tk:6}{K:>9.1f}{spot:>10.2f}{end:>9.2f}{otm:>+7.1f}%  "
          f"{'✅归零' if win else '🔴被行权':10}{lo:>9.2f}{'  ⚠️期间击穿' if lo <= K and win else ''}")

n = len(res)
if n:
    print("-" * 100)
    print(f"  样本 {n} 笔 | **到期未被行权(胜) {sum(res)}/{n} = {sum(res)/n*100:.1f}%**")
    print(f"  期间曾被击穿行权价: {breached}/{n} = {breached/n*100:.1f}%  (含最终收回的)")
    if assigned_loss:
        print(f"  被行权那 {len(assigned_loss)} 笔的账面亏损: 均 {st.mean(assigned_loss):+.2f}% | "
              f"中位 {st.median(assigned_loss):+.2f}% | 最差 {min(assigned_loss):+.2f}%")
        print(f"  (未计权利金, 是保守下界; 他常给的抵押回报 ~3-15%, 可覆盖其中一部分)")
    # 分时段看 —— 防止"整段都在涨市"造成的假高胜率
    print("\n  按发布月拆分(检验是否只是踩中涨市):")
    for mo in sorted(bymonth):
        v = bymonth[mo]
        print(f"     {mo}  {sum(v):>2}/{len(v):<2} = {sum(v)/len(v)*100:>5.1f}%")
if untriggered:
    print(f"\n  ⏸ 条件未触发(单子根本没开, 不计入胜率) {len(untriggered)} 笔:")
    for d, tk, K, trig, lo in untriggered:
        print(f"     {d} {tk} ${K:g} — 需先跌到 {trig}, 期间最低仅 {lo:.2f}")

print("\n" + "=" * 100)
print("【正股加仓点位回测】fill-realistic: Low<=点位才成交, 成交价=min(点位, 当日开盘)")
print("=" * 100)
print(f"{'发布':11}{'票':6}{'点位':>9}{'发布日价':>10}{'成交日':>12}{'成交价':>9}{'现价':>9}{'收益':>8}")
print("-" * 100)
fills, nofill = [], []
for d, tk, P in stk_sig:
    B = bars(tk)
    if not B:
        continue
    i = on_or_before(B, d)
    if i is None or i + 1 >= len(B):
        continue
    ent = edate = None
    for k in range(i + 1, len(B)):
        if B[k][3] <= P:
            ent, edate = min(P, B[k][1]), B[k][0]
            break
    if ent is None:
        nofill.append((tk, d, P, B[-1][4]))
        continue
    cur = B[-1][4]
    r = (cur / ent - 1) * 100
    fills.append(r)
    print(f"{d:11}{tk:6}{P:>9.1f}{B[i][4]:>10.2f}{edate:>12}{ent:>9.2f}{cur:>9.2f}{r:>+7.1f}%")
if fills:
    w = sum(1 for x in fills if x > 0)
    print("-" * 100)
    print(f"  成交 {len(fills)}/{len(fills)+len(nofill)} 笔 | 胜率 {w/len(fills)*100:.1f}% | "
          f"均笔 {st.mean(fills):+.2f}% | 中位 {st.median(fills):+.2f}%")
    print(f"  未成交(没回踩到) {len(nofill)} 笔: " +
          " ".join(f"{t}${p:g}" for t, _, p, _ in nofill[:14]))
if skipped:
    print(f"\n⚠️ 跳过(无日线或到期日超出范围): {len(skipped)} — {' '.join(sorted(set(map(str, skipped)))[:20])}")

json.dump({"sell_put": sp_sig, "stock": stk_sig},
          open("output/tiange_radar_signals.json", "w"), ensure_ascii=False, indent=1)
print("\n→ 解析结果存 output/tiange_radar_signals.json")
