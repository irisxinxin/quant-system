#!/usr/bin/env python3
"""
zhaoge_audit.py — 审计 #股票赵哥-日内 半年播报的自报成交 (纯文本配对, 不拉行情)。

他的格式 (中文, 自己的成交播报, [美股] 标签):
  买: "217.8买了三分之一常规仓的ibm" / "38.9 加了 40.2出掉的iren那部分" / "81.7加了三分之一常规仓的crwv"
  卖: "12.02出掉 11.65剩下一半nok" / "38.95平出盘中38.9的那个批次的iren" / "40.2出了一半iren"

审计方法 (两层, 都只用他自己报的数字):
  A. 自配对: 卖出消息里同时含卖价+成本价("38.95平出...38.9的那个批次") → 直接得单笔收益%
  B. FIFO配对: 只有单价的买/卖消息, 按票FIFO配对 (买入队列→卖出冲销)
诚实边界: 这是"他自报数字"的审计, 不是可跟单收益; 无法验证他是否漏报亏损单(选择性播报偏差),
          解析覆盖率如实报告; 好票审计通过 ≠ 跟单能赚(还有延迟/滑点)。
"""
import json, re, sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent

# 中文数量词
FRAC = {"三分之一": 1/3, "三分之二": 2/3, "四分之一": 0.25, "一半": 0.5, "半仓": 0.5,
        "全部": 1.0, "剩下": 0.5}
PRICE = r"(\d{1,5}\.?\d{0,3})"
TICKER = r"([a-z]{1,5})\b"

BUY_RE = re.compile(PRICE + r"\s*(?:又|再)?(?:买|加|接|进|补)")
SELL_RE = re.compile(PRICE + r"\s*(?:出|平|卖|清|止盈|止损|跑)")
TICK_RE = re.compile(r"(?:的|仓|批次|部分|盘中)?\s*([a-z]{1,6})(?:那部分|的部分)?")
# 卖出自带成本: "38.95平出盘中38.9的那个批次" / "12.02出掉 11.65剩下一半"
PAIR_RE = re.compile(PRICE + r"\s*(?:出掉|出了|平出|平掉|卖出|卖掉|清掉|出|平)[^\d]{0,8}" + PRICE)

NOT_TICKER = {"png", "jpg", "jpeg", "gif", "http", "https", "com", "cn", "etf", "ai", "cpi", "ppi",
              "fomc", "gdp", "pce", "sec", "fda", "ipo", "ceo", "cfo", "qqq", "vix"}  # qqq/vix留意:他很少做


def norm(t: str) -> str:
    return " ".join(t.replace("**", " ").replace("[美股]", " ").split())


def find_ticker(t: str):
    # ⚠ \b 在中文↔字母之间不成立("的lite"), 用只对拉丁字母的边界
    cands = [x for x in re.findall(r"(?<![a-z0-9])([a-z]{1,6})(?![a-z0-9])", t)
             if x not in NOT_TICKER]
    return cands[-1] if cands else None   # ticker几乎总在句尾


def main():
    msgs = json.load(open(ROOT / "output" / "zhaoge_history.json"))
    us = [m for m in msgs if "[美股]" in m["text"]]
    print(f"总消息 {len(msgs)} | [美股]标签 {len(us)} 条 ({msgs[0]['ts'][:10]} ~ {msgs[-1]['ts'][:10]})")

    trades = []          # (ts, ticker, ret%)  A层: 自配对
    fifo_buys = defaultdict(deque)   # B层
    fifo_trades = []
    n_buy = n_sell = n_pair = n_unparsed = 0
    seen_txt = set()

    for m in us:
        t = norm(m["text"]).lower()
        if t in seen_txt:            # 他常double-post同一条
            continue
        seen_txt.add(t)
        ts = m["ts"][:10]
        tk = find_ticker(t)
        if tk is None:
            continue
        # A. 卖出自带成本价
        mp = PAIR_RE.search(t)
        did = False
        if mp:
            sell, cost = float(mp.group(1)), float(mp.group(2))
            # 合理性: 同票两价应在±35%内 (排除"38.9加了40.2出掉的那部分"这种是买入回补)
            if 0.65 <= sell / cost <= 1.55 and not BUY_RE.search(t[:mp.start() + 6]):
                trades.append((ts, tk, (sell / cost - 1) * 100))
                n_pair += 1
                did = True
        if not did:
            mb, ms_ = BUY_RE.search(t), SELL_RE.search(t)
            if mb and (not ms_ or mb.start() < ms_.start()):
                fifo_buys[tk].append(float(mb.group(1)))
                n_buy += 1
            elif ms_:
                px = float(ms_.group(1))
                if fifo_buys[tk]:
                    cost = fifo_buys[tk].popleft()
                    if 0.5 <= px / cost <= 2.0:
                        fifo_trades.append((ts, tk, (px / cost - 1) * 100))
                n_sell += 1
            else:
                n_unparsed += 1

    print(f"解析: 自配对成交 {n_pair} | 单边买 {n_buy} / 单边卖 {n_sell} (FIFO配出 {len(fifo_trades)}) | 未解析 {n_unparsed}")

    for name, arr in [("A. 自配对(他消息里自带买卖价, 最可信)", trades),
                      ("B. FIFO配对(单边消息推断, 参考)", fifo_trades),
                      ("A+B 合并", trades + fifo_trades)]:
        if not arr:
            continue
        rets = [r for _, _, r in arr]
        wins = [r for r in rets if r > 0]
        print(f"\n═══ {name} ═══  n={len(rets)}")
        print(f"  胜率 {len(wins)/len(rets)*100:.0f}% | 平均每笔 {sum(rets)/len(rets):+.2f}% | "
              f"中位 {sorted(rets)[len(rets)//2]:+.2f}% | 最好 {max(rets):+.1f}% / 最差 {min(rets):+.1f}%")
        # 按月
        bym = defaultdict(list)
        for ts, _, r in arr:
            bym[ts[:7]].append(r)
        print("  按月: " + " | ".join(f"{m} n={len(v)} 均{sum(v)/len(v):+.2f}%" for m, v in sorted(bym.items())))
        # 热门票
        byt = defaultdict(list)
        for _, tk, r in arr:
            byt[tk].append(r)
        top = sorted(byt.items(), key=lambda kv: -len(kv[1]))[:8]
        print("  热门票: " + " | ".join(f"{tk}×{len(v)}(均{sum(v)/len(v):+.1f}%)" for tk, v in top))

    # 样本抽查 (人工核对解析对不对)
    print("\n── A层样本抽查 (前8笔, 核对原文):")
    shown = 0
    for m in us:
        t = norm(m["text"]).lower()
        mp = PAIR_RE.search(t)
        if mp and shown < 8:
            sell, cost = float(mp.group(1)), float(mp.group(2))
            if 0.65 <= sell / cost <= 1.55 and not BUY_RE.search(t[:mp.start() + 6]):
                print(f"  [{m['ts'][:10]}] 卖{sell} 成本{cost} → {(sell/cost-1)*100:+.1f}%  ← {t[:70]}")
                shown += 1


if __name__ == "__main__":
    main()
