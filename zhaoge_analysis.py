#!/usr/bin/env python3
"""
zhaoge_analysis.py — #股票赵哥-日内 6个月消息 → 提取自含成本的回合交易 + 真实日K交叉验证。

他的卖出消息常自带成本价 (`187.9出掉184的glw` = 184买 187.9卖), 每条即一笔完整回合。
Tier1 = 卖价+成本价+票 三要素齐的消息 (高置信); 其余买/卖单腿只计数。
验证: 卖价必须落在当日日K [low,high] 内(±0.5%), 成本价落在此前15个交易日价格范围内。
"""
import json, re, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")

UTC = timezone.utc
PX = r"(\d+(?:\.\d+)?)"
TK = r"([a-zA-Z]{2,5})"

# 卖出自含成本的句式 (按特异性排序; sell, cost, ticker)
SELL_COST_PATTERNS = [
    re.compile(PX + r"\s*(?:平出|出掉|出)\s*(?:盘中|夜盘)?" + PX + r"(?:的那个批次的|的剩下的?|剩下(?:一半|的)?(?:的)?|的)\s*\.?" + TK),
    re.compile(PX + r"\s*出一半\s*(?:盘中|夜盘)?" + PX + r"的\s*" + TK),
    re.compile(PX + r"\s*出一半\s*" + PX + r"\.?\s*" + TK),        # 12.63出一半12.54.tsll
    re.compile(TK + r"\s*" + PX + r"\s*(?:附近)?出一半\s*" + PX),   # ticker在前
]
# 单腿 (无成本)
SELL_ONE = re.compile(PX + r"\s*(?:附近)?\s*(?:出一半|出掉|平出|出|清)")
BUY_ONE = re.compile(PX + r"\s*(?:附近)?\s*(?:买了|买入|买|加了|加回了|加仓|加|接了|接)")
TK_ANY = re.compile(r"\b[a-zA-Z]{2,5}\b")
CN_NAME = re.compile(r"双倍|三倍|微软|特斯拉|英伟达")

STOPWORDS = {"spx", "spy", "qqq", "vix", "cpi", "ppi", "fomc", "gdp", "ipo", "etf", "ath",
             "ceo", "gtc", "ai", "cpu", "gpu", "nbsp", "amp", "http", "https", "png", "jpg",
             "com", "cdn", "www", "sk", "pce", "adp"}


def strip_prefix(t: str) -> str:
    t = re.sub(r"\*\*\[.+?\]\*\*", " ", t)
    t = re.sub(r"\*\*.+?\*\*:?", " ", t)
    return t.strip()


def extract(msgs):
    trips, sell_legs, buy_legs, cn_only, unparsed_sell = [], [], [], 0, []
    for m in msgs:
        raw = strip_prefix(m["text"])
        if len(raw) > 200 or "http" in raw:      # 长评论/图链跳过
            continue
        d = m["ts"][:10]
        got = False
        for pat in SELL_COST_PATTERNS:
            mm = pat.search(raw)
            if not mm:
                continue
            g = mm.groups()
            if g[0][0].isalpha() or (len(g) == 3 and pat.pattern.startswith("([a-zA-Z]")):
                # ticker在前句式: (tk, sell, cost)
                tk, sell, cost = g[0].lower(), float(g[1]), float(g[2])
            else:
                sell, cost, tk = float(g[0]), float(g[1]), g[2].lower()
            if tk in STOPWORDS or not (0.5 <= sell / cost <= 2.0):
                continue
            trips.append(dict(date=d, ticker=tk, sell=sell, cost=cost,
                              ret=(sell / cost - 1) * 100, raw=" ".join(raw.split())[:80]))
            got = True
            break
        if got:
            continue
        if SELL_ONE.search(raw):
            tks = [w.lower() for w in TK_ANY.findall(raw) if w.lower() not in STOPWORDS]
            if tks:
                sell_legs.append((d, tks[0]))
            elif CN_NAME.search(raw):
                cn_only += 1
            else:
                unparsed_sell.append(raw[:70])
        elif BUY_ONE.search(raw):
            tks = [w.lower() for w in TK_ANY.findall(raw) if w.lower() not in STOPWORDS]
            if tks:
                buy_legs.append((d, tks[0]))
            elif CN_NAME.search(raw):
                cn_only += 1
    return trips, sell_legs, buy_legs, cn_only, unparsed_sell


def verify(trips):
    """真实日K验证: 卖价在当日[low,high]±0.5%; 成本在近15交易日range内。"""
    from longport.openapi import Config, QuoteContext, Period, AdjustType
    q = QuoteContext(Config.from_env())
    daily = {}
    def get_daily(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk.upper()}.US", Period.Day, 300, AdjustType.NoAdjust)
                daily[tk] = {str(x.timestamp.date()): (float(x.low), float(x.high)) for x in b}
            except Exception:
                daily[tk] = None
        return daily[tk]
    ok = bad = nodata = 0
    bad_rows = []
    for t in trips:
        db = get_daily(t["ticker"])
        if not db:
            nodata += 1; t["verify"] = "无日K"; continue
        days = sorted(db)
        if t["date"] not in db:
            # 周末发的复盘? 找最近前一交易日
            prior = [x for x in days if x <= t["date"]]
            if not prior:
                nodata += 1; t["verify"] = "无日K"; continue
            dd = prior[-1]
        else:
            dd = t["date"]
        lo, hi = db[dd]
        sell_ok = lo * 0.995 <= t["sell"] <= hi * 1.005
        idx = days.index(dd)
        win = [db[x] for x in days[max(0, idx - 15):idx + 1]]
        clo, chi = min(w[0] for w in win), max(w[1] for w in win)
        cost_ok = clo * 0.99 <= t["cost"] <= chi * 1.01
        if sell_ok and cost_ok:
            ok += 1; t["verify"] = "✅"
        else:
            bad += 1; t["verify"] = f"❌sell_in_range={sell_ok} cost_in_range={cost_ok}"
            bad_rows.append(t)
    return ok, bad, nodata, bad_rows


def main():
    msgs = json.load(open("output/zhaoge_history.json"))
    trips, sell_legs, buy_legs, cn_only, unp = extract(msgs)
    print(f"消息总数 {len(msgs)} | Tier1自含成本回合 {len(trips)} | 单腿卖 {len(sell_legs)} | "
          f"单腿买 {len(buy_legs)} | 中文名票 {cn_only} | 未解析卖 {len(unp)}")

    if "--no-verify" not in sys.argv:
        ok, bad, nodata, bad_rows = verify(trips)
        print(f"\n真实日K交叉验证: ✅{ok} / ❌{bad} / 无数据{nodata}  (可信度 {ok/(ok+bad)*100 if ok+bad else 0:.0f}%)")
        for t in bad_rows[:8]:
            print(f"   ❌ [{t['date']}] {t['ticker']} 卖{t['sell']} 成本{t['cost']}: {t['verify']} ← {t['raw']}")
        trips_ok = [t for t in trips if t.get("verify") == "✅"]
    else:
        trips_ok = trips

    if not trips_ok:
        print("无可信回合"); return
    import statistics as st
    rets = [t["ret"] for t in trips_ok]
    wins = [r for r in rets if r > 0]
    print(f"\n═══ 赵哥自报回合 (仅日K验证通过的 {len(trips_ok)} 笔) ═══")
    print(f"胜率 {len(wins)/len(rets)*100:.0f}% | 平均 {st.mean(rets):+.2f}% | 中位 {st.median(rets):+.2f}% | "
          f"最大赢 {max(rets):+.1f}% / 最大亏 {min(rets):+.1f}%")
    print(f"简单等权累计 (每笔满仓复利不现实, 仅参考): {sum(rets):+.1f}% (算术和)")
    from collections import Counter
    mon = Counter(t["date"][:7] for t in trips_ok)
    print(f"月度笔数: {dict(sorted(mon.items()))}")
    by_m = {}
    for t in trips_ok:
        by_m.setdefault(t["date"][:7], []).append(t["ret"])
    print("月度算术和: " + " | ".join(f"{k}: {sum(v):+.1f}%" for k, v in sorted(by_m.items())))
    tk_c = Counter(t["ticker"] for t in trips_ok)
    print(f"高频票: {tk_c.most_common(10)}")
    print("\n-- 样本 (前10) --")
    for t in trips_ok[:10]:
        print(f"[{t['date']}] {t['ticker']:5} {t['cost']}→{t['sell']} ({t['ret']:+.1f}%) ← {t['raw']}")


if __name__ == "__main__":
    main()
