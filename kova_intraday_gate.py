"""
kova_intraday_gate.py — 用 Kova 日线状态给日内实盘做"质量闸门", 诚实回测能否提升。

思路(用户): 每天先用 Kova 高分/健康度筛票, 日内策略只在好票好状态上找入场点。
验证: 对每只票跑当前最优日内策略 → 每笔 FILLED 交易 join 它【前一日收盘】算的 Kova 上下文
      (开盘前已知, 绝无未来函数) → 按 Kova 状态分桶比 赢率/盈亏比(PF)/平均盈亏/期望。

铁律: fill-realistic(只用引擎判定 FILLED 的真实交易); 不伪造信号; 如实报样本数(闸门砍掉多少笔).

用法: python3 kova_intraday_gate.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from backtest_engine import run_backtest, load_5m_data
from signals.strategies.intraday_pool import STRATEGIES

REDUCE_DEV = 2.93


def _ctx():
    from longport.openapi import Config, QuoteContext
    return QuoteContext(Config.from_env())


def _pull_daily(ctx, tk, n=600):
    from longport.openapi import Period, AdjustType
    try:
        b = list(ctx.candlesticks(f"{tk}.US", Period.Day, n, AdjustType.ForwardAdjust))
        df = pd.DataFrame({"High": [float(x.high) for x in b], "Low": [float(x.low) for x in b],
                           "Close": [float(x.close) for x in b], "Volume": [float(x.volume) for x in b]},
                          index=[x.timestamp for x in b]).sort_index()
        df.index = (df.index.tz_localize(None) if df.index.tz else df.index).normalize()
        return df
    except Exception:
        return None


def _atr(df, n=14):
    h, l, c = df.High, df.Low, df.Close
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1).rolling(n).mean()


def build_kova_daily(df, qqq):
    """逐日 Kova 上下文(每行用截至当日收盘的数据算)。调用方对交易日 D 取 < D 的最后一行=前日收盘已知。"""
    c, h, l, v = df.Close, df.High, df.Low, df.Volume
    e10 = c.ewm(span=10).mean(); e20 = c.ewm(span=20).mean(); e21 = c.ewm(span=21).mean()
    s50 = c.rolling(50).mean(); s200 = c.rolling(200).mean()
    atr = _atr(df); hi250 = c.rolling(250).max()
    a10 = c > e10; a20 = c > e20; a50 = c > s50; a200 = c > s200
    distHi = (c / hi250 - 1) * 100
    d20 = (c / e20 - 1) * 100
    dev21 = (c - e21) / atr
    vr = v / v.rolling(20).mean()
    # rsHi vs QQQ
    idx = c.index.intersection(qqq.index)
    ratio = (c.reindex(idx) / qqq.Close.reindex(idx))
    rsHi = (ratio / ratio.rolling(252).max() - 1) * 100
    rsHi = rsHi.reindex(c.index)
    score = (71.6 + 27.6 * a200.astype(int) + 1.05 * rsHi).clip(1, 99)
    # 动能 z
    clv = ((c - l) - (h - c)) / (h - l + 1e-9); ad = (clv * v).cumsum()
    wave = ad.ewm(span=9).mean() - ad.ewm(span=21).mean()
    momz = (wave - wave.rolling(120).mean()) / wave.rolling(120).std()
    # 健康度
    unhealthy = ~(a20 & a50 & a200)
    watch = (~unhealthy) & (~a10) & (d20 >= 0) & (d20 < 3) & (distHi > -6) & (vr >= 1.5)
    health = np.where(unhealthy, "UNHEALTHY", np.where(watch, "WATCH", "HEALTHY"))
    healthy = (~unhealthy) & (~watch)
    reduce = healthy & (dev21 >= REDUCE_DEV)
    out = pd.DataFrame({"score": score, "health": health, "rsHi": rsHi, "dev21": dev21,
                        "reduce": reduce, "momz": momz, "a200": a200.astype(int)}, index=c.index)
    return out


def lookup(kdf, d):
    """交易日 d(date) 的前一日已知 Kova 上下文。"""
    ts = pd.Timestamp(d)
    prior = kdf[kdf.index < ts]
    if len(prior) == 0:
        return None
    return prior.iloc[-1]


def pf_of(pnls):
    g = pnls[pnls > 0].sum(); b = -pnls[pnls < 0].sum()
    return (g / b) if b > 0 else float("inf")


def main():
    best = pd.read_csv("output/best_strategy_per_ticker.csv")
    best = best[best["verdict"] == "OK"]
    ctx = _ctx()
    qqq = _pull_daily(ctx, "QQQ", 700)
    daily_cache = {}
    all_trades = []
    for _, r in best.iterrows():
        sym = r["symbol"]; strat = r["strategy"]
        if strat not in STRATEGIES:
            continue
        tk = sym.replace(".US", "")
        if load_5m_data(sym).empty:
            continue
        try:
            recs = run_backtest(sym, STRATEGIES[strat], strat)
        except Exception as e:
            print(f"  ⚠ {sym} 回测失败: {e}"); continue
        if tk not in daily_cache:
            dd = _pull_daily(ctx, tk)
            daily_cache[tk] = build_kova_daily(dd, qqq) if dd is not None else None
        kdf = daily_cache[tk]
        if kdf is None:
            continue
        for t in recs:
            if t.fill_status != "FILLED" or t.pnl_pct is None:
                continue
            k = lookup(kdf, t.day)
            if k is None or pd.isna(k["score"]):
                continue
            all_trades.append(dict(symbol=sym, strategy=strat, day=t.day, side=t.side,
                                   pnl=t.pnl_pct, win=int(t.pnl_pct > 0), reason=t.exit_reason,
                                   score=float(k["score"]), health=k["health"], rsHi=float(k["rsHi"]),
                                   reduce=bool(k["reduce"]), momz=float(k["momz"]) if not pd.isna(k["momz"]) else np.nan,
                                   a200=int(k["a200"])))
    tr = pd.DataFrame(all_trades)
    print(f"=== 总样本: {len(tr)} 笔 FILLED 交易 (跨 {tr.symbol.nunique()} 票), 都join了前一日Kova上下文 ===")
    base_pf = pf_of(tr.pnl); base_win = tr.win.mean() * 100; base_avg = tr.pnl.mean()
    print(f"基线(全部交易): n={len(tr)} 胜率={base_win:.1f}% PF={base_pf:.2f} 平均={base_avg:+.3f}% 总={tr.pnl.sum():+.0f}%")

    def bucket_report(name, groups):
        print(f"\n── 按 {name} 分桶 ──")
        print(f"{'桶':18}{'n':>6}{'胜率':>8}{'PF':>7}{'平均%':>9}{'总%':>9}")
        for label, mask in groups:
            sub = tr[mask]
            if len(sub) < 10:
                print(f"{label:18}{len(sub):>6}  (样本不足)"); continue
            print(f"{label:18}{len(sub):>6}{sub.win.mean()*100:>7.1f}%{pf_of(sub.pnl):>7.2f}{sub.pnl.mean():>+9.3f}{sub.pnl.sum():>+9.0f}")

    bucket_report("Kova Score", [
        ("Score<50", tr.score < 50), ("50≤Score<70", (tr.score >= 50) & (tr.score < 70)),
        ("70≤Score<85", (tr.score >= 70) & (tr.score < 85)), ("Score≥85", tr.score >= 85)])
    bucket_report("健康度", [("HEALTHY", tr.health == "HEALTHY"),
        ("WATCH", tr.health == "WATCH"), ("UNHEALTHY", tr.health == "UNHEALTHY")])
    bucket_report("动能z(资金流向)", [("z<-0.5 红撤出", tr.momz < -0.5),
        ("-0.5~0.5 中性", (tr.momz >= -0.5) & (tr.momz < 0.5)), ("z≥0.5 蓝进场", tr.momz >= 0.5)])
    bucket_report("REDUCE状态", [("非REDUCE", ~tr.reduce), ("REDUCE中", tr.reduce)])
    bucket_report("站200S", [("站上200S", tr.a200 == 1), ("跌破200S", tr.a200 == 0)])

    # 组合闸门候选
    print("\n=== 闸门候选 (vs 基线 PF={:.2f}, n={}) ===".format(base_pf, len(tr)))
    gates = {
        "Score≥70": tr.score >= 70,
        "Score≥70 且 HEALTHY": (tr.score >= 70) & (tr.health == "HEALTHY"),
        "HEALTHY 且 非REDUCE": (tr.health == "HEALTHY") & (~tr.reduce),
        "Score≥70 且 动能z≥0": (tr.score >= 70) & (tr.momz >= 0),
        "Score≥70 且 HEALTHY 且 动能z≥0": (tr.score >= 70) & (tr.health == "HEALTHY") & (tr.momz >= 0),
    }
    print(f"{'闸门':32}{'n':>6}{'留存%':>7}{'胜率':>8}{'PF':>7}{'平均%':>9}{'总%':>9}")
    for name, mask in gates.items():
        sub = tr[mask]
        if len(sub) < 10:
            print(f"{name:32}{len(sub):>6}  (样本不足)"); continue
        print(f"{name:32}{len(sub):>6}{len(sub)/len(tr)*100:>6.0f}%{sub.win.mean()*100:>7.1f}%{pf_of(sub.pnl):>7.2f}{sub.pnl.mean():>+9.3f}{sub.pnl.sum():>+9.0f}")

    tr.to_csv("kova_intraday_gate_trades.csv", index=False)
    print(f"\n逐笔(含Kova上下文)存档 → kova_intraday_gate_trades.csv ({len(tr)} 行)")


if __name__ == "__main__":
    main()
