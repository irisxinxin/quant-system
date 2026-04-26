"""
longport_stress_3_tests.py — 18个月数据上跑3项压力测试

1. 滑点翻倍 (entry 0.3% + stop 0.6% 小盘 / 0.06% + 0.16% 大盘)
2. Walk-Forward (前12月训练 vs 后6月样本外)
3. 月度收益分布 (验证盈利是否稳定)
"""
import sys
import pickle
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from backtest_orb_validated import orb_backtest_v2, buy_and_hold_return

CACHE_DIR = Path(__file__).parent / "cache" / "longport_history"

TICKERS = [
    ("OKLO.US", 0.0015, 0.0030, "核能小盘"),
    ("IREN.US", 0.0015, 0.0030, "BTC挖矿"),
    ("CIFR.US", 0.0015, 0.0030, "BTC挖矿"),
    ("EOSE.US", 0.0025, 0.0050, "储能小盘"),
    ("PLTR.US", 0.0003, 0.0008, "AI大盘"),
    ("AMZN.US", 0.0002, 0.0005, "超大盘"),
    ("TSLL.US", 0.0005, 0.0010, "2x特斯拉"),
    ("NVDL.US", 0.0005, 0.0010, "2x英伟达"),
    ("SOXL.US", 0.0003, 0.0008, "3x半导体"),
]

base_params = {
    "or_bars": 3,
    "target_r": 2.0,
    "use_volume_filter": True,
    "rvol_threshold": 1.5,
    "commission": 0.0,
    "normal_slip": 0.0005,
    "long_only": True,
}


def load_cached(symbol):
    """从缓存加载已下载的数据 (已经过滤了 regular hours)"""
    f = CACHE_DIR / f"{symbol.replace('.', '_')}_5m.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        df = pickle.load(fh)
    # 转换 ET 时区到 naive (orb_backtest_v2 期望 naive index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ════════════════════════════════════════════════════════════════════
# 测试 1: 滑点翻倍
# ════════════════════════════════════════════════════════════════════
def test_1_slippage_doubled():
    print("=" * 110)
    print("【测试 1】滑点翻倍 (entry/stop 都 ×2) — 评估真实交易成本下的稳健性")
    print("=" * 110)
    print(f"{'Ticker':<10} {'类型':<10} {'基准':>10} {'2x滑点':>10} {'衰减':>9} {'胜率':>7} "
          f"{'交易':>5} {'回撤':>8} {'结论':<10}")
    print("-" * 110)

    results = {}
    for sym, e_slip, s_slip, note in TICKERS:
        df = load_cached(sym)
        if df is None or len(df) < 500:
            continue

        # 基准
        base = orb_backtest_v2(df.copy(), entry_slip=e_slip, stop_slip=s_slip, **base_params)
        # 2x 滑点
        stress = orb_backtest_v2(df.copy(), entry_slip=e_slip*2, stop_slip=s_slip*2, **base_params)

        b_ret = float(base["metrics"]["总收益率"].rstrip("%"))
        s_ret = float(stress["metrics"]["总收益率"].rstrip("%"))
        decay = b_ret - s_ret
        win = stress["metrics"]["真实胜率"]
        trades = stress["metrics"]["交易次数"]
        dd = stress["metrics"]["最大回撤"]

        verdict = "✅ 稳健" if s_ret > 50 else "⚠️ 衰减大" if s_ret > 0 else "❌ 失败"
        results[sym] = {"base": b_ret, "stress": s_ret, "decay": decay}

        print(f"{sym:<10} {note:<10} {b_ret:>+9.1f}% {s_ret:>+9.1f}% {decay:>+8.1f}% "
              f"{win:>7} {trades:>5} {dd:>8} {verdict:<10}")

    return results


# ════════════════════════════════════════════════════════════════════
# 测试 2: Walk-Forward
# ════════════════════════════════════════════════════════════════════
def test_2_walk_forward():
    print(f"\n{'=' * 110}")
    print("【测试 2】Walk-Forward — 前 12 月 (训练) vs 后 6 月 (样本外)")
    print("=" * 110)
    print(f"{'Ticker':<10} {'前12月':>10} {'后6月':>10} {'前BH':>9} {'后BH':>9} "
          f"{'前α':>9} {'后α':>9} {'胜率(后)':>9} {'结论'}")
    print("-" * 110)

    results = {}
    for sym, e_slip, s_slip, note in TICKERS:
        df = load_cached(sym)
        if df is None or len(df) < 1000:
            continue

        # 切分: 前 2/3 训练, 后 1/3 测试 (≈ 前12月 后6月)
        split = int(len(df) * 2 / 3)
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        train_res = orb_backtest_v2(train_df.copy(), entry_slip=e_slip, stop_slip=s_slip, **base_params)
        test_res = orb_backtest_v2(test_df.copy(), entry_slip=e_slip, stop_slip=s_slip, **base_params)
        train_bh = buy_and_hold_return(train_df) * 100
        test_bh = buy_and_hold_return(test_df) * 100

        train_ret = float(train_res["metrics"]["总收益率"].rstrip("%"))
        test_ret = float(test_res["metrics"]["总收益率"].rstrip("%"))
        train_alpha = train_ret - train_bh
        test_alpha = test_ret - test_bh
        test_win = test_res["metrics"]["真实胜率"]

        # 判定: 后 6 月仍盈利 + 仍跑赢 BH = 稳健
        if test_ret > 30 and test_alpha > 0:
            verdict = "✅ 稳健"
        elif test_ret > 0 and test_alpha > 0:
            verdict = "🟡 衰退但盈利"
        elif test_ret > 0:
            verdict = "⚠️ 跑输 BH"
        else:
            verdict = "❌ 后 6 月亏损"

        results[sym] = {
            "train": train_ret, "test": test_ret,
            "train_alpha": train_alpha, "test_alpha": test_alpha,
            "verdict": verdict,
        }

        print(f"{sym:<10} {train_ret:>+9.1f}% {test_ret:>+9.1f}% {train_bh:>+8.1f}% {test_bh:>+8.1f}% "
              f"{train_alpha:>+8.1f}% {test_alpha:>+8.1f}% {test_win:>9} {verdict}")

    return results


# ════════════════════════════════════════════════════════════════════
# 测试 3: 月度收益分布
# ════════════════════════════════════════════════════════════════════
def test_3_monthly_distribution():
    print(f"\n{'=' * 110}")
    print("【测试 3】月度收益分布 — 收益是否稳定 vs 集中在少数月份")
    print("=" * 110)

    for sym, e_slip, s_slip, note in TICKERS:
        df = load_cached(sym)
        if df is None or len(df) < 1000:
            continue

        result = orb_backtest_v2(df.copy(), entry_slip=e_slip, stop_slip=s_slip, **base_params)
        td = result["trades"]
        if td.empty:
            continue

        # 按月聚合
        td["exit_dt"] = pd.to_datetime(td["exit_date"])
        td["month"] = td["exit_dt"].dt.to_period("M")
        monthly = td.groupby("month").agg(
            笔数=("pnl_pct", "count"),
            总盈亏=("pnl_pct", "sum"),
            胜笔=("pnl_pct", lambda x: (x > 0).sum()),
        )
        monthly["月度复利"] = (1 + td.groupby("month")["pnl_pct"].apply(
            lambda x: (1 + x/100).prod() - 1) * 100).pow(1) - 1
        monthly["月度复利"] = td.groupby("month")["pnl_pct"].apply(
            lambda x: ((1 + x/100).prod() - 1) * 100)

        # 统计
        n_months = len(monthly)
        positive_months = (monthly["月度复利"] > 0).sum()
        negative_months = (monthly["月度复利"] < 0).sum()
        best_month = monthly["月度复利"].max()
        worst_month = monthly["月度复利"].min()
        median_month = monthly["月度复利"].median()

        print(f"\n── {sym} ({note}) ──")
        print(f"  总月数: {n_months}  正收益: {positive_months}  负收益: {negative_months}  "
              f"正比例: {positive_months/n_months*100:.0f}%")
        print(f"  最佳月: {best_month:+.1f}%  最差月: {worst_month:+.1f}%  中位数月: {median_month:+.2f}%")

        # 打印每月概况 (只打印近 12 月避免太长)
        recent_months = monthly.tail(12)
        print(f"  近 12 月明细:")
        for month, row in recent_months.iterrows():
            print(f"    {month}: {row['月度复利']:+7.2f}% ({row['笔数']:>2.0f}笔, 胜 {row['胜笔']:>2.0f})")


if __name__ == "__main__":
    test_1_slippage_doubled()
    test_2_walk_forward()
    test_3_monthly_distribution()

    # 综合判定
    print(f"\n{'=' * 110}")
    print("【综合判定】")
    print("=" * 110)
    print("""
判定标准:
  ✅ 全部通过: 滑点翻倍仍 >50%, 后 6 月仍盈利, 月正比 ≥ 60%
  🟡 部分通过: 至少 2/3 通过
  ❌ 不通过: 至少 2/3 不达标

最终意见见上方各项 "结论" 列
""")
