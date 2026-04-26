"""
backtest_orb_stress.py — ORB 策略极限压力测试

三个杀手级测试（目标：打破策略）:
  1. 滑点翻倍：entry 0.3% / stop 0.6%（小盘薄盘口最糟场景）
  2. 剔除最好 10% 交易：验证是否靠运气爆发单撑起
  3. Monte Carlo Bootstrap：打乱交易顺序 1000 次，求 95% 置信区间

只测最有希望的 4 个组合:
  - OKLO / IREN / HOOD / TSLL（都是双向，前一步全部达标）

标准：通过压力测试的要求
  - 2x 滑点下仍盈利 AND 胜率下降 <15%
  - 剔除最好 10% 后仍跑赢 B&H
  - Monte Carlo 95% 置信区间下限 > 0%
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from backtest_soxl_5m import fetch_intraday
from backtest_orb_validated import orb_backtest_v2, buy_and_hold_return


def monte_carlo_bootstrap(trade_pnls: np.ndarray, n_samples: int = 1000) -> dict:
    """对每笔交易的 pnl_pct 做 bootstrap 采样，求终收益的分布"""
    if len(trade_pnls) == 0:
        return {}
    n = len(trade_pnls)
    final_returns = np.zeros(n_samples)
    rng = np.random.default_rng(42)
    for k in range(n_samples):
        idx = rng.choice(n, size=n, replace=True)  # 有放回
        sample = trade_pnls[idx] / 100  # 百分比转小数
        final_returns[k] = (1 + sample).prod() - 1
    return {
        "mean": final_returns.mean() * 100,
        "median": np.median(final_returns) * 100,
        "std": final_returns.std() * 100,
        "p5": np.percentile(final_returns, 5) * 100,
        "p25": np.percentile(final_returns, 25) * 100,
        "p75": np.percentile(final_returns, 75) * 100,
        "p95": np.percentile(final_returns, 95) * 100,
        "pct_positive": (final_returns > 0).sum() / n_samples * 100,
    }


def exclude_top_n_pct(trade_pnls: np.ndarray, top_pct: float = 10) -> float:
    """剔除 top_pct% 最好交易后的累积收益"""
    if len(trade_pnls) == 0:
        return 0
    sorted_pnls = np.sort(trade_pnls)
    n_exclude = int(len(trade_pnls) * top_pct / 100)
    kept = sorted_pnls[:-n_exclude] if n_exclude > 0 else sorted_pnls
    return (1 + kept / 100).prod() - 1


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    tickers_to_test = [
        # (ticker, base_entry_slip, base_stop_slip, description)
        ("OKLO", 0.0015, 0.0030, "小盘高波动核能"),
        ("IREN", 0.0015, 0.0030, "小盘比特币挖矿"),
        ("HOOD", 0.0008, 0.0015, "券商中等流动性"),
        ("TSLL", 0.0008, 0.0015, "Tesla 2x杠杆ETF"),
        ("PLTR", 0.0005, 0.0010, "大盘AI股高流动"),
        ("NBIS", 0.0010, 0.0020, "AI基础设施中盘"),
        ("CIFR", 0.0015, 0.0030, "比特币挖矿小盘"),
        ("CRWV", 0.0010, 0.0020, "AI算力新IPO"),
    ]

    base_params = {
        "or_bars": 3,
        "target_r": 2.0,
        "use_volume_filter": True,
        "rvol_threshold": 1.5,
        "commission": 0.0,
        "normal_slip": 0.0005,
        "long_only": False,
    }

    # 下载数据
    print("=" * 110)
    print("ORB 压力测试 | 滑点翻倍 + 剔除top10% + Monte Carlo Bootstrap")
    print("=" * 110)
    data = {}
    for t, *_ in tickers_to_test:
        df = fetch_intraday(t, period="60d", interval="5m")
        if not df.empty:
            data[t] = df
            print(f"  {t}: {len(df)} 根5m")

    # ============================================================
    # 测试1：滑点翻倍
    # ============================================================
    print(f"\n{'=' * 110}")
    print("【压力测试1】滑点翻倍（小盘 entry 0.3% / stop 0.6%，液性好的 0.16% / 0.3%）")
    print("=" * 110)
    print(f"{'Ticker':<7} {'基准收益':>9} {'基准胜率':>9} {'2x滑点收益':>11} {'2x胜率':>8} "
          f"{'收益衰减':>9} {'交易数':>7} {'结论':<15}")
    print("-" * 110)

    slip_results = {}
    for t, e_slip, s_slip, note in tickers_to_test:
        if t not in data:
            continue
        # 基准 (1x 滑点)
        base_res = orb_backtest_v2(data[t].copy(), entry_slip=e_slip, stop_slip=s_slip, **base_params)
        # 2x 滑点
        stress_res = orb_backtest_v2(data[t].copy(), entry_slip=e_slip*2, stop_slip=s_slip*2, **base_params)

        base_ret = float(base_res["metrics"]["总收益率"].rstrip('%'))
        base_win = float(base_res["metrics"]["真实胜率"].rstrip('%'))
        stress_ret = float(stress_res["metrics"]["总收益率"].rstrip('%'))
        stress_win = float(stress_res["metrics"]["真实胜率"].rstrip('%'))
        win_drop = base_win - stress_win
        decay = base_ret - stress_ret

        verdict = "✅ 稳健" if stress_ret > 0 and win_drop < 15 else \
                  "⚠️ 边缘" if stress_ret > 0 else "❌ 失败"

        slip_results[t] = {
            "base": base_res, "stress": stress_res,
            "base_ret": base_ret, "stress_ret": stress_ret,
            "verdict": verdict,
        }

        print(f"{t:<7} {base_ret:>+8.2f}% {base_win:>8.1f}% {stress_ret:>+10.2f}% "
              f"{stress_win:>7.1f}% {decay:>+8.2f}% {base_res['metrics']['交易次数']:>7} {verdict:<15}")

    # ============================================================
    # 测试2：剔除最好 10% 交易
    # ============================================================
    print(f"\n{'=' * 110}")
    print("【压力测试2】剔除最好 10% 交易 → 验证是否靠运气爆发单")
    print("=" * 110)
    print(f"{'Ticker':<7} {'原始收益':>9} {'剔除top10%':>11} {'剔除top20%':>11} "
          f"{'B&H':>8} {'剔top10%仍胜B&H?':<18}")
    print("-" * 110)

    exclusion_results = {}
    for t, e_slip, s_slip, note in tickers_to_test:
        if t not in data:
            continue
        res = slip_results[t]["base"]
        td = res["trades"]
        if td.empty:
            continue

        trade_pnls = td["pnl_pct"].values
        orig_ret = (1 + trade_pnls/100).prod() - 1
        ret_top10 = exclude_top_n_pct(trade_pnls, 10)
        ret_top20 = exclude_top_n_pct(trade_pnls, 20)
        bh = buy_and_hold_return(data[t])

        verdict = "✅ 胜 B&H" if ret_top10 > bh else "❌ 不如 B&H"
        exclusion_results[t] = {
            "orig_ret": orig_ret * 100,
            "ret_top10": ret_top10 * 100,
            "ret_top20": ret_top20 * 100,
            "bh": bh * 100,
            "verdict": verdict,
        }
        print(f"{t:<7} {orig_ret*100:>+8.2f}% {ret_top10*100:>+10.2f}% "
              f"{ret_top20*100:>+10.2f}% {bh*100:>+7.2f}% {verdict:<18}")

    # ============================================================
    # 测试3：Monte Carlo Bootstrap（1000 次重采样）
    # ============================================================
    print(f"\n{'=' * 110}")
    print("【压力测试3】Monte Carlo Bootstrap 1000 次 → 95% 置信区间")
    print("=" * 110)
    print(f"{'Ticker':<7} {'中位数':>9} {'均值':>9} {'P5(下限)':>10} {'P95(上限)':>10} "
          f"{'正收益概率':>12} {'结论':<18}")
    print("-" * 110)

    mc_results = {}
    for t, e_slip, s_slip, note in tickers_to_test:
        if t not in data:
            continue
        res = slip_results[t]["base"]
        td = res["trades"]
        if td.empty:
            continue

        mc = monte_carlo_bootstrap(td["pnl_pct"].values, n_samples=1000)
        mc_results[t] = mc

        verdict = "✅ 显著为正" if mc["p5"] > 0 else \
                  "⚠️ 有亏损风险" if mc["p5"] > -20 else "❌ 高风险"

        print(f"{t:<7} {mc['median']:>+8.2f}% {mc['mean']:>+8.2f}% "
              f"{mc['p5']:>+9.2f}% {mc['p95']:>+9.2f}% "
              f"{mc['pct_positive']:>10.1f}% {verdict:<18}")

    # ============================================================
    # 综合结论
    # ============================================================
    print(f"\n{'=' * 110}")
    print("【综合结论】三项压力测试汇总")
    print("=" * 110)
    print(f"{'Ticker':<7} {'滑点翻倍':<12} {'剔除top10%':<14} {'MC 下限>0':<12} {'综合':<20}")
    print("-" * 110)

    passing_tickers = []
    for t, *_ in tickers_to_test:
        if t not in data:
            continue
        s1 = "✅" if "稳健" in slip_results[t]["verdict"] else ("⚠️" if "边缘" in slip_results[t]["verdict"] else "❌")
        s2 = "✅" if exclusion_results[t]["ret_top10"] > exclusion_results[t]["bh"] else "❌"
        s3 = "✅" if mc_results[t]["p5"] > 0 else "❌"
        passes = sum(1 for s in [s1, s2, s3] if s == "✅")

        if passes == 3:
            overall = "🎯 全部通过，强推"
            passing_tickers.append(t)
        elif passes == 2:
            overall = "✅ 通过2/3，可尝试"
        else:
            overall = "❌ 不建议"
        print(f"{t:<7} {s1:<12} {s2:<14} {s3:<12} {overall:<20}")

    # 最终建议
    print(f"\n{'=' * 110}")
    print("【最终建议】")
    print("=" * 110)
    if passing_tickers:
        print(f"\n✅ 通过全部压力测试的标的: {passing_tickers}")
        print(f"   建议行动:")
        print(f"   1. 这些标的可以考虑上 Paper Trading 30 天验证实盘执行")
        print(f"   2. 上付费 API (Polygon.io) 拉 6 个月历史数据做更长周期验证")
        print(f"   3. 同时研究不同市场环境下的表现（VIX 高/低日分别分析）")
    else:
        print("\n⚠️ 没有标的通过全部三项测试")

    # 附: 关键数据导出供后续分析
    print(f"\n{'=' * 110}")
    print("【原始交易样本】每个通过标的的前5笔 + 后5笔交易")
    print("=" * 110)
    for t in passing_tickers[:3]:  # 最多显示3个
        td = slip_results[t]["base"]["trades"]
        if td.empty:
            continue
        print(f"\n── {t} 前5笔 ──")
        print(td.head(5).to_string())
        print(f"── {t} 后5笔 ──")
        print(td.tail(5).to_string())
