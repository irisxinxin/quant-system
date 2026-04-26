"""
backtest_orb_long_only.py — ORB 多头 only 版本测试
现实约束：散户难以做空个股，重新验证只做多时每只股票的表现
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_soxl_5m import fetch_intraday
from backtest_orb_validated import orb_backtest_v2, buy_and_hold_return


if __name__ == "__main__":
    tickers_to_test = [
        ("OKLO", 0.0015, 0.0030),
        ("IREN", 0.0015, 0.0030),
        ("HOOD", 0.0008, 0.0015),
        ("TSLL", 0.0008, 0.0015),
        ("PLTR", 0.0005, 0.0010),
        ("NBIS", 0.0010, 0.0020),
        ("CIFR", 0.0015, 0.0030),
        ("CRWV", 0.0010, 0.0020),
        ("SOXL", 0.0005, 0.0010),
    ]

    base_params = {
        "or_bars": 3,
        "target_r": 2.0,
        "use_volume_filter": True,
        "rvol_threshold": 1.5,
        "commission": 0.0,
        "normal_slip": 0.0005,
    }

    print("=" * 110)
    print("ORB 只做多 vs 多空双向 对比 | 真实滑点")
    print("=" * 110)
    print(f"{'Ticker':<7} {'只多收益':>9} {'只多胜率':>9} {'只多交易':>9} "
          f"{'多空收益':>9} {'多空胜率':>9} {'多空交易':>9} {'B&H':>8} {'只多胜B&H':<10}")
    print("-" * 110)

    for t, e_slip, s_slip in tickers_to_test:
        df = fetch_intraday(t, period="60d", interval="5m")
        if df.empty:
            continue

        # 只做多
        long_res = orb_backtest_v2(df.copy(), long_only=True,
                                    entry_slip=e_slip, stop_slip=s_slip, **base_params)
        # 多空双向
        both_res = orb_backtest_v2(df.copy(), long_only=False,
                                    entry_slip=e_slip, stop_slip=s_slip, **base_params)
        bh = buy_and_hold_return(df) * 100

        l_ret = float(long_res["metrics"]["总收益率"].rstrip("%"))
        l_win = long_res["metrics"]["真实胜率"]
        l_trd = long_res["metrics"]["交易次数"]
        b_ret = float(both_res["metrics"]["总收益率"].rstrip("%"))
        b_win = both_res["metrics"]["真实胜率"]
        b_trd = both_res["metrics"]["交易次数"]

        wins_bh = "✅" if l_ret > bh else "❌"
        print(f"{t:<7} {l_ret:>+8.2f}% {l_win:>9} {l_trd:>9} "
              f"{b_ret:>+8.2f}% {b_win:>9} {b_trd:>9} "
              f"{bh:>+7.2f}% {wins_bh:<10}")

    # 只做多的推荐组合
    print("\n" + "=" * 110)
    print("【只做多 可行性判断】")
    print("=" * 110)
    print("""
推断规则：
  ✅ 强推: 只多收益 > 30% AND 只多胜 B&H AND 交易数 ≥ 15
  🤔 可做: 只多收益 > 10% AND 只多胜 B&H
  ❌ 不行: 只多 < 10% OR 不如 B&H
""")
