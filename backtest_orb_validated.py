"""
backtest_orb_validated.py — ORB 完整验证报告

三件事:
  1. Buy & Hold 基准对比（策略是否真的跑赢简单持有？）
  2. 真实滑点模型（按标的流动性分组）
  3. Walk-Forward 样本外测试（前30天训练，后30天验证）

标的: OKLO / IREN / HOOD / NVDL / TSLL / SOXL
     （SOXL 用户要求只做多，其他双向）
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from backtest_soxl_5m import fetch_intraday


# ============================================================
# ORB 回测（改进版：分离 entry/stop/normal 滑点）
# ============================================================
def orb_backtest_v2(
    df: pd.DataFrame,
    or_bars: int = 3,
    target_r: float = 2.0,
    use_volume_filter: bool = True,
    rvol_threshold: float = 1.5,
    rvol_lookback: int = 20,
    long_only: bool = True,
    eod_exit_bars: int = 4,
    # 真实滑点分三种
    commission: float = 0.0,        # Alpaca 免佣
    entry_slip: float = 0.001,       # 突破入场滑点 0.1%（市价单打穿盘口）
    stop_slip: float = 0.002,        # 止损滑点 0.2%（gap-down 风险）
    normal_slip: float = 0.0005,     # 正常出场滑点 0.05%
    min_or_range_pct: float = 0.003,
) -> dict:
    """
    改进版 ORB:
    - 入场价 = OR 边界 × (1 + entry_slip)（多） 或 × (1 - entry_slip)（空）
    - 止损价触发时，实际成交 = stop_price × (1 - stop_slip)（多）
    - TP/EOD 出场按 normal_slip
    - 手续费单独列（commission）
    """
    if df.empty:
        return {}

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    v = df["Volume"].values
    dates = df.index

    day_ids = pd.Series(dates).dt.date.values
    day_change = np.concatenate([[True], day_ids[1:] != day_ids[:-1]])
    bar_of_day = np.zeros(len(df), dtype=int)
    cnt = 0
    for i, change in enumerate(day_change):
        if change:
            cnt = 0
        bar_of_day[i] = cnt
        cnt += 1
    bars_per_day = pd.Series(bar_of_day).groupby(pd.Series(day_ids)).transform("max").values + 1
    bars_to_close = bars_per_day - bar_of_day - 1

    vol_ma = pd.Series(v).rolling(rvol_lookback, min_periods=1).mean().values

    n = len(df)
    daily_returns = np.zeros(n)

    unique_days = pd.Series(day_ids).unique()
    trades = []
    skipped_narrow = 0
    no_breakout_days = 0

    for day in unique_days:
        day_idx = np.where(day_ids == day)[0]
        if len(day_idx) < or_bars + 5:
            continue

        or_start = day_idx[0]
        or_end = day_idx[or_bars - 1]
        or_high = h[or_start:or_end + 1].max()
        or_low = l[or_start:or_end + 1].min()
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2

        if or_range / or_mid < min_or_range_pct:
            skipped_narrow += 1
            continue

        position = 0
        entry_price_theoretical = np.nan
        entry_price_actual = np.nan
        stop_price = tp_price = np.nan
        entry_idx = None
        breakout_found = False

        for j in range(or_bars, len(day_idx)):
            idx = day_idx[j]

            if position != 0:
                # 日末平仓
                if bars_to_close[idx] <= eod_exit_bars:
                    raw_exit = c[idx]
                    # 正常滑点（不利方向）
                    exit_px = raw_exit * (1 - normal_slip) if position == 1 else raw_exit * (1 + normal_slip)
                    pnl_pct = position * (exit_px - entry_price_actual) / entry_price_actual - 2 * commission
                    daily_returns[idx] = pnl_pct
                    trades.append({
                        "day": day, "entry_date": dates[entry_idx], "exit_date": dates[idx],
                        "result": "日末平仓",
                        "entry": round(entry_price_actual, 3),
                        "stop": round(stop_price, 3),
                        "tp": round(tp_price, 3),
                        "exit": round(exit_px, 3),
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "or_range_pct": round(or_range / or_mid * 100, 2),
                        "direction": "多" if position == 1 else "空",
                    })
                    position = 0
                    break

                stop_hit = (position == 1 and l[idx] <= stop_price) or \
                           (position == -1 and h[idx] >= stop_price)
                tp_hit = (position == 1 and h[idx] >= tp_price) or \
                         (position == -1 and l[idx] <= tp_price)

                if stop_hit:
                    # 止损滑点（最糟糕，gap风险）
                    raw_exit = stop_price
                    exit_px = raw_exit * (1 - stop_slip) if position == 1 else raw_exit * (1 + stop_slip)
                    pnl_pct = position * (exit_px - entry_price_actual) / entry_price_actual - 2 * commission
                    daily_returns[idx] = pnl_pct
                    trades.append({
                        "day": day, "entry_date": dates[entry_idx], "exit_date": dates[idx],
                        "result": "止损",
                        "entry": round(entry_price_actual, 3),
                        "stop": round(stop_price, 3),
                        "tp": round(tp_price, 3),
                        "exit": round(exit_px, 3),
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "or_range_pct": round(or_range / or_mid * 100, 2),
                        "direction": "多" if position == 1 else "空",
                    })
                    position = 0
                    break
                elif tp_hit:
                    raw_exit = tp_price
                    exit_px = raw_exit * (1 - normal_slip) if position == 1 else raw_exit * (1 + normal_slip)
                    pnl_pct = position * (exit_px - entry_price_actual) / entry_price_actual - 2 * commission
                    daily_returns[idx] = pnl_pct
                    trades.append({
                        "day": day, "entry_date": dates[entry_idx], "exit_date": dates[idx],
                        "result": "止盈",
                        "entry": round(entry_price_actual, 3),
                        "stop": round(stop_price, 3),
                        "tp": round(tp_price, 3),
                        "exit": round(exit_px, 3),
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "or_range_pct": round(or_range / or_mid * 100, 2),
                        "direction": "多" if position == 1 else "空",
                    })
                    position = 0
                    break
                continue

            if breakout_found:
                continue

            if use_volume_filter:
                rvol = v[idx] / max(vol_ma[idx], 1)
                if rvol < rvol_threshold:
                    continue

            # 突破上沿做多
            if h[idx] > or_high and c[idx] > or_high:
                entry_price_theoretical = or_high
                entry_price_actual = or_high * (1 + entry_slip)  # 入场滑点
                stop_price = or_low
                tp_price = entry_price_theoretical + target_r * or_range
                position = 1
                entry_idx = idx
                breakout_found = True
            elif (not long_only) and l[idx] < or_low and c[idx] < or_low:
                entry_price_theoretical = or_low
                entry_price_actual = or_low * (1 - entry_slip)
                stop_price = or_high
                tp_price = entry_price_theoretical - target_r * or_range
                position = -1
                entry_idx = idx
                breakout_found = True

        # 当日结束仍持仓则强平
        if position != 0 and entry_idx is not None:
            last_idx = day_idx[-1]
            raw_exit = c[last_idx]
            exit_px = raw_exit * (1 - normal_slip) if position == 1 else raw_exit * (1 + normal_slip)
            pnl_pct = position * (exit_px - entry_price_actual) / entry_price_actual - 2 * commission
            daily_returns[last_idx] = pnl_pct
            trades.append({
                "day": day, "entry_date": dates[entry_idx], "exit_date": dates[last_idx],
                "result": "收盘平仓",
                "entry": round(entry_price_actual, 3),
                "stop": round(stop_price, 3),
                "tp": round(tp_price, 3),
                "exit": round(exit_px, 3),
                "pnl_pct": round(pnl_pct * 100, 3),
                "or_range_pct": round(or_range / or_mid * 100, 2),
                "direction": "多" if position == 1 else "空",
            })

        if not breakout_found and position == 0:
            no_breakout_days += 1

    ret_series = pd.Series(daily_returns, index=dates, name="ORB")
    total_ret = (1 + ret_series).prod() - 1
    num_trades = len(trades)

    if num_trades > 0:
        td = pd.DataFrame(trades)
        wins = (td["result"] == "止盈").sum()
        losses = (td["result"] == "止损").sum()
        eod = ((td["result"] == "日末平仓") | (td["result"] == "收盘平仓")).sum()
        positive_trades = (td["pnl_pct"] > 0).sum()
        win_rate_all = positive_trades / num_trades  # 含日末平仓的真实胜率
        avg_win = td[td["pnl_pct"] > 0]["pnl_pct"].mean() if positive_trades > 0 else 0
        avg_loss = td[td["pnl_pct"] < 0]["pnl_pct"].mean() if (td["pnl_pct"] < 0).any() else 0
        profit_factor = (
            td[td["pnl_pct"] > 0]["pnl_pct"].sum() / abs(td[td["pnl_pct"] < 0]["pnl_pct"].sum())
            if (td["pnl_pct"] < 0).any() else np.inf
        )
    else:
        td = pd.DataFrame()
        win_rate_all = avg_win = avg_loss = profit_factor = 0
        wins = losses = eod = positive_trades = 0

    equity = (1 + ret_series).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    days_in_period = (dates[-1] - dates[0]).days
    years = days_in_period / 365.25 if days_in_period > 0 else 1/365
    ann_ret = (1 + total_ret) ** (1/years) - 1 if total_ret > -1 else -1
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

    metrics = {
        "区间": f"{dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}",
        "总收益率": f"{total_ret*100:.2f}%",
        "年化": f"{ann_ret*100:.2f}%",
        "最大回撤": f"{max_dd*100:.2f}%",
        "Calmar": f"{calmar:.2f}",
        "交易次数": num_trades,
        "真实胜率": f"{win_rate_all*100:.1f}%",  # 含日末平仓正收益
        "止盈次数": int(wins),
        "止损次数": int(losses),
        "日末平仓": int(eod),
        "平均盈利": f"{avg_win:.3f}%",
        "平均亏损": f"{avg_loss:.3f}%",
        "盈亏比": f"{profit_factor:.2f}" if profit_factor != np.inf else "∞",
    }

    return {"returns": ret_series, "trades": td, "metrics": metrics, "equity": equity}


# ============================================================
# Buy & Hold 基准
# ============================================================
def buy_and_hold_return(df: pd.DataFrame) -> float:
    """简单持有收益（首根K开盘到末根K收盘）"""
    if df.empty:
        return 0.0
    start_p = df.iloc[0]["Open"]
    end_p = df.iloc[-1]["Close"]
    return (end_p / start_p) - 1


# ============================================================
# 主入口: 三件事一起做
# ============================================================
if __name__ == "__main__":
    # 每只股票的设置（long_only + 滑点模型）
    ticker_configs = [
        # (ticker, long_only, entry_slip, stop_slip, 说明)
        ("OKLO", False, 0.0015, 0.0030, "小盘高波动，双向"),
        ("IREN", False, 0.0015, 0.0030, "小盘高波动，双向"),
        ("HOOD", False, 0.0008, 0.0015, "中等流动性，双向"),
        ("NVDL", False, 0.0008, 0.0015, "杠杆ETF，双向"),
        ("TSLL", False, 0.0008, 0.0015, "杠杆ETF，双向"),
        ("SOXL", True,  0.0005, 0.0010, "杠杆ETF，只多"),
    ]

    # 通用 ORB 参数
    base_params = {
        "or_bars": 3,
        "target_r": 2.0,
        "use_volume_filter": True,
        "rvol_threshold": 1.5,
        "commission": 0.0,
        "normal_slip": 0.0005,
    }

    print("=" * 110)
    print("ORB 完整验证报告 | 5m数据 | 真实滑点 + Buy&Hold 对比 + Walk-Forward")
    print("=" * 110)

    # 下载数据
    data = {}
    for t, *_ in ticker_configs:
        try:
            df = fetch_intraday(t, period="60d", interval="5m")
            if not df.empty and len(df) > 100:
                data[t] = df
                print(f"  {t}: {len(df)} 根5m ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as e:
            print(f"  {t}: 下载失败 {e}")

    # ===== 第1节：全周期回测（含真实滑点）vs Buy & Hold =====
    print(f"\n{'=' * 110}")
    print("【1】全周期回测（60天）vs Buy & Hold")
    print("=" * 110)
    print(f"{'Ticker':<7} {'策略':<6} {'总收益':>9} {'B&H':>9} {'差值':>9} "
          f"{'交易':>5} {'真胜率':>7} {'盈亏比':>7} {'回撤':>8} {'Calmar':>9}")
    print("-" * 110)

    full_results = {}
    for t, long_only, entry_s, stop_s, note in ticker_configs:
        if t not in data:
            continue
        df = data[t]
        result = orb_backtest_v2(df.copy(), long_only=long_only,
                                  entry_slip=entry_s, stop_slip=stop_s, **base_params)
        bh_ret = buy_and_hold_return(df) * 100
        result["bh_return"] = bh_ret
        full_results[t] = result

        m = result["metrics"]
        strat_ret = float(m["总收益率"].rstrip('%'))
        diff = strat_ret - bh_ret
        diff_sign = "+" if diff >= 0 else ""
        strat_name = "多空" if not long_only else "仅多"
        print(f"{t:<7} {strat_name:<6} {m['总收益率']:>9} {bh_ret:>8.2f}% "
              f"{diff_sign}{diff:>7.2f}% {m['交易次数']:>5} "
              f"{m['真实胜率']:>7} {m['盈亏比']:>7} "
              f"{m['最大回撤']:>8} {m['Calmar']:>9}")

    # ===== 第2节：Walk-Forward 样本外测试 =====
    print(f"\n{'=' * 110}")
    print("【2】Walk-Forward 验证：前30天 (训练) vs 后30天 (样本外)")
    print("=" * 110)

    for t, long_only, entry_s, stop_s, note in ticker_configs:
        if t not in data:
            continue
        df = data[t]
        # 按 K线数等分（30天 vs 30天）
        split = len(df) // 2
        train_df = df.iloc[:split]
        test_df = df.iloc[split:]

        train_res = orb_backtest_v2(train_df.copy(), long_only=long_only,
                                      entry_slip=entry_s, stop_slip=stop_s, **base_params)
        test_res = orb_backtest_v2(test_df.copy(), long_only=long_only,
                                     entry_slip=entry_s, stop_slip=stop_s, **base_params)
        bh_train = buy_and_hold_return(train_df) * 100
        bh_test = buy_and_hold_return(test_df) * 100

        print(f"\n── {t} ({note}) ──")
        print(f"{'期间':<6} {'区间':<28} {'策略收益':>9} {'B&H':>9} {'差值':>9} "
              f"{'交易':>5} {'真胜率':>7} {'盈亏比':>7} {'回撤':>8}")
        for label, res, bh in [("前30", train_res, bh_train), ("后30", test_res, bh_test)]:
            m = res["metrics"]
            strat_ret = float(m["总收益率"].rstrip('%'))
            diff = strat_ret - bh
            diff_sign = "+" if diff >= 0 else ""
            print(f"{label:<6} {m['区间']:<28} {m['总收益率']:>9} {bh:>8.2f}% "
                  f"{diff_sign}{diff:>7.2f}% {m['交易次数']:>5} "
                  f"{m['真实胜率']:>7} {m['盈亏比']:>7} {m['最大回撤']:>8}")

        # 稳健性判断
        train_ret = float(train_res["metrics"]["总收益率"].rstrip('%'))
        test_ret = float(test_res["metrics"]["总收益率"].rstrip('%'))
        if train_ret > 10 and test_ret > 10:
            verdict = "✅ 前后稳健（前后都 >+10%）"
        elif train_ret > 10 and test_ret < -5:
            verdict = "❌ 后 30 天崩盘（过拟合嫌疑）"
        elif train_ret > 10 and test_ret < 5:
            verdict = "⚠️ 衰退（前强后弱，需警惕）"
        elif train_ret < 0 and test_ret > 10:
            verdict = "🎲 运气逆转（样本太小）"
        else:
            verdict = f"📊 训练{train_ret:.1f}% → 测试{test_ret:.1f}%"
        print(f"       判定: {verdict}")

    # ===== 第3节：策略是否真的跑赢持有? =====
    print(f"\n{'=' * 110}")
    print("【3】结论：策略是否显著优于 Buy & Hold?")
    print("=" * 110)
    beat_bh_count = 0
    total_count = 0
    for t, res in full_results.items():
        strat_ret = float(res["metrics"]["总收益率"].rstrip('%'))
        bh_ret = res["bh_return"]
        total_count += 1
        if strat_ret > bh_ret:
            beat_bh_count += 1
            status = "✅ 策略胜"
        else:
            status = "❌ 不如持有"
        print(f"  {t:<6} 策略 {strat_ret:>+7.2f}% vs B&H {bh_ret:>+7.2f}% "
              f"(差值 {strat_ret - bh_ret:+.2f}%) {status}")
    print(f"\n  总体: {beat_bh_count}/{total_count} 只跑赢 Buy & Hold")

    # ===== 关键洞察 =====
    print(f"\n{'=' * 110}")
    print("【关键洞察】")
    print("=" * 110)
    print("""
1. 真实滑点（小盘股 0.15% + 止损 gap 0.30%）已计入 — 之前 +197% 的 IREN 会打折扣
2. Walk-Forward 是验证过拟合的金标准 — 如果前30天赚后30天亏，说明策略不稳
3. 跑赢 B&H 才是真正的 alpha — 如果只是复现标的涨幅，策略价值有限
4. 60天数据样本依然偏小 — 2个月覆盖单一市场环境，牛市/熊市/震荡各需独立验证
""")
