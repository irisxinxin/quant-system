"""
backtest_15m_pa_vs_momentum.py — 15分钟级别 裸K vs 动量延续 策略对比

测试股票: OKLO / IREN / HOOD （用户原始想做的高波动波段股）

策略A: 裸K反转（沿用 backtest_soxl_5m 的 backtest_intraday_pa）
  - 触发: 锤子/吞没/晨星/射击之星/黄昏之星 等反转形态
  - 逻辑: 预期反转，做逆势

策略B: 动量延续
  - 触发: 连续 N 根 (默认3) 同方向大实体K线（body_ratio > 0.5）
  - 逻辑: 预期趋势延续，做顺势
  - 入场: 第 N+1 根回调 retrace_pct 处挂限价单
  - 止损: 连续动量段的最低(多)/最高(空)点 或 入场 ∓ 1×ATR 更保守者
  - 止盈: 满足 R:R≥2 的前方摆动高(多)/低(空)点

两策略共用: R:R≥2 门槛 + 最大等待 + 不跨日 + 开收盘跳过 + 交易成本
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yfinance as yf

from signals.price_action import _calc_atr
from backtest_soxl_5m import fetch_intraday, backtest_intraday_pa


# ============================================================
# 策略B: 动量延续
# ============================================================
def detect_momentum_bars(df: pd.DataFrame, n_bars: int = 3,
                          body_ratio_threshold: float = 0.5) -> tuple:
    """
    识别连续 N 根同方向大实体K线
    返回: (bull_momentum, bear_momentum) 两个 bool Series
    """
    o = df["Open"].values
    c = df["Close"].values
    h = df["High"].values
    l = df["Low"].values

    body = np.abs(c - o)
    total = h - l
    # 避免除零
    body_ratio = np.where(total > 0, body / total, 0)
    is_bull = c > o
    is_bear = c < o
    is_large = body_ratio > body_ratio_threshold

    n = len(df)
    bull_momentum = np.zeros(n, dtype=bool)
    bear_momentum = np.zeros(n, dtype=bool)

    for i in range(n_bars - 1, n):
        window = slice(i - n_bars + 1, i + 1)
        if is_bull[window].all() and is_large[window].all():
            bull_momentum[i] = True
        if is_bear[window].all() and is_large[window].all():
            bear_momentum[i] = True

    return bull_momentum, bear_momentum


def backtest_momentum_continuation(
    df: pd.DataFrame,
    n_momentum: int = 3,
    body_ratio: float = 0.5,
    retrace_pct: float = 0.38,     # 斐波那契 38.2% 回调挂单
    min_rr: float = 2.0,
    atr_stop_mult: float = 1.0,
    max_wait_bars: int = 3,
    swing_lookback: int = 20,
    long_only: bool = True,
    commission: float = 0.0005,
    slippage: float = 0.0005,
    skip_bars_open: int = 2,
    skip_bars_close: int = 2,
    no_overnight: bool = True,
) -> dict:
    """动量延续 + 回踩入场 回测"""
    if df.empty:
        return {}

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    dates = df.index

    bull_mom, bear_mom = detect_momentum_bars(df, n_bars=n_momentum,
                                               body_ratio_threshold=body_ratio)

    # 每日开/收盘跳过
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
    is_open_bar = bar_of_day < skip_bars_open
    is_close_bar = (bars_per_day - bar_of_day) <= skip_bars_close

    n = len(df)
    daily_returns = np.zeros(n)

    STATE_IDLE     = 0
    STATE_PENDING  = 1
    STATE_IN_TRADE = 2

    state       = STATE_IDLE
    direction   = 0
    entry_price = np.nan
    stop_price  = np.nan
    tp_price    = np.nan
    actual_rr   = np.nan
    signal_day  = None
    entry_date  = None
    wait_bars   = 0
    trades      = []
    skipped_rr  = 0
    round_trip  = (commission + slippage) * 2

    for i in range(max(swing_lookback, n_momentum), n):

        # 开/收盘段保护
        if is_open_bar[i] or is_close_bar[i]:
            if state == STATE_IN_TRADE and is_close_bar[i] and no_overnight:
                # 日末强制平仓
                exit_px = c[i]
                pnl_pct = direction * (exit_px - entry_price) / entry_price - round_trip
                daily_returns[i] = pnl_pct
                trades.append({
                    "entry_date": entry_date, "exit_date": dates[i], "result": "日末平仓",
                    "entry": round(entry_price, 3), "stop": round(stop_price, 3),
                    "tp": round(tp_price, 3), "exit": round(exit_px, 3),
                    "pnl_pct": round(pnl_pct * 100, 3), "actual_rr": round(actual_rr, 2),
                    "direction": "多" if direction == 1 else "空",
                })
                state = STATE_IDLE
            elif state != STATE_IN_TRADE:
                state = STATE_IDLE
            if state != STATE_IN_TRADE:
                continue

        if state == STATE_IDLE:
            # 动量信号触发 → 直接挂回调单（不需要"下一根确认"）
            if bull_mom[i] or (bear_mom[i] and not long_only):
                direction = 1 if bull_mom[i] else -1
                signal_day = day_ids[i]

                # 动量段范围 = 最近 n_momentum 根
                seg_hi = h[i - n_momentum + 1:i + 1].max()
                seg_lo = l[i - n_momentum + 1:i + 1].min()
                seg_range = seg_hi - seg_lo

                atr = _calc_atr(h, l, c, period=14, idx=i)
                if atr == 0 or np.isnan(atr):
                    continue
                min_dist = atr_stop_mult * atr

                resist_lookback = swing_lookback * 2
                recent_lows = l[max(0, i - swing_lookback):i + 1]
                recent_highs = h[max(0, i - swing_lookback):i + 1]
                resist_highs = h[max(0, i - resist_lookback):i + 1]
                resist_lows = l[max(0, i - resist_lookback):i + 1]

                if direction == 1:
                    # 多单：回踩到动量段 (hi - retrace_pct × range)
                    entry_price = seg_hi - retrace_pct * seg_range

                    # 止损: 取三者最低（最保守）
                    #   - swing_lookback 内的摆动支撑位（最近的有效低点）
                    #   - 动量段最低点（结构止损）
                    #   - 入场 - 1×ATR（波动率止损）
                    valid_swing_lows = recent_lows[recent_lows < entry_price - min_dist]
                    stop_candidates = [entry_price - min_dist]
                    if len(valid_swing_lows) > 0:
                        stop_candidates.append(float(valid_swing_lows.max()))
                    stop_candidates.append(seg_lo)
                    stop_price = float(min(stop_candidates))

                    risk = entry_price - stop_price
                    if risk <= 0:
                        continue
                    min_target = entry_price + min_rr * risk

                    valid_highs = resist_highs[resist_highs >= min_target]
                    if len(valid_highs) == 0:
                        skipped_rr += 1
                        continue
                    tp_price = float(valid_highs.min())

                else:
                    entry_price = seg_lo + retrace_pct * seg_range

                    # 止损: 取三者最高（最保守）
                    valid_swing_highs = recent_highs[recent_highs > entry_price + min_dist]
                    stop_candidates = [entry_price + min_dist]
                    if len(valid_swing_highs) > 0:
                        stop_candidates.append(float(valid_swing_highs.min()))
                    stop_candidates.append(seg_hi)
                    stop_price = float(max(stop_candidates))

                    risk = stop_price - entry_price
                    if risk <= 0:
                        continue
                    min_target = entry_price - min_rr * risk
                    valid_lows = resist_lows[resist_lows <= min_target]
                    if len(valid_lows) == 0:
                        skipped_rr += 1
                        continue
                    tp_price = float(valid_lows.max())

                actual_rr = abs(tp_price - entry_price) / abs(entry_price - stop_price)
                entry_date = dates[i]
                wait_bars = 0
                state = STATE_PENDING

        elif state == STATE_PENDING:
            if day_ids[i] != signal_day:
                state = STATE_IDLE
                continue
            wait_bars += 1
            if wait_bars > max_wait_bars:
                state = STATE_IDLE
                continue

            filled = (direction == 1 and l[i] <= entry_price) or \
                     (direction == -1 and h[i] >= entry_price)
            if filled:
                state = STATE_IN_TRADE

        elif state == STATE_IN_TRADE:
            stop_hit = (direction == 1 and l[i] <= stop_price) or \
                       (direction == -1 and h[i] >= stop_price)
            tp_hit = (direction == 1 and h[i] >= tp_price) or \
                     (direction == -1 and l[i] <= tp_price)

            if stop_hit:
                exit_px = stop_price
                pnl_pct = direction * (exit_px - entry_price) / entry_price - round_trip
                daily_returns[i] = pnl_pct
                trades.append({
                    "entry_date": entry_date, "exit_date": dates[i], "result": "止损",
                    "entry": round(entry_price, 3), "stop": round(stop_price, 3),
                    "tp": round(tp_price, 3), "exit": round(exit_px, 3),
                    "pnl_pct": round(pnl_pct * 100, 3), "actual_rr": round(actual_rr, 2),
                    "direction": "多" if direction == 1 else "空",
                })
                state = STATE_IDLE
            elif tp_hit:
                exit_px = tp_price
                pnl_pct = direction * (exit_px - entry_price) / entry_price - round_trip
                daily_returns[i] = pnl_pct
                trades.append({
                    "entry_date": entry_date, "exit_date": dates[i], "result": "止盈",
                    "entry": round(entry_price, 3), "stop": round(stop_price, 3),
                    "tp": round(tp_price, 3), "exit": round(exit_px, 3),
                    "pnl_pct": round(pnl_pct * 100, 3), "actual_rr": round(actual_rr, 2),
                    "direction": "多" if direction == 1 else "空",
                })
                state = STATE_IDLE

    ret_series = pd.Series(daily_returns, index=dates, name="momentum_continuation")

    num_trades = len(trades)
    total_ret = (1 + ret_series).prod() - 1
    if num_trades > 0:
        td = pd.DataFrame(trades)
        wins = (td["result"] == "止盈").sum()
        losses = (td["result"] == "止损").sum()
        timeouts = (td["result"] == "日末平仓").sum()
        win_rate = wins / num_trades
        avg_win = td[td["result"] == "止盈"]["pnl_pct"].mean() if wins > 0 else 0
        avg_loss = td[td["result"] == "止损"]["pnl_pct"].mean() if losses > 0 else 0
        profit_factor = (
            td[td["pnl_pct"] > 0]["pnl_pct"].sum() /
            abs(td[td["pnl_pct"] < 0]["pnl_pct"].sum())
            if (td["pnl_pct"] < 0).any() else np.inf
        )
    else:
        td = pd.DataFrame()
        win_rate = avg_win = avg_loss = profit_factor = wins = losses = timeouts = 0

    equity = (1 + ret_series).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    days_in_period = (dates[-1] - dates[0]).days
    years = days_in_period / 365.25 if days_in_period > 0 else 1 / 365
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if total_ret > -1 else -1
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

    metrics = {
        "区间": f"{dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}",
        "总收益率": f"{total_ret*100:.2f}%",
        "年化收益": f"{ann_ret*100:.2f}%",
        "最大回撤": f"{max_dd*100:.2f}%",
        "Calmar": f"{calmar:.2f}",
        "交易次数": num_trades,
        "止盈次数": int(wins),
        "止损次数": int(losses),
        "日末平仓": int(timeouts),
        "胜率": f"{win_rate*100:.1f}%",
        "平均盈利": f"{avg_win:.3f}%",
        "平均亏损": f"{avg_loss:.3f}%",
        "盈亏比": f"{profit_factor:.2f}" if profit_factor != np.inf else "∞",
        "R:R不足跳过": skipped_rr,
    }

    return {"returns": ret_series, "trades": td, "metrics": metrics, "equity": equity}


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    tickers = ["OKLO", "IREN", "HOOD"]

    print("=" * 90)
    print("15分钟级别 | 裸K反转 vs 动量延续(3根大实体+回踩) | OKLO/IREN/HOOD")
    print("=" * 90)

    all_results = {}
    for t in tickers:
        df = fetch_intraday(t, period="60d", interval="15m")
        if df.empty or len(df) < 100:
            print(f"\n{t}: 数据不足")
            continue

        # 策略A: 裸K反转（多头only, R:R≥2）
        pa_res = backtest_intraday_pa(df.copy(), long_only=True, min_rr=2.0, max_wait_bars=3,
                                        swing_lookback=20)
        # 策略B: 动量延续（多头only, 3根大实体, R:R≥2）
        mo_res = backtest_momentum_continuation(df.copy(), n_momentum=3, body_ratio=0.5,
                                                  long_only=True, min_rr=2.0, max_wait_bars=3,
                                                  swing_lookback=20)

        all_results[t] = {"裸K": pa_res, "动量": mo_res}

        print(f"\n══ {t} ({len(df)} 根15m) ══")
        print(f"{'指标':<10}  {'裸K反转':>15}  {'动量延续':>15}")
        print("-" * 50)
        for k in ["总收益率", "最大回撤", "Calmar", "交易次数",
                  "胜率", "平均盈利", "平均亏损", "盈亏比"]:
            a = pa_res["metrics"].get(k, "N/A")
            b = mo_res["metrics"].get(k, "N/A")
            print(f"{k:<10}  {str(a):>15}  {str(b):>15}")

    # 总汇总
    print("\n" + "=" * 90)
    print("汇总表 (15m, 多头only, R:R≥2)")
    print("=" * 90)
    header = f"{'Ticker':<8} {'策略':<8} {'总收益':>9} {'胜率':>7} {'盈亏比':>7} {'交易':>5} {'Calmar':>7} {'回撤':>8}"
    print(header)
    print("-" * 90)
    for t, res in all_results.items():
        for name, r in res.items():
            m = r["metrics"]
            print(f"{t:<8} {name:<8} {m['总收益率']:>9} {m['胜率']:>7} "
                  f"{m['盈亏比']:>7} {m['交易次数']:>5} "
                  f"{m['Calmar']:>7} {m['最大回撤']:>8}")

    # 如果有盈利标的，打印样本交易
    print("\n" + "=" * 90)
    winners = []
    for t, res in all_results.items():
        for name, r in res.items():
            ret = float(r['metrics']['总收益率'].rstrip('%'))
            if ret > 0:
                winners.append((t, name, r))
    if winners:
        print(f"✅ 盈利组合: {[(t, n) for t, n, _ in winners]}")
        for t, name, r in winners:
            print(f"\n── {t} / {name} 前10笔交易 ──")
            if not r["trades"].empty:
                print(r["trades"].head(10).to_string())
    else:
        print("⚠️ 所有 Ticker × 策略 组合均为亏损")
