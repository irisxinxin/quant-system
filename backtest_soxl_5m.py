"""
backtest_soxl_5m.py — SOXL 5分钟裸K战法回测（独立脚本）

策略逻辑（用户描述）:
  1. 信号K线：锤子/吞没/晨星/射击之星/黄昏之星 等裸K形态
  2. 确认K线：下一根K线收盘方向与信号同向
  3. 入场：确认K线收盘价回调 retrace_pct 处挂限价单
  4. 止损：最近摆动低点 与 入场-1×ATR 取更低者（更保守）
  5. 止盈：上方最近满足 R:R≥2 的摆动高点
  6. 超过 max_wait_bars 未成交则取消

相比日线，5m级别增加的过滤器:
  - 跳过每日开盘前 2 根K线（9:30-9:40 噪音大）
  - 跳过每日收盘前 2 根K线（避免隔夜持仓 gap 风险）
  - 不跨日持仓：交易必须在同一交易日内完成（or 触发强制平仓）
  - 波动过滤：信号K线的 ATR 必须 > 日均 ATR 的 0.5 倍
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import yfinance as yf

from signals.price_action import detect_candle_patterns, _calc_atr


def fetch_intraday(ticker: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """抓取分钟级 OHLCV（yfinance 5m 限 60 天，1m 限 7 天）"""
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df.dropna(subset=["Close"])


def backtest_intraday_pa(
    df: pd.DataFrame,
    retrace_pct: float = 0.30,
    min_rr: float = 2.0,
    atr_stop_mult: float = 1.0,
    max_wait_bars: int = 3,       # 5m 级别缩短等待（15 分钟内若未成交则取消）
    swing_lookback: int = 20,     # 前 20 根 5m K线（约 100 分钟）
    long_only: bool = True,
    commission: float = 0.0005,   # Alpaca 免佣，此处模拟交易所费+点差
    slippage: float = 0.0005,
    skip_bars_open: int = 2,      # 开盘跳过 2 根（10分钟）
    skip_bars_close: int = 2,     # 收盘跳过 2 根
    no_overnight: bool = True,    # 不跨日持仓
) -> dict:
    """5m 级别裸K确认回调回测"""
    if df.empty:
        return {}

    df = detect_candle_patterns(df)

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    v = df["Volume"].values
    bull_pat = df["pa_bullish"].values
    bear_pat = df["pa_bearish"].values
    dates    = df.index

    # 标记每日第一/最后 N 根K线
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
    is_open_bar  = bar_of_day < skip_bars_open
    is_close_bar = (bars_per_day - bar_of_day) <= skip_bars_close

    n = len(df)
    daily_returns = np.zeros(n)

    STATE_IDLE     = 0
    STATE_CONFIRM  = 1
    STATE_PENDING  = 2
    STATE_IN_TRADE = 3

    state       = STATE_IDLE
    direction   = 0
    entry_price = np.nan
    stop_price  = np.nan
    tp_price    = np.nan
    actual_rr   = np.nan
    signal_day  = None
    wait_bars   = 0
    trades      = []
    skipped_rr  = 0
    skipped_filter = 0
    round_trip  = (commission + slippage) * 2

    for i in range(swing_lookback, n):

        if is_open_bar[i] or is_close_bar[i]:
            if state == STATE_IN_TRADE and is_close_bar[i] and no_overnight:
                # 日末强制平仓
                exit_px = c[i]
                pnl_pct = direction * (exit_px - entry_price) / entry_price - round_trip
                daily_returns[i] = pnl_pct
                trades.append({
                    "entry_date": entry_date if 'entry_date' in dir() else dates[i],
                    "exit_date": dates[i], "result": "日末平仓",
                    "entry": round(entry_price, 3), "stop": round(stop_price, 3),
                    "tp": round(tp_price, 3), "exit": round(exit_px, 3),
                    "pnl_pct": round(pnl_pct * 100, 3), "actual_rr": round(actual_rr, 2),
                    "direction": "多" if direction == 1 else "空",
                })
                state = STATE_IDLE
            elif state != STATE_IN_TRADE:
                state = STATE_IDLE  # 非持仓状态在开/收盘段重置
            if state != STATE_IN_TRADE:
                continue

        if state == STATE_IDLE:
            if bull_pat[i]:
                state     = STATE_CONFIRM
                direction = 1
                signal_day = day_ids[i]
            elif bear_pat[i] and not long_only:
                state     = STATE_CONFIRM
                direction = -1
                signal_day = day_ids[i]

        elif state == STATE_CONFIRM:
            # 跨日信号作废
            if day_ids[i] != signal_day:
                state = STATE_IDLE
                continue
            confirmed = (direction == 1 and c[i] > o[i]) or \
                        (direction == -1 and c[i] < o[i])
            if confirmed:
                candle_range = h[i] - l[i]
                recent_lows  = l[max(0, i - swing_lookback):i]
                recent_highs = h[max(0, i - swing_lookback):i]

                atr = _calc_atr(h, l, c, period=14, idx=i)
                if atr == 0 or np.isnan(atr):
                    state = STATE_IDLE
                    continue
                min_dist = atr_stop_mult * atr

                resist_lookback = swing_lookback * 2
                resist_highs = h[max(0, i - resist_lookback):i]
                resist_lows  = l[max(0, i - resist_lookback):i]

                if direction == 1:
                    entry_price = c[i] - retrace_pct * candle_range

                    valid_lows = recent_lows[recent_lows < entry_price - min_dist]
                    if len(valid_lows) == 0:
                        state = STATE_IDLE
                        skipped_rr += 1
                        continue
                    stop_price = float(valid_lows.max())

                    risk       = entry_price - stop_price
                    min_target = entry_price + min_rr * risk

                    valid_highs = resist_highs[resist_highs >= min_target]
                    if len(valid_highs) == 0:
                        state = STATE_IDLE
                        skipped_rr += 1
                        continue
                    resist_price = float(valid_highs.min())

                else:
                    entry_price  = c[i] + retrace_pct * candle_range
                    valid_highs  = recent_highs[recent_highs > entry_price + min_dist]
                    if len(valid_highs) == 0:
                        state = STATE_IDLE
                        skipped_rr += 1
                        continue
                    stop_price   = float(valid_highs.min())
                    risk         = stop_price - entry_price
                    min_target   = entry_price - min_rr * risk
                    valid_lows   = resist_lows[resist_lows <= min_target]
                    if len(valid_lows) == 0:
                        state = STATE_IDLE
                        skipped_rr += 1
                        continue
                    resist_price = float(valid_lows.max())

                risk      = abs(entry_price - stop_price)
                actual_rr = abs(resist_price - entry_price) / risk

                tp_price   = resist_price
                entry_date = dates[i]
                wait_bars  = 0
                state      = STATE_PENDING
            else:
                state = STATE_IDLE

        elif state == STATE_PENDING:
            if day_ids[i] != signal_day:
                state = STATE_IDLE  # 跨日挂单作废
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
            tp_hit   = (direction == 1 and h[i] >= tp_price) or \
                       (direction == -1 and l[i] <= tp_price)

            if stop_hit:
                exit_px  = stop_price
                pnl_pct  = direction * (exit_px - entry_price) / entry_price - round_trip
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
                exit_px  = tp_price
                pnl_pct  = direction * (exit_px - entry_price) / entry_price - round_trip
                daily_returns[i] = pnl_pct
                trades.append({
                    "entry_date": entry_date, "exit_date": dates[i], "result": "止盈",
                    "entry": round(entry_price, 3), "stop": round(stop_price, 3),
                    "tp": round(tp_price, 3), "exit": round(exit_px, 3),
                    "pnl_pct": round(pnl_pct * 100, 3), "actual_rr": round(actual_rr, 2),
                    "direction": "多" if direction == 1 else "空",
                })
                state = STATE_IDLE

    ret_series = pd.Series(daily_returns, index=dates, name="SOXL_5m_裸K")

    # 计算综合指标
    total_ret  = (1 + ret_series).prod() - 1
    num_trades = len(trades)
    if num_trades > 0:
        td = pd.DataFrame(trades)
        wins   = (td["result"] == "止盈").sum()
        losses = (td["result"] == "止损").sum()
        timeouts = (td["result"] == "日末平仓").sum()
        win_rate = wins / num_trades
        avg_win  = td[td["result"] == "止盈"]["pnl_pct"].mean() if wins > 0 else 0
        avg_loss = td[td["result"] == "止损"]["pnl_pct"].mean() if losses > 0 else 0
        profit_factor = (
            td[td["pnl_pct"] > 0]["pnl_pct"].sum() /
            abs(td[td["pnl_pct"] < 0]["pnl_pct"].sum())
            if (td["pnl_pct"] < 0).any() else np.inf
        )
    else:
        td = pd.DataFrame()
        win_rate = avg_win = avg_loss = profit_factor = wins = losses = timeouts = 0

    # 权益曲线
    equity = (1 + ret_series).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    # 年化（60天 → 252/60）
    days_in_period = (dates[-1] - dates[0]).days
    years = days_in_period / 365.25 if days_in_period > 0 else 1/365
    ann_ret = (1 + total_ret) ** (1/years) - 1 if total_ret > -1 else -1
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

    metrics = {
        "区间": f"{dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}",
        "总K线数": n,
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

    return {
        "returns":  ret_series,
        "trades":   td,
        "metrics":  metrics,
        "equity":   equity,
    }


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("SOXL 5分钟裸K战法回测")
    print("=" * 60)

    df = fetch_intraday("SOXL", period="60d", interval="5m")
    print(f"\n数据: {len(df)} 根5m K线")
    print(f"区间: {df.index[0]} → {df.index[-1]}\n")

    # 三组参数对比：多头only / 多空 / 更严格门槛
    configs = [
        {"name": "A: 多头only R:R≥2", "long_only": True,  "min_rr": 2.0, "max_wait_bars": 3},
        {"name": "B: 多空 R:R≥2",     "long_only": False, "min_rr": 2.0, "max_wait_bars": 3},
        {"name": "C: 多空 R:R≥3",     "long_only": False, "min_rr": 3.0, "max_wait_bars": 3},
        {"name": "D: 多头 等待5根",    "long_only": True,  "min_rr": 2.0, "max_wait_bars": 5},
    ]

    all_results = {}
    for cfg in configs:
        name = cfg.pop("name")
        result = backtest_intraday_pa(df.copy(), **cfg)
        all_results[name] = result
        print(f"\n── {name} ──")
        for k, v in result["metrics"].items():
            print(f"  {k:12} {v}")

    # 打印最佳方案的 trades 样本
    print("\n" + "=" * 60)
    best = max(all_results.items(), key=lambda x: x[1]["metrics"].get("交易次数", 0))
    print(f"样本交易 ({best[0]}) — 前10笔:")
    print("=" * 60)
    if not best[1]["trades"].empty:
        print(best[1]["trades"].head(10).to_string())
    else:
        print("无交易")
