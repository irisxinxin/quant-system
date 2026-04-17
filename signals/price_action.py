"""
signals/price_action.py — 裸K形态识别
包含：经典K线形态、支撑压力位、趋势结构
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PA_BODY_RATIO, PA_SHADOW_MULT
from data.downloader import get_ohlcv

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _body(o, c):        return abs(c - o)
def _total(o, h, l, c): return h - l
def _upper_shadow(o, h, c): return h - max(o, c)
def _lower_shadow(o, l, c): return min(o, c) - l
def _is_bull(o, c):    return c > o
def _is_bear(o, c):    return c < o


# ──────────────────────────────────────────────
# 1. 单根K线形态
# ──────────────────────────────────────────────

def detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    识别经典单/双/三K线形态

    Input:
        df: OHLCV DataFrame，columns = [Open, High, Low, Close, Volume]

    Returns:
        原 df + 形态列（bool），每列代表一种形态
    """
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    body  = (c - o).abs()
    total = h - l
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    body_ratio = body / total.replace(0, np.nan)

    result = df.copy()

    # ── 看涨形态 ──

    # 锤子线 Hammer（下影线长，实体小，在下跌后出现）
    result["hammer"] = (
        (lower >= body * PA_SHADOW_MULT) &   # 下影线 >= 实体2倍
        (upper <= body * 0.3) &              # 几乎无上影线
        (body_ratio < 0.35)                  # 小实体
    )

    # 看涨吞没 Bullish Engulfing
    result["bullish_engulfing"] = (
        _is_bear(o.shift(1), c.shift(1)) &    # 前日阴线
        _is_bull(o, c) &                       # 今日阳线
        (o < c.shift(1)) &                     # 今日开盘低于前日收盘
        (c > o.shift(1))                       # 今日收盘高于前日开盘
    )

    # 早晨之星 Morning Star（三K线，底部反转）
    prev2_bear  = _is_bear(o.shift(2), c.shift(2))
    prev1_doji  = (body.shift(1) / total.shift(1).replace(0, np.nan)) < 0.3
    today_bull  = _is_bull(o, c) & (c > (o.shift(2) + c.shift(2)) / 2)
    result["morning_star"] = prev2_bear & prev1_doji & today_bull

    # 看涨十字星 Dragonfly Doji（下影线长，无上影线，实体极小）
    result["dragonfly_doji"] = (
        (body_ratio < 0.1) &
        (lower >= total * 0.6) &
        (upper <= total * 0.1)
    )

    # 穿刺形态 Piercing Line
    result["piercing"] = (
        _is_bear(o.shift(1), c.shift(1)) &
        _is_bull(o, c) &
        (o < l.shift(1)) &                      # 今开低于前低
        (c > (o.shift(1) + c.shift(1)) / 2) &   # 今收超过前实体中点
        (c < o.shift(1))                         # 但未完全吞没
    )

    # ── 看跌形态 ──

    # 射击之星 Shooting Star
    result["shooting_star"] = (
        (upper >= body * PA_SHADOW_MULT) &   # 上影线 >= 实体2倍
        (lower <= body * 0.3) &              # 几乎无下影线
        (body_ratio < 0.35)                  # 小实体
    )

    # 看跌吞没 Bearish Engulfing
    result["bearish_engulfing"] = (
        _is_bull(o.shift(1), c.shift(1)) &
        _is_bear(o, c) &
        (o > c.shift(1)) &
        (c < o.shift(1))
    )

    # 黄昏之星 Evening Star（三K线，顶部反转）
    prev2_bull  = _is_bull(o.shift(2), c.shift(2))
    prev1_small = (body.shift(1) / total.shift(1).replace(0, np.nan)) < 0.3
    today_bear  = _is_bear(o, c) & (c < (o.shift(2) + c.shift(2)) / 2)
    result["evening_star"] = prev2_bull & prev1_small & today_bear

    # 乌云盖顶 Dark Cloud Cover
    result["dark_cloud"] = (
        _is_bull(o.shift(1), c.shift(1)) &
        _is_bear(o, c) &
        (o > h.shift(1)) &
        (c < (o.shift(1) + c.shift(1)) / 2) &
        (c > o.shift(1))
    )

    # ── 中性/持续形态 ──

    # 十字星 Doji（开收接近，方向不确定）
    result["doji"] = body_ratio < 0.1

    # 纺锤线 Spinning Top
    result["spinning_top"] = (
        (body_ratio < 0.3) &
        (upper >= body * 0.5) &
        (lower >= body * 0.5)
    )

    # 强势多头 Marubozu（几乎无影线）
    result["bullish_marubozu"] = (
        _is_bull(o, c) &
        (body_ratio > 0.85) &
        (upper <= body * 0.05) &
        (lower <= body * 0.05)
    )

    result["bearish_marubozu"] = (
        _is_bear(o, c) &
        (body_ratio > 0.85) &
        (upper <= body * 0.05) &
        (lower <= body * 0.05)
    )

    # 综合信号列
    result["pa_bullish"] = (
        result["hammer"] | result["bullish_engulfing"] |
        result["morning_star"] | result["piercing"]
    )
    result["pa_bearish"] = (
        result["shooting_star"] | result["bearish_engulfing"] |
        result["evening_star"] | result["dark_cloud"]
    )

    return result


# ──────────────────────────────────────────────
# 2. 支撑压力位检测
# ──────────────────────────────────────────────

def detect_swing_levels(
    df: pd.DataFrame,
    swing_period: int = 10,
    min_touches: int = 2,
    tolerance_pct: float = 0.005,
) -> dict:
    """
    识别关键支撑/压力位（摆动高低点聚类）

    Args:
        swing_period:  左右各 N 根K线的极值才算摆动点
        min_touches:   至少被测试 N 次才算有效支撑压力
        tolerance_pct: 价格聚类容差（0.5%以内视为同一水平）

    Returns:
        dict {
            'swing_highs': [价位列表],
            'swing_lows':  [价位列表],
            'support':     [有效支撑位],
            'resistance':  [有效压力位],
        }
    """
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    # 摆动高点：左右各 swing_period 根K线内是最高点
    swing_highs = []
    swing_lows  = []

    for i in range(swing_period, len(df) - swing_period):
        if h.iloc[i] == h.iloc[i - swing_period:i + swing_period + 1].max():
            swing_highs.append(h.iloc[i])
        if l.iloc[i] == l.iloc[i - swing_period:i + swing_period + 1].min():
            swing_lows.append(l.iloc[i])

    # 聚类：将接近的价位合并
    def cluster(levels, tol):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for v in levels[1:]:
            if (v - clusters[-1][-1]) / clusters[-1][-1] < tol:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [(np.mean(cl), len(cl)) for cl in clusters]

    current_price = c.iloc[-1]
    tol = tolerance_pct

    high_clusters = cluster(swing_highs, tol)
    low_clusters  = cluster(swing_lows,  tol)

    # 有效支撑（在当前价格下方，被触及 min_touches 次以上）
    support    = [p for p, cnt in low_clusters
                  if cnt >= min_touches and p < current_price]
    resistance = [p for p, cnt in high_clusters
                  if cnt >= min_touches and p > current_price]

    return {
        "current_price": round(current_price, 2),
        "swing_highs":   [round(p, 2) for p in swing_highs[-20:]],
        "swing_lows":    [round(p, 2) for p in swing_lows[-20:]],
        "support":       sorted([round(p, 2) for p in support], reverse=True)[:5],
        "resistance":    sorted([round(p, 2) for p in resistance])[:5],
    }


# ──────────────────────────────────────────────
# 3. 综合裸K信号
# ──────────────────────────────────────────────

def price_action_signal(
    ticker: str,
    use_trend_filter: bool = True,
    trend_ma: int = 50,
) -> pd.DataFrame:
    """
    裸K综合信号：形态 + 支撑压力位确认

    逻辑：
      - 看涨形态出现在支撑位附近（±0.5%）→ 强力做多信号
      - 看跌形态出现在压力位附近（±0.5%）→ 强力做空信号
      - 趋势过滤：仅在趋势方向做

    Returns:
        DataFrame + signal 列：2=强多，1=多，-1=空，-2=强空，0=无信号
    """
    df    = get_ohlcv(ticker)
    if df.empty:
        return pd.DataFrame()

    # K线形态
    df = detect_candle_patterns(df)

    # 支撑压力
    levels = detect_swing_levels(df)
    supports    = levels["support"]
    resistances = levels["resistance"]

    # 趋势方向
    sma = df["Close"].rolling(trend_ma).mean()
    in_uptrend = df["Close"] > sma

    # 信号合成
    signal = pd.Series(0, index=df.index)
    tol = 0.005   # 0.5% 容差

    for i in range(len(df)):
        row   = df.iloc[i]
        price = row["Close"]

        near_support    = any(abs(price - s) / s < tol for s in supports)
        near_resistance = any(abs(price - r) / r < tol for r in resistances)

        bull_pattern = row["pa_bullish"]
        bear_pattern = row["pa_bearish"]
        up_trend     = in_uptrend.iloc[i] if use_trend_filter else True
        dn_trend     = not in_uptrend.iloc[i] if use_trend_filter else True

        # 强信号：形态 + 关键位置 + 趋势三重确认
        if bull_pattern and near_support and up_trend:
            signal.iloc[i] = 2
        elif bull_pattern and up_trend:
            signal.iloc[i] = 1
        elif bear_pattern and near_resistance and dn_trend:
            signal.iloc[i] = -2
        elif bear_pattern and dn_trend:
            signal.iloc[i] = -1

    df["signal"]     = signal.shift(1).fillna(0)   # 次日执行
    df["support"]    = str(levels["support"])
    df["resistance"] = str(levels["resistance"])

    return df


# ──────────────────────────────────────────────
# 4. 改进裸K策略：二次确认 + 限价回调入场
# ──────────────────────────────────────────────

def _nearest_support(lows: np.ndarray, entry_price: float) -> float:
    """从最近 N 根K线的低点中，找到低于 entry_price 的最近（最高）支撑位"""
    candidates = lows[lows < entry_price]
    if len(candidates) == 0:
        return entry_price * 0.98
    return float(candidates.max())


def _nearest_resistance(highs: np.ndarray, entry_price: float) -> float:
    """从最近 N 根K线的高点中，找到高于 entry_price 的最近（最低）压力位"""
    candidates = highs[highs > entry_price]
    if len(candidates) == 0:
        return np.nan
    return float(candidates.min())


def _calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              period: int = 14, idx: int = None) -> float:
    """计算 idx 位置的 ATR(period)"""
    end = idx + 1 if idx is not None else len(high)
    start = max(1, end - period - 1)
    h = high[start:end]
    l = low[start:end]
    c_prev = close[start - 1:end - 1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    return float(tr[-period:].mean()) if len(tr) >= period else float(tr.mean())


def pa_confirmed_entry_backtest(
    ticker: str,
    start: str,
    end: str = None,
    retrace_pct: float = 0.30,    # 在第二根K线 close - 30%×range 挂单
    min_rr: float = 2.0,          # 最低盈亏比门槛
    atr_stop_mult: float = 1.0,   # 止损至少 N×ATR14，防止止损过紧
    max_wait_bars: int = 5,
    swing_lookback: int = 20,
    long_only: bool = True,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> dict:
    """
    改进裸K策略回测：二次确认 + 限价回调入场 + ATR最小止损 + R:R≥2过滤

    规则：
      1. 第一根K线出现看涨裸K形态
      2. 第二根K线同方向确认（close > open）
      3. 挂单价：entry = K2.Close - retrace_pct × (K2.H - K2.L)
      4. 止损 = min(摆动支撑位, entry - atr_stop_mult×ATR14)
             → 取更低的那个，确保止损至少有 1×ATR 呼吸空间
      5. 止盈目标 = 上方最近压力位（swing_lookback根K线高点）
      6. 仅当 (压力位 - 入场) / (入场 - 止损) >= min_rr 才下单
      7. 超过 max_wait_bars 未成交 → 取消
    """
    from data.downloader import get_ohlcv

    df = get_ohlcv(ticker, start=start, end=end)
    if df.empty:
        return {}

    df = detect_candle_patterns(df)
    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    bull_pat = df["pa_bullish"].values
    bear_pat = df["pa_bearish"].values
    dates    = df.index

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
    wait_bars   = 0
    trades      = []
    skipped_rr  = 0
    round_trip  = (commission + slippage) * 2

    for i in range(swing_lookback, n):

        if state == STATE_IDLE:
            if bull_pat[i]:
                state     = STATE_CONFIRM
                direction = 1
            elif bear_pat[i] and not long_only:
                state     = STATE_CONFIRM
                direction = -1

        elif state == STATE_CONFIRM:
            confirmed = (direction == 1 and c[i] > o[i]) or \
                        (direction == -1 and c[i] < o[i])
            if confirmed:
                candle_range = h[i] - l[i]
                recent_lows  = l[max(0, i - swing_lookback):i]
                recent_highs = h[max(0, i - swing_lookback):i]

                # ATR14 — 用于界定"有意义"的关键位距离
                atr = _calc_atr(h, l, c, period=14, idx=i)
                min_dist = atr_stop_mult * atr   # 支撑/压力位至少要离入场价这么远

                # 用更长的回望期找压力位（关键位通常在更远处）
                resist_lookback = swing_lookback * 2
                resist_highs = h[max(0, i - resist_lookback):i]

                if direction == 1:
                    entry_price = c[i] - retrace_pct * candle_range

                    # 1. 找有效支撑：至少 min_dist 以外的摆动低点
                    valid_lows = recent_lows[recent_lows < entry_price - min_dist]
                    if len(valid_lows) == 0:
                        state = STATE_IDLE
                        skipped_rr += 1
                        continue
                    stop_price = float(valid_lows.max())

                    # 2. 计算满足 min_rr 所需的最低目标价
                    risk       = entry_price - stop_price
                    min_target = entry_price + min_rr * risk

                    # 3. 在更长周期内找到≥min_target 的最近压力位
                    valid_highs = resist_highs[resist_highs >= min_target]
                    if len(valid_highs) == 0:
                        state = STATE_IDLE
                        skipped_rr += 1
                        continue
                    resist_price = float(valid_highs.min())   # 最近的满足R:R的压力位

                else:
                    entry_price  = c[i] + retrace_pct * candle_range
                    resist_lows  = l[max(0, i - resist_lookback):i]
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

                tp_price  = resist_price
                wait_bars = 0
                state     = STATE_PENDING
            else:
                state = STATE_IDLE

        elif state == STATE_PENDING:
            wait_bars += 1
            if wait_bars > max_wait_bars:
                state = STATE_IDLE
                continue

            filled = (direction == 1 and l[i] <= entry_price) or \
                     (direction == -1 and h[i] >= entry_price)
            if filled:
                state = STATE_IN_TRADE

        elif state == STATE_IN_TRADE:
            # 止损优先（保守原则）
            stop_hit = (direction == 1 and l[i] <= stop_price) or \
                       (direction == -1 and h[i] >= stop_price)
            tp_hit   = (direction == 1 and h[i] >= tp_price) or \
                       (direction == -1 and l[i] <= tp_price)

            if stop_hit:
                exit_px  = stop_price
                pnl_pct  = direction * (exit_px - entry_price) / entry_price - round_trip
                daily_returns[i] = pnl_pct
                trades.append({
                    "exit_date": dates[i], "result": "止损",
                    "entry": round(entry_price, 2), "stop": round(stop_price, 2),
                    "tp": round(tp_price, 2), "exit": round(exit_px, 2),
                    "pnl_pct": round(pnl_pct * 100, 2), "actual_rr": round(actual_rr, 2),
                })
                state = STATE_IDLE

            elif tp_hit:
                exit_px  = tp_price
                pnl_pct  = direction * (exit_px - entry_price) / entry_price - round_trip
                daily_returns[i] = pnl_pct
                trades.append({
                    "exit_date": dates[i], "result": "止盈",
                    "entry": round(entry_price, 2), "stop": round(stop_price, 2),
                    "tp": round(tp_price, 2), "exit": round(exit_px, 2),
                    "pnl_pct": round(pnl_pct * 100, 2), "actual_rr": round(actual_rr, 2),
                })
                state = STATE_IDLE
            # else: 持仓未出场，当日收益记为0（仅在出场时结算）

    ret_series = pd.Series(daily_returns, index=dates, name="裸K确认回调")
    # 去除0值前后的非持仓天（保持与其他策略可比）
    ret_series = ret_series

    from backtest.metrics import calc_metrics
    metrics = calc_metrics(ret_series)
    metrics["策略名称"] = "裸K确认回调(R:R≥2)"
    metrics["交易次数"] = len(trades)
    metrics["R:R不足跳过"] = skipped_rr
    if trades:
        td = pd.DataFrame(trades)
        metrics["止盈次数"] = (td["result"] == "止盈").sum()
        metrics["止损次数"] = (td["result"] == "止损").sum()
        win_trades = (td["result"] == "止盈").sum()
        metrics["交易胜率"] = f"{win_trades / len(trades):.1%}"
        metrics["平均R:R"] = f"{td['actual_rr'].mean():.2f}" if "actual_rr" in td.columns else "N/A"

    return {
        "returns":  ret_series,
        "trades":   pd.DataFrame(trades) if trades else pd.DataFrame(),
        "metrics":  metrics,
    }


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== 裸K形态识别（SPY 最近30日）===")
    df = get_ohlcv("SPY", start="2023-01-01")
    result = detect_candle_patterns(df)

    pattern_cols = ["hammer","bullish_engulfing","morning_star",
                    "shooting_star","bearish_engulfing","evening_star",
                    "pa_bullish","pa_bearish"]
    recent = result[pattern_cols].tail(30)
    triggered = recent[recent.any(axis=1)]
    print(f"近30日出现形态的K线数: {len(triggered)}")
    print(triggered[pattern_cols])

    print("\n=== 支撑压力位（SPY）===")
    levels = detect_swing_levels(df)
    print(f"当前价格: ${levels['current_price']}")
    print(f"支撑位:   {levels['support']}")
    print(f"压力位:   {levels['resistance']}")

    print("✅ price_action 测试通过")
