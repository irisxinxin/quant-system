"""
indicators.py — 技术指标共享库 (用于 8 大策略)

所有指标都用 .shift(1) 或同根 close 完成, 无 look-ahead bias.
"""
import numpy as np
import pandas as pd


def vwap_intraday(day_df: pd.DataFrame) -> pd.Series:
    """日内 VWAP. day_df 必须是同一天的 5m bars."""
    typical = (day_df["High"] + day_df["Low"] + day_df["Close"]) / 3
    pv = typical * day_df["Volume"]
    return pv.cumsum() / day_df["Volume"].cumsum().replace(0, np.nan)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_dn = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = (high.diff()).where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_v = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_v.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_v.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20, mult: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return mid, mid + mult * sd, mid - mult * sd


def keltner(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 20, atr_mult: float = 1.5):
    mid = close.ewm(span=period, adjust=False).mean()
    a = atr(high, low, close, period)
    return mid, mid + atr_mult * a, mid - atr_mult * a


def donchian(high: pd.Series, low: pd.Series, period: int = 20):
    """返回 (upper, lower, mid). 用 shift(1) 防止 look-ahead (当根 bar 不算)."""
    upper = high.shift(1).rolling(period).max()
    lower = low.shift(1).rolling(period).min()
    return upper, lower, (upper + lower) / 2


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, mult: float = 3.0):
    """
    Supertrend (Pine Script 经典实现). 返回 (st_value, direction).
      direction: +1 = 多头, -1 = 空头
    """
    a = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_basic = hl2 + mult * a
    lower_basic = hl2 - mult * a

    n = len(close)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    st = np.full(n, np.nan)
    dir_ = np.full(n, 1, dtype=int)   # 初始多头

    cl = close.values
    for i in range(n):
        if i == 0 or pd.isna(upper_basic.iloc[i]):
            upper_band[i] = upper_basic.iloc[i] if not pd.isna(upper_basic.iloc[i]) else cl[i]
            lower_band[i] = lower_basic.iloc[i] if not pd.isna(lower_basic.iloc[i]) else cl[i]
            dir_[i] = 1
            st[i] = lower_band[i]
            continue

        # Trailing 上轨: 只收紧, 不扩张 (除非前一根价格已突破上轨)
        if upper_basic.iloc[i] < upper_band[i - 1] or cl[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic.iloc[i]
        else:
            upper_band[i] = upper_band[i - 1]

        if lower_basic.iloc[i] > lower_band[i - 1] or cl[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic.iloc[i]
        else:
            lower_band[i] = lower_band[i - 1]

        # 方向切换
        if cl[i] > upper_band[i - 1]:
            dir_[i] = 1
        elif cl[i] < lower_band[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]

        st[i] = lower_band[i] if dir_[i] == 1 else upper_band[i]

    return pd.Series(st, index=close.index), pd.Series(dir_, index=close.index)


def daily_ema_from_5m(df_5m: pd.DataFrame, period: int = 200) -> pd.Series:
    """从 5m 数据聚合到日线再算 EMA. 返回每个 5m bar 对应的当日 EMA 值."""
    daily_close = df_5m["Close"].resample("1D").last().dropna()
    daily_ema = daily_close.ewm(span=period, adjust=False).mean()
    # 每个 5m bar 用前一日的 EMA (避免 look-ahead)
    daily_ema_shifted = daily_ema.shift(1)
    # broadcast 到 5m 频率
    expanded = daily_ema_shifted.reindex(df_5m.index, method="ffill")
    return expanded
