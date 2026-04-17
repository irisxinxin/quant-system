"""
signals/mean_reversion.py — 均值回归策略信号
包含：RSI均值回归、布林带、配对交易（统计套利）
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (MR_RSI_PERIOD, MR_RSI_OB, MR_RSI_OS,
                    MR_BB_PERIOD, MR_BB_STD, MR_TREND_MA)
from data.downloader import get_prices, get_ohlcv

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 辅助：RSI 计算
# ──────────────────────────────────────────────

def _rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ──────────────────────────────────────────────
# 1. RSI 均值回归（主力策略）
# ──────────────────────────────────────────────

def rsi_mean_reversion_signal(
    prices: pd.Series,
    rsi_period: int = MR_RSI_PERIOD,
    ob: float = MR_RSI_OB,
    os_: float = MR_RSI_OS,
    trend_ma: int = MR_TREND_MA,
    use_trend_filter: bool = True,
) -> pd.DataFrame:
    """
    RSI(10) 均值回归
    只在趋势方向做：多头市场做超卖反弹，空头市场做超买回落

    Entry:
      Long:  RSI < os_  AND (price > SMA(200) 或不用趋势过滤)
      Short: RSI > ob   AND (price < SMA(200) 或不用趋势过滤)
    Exit:
      Long:  RSI > 55（恢复正常区间）或 RSI > 70（超买）
      Short: RSI < 45（恢复正常区间）或 RSI < 30（超卖）

    Returns:
        DataFrame，columns = [rsi, trend, signal, exit_long, exit_short]
    """
    rsi  = _rsi_series(prices, rsi_period)
    sma  = prices.rolling(trend_ma).mean()

    in_uptrend   = prices > sma    # 多头市场
    in_downtrend = prices < sma    # 空头市场

    # 入场信号（次日执行，shift(1)）
    long_entry  = pd.Series(False, index=prices.index)
    short_entry = pd.Series(False, index=prices.index)

    if use_trend_filter:
        long_entry  = (rsi < os_)  & in_uptrend    # 超卖 + 上升趋势
        short_entry = (rsi > ob)   & in_downtrend  # 超买 + 下降趋势
    else:
        long_entry  = rsi < os_
        short_entry = rsi > ob

    # 出场信号
    exit_long  = rsi > 55  # 多头仓位，RSI回到中位区
    exit_short = rsi < 45  # 空头仓位，RSI回到中位区

    df = pd.DataFrame({
        "price":       prices,
        "rsi":         rsi,
        "sma":         sma,
        "trend":       np.where(in_uptrend, "UP", "DOWN"),
        "long_entry":  long_entry.shift(1).fillna(False),
        "short_entry": short_entry.shift(1).fillna(False),
        "exit_long":   exit_long.shift(1).fillna(False),
        "exit_short":  exit_short.shift(1).fillna(False),
    })
    return df


def rsi_signal_series(
    prices: pd.Series,
    rsi_period: int = MR_RSI_PERIOD,
    ob: float = MR_RSI_OB,
    os_: float = MR_RSI_OS,
    trend_ma: int = MR_TREND_MA,
) -> pd.Series:
    """
    简化版：返回 +1 / -1 / 0 信号 Series（供回测引擎使用）
    """
    df     = rsi_mean_reversion_signal(prices, rsi_period, ob, os_, trend_ma)
    signal = pd.Series(0, index=prices.index)
    pos    = 0

    for i in range(len(df)):
        row = df.iloc[i]
        if pos == 0:
            if row["long_entry"]:
                pos = 1
            elif row["short_entry"]:
                pos = -1
        elif pos == 1 and row["exit_long"]:
            pos = 0
        elif pos == -1 and row["exit_short"]:
            pos = 0
        signal.iloc[i] = pos

    return signal


# ──────────────────────────────────────────────
# 2. 布林带均值回归
# ──────────────────────────────────────────────

def bollinger_signal(
    prices: pd.Series,
    period: int = MR_BB_PERIOD,
    std_mult: float = MR_BB_STD,
    rsi_confirm: bool = True,
    rsi_period: int = 14,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    布林带 + RSI 双重确认均值回归

    Long:  价格触及下轨 AND（RSI < 35 可选）
    Short: 价格触及上轨 AND（RSI > 65 可选）；long_only=True 时跳过做空
    Exit:  价格回到中轨（SMA）

    Returns:
        DataFrame with [mid, upper, lower, signal, position]
    """
    mid   = prices.rolling(period).mean()
    std   = prices.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    rsi   = _rsi_series(prices, rsi_period)

    # 基础信号
    long_cond  = prices < lower
    short_cond = prices > upper

    if rsi_confirm:
        long_cond  = long_cond  & (rsi < 35)
        short_cond = short_cond & (rsi > 65)

    # 出场：回到中轨
    exit_long  = prices >= mid
    exit_short = prices <= mid

    # 构建持仓序列
    signal = pd.Series(0, index=prices.index)
    pos    = 0
    for i in range(1, len(prices)):
        prev_long  = long_cond.iloc[i - 1]
        prev_short = short_cond.iloc[i - 1]
        at_mid_l   = exit_long.iloc[i - 1]
        at_mid_s   = exit_short.iloc[i - 1]

        if pos == 0:
            if prev_long:
                pos = 1
            elif prev_short and not long_only:
                pos = -1
            # long_only=True 时触碰上轨直接忽略，保持空仓
        elif pos == 1 and at_mid_l:
            pos = 0
        elif pos == -1 and at_mid_s:
            pos = 0
        signal.iloc[i] = pos

    return pd.DataFrame({
        "price":  prices,
        "mid":    mid,
        "upper":  upper,
        "lower":  lower,
        "rsi":    rsi,
        "signal": signal,
    })


# ──────────────────────────────────────────────
# 3. 配对交易（统计套利）
# ──────────────────────────────────────────────

def find_cointegrated_pairs(
    tickers: list,
    significance: float = 0.05,
    min_corr: float = 0.70,
) -> pd.DataFrame:
    """
    从股票列表中找出协整对

    Returns:
        DataFrame，columns = [ticker_a, ticker_b, pvalue, correlation, hedge_ratio]
        按 pvalue 升序排列
    """
    prices = {}
    for t in tickers:
        p = get_prices(t)
        if not p.empty and len(p) > 252:
            prices[t] = p

    if len(prices) < 2:
        return pd.DataFrame()

    price_df  = pd.DataFrame(prices).dropna(how="any")
    log_price = np.log(price_df)
    tickers   = list(price_df.columns)
    pairs     = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]

            # 相关性预过滤（快速）
            corr = price_df[a].corr(price_df[b])
            if corr < min_corr:
                continue

            # 协整检验
            try:
                score, pvalue, _ = coint(log_price[a], log_price[b])
            except Exception:
                continue

            if pvalue < significance:
                # 计算对冲比率（OLS）
                beta = np.polyfit(log_price[b], log_price[a], 1)[0]
                pairs.append({
                    "ticker_a":    a,
                    "ticker_b":    b,
                    "pvalue":      round(pvalue, 4),
                    "correlation": round(corr, 3),
                    "hedge_ratio": round(beta, 4),
                })

    return pd.DataFrame(pairs).sort_values("pvalue").reset_index(drop=True)


def pairs_spread(
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float = None,
) -> pd.Series:
    """
    计算配对价差（spread）
    spread = log(A) - hedge_ratio × log(B)
    """
    pa = get_prices(ticker_a)
    pb = get_prices(ticker_b)

    common = pa.index.intersection(pb.index)
    log_a  = np.log(pa.loc[common])
    log_b  = np.log(pb.loc[common])

    if hedge_ratio is None:
        hedge_ratio = np.polyfit(log_b, log_a, 1)[0]

    spread = log_a - hedge_ratio * log_b
    return spread


def pairs_trading_signal(
    ticker_a: str,
    ticker_b: str,
    hedge_ratio: float = None,
    entry_z: float = 1.5,
    exit_z:  float = 0.5,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    配对交易信号

    Long spread:  Z-score < -entry_z（A 相对便宜）做多 A，做空 B
    Short spread: Z-score > +entry_z（A 相对贵）做空 A，做多 B
    Exit:         |Z-score| < exit_z

    Returns:
        DataFrame，columns = [spread, zscore, signal_a, signal_b]
        signal_a/b: +1 做多，-1 做空，0 空仓
    """
    spread = pairs_spread(ticker_a, ticker_b, hedge_ratio)
    mean   = spread.rolling(lookback).mean()
    std    = spread.rolling(lookback).std()
    zscore = (spread - mean) / std.replace(0, np.nan)

    signal_a = pd.Series(0, index=spread.index)
    signal_b = pd.Series(0, index=spread.index)
    pos      = 0

    for i in range(lookback, len(zscore)):
        z = zscore.iloc[i - 1]   # 用前一日信号，次日执行
        if np.isnan(z):
            continue

        if pos == 0:
            if z < -entry_z:
                pos =  1    # 做多 spread：多 A 空 B
            elif z > entry_z:
                pos = -1    # 做空 spread：空 A 多 B
        elif pos == 1 and abs(z) < exit_z:
            pos = 0
        elif pos == -1 and abs(z) < exit_z:
            pos = 0

        signal_a.iloc[i] =  pos
        signal_b.iloc[i] = -pos

    return pd.DataFrame({
        "spread":   spread,
        "zscore":   zscore,
        "signal_a": signal_a,
        "signal_b": signal_b,
    })


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== RSI 均值回归信号（SPY）===")
    spy_px = get_prices("SPY", start="2021-01-01")
    df_rsi = rsi_mean_reversion_signal(spy_px)
    entries = df_rsi[df_rsi["long_entry"] | df_rsi["short_entry"]]
    print(f"入场信号数: {len(entries)}")
    print(entries[["price", "rsi", "trend", "long_entry", "short_entry"]].tail(5))

    print("\n=== 布林带信号（QQQ）===")
    qqq_px = get_prices("QQQ", start="2021-01-01")
    df_bb = bollinger_signal(qqq_px)
    print(df_bb[["price", "mid", "upper", "lower", "signal"]].tail(5))

    print("\n=== 配对交易（XOM vs CVX）===")
    pt = pairs_trading_signal("XOM", "CVX")
    trades = pt[(pt["signal_a"] != 0)]
    print(f"交易次数: {len(trades)}")
    print(pt[["spread", "zscore", "signal_a", "signal_b"]].tail(5))

    print("✅ mean_reversion 测试通过")
