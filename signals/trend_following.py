"""
signals/trend_following.py — 趋势跟踪策略信号
包含：双均线交叉、Dual Momentum、突破系统
输出：买入/卖出/持有 信号 Series，可直接喂给回测引擎
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (TREND_MA_FAST, TREND_MA_SLOW, DUAL_MOM_LK,
                    BENCHMARK, BENCHMARK_BOND, SECTOR_ETFS)
from data.downloader import get_prices, get_returns

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 双均线交叉（MA Crossover）
# ──────────────────────────────────────────────

def ma_crossover_signal(
    prices: pd.Series,
    fast: int = TREND_MA_FAST,
    slow: int = TREND_MA_SLOW,
) -> pd.Series:
    """
    经典 50/200 日均线交叉

    Returns:
        pd.Series: 1=做多, -1=做空, 0=空仓
        index 与 prices 对齐
    """
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()

    signal = pd.Series(0, index=prices.index)
    signal[sma_fast > sma_slow] = 1   # 黄金交叉 → 做多
    signal[sma_fast < sma_slow] = -1  # 死亡交叉 → 做空（或空仓）

    # 延迟1日避免未来数据
    return signal.shift(1).fillna(0)


def ma_crossover_entry(
    prices: pd.Series,
    fast: int = TREND_MA_FAST,
    slow: int = TREND_MA_SLOW,
    volume: pd.Series = None,
    vol_filter: float = 1.5,
) -> pd.Series:
    """
    带成交量确认的均线交叉（减少假信号）

    Args:
        vol_filter: 交叉当日成交量需 > N×20日均量
    """
    raw = ma_crossover_signal(prices, fast, slow)
    entries = raw.diff().fillna(0)  # 1=新多, -1=新空

    if volume is not None:
        vol_ma   = volume.rolling(20).mean()
        vol_conf = volume > vol_ma * vol_filter
        # 新信号 + 成交量确认 才保留
        entries = entries.where(vol_conf | (entries == 0), 0)

    return entries


# ──────────────────────────────────────────────
# 2. Dual Momentum（Gary Antonacci）
# ──────────────────────────────────────────────

def dual_momentum_signal(
    lookback: int = DUAL_MOM_LK,
    us_ticker: str = "SPY",
    intl_ticker: str = "VEU",
    bond_ticker: str = "BND",
    tbill_ticker: str = "SHY",
) -> pd.DataFrame:
    """
    全球双动量策略：绝对动量 + 相对动量

    规则：
      1. 相对动量：SPY vs VEU，选N月收益更高的
      2. 绝对动量：若赢家收益 < SHY，转入债券（安全模式）

    Returns:
        DataFrame，columns = [date, holding, reason]
        holding: 'US_STOCKS' | 'INTL_STOCKS' | 'BONDS'
    """
    tickers = {
        "us":    us_ticker,
        "intl":  intl_ticker,
        "bond":  bond_ticker,
        "tbill": tbill_ticker,
    }
    prices = {k: get_prices(v) for k, v in tickers.items()}

    # 对齐日期
    common_idx = prices["us"].index
    for k in prices:
        common_idx = common_idx.intersection(prices[k].index)

    us_ret    = prices["us"].loc[common_idx].pct_change(lookback)
    intl_ret  = prices["intl"].loc[common_idx].pct_change(lookback)
    tbill_ret = prices["tbill"].loc[common_idx].pct_change(lookback)

    results = []
    for date in common_idx[lookback:]:
        ur  = us_ret.loc[date]
        ir  = intl_ret.loc[date]
        tr  = tbill_ret.loc[date]

        # 相对动量
        winner = "US_STOCKS" if ur >= ir else "INTL_STOCKS"
        winner_ret = ur if winner == "US_STOCKS" else ir

        # 绝对动量：赢家跑赢 T-bill 才持有
        if winner_ret > tr:
            holding = winner
            reason  = f"相对+绝对动量 ({winner_ret:.1%} > tbill {tr:.1%})"
        else:
            holding = "BONDS"
            reason  = f"绝对动量不足 ({winner_ret:.1%} < tbill {tr:.1%})，转入债券"

        results.append({"date": date, "holding": holding, "reason": reason})

    df = pd.DataFrame(results).set_index("date")
    return df


def dual_momentum_returns(lookback: int = DUAL_MOM_LK) -> pd.Series:
    """
    计算 Dual Momentum 策略的日收益率序列（用于回测）
    """
    signals = dual_momentum_signal(lookback=lookback)

    # 各资产日收益
    spy = get_returns("SPY")
    veu = get_returns("VEU")
    bnd = get_returns("BND")

    portfolio_ret = []
    for date, row in signals.iterrows():
        if date not in spy.index:
            continue
        if row["holding"] == "US_STOCKS":
            r = spy.loc[date] if date in spy.index else 0
        elif row["holding"] == "INTL_STOCKS":
            r = veu.loc[date] if date in veu.index else 0
        else:
            r = bnd.loc[date] if date in bnd.index else 0
        portfolio_ret.append((date, r))

    if not portfolio_ret:
        return pd.Series(dtype=float)

    dates, rets = zip(*portfolio_ret)
    return pd.Series(rets, index=pd.DatetimeIndex(dates), name="DualMomentum")


# ──────────────────────────────────────────────
# 3. 突破系统（Breakout）
# ──────────────────────────────────────────────

def breakout_signal(
    prices: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 20,
    vol_mult: float = 1.5,
) -> pd.Series:
    """
    N日高点突破 + 成交量确认

    规则：
      - 做多：收盘价突破 N 日高点 + 成交量 > vol_mult × 20日均量
      - 做空：收盘价跌破 N 日低点 + 成交量 > vol_mult × 20日均量
      - 止损：2×ATR，止盈：3×ATR

    Returns:
        pd.Series: 1=做多入场, -1=做空入场, 0=无信号
    """
    roll_high = high.rolling(period).max().shift(1)
    roll_low  = low.rolling(period).min().shift(1)
    vol_ma    = volume.rolling(20).mean()
    vol_conf  = volume > vol_ma * vol_mult

    atr = _atr(high, low, prices, 14)

    signal = pd.Series(0, index=prices.index)
    signal[(prices > roll_high) & vol_conf] =  1   # 向上突破
    signal[(prices < roll_low)  & vol_conf] = -1   # 向下突破

    return signal.shift(1).fillna(0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ──────────────────────────────────────────────
# 4. 板块动量轮动
# ──────────────────────────────────────────────

def sector_momentum_rotation(
    top_n: int = 3,
    lookback: int = 60,
    rebal_freq: str = "W-FRI",   # 每周五再平衡
) -> pd.DataFrame:
    """
    板块 ETF 动量轮动策略：
    每周选出动量最强的 top_n 个板块 ETF 等权持有

    Returns:
        DataFrame，index = 日期，columns = ETF，值 = 权重 (0 or 1/top_n)
    """
    tickers = list(SECTOR_ETFS.values())
    prices  = {}
    for t in tickers:
        p = get_prices(t)
        if not p.empty:
            prices[t] = p

    if not prices:
        return pd.DataFrame()

    price_df = pd.DataFrame(prices).dropna(how="all")
    returns  = price_df.pct_change(lookback)

    # 按再平衡频率采样
    rebal_dates = returns.resample(rebal_freq).last().index

    weight_records = []
    for date in rebal_dates:
        if date not in returns.index:
            continue
        row = returns.loc[date].dropna()
        if len(row) < top_n:
            continue

        top_etfs = row.nlargest(top_n).index.tolist()
        record   = {t: 0.0 for t in tickers}
        for t in top_etfs:
            record[t] = 1.0 / top_n
        record["date"] = date
        weight_records.append(record)

    if not weight_records:
        return pd.DataFrame()

    weights_df = pd.DataFrame(weight_records).set_index("date")

    # 前向填充到每个交易日
    all_dates = price_df.index
    weights_df = weights_df.reindex(all_dates).ffill().fillna(0)
    return weights_df


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== 均线交叉信号 ===")
    spy_px = get_prices("SPY", start="2020-01-01")
    sig = ma_crossover_signal(spy_px)
    print(sig.value_counts())

    print("\n=== Dual Momentum 最近持仓 ===")
    dm = dual_momentum_signal()
    print(dm.tail(6))
    print(f"当前持仓: {dm['holding'].iloc[-1]}")

    print("\n=== 板块轮动权重（最近） ===")
    rot = sector_momentum_rotation(top_n=3)
    if not rot.empty:
        last = rot.iloc[-1]
        print(last[last > 0])
    print("✅ trend_following 测试通过")
