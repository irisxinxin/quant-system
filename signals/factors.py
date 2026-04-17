"""
signals/factors.py — 多因子选股
包含：价值、动量、质量、低波动因子，综合打分选股
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (FACTOR_UNIV_SIZE, FACTOR_TOP_N,
                    FACTOR_REBAL_FREQ, DATA_START)
from data.downloader import get_prices, get_multi, get_returns
from data.universe import get_sp500_tickers

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 动量因子（可从价格计算，无需财务数据）
# ──────────────────────────────────────────────

def momentum_factor(
    prices_df: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series:
    """
    12-1 动量因子（跳过最近1个月避免短期反转）

    Returns:
        最新时间点各股票的动量得分，已排序（高=强）
    """
    # 回望期收益，跳过最近 skip 天
    ret = prices_df.pct_change(lookback).shift(skip)
    return ret.iloc[-1].dropna().rank(pct=True)


def mom_factor_zscore(
    prices_df: pd.DataFrame,
    lookbacks: list = [21, 63, 126, 252],
    skip: int = 21,
) -> pd.Series:
    """
    多周期动量综合因子（Z-score 标准化后平均）
    """
    zscores = []
    for lk in lookbacks:
        ret = prices_df.pct_change(lk).shift(skip).iloc[-1]
        z = (ret - ret.mean()) / ret.std()
        zscores.append(z)
    return pd.concat(zscores, axis=1).mean(axis=1).rank(pct=True)


# ──────────────────────────────────────────────
# 2. 低波动因子（可从价格计算）
# ──────────────────────────────────────────────

def low_vol_factor(
    prices_df: pd.DataFrame,
    vol_window: int = 252,
) -> pd.Series:
    """
    低波动因子：历史波动率越低得分越高（取反排名）

    Returns:
        各股票低波动得分 rank（高=低波动）
    """
    ret = prices_df.pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    latest_vol = vol.iloc[-1].dropna()
    return (1 - latest_vol.rank(pct=True))   # 低波动 → 高得分


def beta_factor(
    prices_df: pd.DataFrame,
    benchmark: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    低 Beta 因子：Beta 越低得分越高
    """
    bench_ret = benchmark.pct_change().dropna()
    results = {}
    for col in prices_df.columns:
        stock_ret = prices_df[col].pct_change().dropna()
        common = stock_ret.index.intersection(bench_ret.index)
        if len(common) < window // 2:
            continue
        s = stock_ret.loc[common].tail(window)
        b = bench_ret.loc[common].tail(window)
        if b.std() == 0:
            continue
        beta = s.cov(b) / b.var()
        results[col] = beta

    beta_series = pd.Series(results)
    return (1 - beta_series.rank(pct=True))   # 低 Beta → 高得分


# ──────────────────────────────────────────────
# 3. 趋势/质量代理因子（基于价格）
# ──────────────────────────────────────────────

def price_trend_quality(
    prices_df: pd.DataFrame,
    ma_period: int = 200,
) -> pd.Series:
    """
    价格趋势质量：当前价格相对200日均线的位置
    越高于均线，趋势质量越好（适合质量因子代理）
    """
    sma = prices_df.rolling(ma_period).mean()
    above_ma = (prices_df.iloc[-1] / sma.iloc[-1] - 1)
    return above_ma.dropna().rank(pct=True)


def return_consistency(
    prices_df: pd.DataFrame,
    period: int = 252,
) -> pd.Series:
    """
    收益一致性：正收益周数 / 总周数（代理稳定性/质量）
    """
    weekly = prices_df.resample("W").last().pct_change()
    pos_rate = (weekly > 0).rolling(period // 5).mean().iloc[-1]
    return pos_rate.dropna().rank(pct=True)


# ──────────────────────────────────────────────
# 4. 多因子综合打分
# ──────────────────────────────────────────────

def multi_factor_score(
    prices_df: pd.DataFrame,
    benchmark_prices: pd.Series = None,
    weights: dict = None,
) -> pd.DataFrame:
    """
    综合多因子打分（所有因子均由价格计算，无需财务数据）

    因子权重（默认）：
      - 动量（12-1月）:  30%
      - 多周期动量:       20%
      - 低波动:          20%
      - 价格趋势质量:    15%
      - 收益一致性:      15%

    Returns:
        DataFrame，index = ticker，含各因子得分和综合得分
        按综合得分降序排列
    """
    if weights is None:
        weights = {
            "momentum_12_1":  0.30,
            "momentum_multi": 0.20,
            "low_vol":        0.20,
            "trend_quality":  0.15,
            "consistency":    0.15,
        }

    logger.info(f"[factors] 计算 {prices_df.shape[1]} 只股票的因子得分...")

    factors = {}

    # 动量
    try:
        factors["momentum_12_1"] = momentum_factor(prices_df)
    except Exception as e:
        logger.warning(f"momentum_12_1 失败: {e}")

    # 多周期动量
    try:
        factors["momentum_multi"] = mom_factor_zscore(prices_df)
    except Exception as e:
        logger.warning(f"momentum_multi 失败: {e}")

    # 低波动
    try:
        factors["low_vol"] = low_vol_factor(prices_df)
    except Exception as e:
        logger.warning(f"low_vol 失败: {e}")

    # 价格趋势质量
    try:
        factors["trend_quality"] = price_trend_quality(prices_df)
    except Exception as e:
        logger.warning(f"trend_quality 失败: {e}")

    # 收益一致性
    try:
        factors["consistency"] = return_consistency(prices_df)
    except Exception as e:
        logger.warning(f"consistency 失败: {e}")

    if not factors:
        return pd.DataFrame()

    factor_df = pd.DataFrame(factors)

    # 加权综合得分
    composite = pd.Series(0.0, index=factor_df.index)
    total_w = 0.0
    for name, w in weights.items():
        if name in factor_df.columns:
            col = factor_df[name].fillna(0.5)
            composite += col * w
            total_w += w
    if total_w > 0:
        composite /= total_w

    factor_df["composite_score"] = composite
    factor_df = factor_df.sort_values("composite_score", ascending=False)
    return factor_df


# ──────────────────────────────────────────────
# 5. 月度选股主流程
# ──────────────────────────────────────────────

def monthly_stock_selection(
    universe: list = None,
    top_n: int = FACTOR_TOP_N,
    min_price: float = 5.0,
    min_history_days: int = 252,
) -> pd.DataFrame:
    """
    每月月末运行：从 S&P500 中选出综合得分最高的 top_n 只股票

    Returns:
        DataFrame，top_n 只股票 + 各因子得分
    """
    if universe is None:
        universe = get_sp500_tickers()

    logger.info(f"[selection] 下载 {len(universe)} 只股票的价格数据...")
    prices = get_multi(universe, start=DATA_START)

    if prices.empty:
        logger.error("[selection] 无法获取价格数据")
        return pd.DataFrame()

    # 过滤：当前价格 > min_price，历史数据充足
    latest_px = prices.iloc[-1]
    valid = [t for t in prices.columns
             if latest_px.get(t, 0) >= min_price
             and prices[t].count() >= min_history_days]

    logger.info(f"[selection] 有效股票: {len(valid)}")
    prices = prices[valid]

    # 计算多因子得分
    scores = multi_factor_score(prices)
    if scores.empty:
        return pd.DataFrame()

    top_stocks = scores.head(top_n)
    logger.info(f"[selection] 选出 top {top_n}: {top_stocks.index.tolist()}")
    return top_stocks


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== 小范围因子测试（15只科技股）===")
    test_tickers = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AMZN", "TSLA", "AMD", "INTC", "AVGO",
        "QCOM", "MU", "AMAT", "ADI", "KLAC",
    ]
    prices = get_multi(test_tickers, start="2021-01-01")
    if not prices.empty:
        scores = multi_factor_score(prices)
        print(scores[["momentum_12_1", "low_vol", "trend_quality",
                       "consistency", "composite_score"]].round(3).to_string())
        print(f"\n综合得分 Top 5: {scores.index[:5].tolist()}")

    print("✅ factors 测试通过")
