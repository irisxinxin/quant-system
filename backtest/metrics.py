"""
backtest/metrics.py — 策略绩效指标计算
"""
import numpy as np
import pandas as pd
from typing import Union


def calc_metrics(returns: pd.Series, rf: float = 0.05) -> dict:
    """
    计算完整绩效指标

    Args:
        returns: 日收益率 Series
        rf:      无风险利率（年化），默认 5%

    Returns:
        dict with all performance metrics
    """
    if returns.empty or len(returns) < 2:
        return {}

    r = returns.dropna()
    cum = (1 + r).cumprod()
    n_days = len(r)
    n_years = n_days / 252

    # 年化收益
    total_ret   = cum.iloc[-1] - 1
    cagr        = (cum.iloc[-1]) ** (1 / n_years) - 1

    # 波动率
    daily_vol   = r.std()
    ann_vol     = daily_vol * np.sqrt(252)

    # Sharpe
    daily_rf    = rf / 252
    excess      = r - daily_rf
    sharpe      = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Sortino（只用下行波动率）
    downside    = r[r < daily_rf]
    down_std    = downside.std() * np.sqrt(252) if len(downside) > 1 else ann_vol
    sortino     = (cagr - rf) / down_std if down_std > 0 else 0

    # 最大回撤
    roll_max    = cum.cummax()
    drawdown    = (cum - roll_max) / roll_max
    max_dd      = drawdown.min()

    # 回撤恢复时间
    dd_duration = _max_dd_duration(cum)

    # Calmar
    calmar      = cagr / abs(max_dd) if max_dd < 0 else 0

    # 胜率
    win_rate    = (r > 0).mean()

    # 盈亏比
    avg_win     = r[r > 0].mean() if (r > 0).any() else 0
    avg_loss    = r[r < 0].mean() if (r < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # VaR / CVaR（95%）
    var_95  = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean()

    # 月度命中率
    monthly_ret = (1 + r).resample("ME").prod() - 1 if hasattr(r.index, 'freq') else _resample_monthly(r)
    monthly_win = (monthly_ret > 0).mean()

    return {
        "总收益":      f"{total_ret:.1%}",
        "年化收益(CAGR)": f"{cagr:.1%}",
        "年化波动率":   f"{ann_vol:.1%}",
        "Sharpe比率":   round(sharpe, 2),
        "Sortino比率":  round(sortino, 2),
        "Calmar比率":   round(calmar, 2),
        "最大回撤":     f"{max_dd:.1%}",
        "最长回撤天数": dd_duration,
        "日胜率":       f"{win_rate:.1%}",
        "月胜率":       f"{monthly_win:.1%}",
        "盈亏比":       round(profit_factor, 2),
        "日均收益":     f"{r.mean():.4%}",
        "VaR(95%)":    f"{var_95:.2%}",
        "CVaR(95%)":   f"{cvar_95:.2%}",
        "交易天数":     n_days,
        "年化天数":     round(n_years, 1),
    }


def _max_dd_duration(cum: pd.Series) -> int:
    """最长回撤持续天数"""
    roll_max = cum.cummax()
    in_dd    = cum < roll_max
    duration = 0
    max_dur  = 0
    for v in in_dd:
        duration = duration + 1 if v else 0
        max_dur  = max(max_dur, duration)
    return max_dur


def _resample_monthly(returns: pd.Series) -> pd.Series:
    """安全的月度收益计算"""
    try:
        return (1 + returns).resample("ME").prod() - 1
    except Exception:
        return (1 + returns).resample("M").prod() - 1


def compare_strategies(strategy_returns: dict, benchmark: pd.Series = None) -> pd.DataFrame:
    """
    多策略绩效对比表

    Args:
        strategy_returns: {策略名: 日收益率Series}
        benchmark:        基准日收益率

    Returns:
        DataFrame，行=策略，列=各绩效指标
    """
    rows = {}
    all_returns = dict(strategy_returns)
    if benchmark is not None:
        all_returns["📊 基准(SPY)"] = benchmark

    for name, ret in all_returns.items():
        metrics = calc_metrics(ret)
        rows[name] = metrics

    df = pd.DataFrame(rows).T
    # 按 Sharpe 降序排列
    if "Sharpe比率" in df.columns:
        df = df.sort_values("Sharpe比率", ascending=False)
    return df


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """滚动Sharpe比率"""
    roll_mean = returns.rolling(window).mean() * 252
    roll_std  = returns.rolling(window).std() * np.sqrt(252)
    return (roll_mean / roll_std.replace(0, np.nan)).rename("rolling_sharpe")


def drawdown_series(returns: pd.Series) -> pd.Series:
    """回撤序列"""
    cum      = (1 + returns).cumprod()
    roll_max = cum.cummax()
    return (cum - roll_max) / roll_max
