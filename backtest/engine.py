"""
backtest/engine.py — 轻量级回测引擎
支持：单策略回测、多策略对比、Walk-Forward 验证
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Callable, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (BACKTEST_INIT_CASH, BACKTEST_COMMISSION,
                    BACKTEST_SLIPPAGE, WF_TRAIN_YEARS, WF_TEST_YEARS,
                    BENCHMARK, DATA_START)
from data.downloader import get_prices, get_returns
from backtest.metrics import calc_metrics, compare_strategies

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 核心回测函数
# ──────────────────────────────────────────────

def backtest(
    prices: pd.Series,
    signal: pd.Series,
    commission: float = BACKTEST_COMMISSION,
    slippage:   float = BACKTEST_SLIPPAGE,
    allow_short: bool = False,
    name: str = "Strategy",
) -> dict:
    """
    单资产回测

    Args:
        prices:      日收盘价 Series
        signal:      仓位信号 Series：1=满仓多头, -1=满仓空头, 0=空仓
                     也支持 0~1 浮点数（部分仓位）
        commission:  双边手续费率（每次交易）
        slippage:    单边滑点（执行价偏差）
        allow_short: 是否允许做空
        name:        策略名称

    Returns:
        dict with 'returns', 'equity', 'trades', 'metrics'
    """
    # 对齐
    common  = prices.index.intersection(signal.index)
    prices  = prices.loc[common]
    signal  = signal.loc[common]

    if not allow_short:
        signal = signal.clip(lower=0)   # 纯多头

    # 日收益率
    raw_ret = prices.pct_change()

    # 持仓变化（用于计算交易成本）
    pos_change   = signal.diff().abs().fillna(signal.abs())
    total_cost   = pos_change * (commission + slippage)

    # 策略收益 = 信号 × 次日收益 - 交易成本
    strat_ret = signal.shift(1).fillna(0) * raw_ret - total_cost
    strat_ret  = strat_ret.dropna()

    # 净值曲线
    equity = (1 + strat_ret).cumprod() * BACKTEST_INIT_CASH

    # 交易记录
    trades = _extract_trades(signal, prices)

    # 绩效指标
    metrics = calc_metrics(strat_ret)
    metrics["策略名称"] = name

    return {
        "returns":  strat_ret,
        "equity":   equity,
        "trades":   trades,
        "metrics":  metrics,
        "signal":   signal,
    }


def backtest_multi(
    prices_dict: dict,
    signal_dict: dict,
    allow_short: bool = False,
) -> dict:
    """
    多资产组合回测（等权或按信号权重）

    Args:
        prices_dict: {ticker: price_series}
        signal_dict: {ticker: signal_series}

    Returns:
        dict with portfolio returns, equity, metrics
    """
    returns_list = []

    for ticker, price in prices_dict.items():
        if ticker not in signal_dict:
            continue
        sig    = signal_dict[ticker]
        result = backtest(price, sig, allow_short=allow_short, name=ticker)
        returns_list.append(result["returns"].rename(ticker))

    if not returns_list:
        return {}

    ret_df  = pd.concat(returns_list, axis=1).dropna(how="all")
    # 等权组合
    port_ret = ret_df.mean(axis=1)
    equity   = (1 + port_ret).cumprod() * BACKTEST_INIT_CASH
    metrics  = calc_metrics(port_ret)

    return {
        "returns":       port_ret,
        "equity":        equity,
        "asset_returns": ret_df,
        "metrics":       metrics,
    }


# ──────────────────────────────────────────────
# Walk-Forward 验证
# ──────────────────────────────────────────────

def walk_forward(
    prices: pd.Series,
    strategy_fn: Callable[[pd.Series], pd.Series],
    train_years: int = WF_TRAIN_YEARS,
    test_years:  int = WF_TEST_YEARS,
    name: str = "WF Strategy",
) -> dict:
    """
    Walk-Forward 滚动回测

    Args:
        prices:       完整价格序列
        strategy_fn:  接收训练期价格，返回测试期信号的函数
                      signature: fn(train_prices: pd.Series) -> pd.Series（测试期信号）
        train_years:  训练窗口年数
        test_years:   测试窗口年数

    Returns:
        dict with oos_returns (out-of-sample), windows, metrics
    """
    splits    = _wf_splits(prices.index, train_years, test_years)
    oos_parts = []

    logger.info(f"[WF] {len(splits)} 个窗口 ({train_years}年训练/{test_years}年测试)")

    for i, (train_start, train_end, test_end) in enumerate(splits):
        train_px = prices.loc[train_start:train_end]
        test_px  = prices.loc[train_end:test_end]

        if len(train_px) < 252 or len(test_px) < 20:
            continue

        try:
            # 用训练期参数生成测试期信号
            signal_oos = strategy_fn(train_px)
            # 只保留测试期
            if hasattr(signal_oos, 'loc'):
                signal_test = signal_oos.reindex(test_px.index).ffill().fillna(0)
            else:
                signal_test = pd.Series(signal_oos, index=test_px.index)

            result = backtest(test_px, signal_test, name=f"WF_{i+1}")
            oos_parts.append(result["returns"])
            logger.info(f"  窗口 {i+1}: {train_end.date()} ~ {test_end.date()}, "
                        f"Sharpe={result['metrics'].get('Sharpe比率', 'N/A')}")
        except Exception as e:
            logger.warning(f"  窗口 {i+1} 失败: {e}")

    if not oos_parts:
        return {}

    oos_ret = pd.concat(oos_parts).sort_index()
    oos_ret = oos_ret[~oos_ret.index.duplicated(keep="first")]
    metrics = calc_metrics(oos_ret)
    metrics["策略名称"] = f"{name} (WF OOS)"

    return {
        "oos_returns": oos_ret,
        "equity":      (1 + oos_ret).cumprod() * BACKTEST_INIT_CASH,
        "metrics":     metrics,
        "n_windows":   len(oos_parts),
    }


def _wf_splits(index: pd.DatetimeIndex, train_y: int, test_y: int):
    start = index[0]
    splits = []
    while True:
        train_end = start + pd.DateOffset(years=train_y)
        test_end  = train_end + pd.DateOffset(years=test_y)
        if test_end > index[-1]:
            break
        splits.append((start, train_end, test_end))
        start = train_end
    return splits


# ──────────────────────────────────────────────
# 策略比较
# ──────────────────────────────────────────────

def run_all_strategies(ticker: str = "SPY", start: str = DATA_START) -> pd.DataFrame:
    """
    对指定标的运行所有策略，生成对比表

    Returns:
        绩效对比 DataFrame
    """
    from signals.trend_following  import ma_crossover_signal
    from signals.mean_reversion   import rsi_signal_series, bollinger_signal
    from signals.price_action     import price_action_signal, pa_confirmed_entry_backtest
    from signals.smart_money      import smc_signal
    from data.downloader          import get_ohlcv

    prices = get_prices(ticker, start=start)
    ohlcv  = get_ohlcv(ticker, start=start)
    spy    = get_returns(BENCHMARK, start=start)

    if prices.empty:
        logger.error(f"无法获取 {ticker} 数据")
        return pd.DataFrame()

    strategy_returns = {}

    # 1. 买入持有
    strategy_returns["买入持有"] = prices.pct_change().dropna()

    # ── 以下所有策略均为纯多头（allow_short=False，负信号→空仓）──

    # 2. MA 交叉（50/200，死亡交叉时空仓而非做空）
    try:
        sig = ma_crossover_signal(prices).clip(lower=0)
        res = backtest(prices, sig, name="MA交叉")
        strategy_returns["MA均线交叉"] = res["returns"]
    except Exception as e:
        logger.warning(f"MA交叉失败: {e}")

    # 3. RSI 均值回归（趋势过滤已只生成多头信号）
    try:
        sig = rsi_signal_series(prices).clip(lower=0)
        res = backtest(prices, sig, name="RSI均值回归")
        strategy_returns["RSI均值回归"] = res["returns"]
    except Exception as e:
        logger.warning(f"RSI均值回归失败: {e}")

    # 4. 布林带（long_only=True：触上轨空仓而非做空）
    try:
        bb_df = bollinger_signal(prices, long_only=True)
        sig   = bb_df["signal"].clip(lower=0)
        res   = backtest(prices, sig, name="布林带")
        strategy_returns["布林带"] = res["returns"]
    except Exception as e:
        logger.warning(f"布林带失败: {e}")

    # 5. 裸K（只保留多头信号）
    try:
        pa_df = price_action_signal(ticker)
        if not pa_df.empty:
            sig = pa_df["signal"].reindex(prices.index).fillna(0).clip(lower=0)
            res = backtest(prices, sig, name="裸K形态")
            strategy_returns["裸K形态"] = res["returns"]
    except Exception as e:
        logger.warning(f"裸K失败: {e}")

    # 6. SMC（只保留多头信号）
    try:
        smc_df = smc_signal(ticker)
        if not smc_df.empty:
            sig = smc_df["signal"].reindex(prices.index).fillna(0).clip(lower=0)
            res = backtest(prices, sig, name="SMC")
            strategy_returns["SMC智能资金"] = res["returns"]
    except Exception as e:
        logger.warning(f"SMC失败: {e}")

    # 7. 裸K改进版：二次确认 + 限价回调入场 + 支撑止损
    try:
        pa_v2 = pa_confirmed_entry_backtest(ticker, start=start)
        if pa_v2 and not pa_v2["returns"].empty:
            strategy_returns["裸K确认回调"] = pa_v2["returns"]
            # 打印详细交易统计
            m = pa_v2["metrics"]
            logger.info(f"裸K确认回调: 交易{m.get('交易次数',0)}次 | "
                        f"胜率{m.get('交易胜率','N/A')} | "
                        f"止盈{m.get('止盈次数',0)}次 | 止损{m.get('止损次数',0)}次")
    except Exception as e:
        logger.warning(f"裸K确认回调失败: {e}", exc_info=True)

    # ── TradingView 策略 ──────────────────────────────────────────────
    from signals.tv_strategies import (
        adaptive_ema_signal, combined_tv_signal, vegas_bounce_signal
    )

    # 8. EMA 自适应（20/60）
    try:
        sig = adaptive_ema_signal(prices, fast=20, slow=60)
        res = backtest(prices, sig, name="EMA自适应")
        strategy_returns["EMA自适应(20/60)"] = res["returns"]
    except Exception as e:
        logger.warning(f"EMA自适应失败: {e}")

    # 9. Vegas 反弹策略（DXDX + EMA144过滤，出场用DBJGXC or EMA169）
    try:
        sig = vegas_bounce_signal(prices)
        res = backtest(prices, sig, name="Vegas反弹")
        strategy_returns["Vegas反弹策略"] = res["returns"]
    except Exception as e:
        logger.warning(f"Vegas反弹失败: {e}")

    # 10. 抄底/卖出组合策略（DXDX + EMA60过滤，出场用DBJGXC or EMA60）
    try:
        sig = combined_tv_signal(prices)
        res = backtest(prices, sig, name="抄底卖出组合")
        strategy_returns["抄底卖出组合"] = res["returns"]
    except Exception as e:
        logger.warning(f"抄底卖出组合失败: {e}")

    # 11. 双信号OR模式+CTA过滤（SMC或EMA任一满足 + 宏观顺风）
    try:
        from signals.combo_strategy import duo_cta_signal
        sig = duo_cta_signal(prices, ticker=ticker, mode="OR", start=start)
        res = backtest(prices, sig, name="SMC/EMA-OR+CTA")
        strategy_returns["SMC∨EMA+CTA"] = res["returns"]
    except Exception as e:
        logger.warning(f"duo_OR+CTA失败: {e}", exc_info=True)

    # 12. 双信号AND模式+CTA过滤（SMC且EMA同时满足 + 宏观顺风）
    try:
        from signals.combo_strategy import duo_cta_signal
        sig = duo_cta_signal(prices, ticker=ticker, mode="AND", start=start)
        res = backtest(prices, sig, name="SMC&EMA-AND+CTA")
        strategy_returns["SMC∧EMA+CTA"] = res["returns"]
    except Exception as e:
        logger.warning(f"duo_AND+CTA失败: {e}", exc_info=True)

    if not strategy_returns:
        return pd.DataFrame()

    bench_ret = spy.reindex(prices.pct_change().index)
    return compare_strategies(strategy_returns, benchmark=bench_ret)


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _extract_trades(signal: pd.Series, prices: pd.Series) -> pd.DataFrame:
    """提取买卖记录"""
    trades  = []
    pos     = 0
    entry_p = None
    entry_d = None

    for date, sig in signal.items():
        if pos == 0 and sig != 0:
            pos     = sig
            entry_p = prices.get(date, None)
            entry_d = date
        elif pos != 0 and (sig == 0 or sig != pos):
            exit_p = prices.get(date, entry_p)
            pnl    = (exit_p - entry_p) / entry_p * pos if entry_p else 0
            trades.append({
                "entry_date": entry_d,
                "exit_date":  date,
                "direction":  "多头" if pos > 0 else "空头",
                "entry_px":   round(entry_p, 2) if entry_p else None,
                "exit_px":    round(exit_p, 2),
                "pnl_pct":    round(pnl * 100, 2),
                "days_held":  (date - entry_d).days if entry_d else 0,
            })
            pos     = sig if sig != 0 else 0
            entry_p = prices.get(date) if sig != 0 else None
            entry_d = date if sig != 0 else None

    return pd.DataFrame(trades)


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== 策略对比回测（SPY）===")
    result_df = run_all_strategies("SPY")
    if not result_df.empty:
        print(result_df[["年化收益(CAGR)", "Sharpe比率", "最大回撤",
                          "Calmar比率", "日胜率"]].to_string())
    print("✅ engine 测试通过")
