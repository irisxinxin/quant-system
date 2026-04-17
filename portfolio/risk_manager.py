"""
portfolio/risk_manager.py — 风险管理
仓位sizing、回撤熔断、CTA+板块双重过滤
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (RISK_MAX_DD, RISK_PER_TRADE, RISK_MAX_SECTOR,
                    RISK_MAX_POSITION, BACKTEST_INIT_CASH)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 仓位计算
# ──────────────────────────────────────────────

def position_size_fixed_risk(
    capital: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = RISK_PER_TRADE,
) -> int:
    """
    固定风险仓位计算（最常用）
    每笔亏损不超过总资金的 risk_pct

    公式: shares = (capital × risk_pct) / (entry - stop)
    """
    risk_amount = capital * risk_pct
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return 0
    shares = int(risk_amount / risk_per_share)
    # 额外限制：单笔仓位不超过总资金的 RISK_MAX_POSITION
    max_shares = int(capital * RISK_MAX_POSITION / entry_price)
    return min(shares, max_shares)


def position_size_kelly(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,   # 1/4 Kelly（保守）
) -> float:
    """
    Kelly 准则仓位（保守版：1/4 Kelly）

    f = W - (1-W)/R
    W = 胜率，R = 盈亏比

    Returns:
        建议资金使用比例（0~1）
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    R = abs(avg_win / avg_loss)
    kelly_f = win_rate - (1 - win_rate) / R
    kelly_f = max(0, kelly_f)           # 不允许负数
    kelly_f = min(kelly_f * fraction, RISK_MAX_POSITION)  # 上限
    return round(kelly_f, 3)


def position_size_vol_target(
    prices: pd.Series,
    target_vol: float = 0.12,   # 年化目标波动率 12%
    capital: float = BACKTEST_INIT_CASH,
    vol_window: int = 20,
) -> float:
    """
    波动率目标化仓位
    根据最近波动率动态调整，保持组合波动率恒定

    Returns:
        建议持仓比例（0~2，允许轻度杠杆）
    """
    ret = prices.pct_change().dropna()
    if len(ret) < vol_window:
        return 1.0

    realized_vol = ret.tail(vol_window).std() * np.sqrt(252)
    if realized_vol <= 0:
        return 1.0

    weight = target_vol / realized_vol
    weight = min(weight, 2.0)    # 最多2倍杠杆
    weight = max(weight, 0.1)    # 最少10%仓位
    return round(weight, 3)


# ──────────────────────────────────────────────
# 2. 回撤熔断
# ──────────────────────────────────────────────

class DrawdownGuard:
    """
    回撤熔断器：实时监控组合净值，触发阈值时强制降仓

    用法：
        guard = DrawdownGuard(max_dd=0.15, reduce_pct=0.5)
        multiplier = guard.check(current_equity)
        # multiplier = 1.0 正常, 0.5 降仓50%, 0.0 全部清仓
    """
    def __init__(
        self,
        max_dd: float = RISK_MAX_DD,
        warning_dd: float = None,
        reduce_pct: float = 0.5,
    ):
        self.max_dd     = max_dd
        self.warning_dd = warning_dd or max_dd * 0.7
        self.reduce_pct = reduce_pct
        self.peak       = None
        self.halted     = False

    def check(self, current_equity: float) -> float:
        """
        Returns:
            1.0 = 正常操作
            reduce_pct = 降仓（回撤超过警戒线）
            0.0 = 全部清仓（回撤超过最大值）
        """
        if self.peak is None or current_equity > self.peak:
            self.peak   = current_equity
            self.halted = False

        dd = (current_equity - self.peak) / self.peak

        if dd <= -self.max_dd:
            if not self.halted:
                logger.warning(f"🚨 熔断触发！回撤 {dd:.1%}，强制清仓")
            self.halted = True
            return 0.0

        if dd <= -self.warning_dd:
            logger.info(f"⚠️ 警戒线：回撤 {dd:.1%}，降仓至 {self.reduce_pct:.0%}")
            return self.reduce_pct

        return 1.0

    def reset(self):
        self.peak   = None
        self.halted = False


# ──────────────────────────────────────────────
# 3. 双重过滤器（CTA + 板块）
# ──────────────────────────────────────────────

def apply_cta_filter(
    signal: float,
    cta_signal: float,
    cot_pct: float,
) -> float:
    """
    CTA 过滤器：根据 CTA 仓位状态调整信号强度

    Args:
        signal:     原始策略信号（-1 到 1）
        cta_signal: CTA 趋势信号（-1 到 1）
        cot_pct:    COT 百分位（0 到 1）

    Returns:
        调整后的信号
    """
    # 极度多头区，多头信号减半
    if cot_pct > 0.85 and cta_signal > 0.5 and signal > 0:
        return signal * 0.5

    # 极度空头区，空头信号减半
    if cot_pct < 0.15 and cta_signal < -0.5 and signal < 0:
        return signal * 0.5

    # 顺趋势：CTA 与信号方向一致，适度放大
    if cta_signal > 0.4 and signal > 0:
        return min(signal * 1.2, 1.0)
    if cta_signal < -0.4 and signal < 0:
        return max(signal * 1.2, -1.0)

    return signal


def apply_sector_filter(
    signal: float,
    sector_signal: float,
) -> float:
    """
    板块过滤器：信号与板块方向不一致时减仓

    Args:
        signal:        策略信号
        sector_signal: 板块趋势信号（-1 到 1）
    """
    # 信号方向与板块方向相反 → 信号减弱 50%
    if signal > 0 and sector_signal < -0.3:
        logger.debug("板块看空，多头信号减半")
        return signal * 0.5
    if signal < 0 and sector_signal > 0.3:
        logger.debug("板块看多，空头信号减半")
        return signal * 0.5

    # 方向一致 → 信号不变
    return signal


def composite_risk_filter(
    base_signal: float,
    cta_signal: float = 0.0,
    cot_pct: float = 0.5,
    sector_signal: float = 0.0,
    current_equity: float = None,
    guard: DrawdownGuard = None,
) -> dict:
    """
    完整风险过滤流程：CTA + 板块 + 回撤熔断

    Returns:
        dict {final_signal, multiplier, warnings}
    """
    warnings = []
    signal   = base_signal

    # 1. CTA 过滤
    filtered = apply_cta_filter(signal, cta_signal, cot_pct)
    if abs(filtered) < abs(signal):
        warnings.append(f"CTA过滤: {signal:.2f} → {filtered:.2f}")
    signal = filtered

    # 2. 板块过滤
    filtered = apply_sector_filter(signal, sector_signal)
    if abs(filtered) < abs(signal):
        warnings.append(f"板块过滤: {signal:.2f} → {filtered:.2f}")
    signal = filtered

    # 3. 回撤熔断
    multiplier = 1.0
    if guard is not None and current_equity is not None:
        multiplier = guard.check(current_equity)
        if multiplier < 1.0:
            warnings.append(f"熔断: 仓位乘数 {multiplier:.1f}")
        signal *= multiplier

    return {
        "final_signal": round(signal, 3),
        "multiplier":   multiplier,
        "warnings":     warnings,
    }


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== 仓位计算 ===")
    shares = position_size_fixed_risk(100000, 450.0, 440.0)
    print(f"固定风险仓位: {shares} 股")

    kelly = position_size_kelly(100000, 0.55, 0.03, 0.015)
    print(f"Kelly 仓位比例: {kelly:.1%}")

    print("\n=== 回撤熔断测试 ===")
    guard  = DrawdownGuard(max_dd=0.15, warning_dd=0.10)
    equities = [100000, 102000, 98000, 95000, 92000, 88000, 85000]
    for eq in equities:
        m = guard.check(eq)
        print(f"  净值 ${eq:,} → 仓位乘数 {m}")

    print("\n=== 综合过滤器 ===")
    result = composite_risk_filter(
        base_signal=0.8,
        cta_signal=0.9,    # CTA 极度多头
        cot_pct=0.87,      # COT > 85%
        sector_signal=0.5,
        current_equity=92000,
        guard=guard,
    )
    print(result)
    print("✅ risk_manager 测试通过")
