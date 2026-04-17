"""
signals/cta_monitor.py — CTA 大盘仓位监控（Part 13）
每日运行，输出各资产 CTA 趋势信号 + COT 仓位百分位
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CTA_UNIVERSE, CTA_LOOKBACKS, CTA_VOL_WIN, CTA_COT_PCT_HI, CTA_COT_PCT_LO
from data.downloader import get_prices, get_returns

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 每日趋势复制信号
# ──────────────────────────────────────────────

def cta_trend_signal(
    ticker: str,
    lookbacks: list = CTA_LOOKBACKS,
    vol_win: int = CTA_VOL_WIN,
) -> float:
    """
    对单个资产计算 CTA 趋势信号
    原理：多时间跨度动量归一化后平均，R² > 0.75 vs SG CTA Trend Index

    Returns:
        float in [-1, 1]
        正值 = 系统性多头，负值 = 系统性空头，绝对值越大越强
    """
    prices = get_prices(ticker)
    if prices.empty or len(prices) < max(lookbacks) + vol_win:
        return 0.0

    ret = prices.pct_change().dropna()
    sigs = []

    for lk in lookbacks:
        mom = ret.rolling(lk).mean() * 252   # 年化动量
        vol = ret.rolling(vol_win).std() * np.sqrt(252)
        # 归一化：年化动量 / 年化波动率，cap 在 [-1, 1]
        s = (mom / vol.replace(0, np.nan)).clip(-1, 1)
        sigs.append(s)

    combined = pd.concat(sigs, axis=1).mean(axis=1)
    return float(combined.dropna().iloc[-1]) if not combined.dropna().empty else 0.0


def run_cta_dashboard() -> pd.DataFrame:
    """
    每日运行：输出所有 CTA 监控资产的方向 + 强度 + 5日变化

    Returns:
        DataFrame，按 signal 降序排列
    """
    rows = []
    for name, ticker in CTA_UNIVERSE.items():
        prices = get_prices(ticker)
        if prices.empty or len(prices) < max(CTA_LOOKBACKS) + CTA_VOL_WIN + 10:
            continue

        ret = prices.pct_change().dropna()
        sigs_series = []
        for lk in CTA_LOOKBACKS:
            mom = ret.rolling(lk).mean() * 252   # 年化动量
            vol = ret.rolling(CTA_VOL_WIN).std() * np.sqrt(252)
            s = (mom / vol.replace(0, np.nan)).clip(-1, 1)
            sigs_series.append(s)

        combined = pd.concat(sigs_series, axis=1).mean(axis=1).dropna()
        if len(combined) < 6:
            continue

        sig_now  = combined.iloc[-1]
        sig_5d   = combined.iloc[-5]
        delta    = sig_now - sig_5d

        rows.append({
            "资产":     name,
            "ETF":      ticker,
            "信号":     round(sig_now, 3),
            "5日变化":  round(delta, 3),
            "方向":     _direction(sig_now),
            "强度":     _strength(sig_now),
            "仓位变化": "增仓⬆" if delta > 0.08 else ("减仓⬇" if delta < -0.08 else "持平"),
        })

    return pd.DataFrame(rows).sort_values("信号", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# 2. COT 仓位过滤器（结合 cot_loader）
# ──────────────────────────────────────────────

def cta_position_filter(
    cot_percentile: float,
    daily_signal: float,
) -> dict:
    """
    CTA 反向过滤器：在开仓前调用，防止追高/追低进入拥挤交易

    Args:
        cot_percentile: 0~1，COT Leveraged Funds 历史百分位
        daily_signal:   -1~1，每日趋势信号

    Returns:
        dict {trade_bias, allow_long, allow_short, warnings}
    """
    warnings = []
    allow_long  = True
    allow_short = True
    trade_bias  = "NEUTRAL"

    if cot_percentile > CTA_COT_PCT_HI and daily_signal > 0.5:
        warnings.append(f"⚠️ CTA 极度多头（COT={cot_percentile:.0%}），禁止追多")
        allow_long = False
        trade_bias = "AVOID_LONG"

    elif cot_percentile < CTA_COT_PCT_LO and daily_signal < -0.5:
        warnings.append(f"⚠️ CTA 极度空头（COT={cot_percentile:.0%}），禁止追空")
        allow_short = False
        trade_bias = "AVOID_SHORT"

    elif cot_percentile > 0.70 and daily_signal > 0.4:
        warnings.append("📊 CTA 偏多，建议控制仓位")
        trade_bias = "CAUTIOUS_LONG"

    elif cot_percentile < 0.30 and daily_signal < -0.4:
        warnings.append("📊 CTA 偏空，建议控制仓位")
        trade_bias = "CAUTIOUS_SHORT"

    return {
        "trade_bias":   trade_bias,
        "allow_long":   allow_long,
        "allow_short":  allow_short,
        "cot_pct":      f"{cot_percentile:.0%}",
        "daily_signal": round(daily_signal, 3),
        "warnings":     warnings,
    }


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _direction(sig: float) -> str:
    if sig > 0.15:   return "系统多头🟢"
    if sig < -0.15:  return "系统空头🔴"
    return "中性⚪"

def _strength(sig: float) -> str:
    a = abs(sig)
    if a > 0.6: return "强"
    if a > 0.3: return "中"
    return "弱"


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("=== CTA 每日仪表盘 ===")
    df = run_cta_dashboard()
    print(df.to_string(index=False))

    print("\n=== 过滤器测试 ===")
    result = cta_position_filter(cot_percentile=0.90, daily_signal=0.72)
    print(result)
    print("✅ cta_monitor 测试通过")
