"""
signals/breakout.py — 突破系统
包含：
  1. Donchian 通道突破（N日最高/最低价通道 + ATR止损）
  2. VIX 波动率政体过滤（低/高/极端三档）
  3. 完整信号接口，兼容 backtest/engine.py

策略逻辑：
  入场：收盘价突破 N日高点（多头）或跌破 N日低点（空头）
  出场：收盘价跌破 M日低点（通道出场）OR 跌破 止损 = 入场价 - 2×ATR
  VIX过滤：
    VIX < 25  → 正常开仓（仓位×1.0）
    VIX 25-40 → 高波动，仓位减半（×0.5）
    VIX > 40  → 极端，禁止新开仓（×0.0）

适用资产类型：
  ✅ Type A（杠杆ETF）— 趋势突破后弹性最强
  ✅ Type B（成长股）— 上升通道突破
  ⚠️  Type C/D — 可用但需更宽通道（加大period）
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (DONCHIAN_PERIOD, DONCHIAN_EXIT,
                    VIX_HIGH, VIX_EXTREME,
                    BREAKOUT_ATR_STOP, BREAKOUT_ATR_TGT)
from data.downloader import get_prices, get_ohlcv

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. VIX 政体分类
# ──────────────────────────────────────────────

def vix_regime(start: str = "2015-01-01") -> pd.DataFrame:
    """
    VIX 三档政体分类（与价格数据对齐使用）

    Returns:
        DataFrame, columns=[vix, regime, allow_entry, size_mult]
        regime: "LOW" | "HIGH" | "EXTREME"
        size_mult: 1.0 / 0.5 / 0.0
    """
    vix = get_prices("^VIX", start=start)
    if vix.empty:
        logger.warning("[VIX] 无法获取VIX数据，默认政体=LOW")
        return pd.DataFrame()

    regime = pd.Series("LOW", index=vix.index)
    regime[vix >= VIX_HIGH]    = "HIGH"
    regime[vix >= VIX_EXTREME] = "EXTREME"

    size_mult = pd.Series(1.0, index=vix.index)
    size_mult[vix >= VIX_HIGH]    = 0.5
    size_mult[vix >= VIX_EXTREME] = 0.0

    # 使用前一日判断，避免前视偏差
    return pd.DataFrame({
        "vix":         vix,
        "regime":      regime.shift(1).fillna("LOW"),
        "allow_entry": (regime.shift(1).fillna("LOW") != "EXTREME"),
        "size_mult":   size_mult.shift(1).fillna(1.0),
    })


# ──────────────────────────────────────────────
# 2. Donchian 通道计算
# ──────────────────────────────────────────────

def donchian_channels(
    high: pd.Series,
    low: pd.Series,
    period: int = DONCHIAN_PERIOD,
    exit_period: int = DONCHIAN_EXIT,
) -> pd.DataFrame:
    """
    计算 Donchian 入场通道 + 出场通道

    注意：全部 shift(1)，避免当日数据泄露

    Returns:
        DataFrame, columns=[dc_upper, dc_lower, dc_mid, dc_exit_up, dc_exit_lo]
    """
    dc_upper   = high.rolling(period).max().shift(1)
    dc_lower   = low.rolling(period).min().shift(1)
    dc_mid     = (dc_upper + dc_lower) / 2
    dc_exit_up = high.rolling(exit_period).max().shift(1)
    dc_exit_lo = low.rolling(exit_period).min().shift(1)

    return pd.DataFrame({
        "dc_upper":   dc_upper,
        "dc_lower":   dc_lower,
        "dc_mid":     dc_mid,
        "dc_exit_up": dc_exit_up,
        "dc_exit_lo": dc_exit_lo,
    })


# ──────────────────────────────────────────────
# 3. ATR 计算（内部辅助）
# ──────────────────────────────────────────────

def _atr(high: pd.Series, low: pd.Series,
         close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ──────────────────────────────────────────────
# 4. 完整 Donchian 突破信号
# ──────────────────────────────────────────────

def donchian_breakout_signal(
    ticker: str,
    period: int        = DONCHIAN_PERIOD,
    exit_period: int   = DONCHIAN_EXIT,
    atr_stop: float    = BREAKOUT_ATR_STOP,
    vix_filter: bool   = True,
    long_only: bool    = True,
    start: str         = "2015-01-01",
) -> pd.DataFrame:
    """
    完整 Donchian 通道突破策略

    入场：
      多头：Close > dc_upper  AND  VIX允许
      空头：Close < dc_lower  AND  VIX允许  AND  not long_only

    出场：
      多头：Close < dc_exit_lo  OR  Close < entry - atr_stop×ATR（硬止损）
      空头：Close > dc_exit_up  OR  Close > entry + atr_stop×ATR

    Returns:
        DataFrame with [Close, dc_upper, dc_lower, vix, regime,
                        signal, signal_int, size_mult]
        signal_int: +1 / -1 / 0（供回测引擎使用）
    """
    df = get_ohlcv(ticker, start=start)
    if df.empty:
        return pd.DataFrame()

    h, l, c = df["High"], df["Low"], df["Close"]

    # Donchian 通道
    dc = donchian_channels(h, l, period, exit_period)

    # ATR
    atr = _atr(h, l, c, period=14)

    # VIX 政体
    vix_df = vix_regime(start=start)
    if vix_filter and not vix_df.empty:
        vix_aligned = vix_df.reindex(df.index).ffill()
        allow_entry = vix_aligned["allow_entry"].fillna(True)
        size_mult   = vix_aligned["size_mult"].fillna(1.0)
        vix_vals    = vix_aligned["vix"]
        regime_vals = vix_aligned["regime"]
    else:
        allow_entry = pd.Series(True,  index=df.index)
        size_mult   = pd.Series(1.0,   index=df.index)
        vix_vals    = pd.Series(np.nan, index=df.index)
        regime_vals = pd.Series("LOW",  index=df.index)

    # ── 状态机 ──
    n          = len(df)
    signal_arr = np.zeros(n)
    pos        = 0
    entry_px   = np.nan

    c_arr   = c.values
    dc_up   = dc["dc_upper"].values
    dc_lo   = dc["dc_lower"].values
    ex_lo   = dc["dc_exit_lo"].values
    ex_up   = dc["dc_exit_up"].values
    atr_arr = atr.values
    allow   = allow_entry.values
    smult   = size_mult.values

    for i in range(period, n):
        if np.isnan(dc_up[i]) or np.isnan(dc_lo[i]):
            continue

        if pos == 0:
            if allow[i]:
                if c_arr[i] > dc_up[i]:              # 上轨突破 → 多头
                    pos       = 1
                    entry_px  = c_arr[i]
                elif not long_only and c_arr[i] < dc_lo[i]:  # 下轨突破 → 空头
                    pos       = -1
                    entry_px  = c_arr[i]

        elif pos == 1:
            stop_hit  = c_arr[i] < entry_px - atr_stop * atr_arr[i]
            chan_exit  = not np.isnan(ex_lo[i]) and c_arr[i] < ex_lo[i]
            if stop_hit or chan_exit:
                pos = 0
                entry_px = np.nan

        elif pos == -1:
            stop_hit  = c_arr[i] > entry_px + atr_stop * atr_arr[i]
            chan_exit  = not np.isnan(ex_up[i]) and c_arr[i] > ex_up[i]
            if stop_hit or chan_exit:
                pos = 0
                entry_px = np.nan

        signal_arr[i] = pos * smult[i]   # 连续信号（含 VIX 仓位缩放）

    result = df[["Close"]].copy()
    result["dc_upper"]   = dc["dc_upper"]
    result["dc_lower"]   = dc["dc_lower"]
    result["dc_mid"]     = dc["dc_mid"]
    result["vix"]        = vix_vals
    result["regime"]     = regime_vals
    result["size_mult"]  = size_mult
    result["signal"]     = signal_arr
    result["signal_int"] = np.sign(signal_arr).astype(int)

    return result


def breakout_signal_series(
    ticker: str,
    period: int      = DONCHIAN_PERIOD,
    vix_filter: bool = True,
    long_only: bool  = True,
    start: str       = "2015-01-01",
) -> pd.Series:
    """
    精简版：只返回 +1/0/-1 信号 Series（兼容 backtest engine）
    """
    df = donchian_breakout_signal(ticker, period=period,
                                  vix_filter=vix_filter,
                                  long_only=long_only, start=start)
    if df.empty:
        return pd.Series(dtype=int)
    return df["signal_int"].rename(f"donchian_{period}")


# ──────────────────────────────────────────────
# 5. VIX 当日状态（供 router/main 调用）
# ──────────────────────────────────────────────

def current_vix_regime() -> dict:
    """
    返回当前 VIX 政体状态（供每日运行时参考）

    Returns:
        {"vix": float, "regime": str, "allow_entry": bool,
         "size_mult": float, "note": str}
    """
    vix_px = get_prices("^VIX")
    if vix_px.empty:
        return {"vix": None, "regime": "UNKNOWN",
                "allow_entry": True, "size_mult": 1.0, "note": "数据不可用"}

    latest = float(vix_px.iloc[-1])

    if latest >= VIX_EXTREME:
        regime = "EXTREME"
        note   = f"⛔ VIX={latest:.1f} 极端行情，禁止新开仓"
        allow  = False
        mult   = 0.0
    elif latest >= VIX_HIGH:
        regime = "HIGH"
        note   = f"⚠️  VIX={latest:.1f} 高波动，仓位×0.5"
        allow  = True
        mult   = 0.5
    else:
        regime = "LOW"
        note   = f"✅ VIX={latest:.1f} 正常，仓位×1.0"
        allow  = True
        mult   = 1.0

    return {"vix": latest, "regime": regime,
            "allow_entry": allow, "size_mult": mult, "note": note}


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backtest.engine import backtest

    print("=== VIX 当前政体 ===")
    v = current_vix_regime()
    print(v["note"])

    print("\n=== Donchian 突破 回测（SPY 2021起）===")
    for ticker in ["SPY", "QQQ", "NVDA", "SOXL"]:
        sig = breakout_signal_series(ticker, start="2021-01-01")
        if sig.empty:
            print(f"{ticker}: 无信号")
            continue
        from data.downloader import get_prices
        prices = get_prices(ticker, start="2021-01-01")
        res = backtest(prices, sig.clip(lower=0), name=f"Donchian({ticker})")
        m   = res["metrics"]
        in_m = sig.clip(lower=0).mean()
        print(f"{ticker:6s}  CAGR={m.get('年化收益(CAGR)','N/A'):>8s}  "
              f"Sharpe={float(m.get('Sharpe比率',0)):.2f}  "
              f"DD={m.get('最大回撤','N/A'):>8s}  "
              f"Calmar={float(m.get('Calmar比率',0)):.2f}  在市={in_m:.1%}")

    print("\n✅ breakout 测试通过")
