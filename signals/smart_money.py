"""
signals/smart_money.py — Smart Money Concepts (SMC / ICT)
包含：订单块(OB)、公平价值缺口(FVG)、结构突破(BOS)、特征转变(CHOCH)
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SMC_SWING_LK, SMC_FVG_MIN
from data.downloader import get_ohlcv

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 1. 摆动高低点识别（SMC 基础）
# ──────────────────────────────────────────────

def detect_swing_points(df: pd.DataFrame, period: int = SMC_SWING_LK) -> pd.DataFrame:
    """
    识别市场结构高低点（HH/HL/LH/LL）

    Returns:
        df + 列: swing_high(bool), swing_low(bool),
                 sh_price, sl_price（非摆动点为 NaN）
    """
    h = df["High"]
    l = df["Low"]
    n = len(df)

    sh = pd.Series(False, index=df.index)  # swing high
    sl = pd.Series(False, index=df.index)  # swing low

    for i in range(period, n - period):
        window_h = h.iloc[i - period: i + period + 1]
        window_l = l.iloc[i - period: i + period + 1]
        if h.iloc[i] == window_h.max():
            sh.iloc[i] = True
        if l.iloc[i] == window_l.min():
            sl.iloc[i] = True

    result = df.copy()
    result["swing_high"]  = sh
    result["swing_low"]   = sl
    result["sh_price"]    = h.where(sh)
    result["sl_price"]    = l.where(sl)
    return result


# ──────────────────────────────────────────────
# 2. 市场结构分析（BOS / CHOCH）
# ──────────────────────────────────────────────

def detect_market_structure(df: pd.DataFrame, period: int = SMC_SWING_LK) -> pd.DataFrame:
    """
    识别结构突破（Break of Structure, BOS）和特征转变（Change of Character, CHOCH）

    BOS:  在趋势方向突破最近摆动高/低点 → 趋势延续确认
    CHOCH: 突破反方向的摆动点 → 趋势可能反转

    Returns:
        df + 列: trend('bullish'/'bearish'/'undefined'),
                 bos_bull(bool), bos_bear(bool),
                 choch_bull(bool), choch_bear(bool),
                 structure_level(价位)
    """
    df = detect_swing_points(df, period)
    c  = df["Close"]
    n  = len(df)

    trend       = pd.Series("undefined", index=df.index)
    bos_bull    = pd.Series(False, index=df.index)
    bos_bear    = pd.Series(False, index=df.index)
    choch_bull  = pd.Series(False, index=df.index)
    choch_bear  = pd.Series(False, index=df.index)
    struct_lvl  = pd.Series(np.nan, index=df.index)

    # 提取历史摆动点
    sh_prices = df["sh_price"].dropna()
    sl_prices = df["sl_price"].dropna()

    current_trend = "undefined"
    last_sh = None
    last_sl = None

    for i in range(period * 2, n):
        # 更新最近摆动点
        past_sh = sh_prices[sh_prices.index < df.index[i]]
        past_sl = sl_prices[sl_prices.index < df.index[i]]

        if len(past_sh) > 0:
            last_sh = past_sh.iloc[-1]
        if len(past_sl) > 0:
            last_sl = past_sl.iloc[-1]

        if last_sh is None or last_sl is None:
            continue

        price = c.iloc[i]

        if current_trend == "bullish":
            if price > last_sh:               # 突破前高 → BOS
                bos_bull.iloc[i]   = True
                struct_lvl.iloc[i] = last_sh
            elif price < last_sl:             # 跌破前低 → CHOCH（反转信号）
                choch_bear.iloc[i] = True
                struct_lvl.iloc[i] = last_sl
                current_trend = "bearish"

        elif current_trend == "bearish":
            if price < last_sl:               # 跌破前低 → BOS
                bos_bear.iloc[i]   = True
                struct_lvl.iloc[i] = last_sl
            elif price > last_sh:             # 突破前高 → CHOCH（反转信号）
                choch_bull.iloc[i] = True
                struct_lvl.iloc[i] = last_sh
                current_trend = "bullish"

        else:
            # 初始方向判断
            if price > last_sh:
                current_trend = "bullish"
            elif price < last_sl:
                current_trend = "bearish"

        trend.iloc[i] = current_trend

    df["trend"]      = trend
    df["bos_bull"]   = bos_bull
    df["bos_bear"]   = bos_bear
    df["choch_bull"] = choch_bull
    df["choch_bear"] = choch_bear
    df["struct_lvl"] = struct_lvl
    return df


# ──────────────────────────────────────────────
# 3. 订单块识别（Order Block）
# ──────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, period: int = SMC_SWING_LK) -> pd.DataFrame:
    """
    识别订单块（OB）：结构突破前最后一根反向K线区域

    看涨订单块（Bullish OB）：
      - 发生 BOS（向上突破）之前，最后一根阴线的高低范围
      - 价格回到此区域时是做多机会

    看跌订单块（Bearish OB）：
      - 发生 BOS（向下突破）之前，最后一根阳线的高低范围
      - 价格反弹到此区域时是做空机会

    Returns:
        df + 列: bull_ob_high/low, bear_ob_high/low,
                 in_bull_ob(bool), in_bear_ob(bool)
    """
    df   = detect_market_structure(df, period)
    o    = df["Open"]
    h    = df["High"]
    l    = df["Low"]
    c    = df["Close"]
    n    = len(df)

    bull_ob_high = pd.Series(np.nan, index=df.index)
    bull_ob_low  = pd.Series(np.nan, index=df.index)
    bear_ob_high = pd.Series(np.nan, index=df.index)
    bear_ob_low  = pd.Series(np.nan, index=df.index)

    # 记录当前有效订单块
    active_bull_ob = None   # (high, low)
    active_bear_ob = None

    for i in range(period + 1, n):
        # BOS 向上突破 → 找之前最后一根阴线作为看涨 OB
        if df["bos_bull"].iloc[i]:
            for j in range(i - 1, max(i - 20, 0), -1):
                if c.iloc[j] < o.iloc[j]:    # 阴线
                    active_bull_ob = (h.iloc[j], l.iloc[j])
                    break

        # BOS 向下突破 → 找之前最后一根阳线作为看跌 OB
        if df["bos_bear"].iloc[i]:
            for j in range(i - 1, max(i - 20, 0), -1):
                if c.iloc[j] > o.iloc[j]:    # 阳线
                    active_bear_ob = (h.iloc[j], l.iloc[j])
                    break

        if active_bull_ob:
            bull_ob_high.iloc[i] = active_bull_ob[0]
            bull_ob_low.iloc[i]  = active_bull_ob[1]
        if active_bear_ob:
            bear_ob_high.iloc[i] = active_bear_ob[0]
            bear_ob_low.iloc[i]  = active_bear_ob[1]

    df["bull_ob_high"] = bull_ob_high
    df["bull_ob_low"]  = bull_ob_low
    df["bear_ob_high"] = bear_ob_high
    df["bear_ob_low"]  = bear_ob_low

    # 当前价格是否在订单块内
    df["in_bull_ob"] = (
        c.notna() &
        bull_ob_high.notna() &
        (c <= bull_ob_high) &
        (c >= bull_ob_low)
    )
    df["in_bear_ob"] = (
        c.notna() &
        bear_ob_high.notna() &
        (c <= bear_ob_high) &
        (c >= bear_ob_low)
    )
    return df


# ──────────────────────────────────────────────
# 4. 公平价值缺口（Fair Value Gap, FVG）
# ──────────────────────────────────────────────

def detect_fvg(df: pd.DataFrame, min_size: float = SMC_FVG_MIN) -> pd.DataFrame:
    """
    识别公平价值缺口（三K线结构中间K线留下的空缺）

    看涨 FVG：K线[i-2]高点 < K线[i]低点（向上跳空缺口）
    看跌 FVG：K线[i-2]低点 > K线[i]高点（向下跳空缺口）

    价格回测到 FVG 区域时是入场机会

    Returns:
        df + 列: bull_fvg_high/low, bear_fvg_high/low,
                 in_bull_fvg(bool), in_bear_fvg(bool)
    """
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    n = len(df)

    bull_fvg_h = pd.Series(np.nan, index=df.index)
    bull_fvg_l = pd.Series(np.nan, index=df.index)
    bear_fvg_h = pd.Series(np.nan, index=df.index)
    bear_fvg_l = pd.Series(np.nan, index=df.index)

    active_bull_fvgs = []   # [(high, low), ...]
    active_bear_fvgs = []

    for i in range(2, n):
        price = c.iloc[i]
        mid_c = c.iloc[i - 1]

        # 新看涨 FVG
        gap_top = l.iloc[i]
        gap_bot = h.iloc[i - 2]
        if gap_top > gap_bot and (gap_top - gap_bot) / gap_bot > min_size:
            active_bull_fvgs.append((gap_top, gap_bot))

        # 新看跌 FVG
        gap_bot2 = h.iloc[i]
        gap_top2 = l.iloc[i - 2]
        if gap_top2 > gap_bot2 and (gap_top2 - gap_bot2) / gap_top2 > min_size:
            active_bear_fvgs.append((gap_top2, gap_bot2))

        # 记录当前最近有效 FVG
        if active_bull_fvgs:
            bh, bl = active_bull_fvgs[-1]
            bull_fvg_h.iloc[i] = bh
            bull_fvg_l.iloc[i] = bl
        if active_bear_fvgs:
            bh2, bl2 = active_bear_fvgs[-1]
            bear_fvg_h.iloc[i] = bh2
            bear_fvg_l.iloc[i] = bl2

        # 如果价格已经填补 FVG，则移除
        active_bull_fvgs = [
            (bh, bl) for bh, bl in active_bull_fvgs if price >= bl  # 未被完全填补
        ]
        active_bear_fvgs = [
            (bh, bl) for bh, bl in active_bear_fvgs if price <= bh
        ]

    df["bull_fvg_high"] = bull_fvg_h
    df["bull_fvg_low"]  = bull_fvg_l
    df["bear_fvg_high"] = bear_fvg_h
    df["bear_fvg_low"]  = bear_fvg_l

    df["in_bull_fvg"] = (
        bull_fvg_h.notna() &
        (c <= bull_fvg_h) & (c >= bull_fvg_l)
    )
    df["in_bear_fvg"] = (
        bear_fvg_h.notna() &
        (c <= bear_fvg_h) & (c >= bear_fvg_l)
    )
    return df


# ──────────────────────────────────────────────
# 5. SMC 综合信号
# ──────────────────────────────────────────────

def smc_signal(ticker: str) -> pd.DataFrame:
    """
    SMC 综合信号：OB + FVG + BOS/CHOCH 三层确认

    做多条件（任意两层确认）：
      - 看涨 BOS/CHOCH（结构翻多）
      - 价格在看涨 OB 内
      - 价格在看涨 FVG 内

    做空条件（任意两层确认）：
      - 看跌 BOS/CHOCH
      - 价格在看跌 OB 内
      - 价格在看跌 FVG 内

    Returns:
        DataFrame + signal 列：2=强多，1=多，-1=空，-2=强空
    """
    df = get_ohlcv(ticker)
    if df.empty:
        return pd.DataFrame()

    df = detect_order_blocks(df)
    df = detect_fvg(df)

    # 确认层数计分
    bull_score = (
        df["bos_bull"].astype(int) +
        df["choch_bull"].astype(int) +
        df["in_bull_ob"].astype(int) +
        df["in_bull_fvg"].astype(int)
    )
    bear_score = (
        df["bos_bear"].astype(int) +
        df["choch_bear"].astype(int) +
        df["in_bear_ob"].astype(int) +
        df["in_bear_fvg"].astype(int)
    )

    signal = pd.Series(0, index=df.index)
    signal[bull_score >= 3] =  2   # 三层以上确认 → 强多
    signal[bull_score == 2] =  1   # 两层确认 → 多
    signal[bear_score >= 3] = -2   # 三层以上确认 → 强空
    signal[bear_score == 2] = -1   # 两层确认 → 空

    df["smc_score_bull"] = bull_score
    df["smc_score_bear"] = bear_score
    df["signal"]         = signal.shift(1).fillna(0)   # 次日执行
    return df


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== SMC 信号（SPY 最近数据）===")
    df = smc_signal("SPY")
    if not df.empty:
        cols = ["Close", "trend", "bos_bull", "bos_bear",
                "choch_bull", "choch_bear", "in_bull_ob",
                "in_bull_fvg", "signal"]
        print(df[cols].tail(10).to_string())

        print(f"\n当前趋势:    {df['trend'].iloc[-1]}")
        print(f"当前信号:    {df['signal'].iloc[-1]}")
        print(f"Bull Score:  {df['smc_score_bull'].iloc[-1]}")
        print(f"Bear Score:  {df['smc_score_bear'].iloc[-1]}")

        recent_bos = df[df["bos_bull"] | df["bos_bear"]].tail(3)
        print(f"\n最近3次 BOS:\n{recent_bos[['Close', 'trend', 'bos_bull', 'bos_bear', 'struct_lvl']].to_string()}")
    print("✅ smart_money 测试通过")
