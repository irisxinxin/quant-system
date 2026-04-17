"""
signals/tv_strategies.py — TradingView 策略移植
包含：
  1. Adaptive EMA 信号（EMA20/60）
  2. Vegas Tunnel（EMA144/169）
  3. MACD 背离抄底/卖出信号（含状态机）
  4. Combined_TV 组合策略
  5. Vegas_Bounce 策略
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 辅助：EMA
# ──────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ──────────────────────────────────────────────
# 1. Adaptive EMA 信号
# ──────────────────────────────────────────────

def adaptive_ema_signal(prices: pd.Series, fast: int = 20, slow: int = 60) -> pd.Series:
    """
    EMA 快慢线趋势跟踪：EMA_fast > EMA_slow → 做多，否则空仓

    Returns:
        pd.Series: 0/1 信号（纯多头）
    """
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)

    signal = pd.Series(0, index=prices.index, dtype=int)
    signal[ema_fast > ema_slow] = 1
    return signal


# ──────────────────────────────────────────────
# 2. MACD 背离信号（状态机，必须用循环）
# ──────────────────────────────────────────────

def macd_divergence_signals(prices: pd.Series) -> pd.DataFrame:
    """
    计算 MACD 背离抄底（DXDX）和顶背离卖出（DBJGXC）信号

    MACD 定义：
        D = EMA(close,12) - EMA(close,26)   # MACD 线
        A = EMA(D, 9)                         # 信号线
        M = (D - A) * 2                       # 柱子×2（放大）

    底背离追踪（负柱区间内）：
        - crossDown（柱子从正转负）时：轮换存档 close 和 D 的历史低点
        - M<0 期间：持续更新当前轮次的最低 close 和最低 D
        - 2 轮背离(AAA)：close1 < close2 且 D1 > D2（价低 MACD 高）
        - 3 轮背离(BBB)：close1 < close3 且 D1 < D2 且 D1 > D3
        - CCC = (AAA or BBB) and D < 0
        - JJJ = CCC[前bar] and abs(D[前bar]) >= abs(D_now) * 1.01
        - DXDX = JJJ 首次出现（not JJJ[前bar] and JJJ）

    顶背离追踪（正柱区间内）：镜像逻辑
        - DBJGXC：顶背离卖出信号

    Returns:
        DataFrame，列：bottom(bool), sell(bool)，index 与 prices 对齐
    """
    n = len(prices)
    close_arr = prices.values.astype(float)
    idx = prices.index

    # MACD 计算
    D_series = _ema(prices, 12) - _ema(prices, 26)
    A_series = _ema(D_series, 9)
    M_series = (D_series - A_series) * 2

    D_arr = D_series.values
    M_arr = M_series.values

    # 输出数组
    bottom_arr = np.zeros(n, dtype=bool)
    sell_arr   = np.zeros(n, dtype=bool)

    # ── 底背离状态 ──────────────────────────────
    # 存储 3 轮负柱区间的最低 close 和最低 D（1=最新, 2=上一轮, 3=再上一轮）
    b_close1 = np.nan; b_close2 = np.nan; b_close3 = np.nan
    b_D1     = np.nan; b_D2     = np.nan; b_D3     = np.nan

    # ── 顶背离状态 ──────────────────────────────
    t_close1 = np.nan; t_close2 = np.nan; t_close3 = np.nan
    t_D1     = np.nan; t_D2     = np.nan; t_D3     = np.nan

    # JJJ / DBJG 前值（用于判断"首次出现"）
    prev_JJJ   = False
    prev_DBJG  = False

    # CCC / DBBL 前值（JJJ 需要用到前一 bar 的 CCC）
    prev_CCC  = False
    prev_DBBL = False

    for i in range(1, n):
        m_prev = M_arr[i - 1]
        m_now  = M_arr[i]
        d_now  = D_arr[i]
        c_now  = close_arr[i]

        if np.isnan(m_prev) or np.isnan(m_now) or np.isnan(d_now):
            prev_JJJ  = False
            prev_DBJG = False
            prev_CCC  = False
            prev_DBBL = False
            continue

        # ── 柱子方向切换检测 ──
        crossDown = (m_prev >= 0) and (m_now < 0)   # 进入负柱区间
        crossUp   = (m_prev <= 0) and (m_now > 0)   # 进入正柱区间

        # ── 底背离：负柱区间状态更新 ──
        if crossDown:
            # 轮换：3←2, 2←1, 1←当前
            b_close3 = b_close2; b_D3 = b_D2
            b_close2 = b_close1; b_D2 = b_D1
            b_close1 = c_now;    b_D1 = d_now

        if m_now < 0:
            # 持续更新当前轮次最低 close 和最低 D
            if np.isnan(b_close1):
                b_close1 = c_now; b_D1 = d_now
            else:
                if c_now < b_close1:
                    b_close1 = c_now
                if d_now < b_D1:
                    b_D1 = d_now

        # ── 顶背离：正柱区间状态更新 ──
        if crossUp:
            t_close3 = t_close2; t_D3 = t_D2
            t_close2 = t_close1; t_D2 = t_D1
            t_close1 = c_now;    t_D1 = d_now

        if m_now > 0:
            if np.isnan(t_close1):
                t_close1 = c_now; t_D1 = d_now
            else:
                if c_now > t_close1:
                    t_close1 = c_now
                if d_now > t_D1:
                    t_D1 = d_now

        # ── 底背离条件计算 ──
        AAA = False; BBB = False
        if not (np.isnan(b_close1) or np.isnan(b_close2)):
            # 2 轮：价格新低，MACD 不创新低（背离）
            AAA = (b_close1 < b_close2) and (b_D1 > b_D2) and (m_prev < 0) and (d_now < 0)
        if not (np.isnan(b_close1) or np.isnan(b_close2) or np.isnan(b_close3)):
            # 3 轮：价格新低，MACD 递减但幅度收窄
            BBB = (b_close1 < b_close3) and (b_D1 < b_D2) and (b_D1 > b_D3) and (m_prev < 0) and (d_now < 0)

        CCC = (AAA or BBB) and (d_now < 0)

        # JJJ：前一 bar 有 CCC，且当前 MACD 绝对值变小（背离在减弱）
        JJJ = prev_CCC and (abs(D_arr[i - 1]) >= abs(d_now) * 1.01)

        # DXDX：JJJ 首次出现
        DXDX = (not prev_JJJ) and JJJ
        if DXDX:
            bottom_arr[i] = True

        # ── 顶背离条件计算 ──
        ZJDBL = False; GXDBL = False
        if not (np.isnan(t_close1) or np.isnan(t_close2)):
            ZJDBL = (t_close1 > t_close2) and (t_D1 < t_D2) and (m_prev > 0) and (d_now > 0)
        if not (np.isnan(t_close1) or np.isnan(t_close2) or np.isnan(t_close3)):
            GXDBL = (t_close1 > t_close3) and (t_D1 > t_D2) and (t_D1 < t_D3) and (m_prev > 0) and (d_now > 0)

        DBBL = (ZJDBL or GXDBL) and (d_now > 0)

        # DBJG：前一 bar 有 DBBL，且当前 MACD 值有所下降
        DBJG = prev_DBBL and (D_arr[i - 1] >= d_now * 1.01)

        # DBJGXC：DBJG 首次出现
        DBJGXC = (not prev_DBJG) and DBJG
        if DBJGXC:
            sell_arr[i] = True

        # 滚动前值
        prev_JJJ  = JJJ
        prev_DBJG = DBJG
        prev_CCC  = CCC
        prev_DBBL = DBBL

    result = pd.DataFrame({
        "bottom": bottom_arr,
        "sell":   sell_arr,
    }, index=idx)
    return result


# ──────────────────────────────────────────────
# 3. Combined_TV 组合策略
# ──────────────────────────────────────────────

def combined_tv_signal(prices: pd.Series) -> pd.Series:
    """
    Combined_TV 策略（纯多头状态机）：
      入场：DXDX（底背离）信号触发即入场（不要求价格在EMA60上方，
            因为DXDX在MACD负值区触发，与EMA60趋势过滤互斥）
      出场：DBJGXC（顶背离卖出）OR 收盘价跌破 EMA60

    Note: 原版入场需 price > EMA60，但 DXDX 在 MACD<0（价格走弱）时触发，
          两者在高波动品种上几乎永远不会同时满足，导致 0 笔交易。
          修正：EMA60 仅用于出场保护，不作为入场过滤。

    Returns:
        pd.Series: 0/1 信号
    """
    ema60  = _ema(prices, 60)
    div_df = macd_divergence_signals(prices)
    bottom = div_df["bottom"]
    sell   = div_df["sell"]

    signal   = pd.Series(0, index=prices.index, dtype=int)
    position = 0

    for i in range(len(prices)):
        c   = prices.iloc[i]
        e60 = ema60.iloc[i]

        if position == 0:
            # 入场条件：底背离信号即可（EMA60不做入场过滤）
            if bottom.iloc[i]:
                position = 1
        else:
            # 出场条件：顶背离卖出 or 跌破 EMA60
            if sell.iloc[i] or c < e60:
                position = 0

        signal.iloc[i] = position

    return signal


# ──────────────────────────────────────────────
# 4. Vegas_Bounce 策略
# ──────────────────────────────────────────────

def vegas_bounce_signal(prices: pd.Series) -> pd.Series:
    """
    Vegas_Bounce 策略（纯多头状态机）：
      Vegas Tunnel：EMA144（内隧道下轨）, EMA169（内隧道上轨）
      入场：DXDX（抄底）信号 AND 收盘价 > EMA144（在隧道上方）
      出场：DBJGXC（顶背离卖出）OR 收盘价跌破 EMA169

    Returns:
        pd.Series: 0/1 信号
    """
    ema144 = _ema(prices, 144)
    ema169 = _ema(prices, 169)
    div_df = macd_divergence_signals(prices)
    bottom = div_df["bottom"]
    sell   = div_df["sell"]

    signal   = pd.Series(0, index=prices.index, dtype=int)
    position = 0

    for i in range(len(prices)):
        c    = prices.iloc[i]
        e144 = ema144.iloc[i]
        e169 = ema169.iloc[i]

        if position == 0:
            # 入场：抄底信号 + 价格在 Vegas 隧道上方
            if bottom.iloc[i] and c > e144:
                position = 1
        else:
            # 出场：顶背离卖出 or 跌破 EMA169
            if sell.iloc[i] or c < e169:
                position = 0

        signal.iloc[i] = position

    return signal


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.downloader import get_prices

    ticker = "NVDL"
    prices = get_prices(ticker, start="2023-01-01")
    print(f"=== {ticker} 价格数据：{len(prices)} 条 ===\n")

    # EMA 信号
    ema_sig = adaptive_ema_signal(prices)
    print(f"EMA20/60 信号：多头占比 {ema_sig.mean():.1%}")

    # MACD 背离
    div_df = macd_divergence_signals(prices)
    print(f"抄底信号次数: {div_df['bottom'].sum()}")
    print(f"卖出信号次数: {div_df['sell'].sum()}")

    # Combined_TV
    sig_ctv = combined_tv_signal(prices)
    print(f"Combined_TV 多头占比: {sig_ctv.mean():.1%}")

    # Vegas Bounce
    sig_vb = vegas_bounce_signal(prices)
    print(f"Vegas_Bounce 多头占比: {sig_vb.mean():.1%}")

    print("\n✅ tv_strategies 测试通过")
