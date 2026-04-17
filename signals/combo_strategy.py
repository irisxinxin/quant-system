"""
signals/combo_strategy.py — 多策略融合 + CTA资金流向过滤

架构：2层 × 3信号投票
  ┌─ 技术层（个股） ──────────────────────────────────────────────┐
  │  Vote A: EMA20 > EMA60          短期趋势方向                   │
  │  Vote B: EMA60 > EMA200         中期结构多头                   │
  │  Vote C: 价格 > EMA200          长期牛市确认                   │
  │  入场门槛 tech_score ≥ 2（2/3 票通过）                         │
  └────────────────────────────────────────────────────────────────┘
  ┌─ CTA宏观层（SPY+QQQ） ─────────────────────────────────────────┐
  │  多周期动量/波动率 → CTA仓位代理（与SG CTA指数相关性>0.75）       │
  │  > +阈值 → 系统性多头，允许开仓                                 │
  │  < -阈值 → 系统性空头，强制平仓                                 │
  └────────────────────────────────────────────────────────────────┘

入场：tech_score ≥ 2 AND cta_signal > 阈值
出场：tech_score < 2 OR cta_signal < -阈值（宏观逆转强平）

板块资金流（sector_flow）：作为参考展示，不强制过滤（噪声太高）
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.downloader import get_prices, get_ohlcv

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Layer 2: CTA 宏观趋势序列（时间序列化，用于回测）
# ──────────────────────────────────────────────

# 各标的对应的宏观参考 ETF（用于计算 CTA 代理信号）
_SECTOR_MAP = {
    "NVDL":  ["SMH", "QQQ"],    # 2x NVIDIA → 半导体 + 纳指
    "TSLL":  ["QQQ", "SPY"],    # 2x TSLA   → 纳指 + 大盘
    "SOXL":  ["SMH", "QQQ"],    # 3x SOX    → 半导体
    "EOSE":  ["ICLN", "SPY"],   # 储能       → 清洁能源 + 大盘
    "IREN":  ["ICLN", "QQQ"],   # 加密矿     → 清洁能源 + 纳指
    "INTC":  ["SMH", "SPY"],    # Intel     → 半导体 + 大盘
    "DEFAULT": ["SPY", "QQQ"],
}


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def cta_macro_series(
    ref_tickers: list,
    lookbacks: list = (80, 160, 260),
    vol_win: int = 90,
    start: str = "2020-01-01",
) -> pd.Series:
    """
    将 CTA 趋势信号转为每日时间序列（用于回测过滤器）

    原理：多周期风险调整后动量均值，与 SG CTA Trend Index 相关性 > 0.75

    > 0  → 宏观顺风（CTA 系统性多头），允许开多仓
    < 0  → 宏观逆风（CTA 系统性空头），禁止新开多仓

    Returns:
        pd.Series, 值域约 [-1, 1]
    """
    all_sigs = []
    for ticker in ref_tickers:
        prices = get_prices(ticker, start=start)
        if prices.empty or len(prices) < max(lookbacks) + vol_win:
            continue
        ret = prices.pct_change().dropna()
        for lk in lookbacks:
            mom = ret.rolling(lk).mean() * 252        # 年化动量
            vol = ret.rolling(vol_win).std() * np.sqrt(252)  # 年化波动
            s   = (mom / vol.replace(0, np.nan)).clip(-1, 1)
            all_sigs.append(s)

    if not all_sigs:
        return pd.Series(dtype=float)

    combined = pd.concat(all_sigs, axis=1).mean(axis=1)
    return combined.dropna()


def sector_flow_series(
    ref_tickers: list,
    lookback: int = 20,
    start: str = "2020-01-01",
) -> pd.Series:
    """
    板块资金流代理：价格涨跌 × 相对成交量，取参考ETF均值

    > 0  → 板块净流入，支持多头
    < 0  → 板块净流出，谨慎做多

    Returns:
        pd.Series（20日滚动均值平滑后）
    """
    flow_list = []
    for ticker in ref_tickers:
        df = get_ohlcv(ticker, start=start)
        if df.empty or len(df) < lookback + 5:
            continue
        pct_chg   = df["Close"].pct_change()
        vol_ratio = df["Volume"] / df["Volume"].rolling(lookback).mean()
        flow      = (pct_chg * vol_ratio).rolling(5).mean()  # 5日平滑
        flow_list.append(flow)

    if not flow_list:
        return pd.Series(dtype=float)

    return pd.concat(flow_list, axis=1).mean(axis=1).dropna()


# ──────────────────────────────────────────────
# Layer 1: 技术信号投票
# ──────────────────────────────────────────────

def _tech_votes(prices: pd.Series) -> pd.DataFrame:
    """
    生成 3 路技术投票

    Vote A: EMA20 > EMA60  （短期趋势）
    Vote B: EMA60 > EMA200 （中期趋势结构）
    Vote C: 价格 > EMA200  （长期牛熊分水岭）

    Returns:
        DataFrame, columns=[vote_a, vote_b, vote_c, score(0~3)]
    """
    e20  = _ema(prices, 20)
    e60  = _ema(prices, 60)
    e200 = _ema(prices, 200)

    va = (e20  > e60).astype(int)
    vb = (e60  > e200).astype(int)
    vc = (prices > e200).astype(int)

    score = va + vb + vc
    return pd.DataFrame({
        "vote_ema_short":  va,
        "vote_ema_mid":    vb,
        "vote_above_200":  vc,
        "tech_score":      score,
    })


# ──────────────────────────────────────────────
# 主函数：组合信号
# ──────────────────────────────────────────────

def combo_signal(
    prices: pd.Series,
    ticker: str = "DEFAULT",
    tech_threshold:  int   = 2,      # 技术投票阈值：≥N票才允许多头
    cta_threshold:   float = 0.10,   # CTA顺风阈值（年化动量/波动率比）
    cta_exit_thresh: float = -0.05,  # CTA空头出场阈值（比入场宽松）
    use_cta:  bool  = True,          # 是否启用CTA宏观过滤
    start:    str   = "2020-01-01",
    lookbacks: tuple = (80, 160, 260),
    vol_win:   int   = 90,
) -> pd.Series:
    """
    两层过滤组合策略（技术投票 + CTA宏观过滤）

    入场条件（两层均满足）：
      L1: tech_score >= tech_threshold（默认2/3票通过）
      L2: cta_signal > cta_threshold  （CTA系统性多头）

    出场条件（任一触发）：
      - tech_score < tech_threshold（技术趋势转弱）
      - cta_signal < cta_exit_thresh（宏观转空头，强制平仓）

    CTA 基准：始终用 SPY+QQQ 作为宏观参考，不跟着板块走
    （板块ETF相关性在高波动品种下噪声太大）

    Returns:
        pd.Series: 0/1 仓位信号
    """
    # ── CTA 宏观始终用大盘参考 ──
    macro_ref = ["SPY", "QQQ"]

    # ── Layer 1: 技术投票 ──
    tech_df = _tech_votes(prices)

    # ── Layer 2: CTA 宏观序列（使用比技术起点更早的数据保证预热）──
    cta_ser = pd.Series(1.0, index=prices.index)  # 默认顺风
    if use_cta:
        early_start = str(int(start[:4]) - 2) + start[4:]   # 多取2年预热
        raw_cta = cta_macro_series(macro_ref, lookbacks=lookbacks,
                                    vol_win=vol_win, start=early_start)
        if not raw_cta.empty:
            cta_ser = raw_cta.reindex(prices.index).ffill().fillna(0.0)

    # ── 状态机：生成 0/1 信号 ──
    n      = len(prices)
    signal = np.zeros(n, dtype=int)
    pos    = 0

    for i in range(1, n):
        ts  = int(tech_df["tech_score"].iloc[i])
        cta = float(cta_ser.iloc[i])

        if pos == 0:
            tech_ok = ts >= tech_threshold
            cta_ok  = (not use_cta) or (cta > cta_threshold)
            if tech_ok and cta_ok:
                pos = 1
        else:
            tech_exit = ts < tech_threshold
            cta_exit  = use_cta and (cta < cta_exit_thresh)
            if tech_exit or cta_exit:
                pos = 0

        signal[i] = pos

    return pd.Series(signal, index=prices.index, name="combo")


# ──────────────────────────────────────────────
# 核心组合策略：SMC + EMA + CTA（双逻辑叠加）
# ──────────────────────────────────────────────

def duo_cta_signal(
    prices: pd.Series,
    ticker: str = "DEFAULT",
    mode: str = "OR",            # "AND": 两信号同时满足; "OR": 任一满足
    cta_threshold:   float = 0.10,
    cta_exit_thresh: float = -0.05,
    start:   str = "2020-01-01",
    lookbacks: tuple = (80, 160, 260),
    vol_win:   int   = 90,
) -> pd.Series:
    """
    双信号融合 + CTA宏观硬过滤

    信号 A：EMA趋势（EMA20 > EMA60 → 多头）
            逻辑：捕捉中期趋势，频率高，持仓时间长
    信号 B：SMC智能资金（机构建仓 FVG/摆动突破 → 多头）
            逻辑：精准捕捉机构动作，交易次数少但质量高

    融合模式：
      OR  模式：A 或 B 任一发出多头信号即入场（高覆盖率）
      AND 模式：A 和 B 同时确认才入场（高精度，低频）

    CTA宏观过滤（硬过滤器，凌驾于技术信号之上）：
      入场：cta_signal > cta_threshold
      强平：cta_signal < cta_exit_thresh（宏观转熊，强制清仓）

    出场逻辑：
      OR  模式：A 和 B 同时转空才出场（宽松出场，减少误出）
      AND 模式：A 或 B 任一转空即出场（严格出场）

    Returns:
        pd.Series: 0/1 仓位信号
    """
    from signals.smart_money import smc_signal
    from signals.tv_strategies import adaptive_ema_signal

    # ── 信号 A：EMA 趋势 ──
    sig_a = adaptive_ema_signal(prices, fast=20, slow=60)

    # ── 信号 B：SMC ──
    try:
        smc_df = smc_signal(ticker)
        if smc_df.empty:
            sig_b = pd.Series(0, index=prices.index)
        else:
            sig_b = smc_df["signal"].reindex(prices.index).fillna(0).clip(lower=0).astype(int)
    except Exception as e:
        logger.warning(f"SMC信号失败: {e}")
        sig_b = pd.Series(0, index=prices.index)

    # ── CTA 宏观序列 ──
    early_start = str(int(start[:4]) - 2) + start[4:]
    raw_cta = cta_macro_series(["SPY", "QQQ"], lookbacks=lookbacks,
                                vol_win=vol_win, start=early_start)
    cta_ser = raw_cta.reindex(prices.index).ffill().fillna(0.0) if not raw_cta.empty \
              else pd.Series(1.0, index=prices.index)

    # ── 状态机 ──
    n      = len(prices)
    signal = np.zeros(n, dtype=int)
    pos    = 0

    for i in range(1, n):
        a   = int(sig_a.iloc[i])
        b   = int(sig_b.iloc[i])
        cta = float(cta_ser.iloc[i])

        cta_ok   = cta > cta_threshold
        cta_kill = cta < cta_exit_thresh   # 宏观强平

        if pos == 0:
            if cta_ok:
                if   mode == "OR"  and (a == 1 or  b == 1): pos = 1
                elif mode == "AND" and (a == 1 and b == 1): pos = 1
        else:
            if cta_kill:
                pos = 0   # 宏观强平，优先级最高
            elif mode == "OR"  and (a == 0 and b == 0): pos = 0
            elif mode == "AND" and (a == 0 or  b == 0): pos = 0

        signal[i] = pos

    return pd.Series(signal, index=prices.index, name=f"duo_{mode}_cta")


# ──────────────────────────────────────────────
# 诊断工具
# ──────────────────────────────────────────────

def combo_diagnosis(
    prices: pd.Series,
    ticker: str = "DEFAULT",
    start: str = "2020-01-01",
) -> pd.DataFrame:
    """
    输出每日三层信号明细，用于调试和理解策略行为

    Returns:
        DataFrame with [price, tech_score, cta_signal, sector_flow, signal]
    """
    ref_tickers = _SECTOR_MAP.get(ticker.upper(), _SECTOR_MAP["DEFAULT"])

    tech_df = _tech_votes(prices)

    cta_ser = pd.Series(np.nan, index=prices.index)
    raw_cta = cta_macro_series(ref_tickers, start=start)
    if not raw_cta.empty:
        cta_ser = raw_cta.reindex(prices.index).ffill()

    flow_ser = pd.Series(np.nan, index=prices.index)
    raw_flow = sector_flow_series(ref_tickers, start=start)
    if not raw_flow.empty:
        flow_ser = raw_flow.reindex(prices.index).ffill()

    sig = combo_signal(prices, ticker=ticker, start=start)

    return pd.DataFrame({
        "price":        prices,
        "tech_score":   tech_df["tech_score"],
        "vote_ema_s":   tech_df["vote_ema_short"],
        "vote_ema_m":   tech_df["vote_ema_mid"],
        "vote_200":     tech_df["vote_above_200"],
        "cta_signal":   cta_ser,
        "sector_flow":  flow_ser,
        "combo_signal": sig,
    })


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backtest.engine import backtest
    from backtest.metrics import calc_metrics

    tickers = ["NVDL", "TSLL", "SOXL", "EOSE", "IREN", "INTC"]
    start   = "2023-01-01"

    for t in tickers:
        prices = get_prices(t, start=start)
        if prices.empty:
            print(f"{t}: 无数据")
            continue

        sig = combo_signal(prices, ticker=t, start=start)
        res = backtest(prices, sig, name=f"Combo({t})")
        m   = res["metrics"]

        print(f"{t:6s}  CAGR={m.get('年化收益(CAGR)','N/A'):>8s}  "
              f"Sharpe={m.get('Sharpe比率',0):.2f}  "
              f"DD={m.get('最大回撤','N/A'):>8s}  "
              f"Calmar={m.get('Calmar比率',0):.2f}  "
              f"多头占比={sig.mean():.1%}")

    print("\n✅ combo_strategy 测试通过")
