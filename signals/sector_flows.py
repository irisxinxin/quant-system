"""
signals/sector_flows.py — 板块资金流 + 趋势信号（Part 14）
每日运行，输出板块热力图
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SECTOR_ETFS, BENCHMARK, CTA_LOOKBACKS, CTA_VOL_WIN
from data.downloader import get_prices, get_ohlcv

logger = logging.getLogger(__name__)

SECTOR_CORE = {k: v for k, v in SECTOR_ETFS.items()}


# ──────────────────────────────────────────────
# 1. 资金流代理
# ──────────────────────────────────────────────

def sector_flow_proxy(lookback: int = 20) -> pd.DataFrame:
    """
    价格 × 相对成交量 估算板块 ETF 资金流方向
    正 = 净流入，负 = 净流出
    """
    rows = []
    for name, ticker in SECTOR_CORE.items():
        df = get_ohlcv(ticker)
        if df.empty or len(df) < lookback + 5:
            continue

        df["pct_chg"]   = df["Close"].pct_change()
        df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(lookback).mean()
        df["flow"]      = df["pct_chg"] * df["vol_ratio"]

        f5   = df["flow"].tail(5).sum()
        f20  = df["flow"].tail(lookback).sum()
        accel = df["flow"].tail(3).mean() > df["flow"].tail(lookback).mean() * 1.5

        rows.append({
            "板块":     name,
            "ETF":      ticker,
            "5日流向":  round(f5, 4),
            "20日流向": round(f20, 4),
            "方向":     "流入🟢" if f20 > 0 else "流出🔴",
            "加速":     "⚡是" if accel else "否",
        })

    return pd.DataFrame(rows).sort_values("20日流向", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# 2. 板块趋势信号（CTA 复制）
# ──────────────────────────────────────────────

def sector_cta_signals(
    short_lk: int = 20,
    mid_lk:   int = 60,
    long_lk:  int = 120,
    vol_win:  int = 60,
) -> pd.DataFrame:
    """
    对各板块 ETF 跑多周期动量趋势信号
    估算系统性资金对各板块的多空倾向
    """
    rows = []
    for name, ticker in SECTOR_CORE.items():
        prices = get_prices(ticker)
        if prices.empty or len(prices) < long_lk + vol_win:
            continue

        ret  = prices.pct_change().dropna()
        sigs = []
        for lk in [short_lk, mid_lk, long_lk]:
            mom = ret.rolling(lk).mean()
            vol = ret.rolling(vol_win).std() * np.sqrt(252)
            s   = (mom / vol.replace(0, np.nan)).clip(-1, 1)
            sigs.append(s)

        combined = pd.concat(sigs, axis=1).mean(axis=1).dropna()
        if len(combined) < 6:
            continue

        sig   = combined.iloc[-1]
        delta = sig - combined.iloc[-5]

        rows.append({
            "板块":     name,
            "ETF":      ticker,
            "趋势信号": round(float(sig), 3),
            "5日变化":  round(float(delta), 3),
            "方向":     "系统多头🟢" if sig > 0.15 else ("系统空头🔴" if sig < -0.15 else "中性⚪"),
            "强度":     "强" if abs(sig) > 0.6 else ("中" if abs(sig) > 0.3 else "弱"),
            "仓位变化": "增仓⬆" if delta > 0.08 else ("减仓⬇" if delta < -0.08 else "持平"),
        })

    return pd.DataFrame(rows).sort_values("趋势信号", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# 3. 综合热力图
# ──────────────────────────────────────────────

def sector_heatmap() -> pd.DataFrame:
    """
    融合趋势信号 + RSI + 相对强弱 → 每日板块全景热力图
    综合评分 = 趋势信号×60% + 相对强弱×40%
    """
    spy_px = get_prices(BENCHMARK)
    spy_3m = spy_px.pct_change(63).iloc[-1] if not spy_px.empty else 0.0

    rows = []
    for name, ticker in SECTOR_CORE.items():
        prices = get_prices(ticker)
        ohlcv  = get_ohlcv(ticker)
        if prices.empty or len(prices) < 130:
            continue

        # RSI (14日)
        rsi_val = _rsi(prices, 14)

        # 相对强弱 vs SPY（3个月）
        ret_3m  = prices.pct_change(63).iloc[-1]
        rel_str = (ret_3m - spy_3m) * 100  # 超额收益（百分点）

        # 趋势信号（60日）
        ret  = prices.pct_change().dropna()
        mom  = ret.rolling(60).mean().iloc[-1]
        vol  = ret.rolling(60).std().iloc[-1] * np.sqrt(252)
        sig  = float(np.clip(mom / vol if vol > 0 else 0, -1, 1))

        # 综合评分
        rs_norm   = float(np.clip(rel_str / 20, -1, 1))
        composite = sig * 0.6 + rs_norm * 0.4

        rows.append({
            "板块":       name,
            "ETF":        ticker,
            "综合评分":   round(composite, 2),
            "趋势信号":   round(sig, 2),
            "RSI":        round(rsi_val, 1),
            "3月超额":    f"{rel_str:+.1f}%",
            "RSI状态":    "🔴超买" if rsi_val > 70 else ("🟢超卖" if rsi_val < 30 else "⚪正常"),
            "整体判断":   _verdict(composite),
        })

    return pd.DataFrame(rows).sort_values("综合评分", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# 4. 板块开仓过滤器
# ──────────────────────────────────────────────

def sector_filter(sector_name: str) -> dict:
    """
    开仓前检查：该板块是否处于极端拥挤区
    Returns dict with allow_long, allow_short, warnings
    """
    signals = sector_cta_signals()
    flows   = sector_flow_proxy()

    row_s = signals[signals["板块"] == sector_name]
    row_f = flows[flows["板块"] == sector_name]

    if row_s.empty:
        return {"allow_long": True, "allow_short": True, "warnings": ["无板块数据"]}

    sig     = row_s["趋势信号"].iloc[0]
    delta   = row_s["5日变化"].iloc[0]
    flow_20 = row_f["20日流向"].iloc[0] if not row_f.empty else 0

    warnings    = []
    allow_long  = True
    allow_short = True

    if sig > 0.75 and flow_20 > 0:
        warnings.append(f"⚠️ {sector_name} 系统性多头极度拥挤（信号{sig:.2f}），追多减半仓")
        allow_long = False
    elif sig < -0.75 and flow_20 < 0:
        warnings.append(f"⚠️ {sector_name} 系统性空头极度拥挤（信号{sig:.2f}），追空减半仓")
        allow_short = False

    if abs(delta) > 0.15:
        dir_str = "快速增仓" if delta > 0 else "快速减仓"
        warnings.append(f"📉 {sector_name} 仓位{dir_str}（5日变化{delta:+.2f}），注意流动性")

    return {
        "sector":       sector_name,
        "allow_long":   allow_long,
        "allow_short":  allow_short,
        "signal":       sig,
        "flow_20d":     flow_20,
        "warnings":     warnings,
    }


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> float:
    delta  = prices.diff()
    gain   = delta.clip(lower=0).rolling(period).mean()
    loss   = (-delta.clip(upper=0)).rolling(period).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = 100 - (100 / (1 + rs))
    return float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50.0


def _verdict(score: float) -> str:
    if score > 0.5:  return "强烈看多🔥"
    if score > 0.2:  return "看多📈"
    if score > -0.2: return "中性➡"
    if score > -0.5: return "看空📉"
    return "强烈看空❄️"


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== 板块趋势信号 ===")
    df_sig = sector_cta_signals()
    print(df_sig.to_string(index=False))

    print("\n=== 板块资金流 ===")
    df_flow = sector_flow_proxy()
    print(df_flow.to_string(index=False))

    print("\n=== 板块综合热力图 ===")
    df_heat = sector_heatmap()
    print(df_heat.to_string(index=False))

    print("\n=== 板块过滤器测试（半导体）===")
    print(sector_filter("半导体"))
    print("✅ sector_flows 测试通过")
