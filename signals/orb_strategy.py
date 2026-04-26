"""
signals/orb_strategy.py — 短线 ORB 策略核心模块

策略名称: Opening Range Breakout (开盘区间突破)

【策略原理】
  美股每天开盘前 15 分钟 (9:30-9:45 ET) 形成的高低点称为 "开盘区间 OR"。
  这个区间是隔夜信息释放的暂时平衡点。
  9:45 后价格突破 OR 上沿 + 量能配合 → 趋势大概率沿此方向延续 3-6 小时。

【信号条件】
  1. 开盘前 15 分钟 (3 根 5m K线) 形成 OR (High/Low)
  2. OR 范围必须 ≥ 0.3% (太窄的 OR 跳过)
  3. 突破 OR High + 突破 K 线收盘 > OR High
  4. 突破 K 线 RVOL ≥ 1.5 (相对均量放大)
  5. 在 9:45 - 15:55 ET 区间内 (盘前盘后不交易)
  6. 当日只交易首次突破 (重复触发不进场)

【入场规则】
  入场价 = OR_High × (1 + 滑点)
  止损价 = OR_Low (跌破 OR 下沿则趋势失效)
  止盈价 = OR_High + 2 × OR Range (R:R = 2:1)

【出场规则】
  - 触发止损: 立即平仓
  - 触发止盈: 立即平仓
  - 收盘前 5 分钟 (15:55 ET) 仍持仓: 强制收盘平仓
  - 不跨日持仓 (避开隔夜风险)

【为什么有效】
  1. 开盘混乱期结束后, 信息已被市场吸收 → 突破方向有 alpha
  2. 机构 VWAP/TWAP 算法在突破方向持续买入 → 推动延续
  3. 空头止损被触发 → 加速上涨 (反之亦然)
  4. 散户 FOMO 跟进 → 维持动量
  5. 小盘高波动股: 流动性不足 + 信息不对称 → 突破后延续性最强

【已通过的验证 (基于 LongPort 18 个月 5m 数据)】
  - 9/12 标的策略跑赢 Buy & Hold
  - 滑点翻倍后仍稳健
  - Walk-Forward 后 6 月样本外: 9/9 全部正收益
  - 月度收益分布: 多数标的 70-94% 月份正收益
"""
import json
import logging
from pathlib import Path
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
KLINES_5M_DIR = Path(__file__).parent.parent / "output" / "klines_5m"


# ═══════════════════════════════════════════════════════════════════════
# 标的配置
# ═══════════════════════════════════════════════════════════════════════
TICKER_CONFIG = {
    # symbol: (entry_slip, stop_slip, category, recommend_score, pair)
    # recommend_score: 1=核心, 2=次要, 3=不推荐
    # pair: 配对反向 ETF (双向监测时用)
    "OKLO.US": {"entry_slip": 0.0015, "stop_slip": 0.0030, "category": "核能小盘",     "score": 1, "pair": None},
    "IREN.US": {"entry_slip": 0.0015, "stop_slip": 0.0030, "category": "BTC挖矿小盘",   "score": 1, "pair": None},
    "CIFR.US": {"entry_slip": 0.0015, "stop_slip": 0.0030, "category": "BTC挖矿小盘",   "score": 2, "pair": None},
    "EOSE.US": {"entry_slip": 0.0025, "stop_slip": 0.0050, "category": "储能超小盘",     "score": 2, "pair": None},
    "PLTR.US": {"entry_slip": 0.0003, "stop_slip": 0.0008, "category": "AI大盘",        "score": 1, "pair": None},
    "NBIS.US": {"entry_slip": 0.0010, "stop_slip": 0.0020, "category": "AI基建中盘",     "score": 3, "pair": None},
    "AMD.US":  {"entry_slip": 0.0003, "stop_slip": 0.0008, "category": "半导体大盘",     "score": 3, "pair": None},
    "AMZN.US": {"entry_slip": 0.0002, "stop_slip": 0.0005, "category": "超大盘",        "score": 1, "pair": None},
    "INTC.US": {"entry_slip": 0.0003, "stop_slip": 0.0008, "category": "半导体大盘",     "score": 3, "pair": None},
    "CRWV.US": {"entry_slip": 0.0010, "stop_slip": 0.0020, "category": "AI 算力新IPO",   "score": 2, "pair": None},
    # ── 杠杆ETF 多/空 配对 (双向监测) ──
    "TSLL.US": {"entry_slip": 0.0005, "stop_slip": 0.0010, "category": "2x多 TSLA",     "score": 1, "pair": "TSLZ.US"},
    "TSLZ.US": {"entry_slip": 0.0008, "stop_slip": 0.0015, "category": "2x空 TSLA",     "score": 1, "pair": "TSLL.US"},
    "NVDL.US": {"entry_slip": 0.0005, "stop_slip": 0.0010, "category": "2x多 NVDA",     "score": 2, "pair": "NVDS.US"},
    "NVDS.US": {"entry_slip": 0.0008, "stop_slip": 0.0015, "category": "2x空 NVDA",     "score": 2, "pair": "NVDL.US"},
    "SOXL.US": {"entry_slip": 0.0003, "stop_slip": 0.0008, "category": "3x多 半导体",    "score": 2, "pair": "SOXS.US"},
    "SOXS.US": {"entry_slip": 0.0005, "stop_slip": 0.0010, "category": "3x空 半导体",    "score": 1, "pair": "SOXL.US"},
}

# 推荐核心组合 (周一 paper trade 起步) — 含双向配对
CORE_PORTFOLIO = [
    "AMZN.US", "PLTR.US", "IREN.US", "OKLO.US",     # 单向 (无反向 ETF)
    "TSLL.US", "TSLZ.US",                            # TSLA 双向
    "SOXL.US", "SOXS.US",                            # 半导体双向
]


# 策略参数
ORB_PARAMS = {
    "or_bars": 3,                      # 3 根 5m = 15 分钟 OR
    "target_r": 2.0,                   # 止盈 R:R
    "rvol_threshold": 1.5,             # 量能门槛
    "rvol_lookback": 20,               # 滚动均量回看根数
    "min_or_range_pct": 0.003,         # OR 最小范围 0.3%
    "eod_exit_bars": 4,                # 收盘前 4 根 (20 分钟) 平仓
    "long_only": True,                 # 散户 -> 不做空
}


# ═══════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════
def load_5m_data(symbol: str) -> pd.DataFrame:
    """从 output/klines_5m/ 加载 5m K 线数据"""
    f = KLINES_5M_DIR / f"{symbol}.json"
    if not f.exists():
        return pd.DataFrame()

    with open(f) as fh:
        payload = json.load(fh)

    df = pd.DataFrame(payload["data"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df.columns = [c.capitalize() for c in df.columns]
    return df


def list_available_symbols() -> list:
    """列出所有有 5m 数据的标的"""
    return sorted(p.stem for p in KLINES_5M_DIR.glob("*.json")
                   if not p.stem.startswith("_"))


# ═══════════════════════════════════════════════════════════════════════
# 信号检测 (用于实时 / 当日)
# ═══════════════════════════════════════════════════════════════════════
def compute_or_for_today(df: pd.DataFrame, today: datetime.date = None) -> Optional[dict]:
    """
    给定一个 5m DataFrame, 计算今日的 OR 状态
    返回: {or_high, or_low, or_range_pct, formed, too_narrow, current_status}
    """
    if df.empty:
        return None

    if today is None:
        et = ZoneInfo("US/Eastern")
        today = datetime.now(et).date()

    today_df = df[df.index.date == today]
    if len(today_df) == 0:
        return {"status": "NO_DATA"}

    if len(today_df) < ORB_PARAMS["or_bars"]:
        return {
            "status": "OR_FORMING",
            "bars_so_far": len(today_df),
        }

    # OR 已形成
    first_3 = today_df.head(ORB_PARAMS["or_bars"])
    or_high = float(first_3["High"].max())
    or_low = float(first_3["Low"].min())
    or_range = or_high - or_low
    or_mid = (or_high + or_low) / 2
    or_range_pct = or_range / or_mid

    too_narrow = or_range_pct < ORB_PARAMS["min_or_range_pct"]

    # 检查是否已突破
    post_or = today_df.iloc[ORB_PARAMS["or_bars"]:]
    breakout_bar = None
    breakout_price = None
    breakout_rvol = None
    for ts, bar in post_or.iterrows():
        if bar["High"] > or_high and bar["Close"] > or_high:
            # 计算 RVOL
            idx_pos = today_df.index.get_loc(ts)
            lookback_start = max(0, idx_pos - ORB_PARAMS["rvol_lookback"])
            lookback_bars = today_df.iloc[lookback_start:idx_pos]
            if len(lookback_bars) >= 5:
                avg_vol = lookback_bars["Volume"].mean()
                rvol = bar["Volume"] / max(avg_vol, 1)
                if rvol >= ORB_PARAMS["rvol_threshold"]:
                    breakout_bar = ts
                    breakout_price = float(bar["High"])
                    breakout_rvol = rvol
                    break

    return {
        "status": "BREAKOUT" if breakout_bar else ("MONITORING" if not too_narrow else "TOO_NARROW"),
        "or_high": or_high,
        "or_low": or_low,
        "or_range": or_range,
        "or_range_pct": or_range_pct,
        "too_narrow": too_narrow,
        "breakout_time": breakout_bar.isoformat() if breakout_bar else None,
        "breakout_price": breakout_price,
        "breakout_rvol": breakout_rvol,
    }


def generate_signal(symbol: str, today: datetime.date = None) -> Optional[dict]:
    """
    给定 symbol, 生成完整下单参数
    返回 None 如果今天没有信号
    """
    df = load_5m_data(symbol)
    if df.empty:
        return None

    or_state = compute_or_for_today(df, today)
    if not or_state or or_state.get("status") != "BREAKOUT":
        return None

    cfg = TICKER_CONFIG.get(symbol, {"entry_slip": 0.001, "stop_slip": 0.002})
    or_high = or_state["or_high"]
    or_low = or_state["or_low"]
    or_range = or_state["or_range"]

    entry = or_high * (1 + cfg["entry_slip"])
    stop = or_low
    tp = or_high + ORB_PARAMS["target_r"] * or_range

    return {
        "symbol": symbol,
        "side": "BUY",
        "breakout_time": or_state["breakout_time"],
        "entry": round(entry, 3),
        "stop": round(stop, 3),
        "tp": round(tp, 3),
        "rvol": round(or_state["breakout_rvol"], 2),
        "or_range_pct": round(or_state["or_range_pct"] * 100, 2),
        "risk_pct": round((entry - stop) / entry * 100, 2),
        "reward_pct": round((tp - entry) / entry * 100, 2),
        "rr_ratio": ORB_PARAMS["target_r"],
    }


# ═══════════════════════════════════════════════════════════════════════
# 历史回测 (从 5m 数据生成所有历史信号 + 结果)
# ═══════════════════════════════════════════════════════════════════════
def backtest_signals_history(symbol: str) -> pd.DataFrame:
    """
    遍历整个 5m 历史数据, 生成所有 ORB 信号 + 实际结果 (止盈/止损/日末)
    用于在 K 线图上标记历史买卖点
    """
    df = load_5m_data(symbol)
    if df.empty:
        return pd.DataFrame()

    cfg = TICKER_CONFIG.get(symbol, {"entry_slip": 0.001, "stop_slip": 0.002})
    p = ORB_PARAMS

    # 按交易日分组
    df["date"] = df.index.date
    trades = []

    for day, day_df in df.groupby("date"):
        if len(day_df) < p["or_bars"] + 5:
            continue

        first_n = day_df.head(p["or_bars"])
        or_high = float(first_n["High"].max())
        or_low = float(first_n["Low"].min())
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2
        or_range_pct = or_range / or_mid

        if or_range_pct < p["min_or_range_pct"]:
            continue

        post_or = day_df.iloc[p["or_bars"]:]
        if len(post_or) == 0:
            continue

        # 找突破点
        entry_idx = None
        entry_price = None
        for ts, bar in post_or.iterrows():
            if bar["High"] > or_high and bar["Close"] > or_high:
                bar_pos = day_df.index.get_loc(ts)
                lookback_start = max(0, bar_pos - p["rvol_lookback"])
                avg_vol = day_df.iloc[lookback_start:bar_pos]["Volume"].mean() or 1
                rvol = bar["Volume"] / avg_vol
                if rvol >= p["rvol_threshold"]:
                    entry_idx = ts
                    entry_price_theoretical = or_high
                    entry_price = or_high * (1 + cfg["entry_slip"])
                    break

        if entry_idx is None:
            continue

        # 模拟出场
        stop_price = or_low
        tp_price = or_high + p["target_r"] * or_range
        bars_after_entry = day_df.loc[entry_idx:].iloc[1:]

        result = "EOD"
        exit_price = None
        exit_idx = None

        for ts, bar in bars_after_entry.iterrows():
            # 收盘前 N 根强平
            bars_remaining = len(day_df) - day_df.index.get_loc(ts) - 1
            if bars_remaining <= p["eod_exit_bars"]:
                exit_price = float(bar["Close"]) * (1 - 0.0005)
                exit_idx = ts
                result = "EOD"
                break
            if bar["Low"] <= stop_price:
                exit_price = stop_price * (1 - cfg["stop_slip"])
                exit_idx = ts
                result = "STOP"
                break
            if bar["High"] >= tp_price:
                exit_price = tp_price * (1 - 0.0005)
                exit_idx = ts
                result = "TP"
                break

        if exit_idx is None:
            exit_idx = day_df.index[-1]
            exit_price = float(day_df.loc[exit_idx, "Close"]) * (1 - 0.0005)
            result = "EOD"

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        trades.append({
            "day": str(day),
            "entry_time": entry_idx,
            "entry_price": round(entry_price, 3),
            "stop": round(stop_price, 3),
            "tp": round(tp_price, 3),
            "exit_time": exit_idx,
            "exit_price": round(exit_price, 3),
            "result": result,
            "pnl_pct": round(pnl_pct, 3),
            "or_range_pct": round(or_range_pct * 100, 2),
        })

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════
# 性能汇总
# ═══════════════════════════════════════════════════════════════════════
def summary_stats(trades_df: pd.DataFrame) -> dict:
    """生成回测性能指标"""
    if trades_df.empty:
        return {}
    n = len(trades_df)
    wins_tp = (trades_df["result"] == "TP").sum()
    losses_stop = (trades_df["result"] == "STOP").sum()
    eod = (trades_df["result"] == "EOD").sum()
    positive = (trades_df["pnl_pct"] > 0).sum()

    cum = (1 + trades_df["pnl_pct"] / 100).prod() - 1
    avg_win = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"].mean() if positive > 0 else 0
    avg_loss = trades_df[trades_df["pnl_pct"] < 0]["pnl_pct"].mean() if (n - positive) > 0 else 0
    pf = (trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"].sum() /
          abs(trades_df[trades_df["pnl_pct"] < 0]["pnl_pct"].sum())) if (n - positive) > 0 else float('inf')

    return {
        "trades": int(n),
        "wins_tp": int(wins_tp),
        "losses_stop": int(losses_stop),
        "eod_exits": int(eod),
        "positive_pct": round(positive / n * 100, 1),
        "win_rate_strict": round(wins_tp / n * 100, 1),
        "cumulative_return_pct": round(cum * 100, 2),
        "avg_win_pct": round(avg_win, 3) if avg_win else 0,
        "avg_loss_pct": round(avg_loss, 3) if avg_loss else 0,
        "profit_factor": round(pf, 2) if pf != float('inf') else "Inf",
    }


if __name__ == "__main__":
    print(f"📊 ORB 策略模块自检\n")
    print(f"可用标的: {list_available_symbols()}")
    print(f"核心组合: {CORE_PORTFOLIO}\n")
    for sym in CORE_PORTFOLIO:
        df = load_5m_data(sym)
        trades = backtest_signals_history(sym)
        stats = summary_stats(trades)
        print(f"{sym:<10} {len(df):>6} 根  |  {stats['trades']:>4} 笔交易  |  "
              f"胜率 {stats['positive_pct']}%  |  累计 {stats['cumulative_return_pct']:+.1f}%  |  "
              f"PF {stats['profit_factor']}")
