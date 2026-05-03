"""
backtest_engine.py — Fill-Realistic 共享回测引擎

设计原则 (来自 memory/feedback_realistic_fill.md):
  1. 限价单 fill: bar.Low <= limit_price 才算成交
  2. 市价单 fill: 按 bar.Open * (1+slip) 成交
  3. 止损/止盈 gap-through: 按 bar.Open 成交而非触发价
  4. Sanity check 红旗: 胜率>65%/PF>3/累计>1000% 立刻告警
  5. Phantom fill 率: 自动统计限价单未成交比例

策略接入方式:
  策略提供 generate_trade_plans(day_df, **params) -> list[TradePlan]
  引擎负责 fill 模拟 + 出场 + 统计

近期权重评分:
  half-life=90 天的指数衰减加权 (用户要求"近期结果比例放最重")
  同时输出 90d/180d/18m 三个时间窗口对比
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
KLINES_5M_DIR = Path(__file__).parent / "output" / "klines_5m"


# ═══════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class TradePlan:
    """策略生成的开仓计划"""
    day: date
    entry_ts: pd.Timestamp           # 触发信号时的 bar 时间戳
    side: str                         # 'long' / 'short'
    order_type: str                   # 'LMT' / 'MKT'
    limit_price: float                # LMT 用; MKT 也填 (作为参考价)
    stop_price: float
    tp_price: float
    max_hold_bars: Optional[int] = None      # 最长持仓 bar 数, None = 不限
    note: str = ""                            # 信号附加信息


@dataclass
class TradeRecord:
    """实际发生的交易"""
    symbol: str
    strategy: str
    day: date
    side: str
    order_type: str
    fill_status: str                  # 'FILLED' / 'PHANTOM_MISS'
    plan_entry_ts: pd.Timestamp
    plan_limit_price: float
    plan_stop: float
    plan_tp: float
    fill_ts: Optional[pd.Timestamp] = None
    fill_price: Optional[float] = None
    exit_ts: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # STOP/TP/CUTOFF/MAX_HOLD/EOD/NO_FILL
    pnl_pct: Optional[float] = None
    hold_bars: Optional[int] = None
    note: str = ""


# ═══════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════
def load_5m_data(symbol: str) -> pd.DataFrame:
    f = KLINES_5M_DIR / f"{symbol}.json"
    if not f.exists():
        return pd.DataFrame()
    with open(f) as fh:
        payload = json.load(fh)
    df = pd.DataFrame(payload["data"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df.columns = [c.capitalize() for c in df.columns]
    df["date"] = df.index.date
    return df


# ═══════════════════════════════════════════════════════════════════════
# 反向 ETF 过滤 (TSLZ/SOXS/NVDS 多头突破必须底层在跌)
# ═══════════════════════════════════════════════════════════════════════
INVERSE_TO_LONG_PAIR = {
    "TSLZ.US": "TSLL.US",
    "SOXS.US": "SOXL.US",
    "NVDS.US": "NVDL.US",
}


def get_inverse_etf_day_allow(symbol: str) -> Optional[dict]:
    """
    反向 ETF 多头交易: 当且仅当配对的多向 ETF 昨日收盘 EMA50 < EMA200 (底层长期下跌)
    才允许当日开仓.

    返回 dict {date: bool} (True=允许). 非反向 ETF 返回 None (不过滤).
    用 shift(1) 避免 look-ahead: 当天决策只能用昨日收盘的 EMA.
    """
    if symbol not in INVERSE_TO_LONG_PAIR:
        return None
    long_sym = INVERSE_TO_LONG_PAIR[symbol]
    df = load_5m_data(long_sym)
    if df.empty:
        logger.warning(f"反向ETF过滤: 配对 {long_sym} 数据缺失, {symbol} 不过滤")
        return None
    daily_close = df["Close"].resample("1D").last().dropna()
    if len(daily_close) < 200:
        logger.warning(f"反向ETF过滤: {long_sym} 不足 200 天, {symbol} 不过滤")
        return None
    ema50 = daily_close.ewm(span=50, adjust=False).mean()
    ema200 = daily_close.ewm(span=200, adjust=False).mean()
    # 当日决策用昨日 EMA
    allow = (ema50.shift(1) < ema200.shift(1))
    return {d.date(): bool(v) for d, v in allow.items()}


# ═══════════════════════════════════════════════════════════════════════
# Fill 模型
# ═══════════════════════════════════════════════════════════════════════
def simulate_entry_fill(plan: TradePlan, day_df: pd.DataFrame,
                        slip: float = 0.0005) -> tuple:
    """
    模拟开仓 fill. 返回 (fill_ts, fill_price, status)
      status: 'FILLED' / 'PHANTOM_MISS'

    LMT 单逻辑 (long, 同理 short 镜像):
      - 在 entry_ts 这根 bar 检查: 如果 bar.Low <= limit_price → fill @ limit_price
      - 如果 bar.Low > limit_price → 这根 bar PHANTOM_MISS (gap-up 跳过限价)
      - 不延后到下一根 bar (实盘 LIT 一般 GTC 但策略层会取消)

    MKT 单逻辑:
      - 按 entry_ts 那根 bar 的 Open * (1+slip) 成交 (long); 反向同理
    """
    if plan.entry_ts not in day_df.index:
        return None, None, "PHANTOM_MISS"
    bar = day_df.loc[plan.entry_ts]

    if plan.order_type == "MKT":
        if plan.side == "long":
            fill_price = float(bar["Open"]) * (1 + slip)
        else:
            fill_price = float(bar["Open"]) * (1 - slip)
        return plan.entry_ts, fill_price, "FILLED"

    # LMT
    if plan.side == "long":
        # 限价买: 价格回到 limit 之下才 fill
        if float(bar["Low"]) <= plan.limit_price:
            return plan.entry_ts, plan.limit_price, "FILLED"
        else:
            return None, None, "PHANTOM_MISS"
    else:
        # 限价卖空: 价格涨到 limit 之上才 fill
        if float(bar["High"]) >= plan.limit_price:
            return plan.entry_ts, plan.limit_price, "FILLED"
        else:
            return None, None, "PHANTOM_MISS"


def simulate_exit(day_df: pd.DataFrame, fill_ts: pd.Timestamp,
                  fill_price: float, side: str,
                  stop_price: float, tp_price: float,
                  cutoff_et: dtime, max_hold_bars: Optional[int] = None,
                  stop_slip: float = 0.0005, tp_slip: float = 0.0005,
                  fill_is_market: bool = False) -> dict:
    """
    模拟出场. 修正 gap-through:
      - 止损: bar.Open 已穿过 stop → 按 bar.Open 成交 (滑点更深)
      - 止盈: bar.Open 已穿过 tp → 按 bar.Open 成交 (滑点反向, 对你有利)
    优先级: STOP > TP > CUTOFF > MAX_HOLD > EOD (假设同一根 bar 内 stop 先到)

    BUG-M 修复: 对 MKT 单 (fill_is_market=True), 入场 bar 当根 fill 之后剩余时间
    仍可能触 stop/tp. 在循环开始前先用入场 bar 的 Low/High 检测一次.
    LMT 单不做此检测 (fill 在 bar 内位置不明, 保守不算当根出场).
    """
    # ── 入场 bar 同根 stop/tp 检测 (仅 MKT 单) ──
    if fill_is_market and fill_ts in day_df.index:
        entry_bar = day_df.loc[fill_ts]
        if side == "long":
            if float(entry_bar["Low"]) <= stop_price:
                exit_p = stop_price * (1 - stop_slip)
                return {"exit_ts": fill_ts, "exit_p": exit_p, "reason": "STOP",
                        "hold_bars": 0}
            if float(entry_bar["High"]) >= tp_price:
                exit_p = tp_price * (1 - tp_slip)
                return {"exit_ts": fill_ts, "exit_p": exit_p, "reason": "TP",
                        "hold_bars": 0}
        else:
            if float(entry_bar["High"]) >= stop_price:
                exit_p = stop_price * (1 + stop_slip)
                return {"exit_ts": fill_ts, "exit_p": exit_p, "reason": "STOP",
                        "hold_bars": 0}
            if float(entry_bar["Low"]) <= tp_price:
                exit_p = tp_price * (1 + tp_slip)
                return {"exit_ts": fill_ts, "exit_p": exit_p, "reason": "TP",
                        "hold_bars": 0}

    bars_after = day_df.loc[fill_ts:].iloc[1:]   # 入场 bar 不参与后续循环

    for i, (ts, bar) in enumerate(bars_after.iterrows()):
        bar_time = ts.time()

        if side == "long":
            # 1. 止损
            if float(bar["Low"]) <= stop_price:
                exit_p = min(float(bar["Open"]), stop_price) * (1 - stop_slip)
                return {"exit_ts": ts, "exit_p": exit_p, "reason": "STOP",
                        "hold_bars": i + 1}
            # 2. 止盈
            if float(bar["High"]) >= tp_price:
                exit_p = max(float(bar["Open"]), tp_price) * (1 - tp_slip)
                return {"exit_ts": ts, "exit_p": exit_p, "reason": "TP",
                        "hold_bars": i + 1}
        else:  # short
            if float(bar["High"]) >= stop_price:
                exit_p = max(float(bar["Open"]), stop_price) * (1 + stop_slip)
                return {"exit_ts": ts, "exit_p": exit_p, "reason": "STOP",
                        "hold_bars": i + 1}
            if float(bar["Low"]) <= tp_price:
                exit_p = min(float(bar["Open"]), tp_price) * (1 + tp_slip)
                return {"exit_ts": ts, "exit_p": exit_p, "reason": "TP",
                        "hold_bars": i + 1}

        # 3. cutoff 强平
        if bar_time >= cutoff_et:
            exit_p = float(bar["Close"]) * (1 - 0.0005 if side == "long" else 1 + 0.0005)
            return {"exit_ts": ts, "exit_p": exit_p, "reason": "CUTOFF",
                    "hold_bars": i + 1}

        # 4. max_hold_bars
        if max_hold_bars is not None and (i + 1) >= max_hold_bars:
            exit_p = float(bar["Close"]) * (1 - 0.0005 if side == "long" else 1 + 0.0005)
            return {"exit_ts": ts, "exit_p": exit_p, "reason": "MAX_HOLD",
                    "hold_bars": i + 1}

    # 5. 数据末尾 / 当日数据末尾
    last = day_df.iloc[-1]
    exit_p = float(last["Close"]) * (1 - 0.0005 if side == "long" else 1 + 0.0005)
    return {"exit_ts": day_df.index[-1], "exit_p": exit_p, "reason": "EOD",
            "hold_bars": len(bars_after)}


def compute_pnl_pct(side: str, entry_price: float, exit_price: float) -> float:
    if side == "long":
        return (exit_price - entry_price) / entry_price * 100
    else:
        return (entry_price - exit_price) / entry_price * 100


# ═══════════════════════════════════════════════════════════════════════
# 主回测循环
# ═══════════════════════════════════════════════════════════════════════
def run_backtest(symbol: str,
                 strategy_fn: Callable,
                 strategy_name: str,
                 cutoff_et: dtime = dtime(15, 55),
                 entry_slip: float = 0.0005,
                 stop_slip: float = 0.0005,
                 tp_slip: float = 0.0005,
                 strategy_params: dict = None) -> list[TradeRecord]:
    """
    通用回测主循环.

    strategy_fn(day_df, full_df, **strategy_params) -> list[TradePlan]
      day_df: 当日 5m K 线 (含 OHLCV)
      full_df: 完整历史 5m (用于跨日指标如 EMA200)
      返回: 当日的 trade plans (长度 0+)
    """
    strategy_params = strategy_params or {}
    df = load_5m_data(symbol)
    if df.empty:
        return []

    days = sorted(df["date"].unique())
    records: list[TradeRecord] = []

    # 反向 ETF 日级过滤 (TSLZ/SOXS/NVDS): 配对底层昨日 EMA50<EMA200 才允许交易
    inv_allow = get_inverse_etf_day_allow(symbol)

    for day in days:
        day_df = df[df["date"] == day]
        if len(day_df) < 5:
            continue

        # BUG-V 修复: 跳过开盘数据不完整的日子 (首根 != 09:30 ET 通常是数据采集起始日)
        first_ts = day_df.index[0]
        if first_ts.time() != dtime(9, 30):
            continue

        # 反向 ETF: 底层趋势不利则跳过当日 (无 plan, 无 record)
        if inv_allow is not None:
            if not inv_allow.get(day, False):
                continue

        plans = strategy_fn(day_df, df, **strategy_params)
        if not plans:
            continue

        # 引擎约束: 同日内只接第一个 plan (实盘单一仓位)
        # 策略可以返回多个, 但引擎只跑首个 fill 成功的
        for plan in plans:
            fill_ts, fill_p, status = simulate_entry_fill(plan, day_df, slip=entry_slip)

            rec = TradeRecord(
                symbol=symbol, strategy=strategy_name, day=day,
                side=plan.side, order_type=plan.order_type,
                fill_status=status,
                plan_entry_ts=plan.entry_ts, plan_limit_price=plan.limit_price,
                plan_stop=plan.stop_price, plan_tp=plan.tp_price,
                note=plan.note,
            )

            if status != "FILLED":
                rec.exit_reason = "NO_FILL"
                records.append(rec)
                continue

            # 模拟出场
            ex = simulate_exit(day_df, fill_ts, fill_p, plan.side,
                                plan.stop_price, plan.tp_price,
                                cutoff_et, plan.max_hold_bars,
                                stop_slip, tp_slip,
                                fill_is_market=(plan.order_type == "MKT"))
            rec.fill_ts = fill_ts
            rec.fill_price = fill_p
            rec.exit_ts = ex["exit_ts"]
            rec.exit_price = ex["exit_p"]
            rec.exit_reason = ex["reason"]
            rec.hold_bars = ex["hold_bars"]
            rec.pnl_pct = compute_pnl_pct(plan.side, fill_p, ex["exit_p"])

            records.append(rec)
            break   # 当日只交易首笔成功 fill 的

    return records


# ═══════════════════════════════════════════════════════════════════════
# 性能统计 (含 recency-weighted)
# ═══════════════════════════════════════════════════════════════════════
def compute_metrics(records: list[TradeRecord],
                    half_life_days: int = 90,
                    asof: Optional[date] = None,
                    data_end_date: Optional[date] = None) -> dict:
    """
    计算综合指标. 默认 half-life 90 天加权 (近期权重最高).

    返回字段:
      filled       — 实际成交笔数
      phantom_miss — 限价单未 fill 笔数
      win_rate, cum_pct, pf, max_dd, avg_pnl  — 全周期等权
      recency_*    — half-life 加权版本 (近期权重大)
      last_90d_*, last_180d_*  — 时间窗口子集
    """
    filled = [r for r in records if r.fill_status == "FILLED"]
    phantom = [r for r in records if r.fill_status == "PHANTOM_MISS"]

    n = len(filled)
    n_total = len(records)
    out = {
        "trades_total_signals": n_total,
        "filled": n,
        "phantom_miss": len(phantom),
        "phantom_rate_pct": round(len(phantom) / n_total * 100, 1) if n_total else 0,
    }

    if n == 0:
        return {**out, "no_trades": True}

    df = pd.DataFrame([{
        "day": r.day, "pnl_pct": r.pnl_pct, "reason": r.exit_reason,
    } for r in filled])
    df["day"] = pd.to_datetime(df["day"])

    # ── 全周期等权 ──
    pos = (df["pnl_pct"] > 0).sum()
    win_pnl = df[df["pnl_pct"] > 0]["pnl_pct"].sum()
    loss_pnl = abs(df[df["pnl_pct"] < 0]["pnl_pct"].sum())
    pf = win_pnl / loss_pnl if loss_pnl > 0 else float("inf")
    cum = (1 + df["pnl_pct"] / 100).prod() - 1
    # BUG-U 修复: cumprod 序列前补 1.0, 否则第一笔大亏时 max_dd 被低估 1-2pp
    eq = pd.concat([pd.Series([1.0]), (1 + df["pnl_pct"] / 100).cumprod()],
                    ignore_index=True)
    dd = (eq / eq.cummax() - 1).min() * 100
    avg = df["pnl_pct"].mean()

    out.update({
        "win_rate": round(pos / n * 100, 1),
        "cum_pct": round(cum * 100, 2),
        "pf": round(pf, 2) if pf != float("inf") else float("inf"),
        "max_dd": round(dd, 2),
        "avg_pnl": round(avg, 3),
        "n_tp": int((df["reason"] == "TP").sum()),
        "n_stop": int((df["reason"] == "STOP").sum()),
        "n_cutoff": int((df["reason"] == "CUTOFF").sum()),
        "n_eod": int((df["reason"] == "EOD").sum()),
    })

    # ── recency-weighted (half-life 衰减) ──
    # BUG-J 修复: 优先用 data_end_date (数据末日) 而非 trade.max()
    # 否则若策略最近半年没出信号, recency 仍按"最近一笔=今天"算 → 失真
    if asof is None:
        if data_end_date is not None:
            asof = data_end_date
        else:
            asof = df["day"].max().date()
    asof_ts = pd.Timestamp(asof)
    df["days_ago"] = (asof_ts - df["day"]).dt.days
    df["weight"] = 0.5 ** (df["days_ago"] / half_life_days)

    w_sum = df["weight"].sum()
    if w_sum > 0:
        recency_avg = (df["pnl_pct"] * df["weight"]).sum() / w_sum
        recency_win = (df["weight"] * (df["pnl_pct"] > 0)).sum() / w_sum * 100
        rec_win = (df["weight"] * df["pnl_pct"].clip(lower=0)).sum()
        rec_loss = (df["weight"] * (-df["pnl_pct"].clip(upper=0))).sum()
        recency_pf = rec_win / rec_loss if rec_loss > 0 else float("inf")
        out.update({
            "recency_avg_pnl": round(recency_avg, 3),
            "recency_win_rate": round(recency_win, 1),
            "recency_pf": round(recency_pf, 2) if recency_pf != float("inf") else float("inf"),
        })

    # ── 时间窗口子集 ──
    for label, days_back in [("last_90d", 90), ("last_180d", 180)]:
        sub = df[df["days_ago"] <= days_back]
        if len(sub) == 0:
            continue
        ns = len(sub)
        ps = (sub["pnl_pct"] > 0).sum()
        wp = sub[sub["pnl_pct"] > 0]["pnl_pct"].sum()
        lp = abs(sub[sub["pnl_pct"] < 0]["pnl_pct"].sum())
        pfs = wp / lp if lp > 0 else float("inf")
        cums = (1 + sub["pnl_pct"] / 100).prod() - 1
        out[f"{label}_n"] = ns
        out[f"{label}_win"] = round(ps / ns * 100, 1)
        out[f"{label}_pf"] = round(pfs, 2) if pfs != float("inf") else float("inf")
        out[f"{label}_cum_pct"] = round(cums * 100, 2)

    return out


# ═══════════════════════════════════════════════════════════════════════
# Sanity Check 红旗系统
# ═══════════════════════════════════════════════════════════════════════
RED_FLAGS = {
    "win_rate_too_high": ("win_rate", lambda v: v > 65, "胜率 > 65% (短线突破基准 50-60%)"),
    "pf_too_high":        ("pf", lambda v: v != float("inf") and v > 3.0,
                           "PF > 3 (顶级量化基金水平, 散户策略可疑)"),
    "cum_too_high":       ("cum_pct", lambda v: v > 1000, "1y 累计 > 1000% (复利幻觉?)"),
    "phantom_too_high":   ("phantom_rate_pct", lambda v: v > 30,
                           "限价单 phantom 率 > 30% (考虑改市价或剔除)"),
}


def check_sanity(metrics: dict, symbol: str = "", strategy: str = "") -> list[str]:
    flags = []
    for name, (key, check, msg) in RED_FLAGS.items():
        v = metrics.get(key)
        if v is not None and check(v):
            flags.append(f"  ⚠ {symbol} / {strategy}: {msg} (值={v})")
    return flags


# ═══════════════════════════════════════════════════════════════════════
# 标的列表
# ═══════════════════════════════════════════════════════════════════════
# TOP9 (用户当前实盘) + 之前剔除的 11 只 + 2026-05 新加 3 只 = 23 只全集
TICKERS_FULL = [
    # TOP9
    "CRWV.US", "RKLB.US", "OKLO.US", "INTC.US", "IREN.US",
    "TSLL.US", "PLTR.US", "NBIS.US", "AMZN.US",
    # 之前剔除的 11 只
    "SOXL.US", "NVDL.US", "TSLZ.US", "NVDS.US", "SOXS.US",
    "HOOD.US", "AMD.US", "EOSE.US", "MSFL.US", "MSFU.US", "CIFR.US",
    # 2026-05 新加 (IPO / 拆分上市的高波动股)
    "CRCL.US", "SNDK.US", "LITE.US",
]


if __name__ == "__main__":
    # Smoke test: 加载一只标的, 验证引擎可用
    print("📊 backtest_engine 自检")
    df = load_5m_data("PLTR.US")
    print(f"PLTR.US: {len(df)} bars, {len(df['date'].unique())} 天")
    print(f"日期范围: {df.index.min()} ~ {df.index.max()}")

    # 简单测试: 第一根 5m bar 直接 BUY @ open, EOD 平仓
    def dummy_strategy(day_df, full_df, **kw):
        if len(day_df) < 2:
            return []
        first = day_df.iloc[1]
        ts = day_df.index[1]
        return [TradePlan(
            day=ts.date(), entry_ts=ts, side="long",
            order_type="MKT", limit_price=float(first["Open"]),
            stop_price=float(first["Low"]) * 0.99,
            tp_price=float(first["High"]) * 1.05,
        )]

    recs = run_backtest("PLTR.US", dummy_strategy, "DUMMY")
    m = compute_metrics(recs)
    print(f"\nDummy strategy: {m['filled']} 笔, 胜率 {m.get('win_rate', '-')}%, "
          f"累计 {m.get('cum_pct', '-')}%, PF {m.get('pf', '-')}")
