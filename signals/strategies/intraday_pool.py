"""
intraday_pool.py — 8 大日内短线策略池

每个策略实现统一接口:
  generate_trade_plans(day_df, full_df, **params) -> list[TradePlan]

所有策略 long-only (用户散户做多, 不做空).
所有策略遵守 fill-realistic 约定: 由引擎处理 LMT/MKT 模拟.
"""
import sys
from datetime import time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backtest_engine import TradePlan
from signals.strategies.indicators import (
    vwap_intraday, rsi, ema, atr, adx, bollinger, keltner,
    donchian, supertrend, daily_ema_from_5m,
)


# ═══════════════════════════════════════════════════════════════════════
# 策略 1: ORB-5min Zarattini 论文版
# ═══════════════════════════════════════════════════════════════════════
def strat_orb_5min_zarattini(day_df: pd.DataFrame, full_df: pd.DataFrame,
                              tp_r: float = 10.0,
                              cutoff_close: dtime = dtime(15, 55),
                              **kw) -> list[TradePlan]:
    """
    OR = 第 1 根 5m bar (09:30-09:35 ET).
    第 2 根 5m bar 开盘:
      - OR 阳线 (close > open) → MKT BUY
      - OR 阴线 → 跳过 (long-only)
    止损 = OR.low; 止盈 = entry + tp_r * (entry - stop)
    """
    if len(day_df) < 2:
        return []
    or_bar = day_df.iloc[0]
    if or_bar["Close"] <= or_bar["Open"]:
        return []  # 阴线 / 十字, long-only 跳过

    next_ts = day_df.index[1]
    or_low = float(or_bar["Low"])
    next_open_ref = float(day_df.iloc[1]["Open"])
    stop_dist = next_open_ref - or_low
    if stop_dist <= 0:
        return []
    tp = next_open_ref + tp_r * stop_dist

    return [TradePlan(
        day=day_df.index[0].date(),
        entry_ts=next_ts,
        side="long",
        order_type="MKT",
        limit_price=next_open_ref,
        stop_price=or_low,
        tp_price=tp,
        note=f"ORB5_Z OR={or_bar['High']:.2f}/{or_low:.2f}",
    )]


# ═══════════════════════════════════════════════════════════════════════
# 策略 2: ORB-15min + VWAP + RVOL (现版翻新, fill 修对)
# ═══════════════════════════════════════════════════════════════════════
def strat_orb_15min_vwap(day_df: pd.DataFrame, full_df: pd.DataFrame,
                         or_bars: int = 3,
                         rvol_lookback: int = 20,
                         rvol_thresh: float = 1.5,
                         min_or_pct: float = 0.003,
                         max_or_pct: float = 0.05,
                         tp_r: float = 2.0,
                         **kw) -> list[TradePlan]:
    """
    OR = 前 3 根 5m (15min). 突破 OR_high + close>OR_high + close>VWAP + RVOL≥1.5
    LMT @ OR_high (限价单 — 跟你 live_executor 一致).
    """
    if len(day_df) < or_bars + 5:
        return []
    or_block = day_df.head(or_bars)
    or_high = float(or_block["High"].max())
    or_low = float(or_block["Low"].min())
    or_mid = (or_high + or_low) / 2
    or_range_pct = (or_high - or_low) / or_mid
    if or_range_pct < min_or_pct or or_range_pct > max_or_pct:
        return []

    vwap = vwap_intraday(day_df)
    post = day_df.iloc[or_bars:]
    for i, (ts, bar) in enumerate(post.iterrows()):
        if not (bar["High"] > or_high and bar["Close"] > or_high):
            continue
        # VWAP 同侧
        if bar["Close"] <= vwap.loc[ts]:
            continue
        # RVOL
        bar_pos = day_df.index.get_loc(ts)
        lb_start = max(0, bar_pos - rvol_lookback)
        avg_v = day_df.iloc[lb_start:bar_pos]["Volume"].mean() or 1
        rvol = bar["Volume"] / avg_v
        if rvol < rvol_thresh:
            continue
        # BUG-AT 修复: 信号 bar.Close>or_high 在 bar 收盘后才确认,
        # 实盘只能在 *下一根 bar* 才有 fill 机会. 当根 bar fill 是 look-ahead.
        bar_pos = day_df.index.get_loc(ts)
        if bar_pos + 1 >= len(day_df):
            return []
        next_ts = day_df.index[bar_pos + 1]
        return [TradePlan(
            day=next_ts.date(),
            entry_ts=next_ts,
            side="long",
            order_type="LMT",
            limit_price=or_high,
            stop_price=or_low,
            tp_price=or_high + tp_r * (or_high - or_low),
            note=f"ORB15V rvol={rvol:.2f} or_pct={or_range_pct*100:.2f}",
        )]
    return []


# ═══════════════════════════════════════════════════════════════════════
# 策略 3: VWAP Pullback (Brian Shannon 风格)
# ═══════════════════════════════════════════════════════════════════════
def strat_vwap_pullback(day_df: pd.DataFrame, full_df: pd.DataFrame,
                        warmup_bars: int = 6,         # 至少 6 根 5m 后才考虑
                        trend_check_bars: int = 3,    # VWAP 上倾确认
                        tp_r: float = 2.0,
                        max_hold_bars: int = 30,
                        **kw) -> list[TradePlan]:
    """
    1. warmup 后, 价格站稳 VWAP 上方 (close > VWAP 持续 N 根)
    2. pullback: 某根 bar.Low <= VWAP (碰到 VWAP)
    3. 反弹: 下一根 close > VWAP 且 close > pullback bar.High
    4. LMT @ pullback bar.High
    止损: pullback bar.Low
    止盈: 2R 或 max_hold_bars
    """
    if len(day_df) < warmup_bars + 5:
        return []
    vwap = vwap_intraday(day_df)

    # 趋势确认: warmup 区间内多数 bar close > vwap
    warm = day_df.iloc[:warmup_bars]
    above_pct = ((warm["Close"] > vwap.iloc[:warmup_bars]).sum()) / warmup_bars
    if above_pct < 0.6:
        return []
    # VWAP 上倾
    if vwap.iloc[warmup_bars - 1] <= vwap.iloc[warmup_bars - trend_check_bars - 1]:
        return []

    # BUG-R 修复: pullback 后 *紧邻下一根* 就检测反弹 (而非隔一根).
    # 改用单循环, 在每根 bar 上判断: 当根是否反弹自上一个 pullback?
    post = day_df.iloc[warmup_bars:]
    pullback_bar = None
    for i in range(len(post)):
        ts = post.index[i]
        bar = post.iloc[i]

        # 1. 已有 pullback → 当根能否作为反弹入场 bar?
        if pullback_bar is not None:
            pb_ts, pb = pullback_bar
            if ts == pb_ts:
                continue   # 同根 (设置 pullback 那根) 不算反弹
            if (float(bar["Close"]) > vwap.loc[ts] and
                float(bar["Close"]) > float(pb["High"])):
                limit = float(pb["High"])
                stop = float(pb["Low"])
                stop_dist = limit - stop
                if stop_dist <= 0:
                    pullback_bar = None
                    continue
                # BUG-AT 修复: 反弹信号 bar.Close>pb.High 在 bar 收盘后才确认,
                # 实盘只能在 *下一根 bar* 才有 fill 机会.
                bar_pos = day_df.index.get_loc(ts)
                if bar_pos + 1 >= len(day_df):
                    return []
                next_ts = day_df.index[bar_pos + 1]
                return [TradePlan(
                    day=next_ts.date(),
                    entry_ts=next_ts,
                    side="long",
                    order_type="LMT",
                    limit_price=limit,
                    stop_price=stop,
                    tp_price=limit + tp_r * stop_dist,
                    max_hold_bars=max_hold_bars,
                    note=f"VWAP_PB pb@{pb_ts.strftime('%H:%M')}",
                )]
            # 当根 close 跌破 VWAP → 趋势失效, 整天弃单
            if float(bar["Close"]) < vwap.loc[ts]:
                return []
            # 否则继续等下一根 (pullback 还有效)
            continue

        # 2. 还没找到 pullback → 当根是否就是 pullback?
        if float(bar["Low"]) <= vwap.loc[ts] and float(bar["Close"]) >= vwap.loc[ts] * 0.999:
            pullback_bar = (ts, bar)
    return []


# ═══════════════════════════════════════════════════════════════════════
# 策略 4: VWAP Mean Reversion (RSI2 oversold)
# ═══════════════════════════════════════════════════════════════════════
def strat_vwap_meanrev_rsi2(day_df: pd.DataFrame, full_df: pd.DataFrame,
                            warmup_bars: int = 6,
                            rsi_period: int = 2,
                            rsi_oversold: float = 10.0,
                            atr_dist_mult: float = 1.5,
                            atr_period: int = 14,
                            stop_atr_mult: float = 1.5,
                            max_hold_bars: int = 12,
                            **kw) -> list[TradePlan]:
    """
    long-only 均值回归:
      - close < VWAP 距离 > atr_dist_mult * ATR
      - RSI(2) < rsi_oversold
      → MKT BUY 下根 open
    止损: 当前 close - stop_atr_mult * ATR
    止盈: 触 VWAP (动态) — 这里简化为 +1.5*ATR 固定 TP
    """
    if len(day_df) < max(warmup_bars, rsi_period + 1, atr_period + 1):
        return []
    vwap = vwap_intraday(day_df)
    r = rsi(day_df["Close"], rsi_period)
    a = atr(day_df["High"], day_df["Low"], day_df["Close"], atr_period)

    post = day_df.iloc[warmup_bars:-1]   # 留一根 next bar
    for i, (ts, bar) in enumerate(post.iterrows()):
        bar_pos = day_df.index.get_loc(ts)
        v = vwap.loc[ts]
        rv = r.loc[ts]
        av = a.loc[ts]
        if pd.isna(rv) or pd.isna(av):
            continue
        # 距 VWAP 下方多远 (ATR 计)
        if bar["Close"] >= v:
            continue
        dist_atr = (v - bar["Close"]) / max(av, 1e-6)
        if dist_atr < atr_dist_mult:
            continue
        if rv >= rsi_oversold:
            continue
        # 触发: 下一根 MKT BUY
        if bar_pos + 1 >= len(day_df):
            continue
        nxt_ts = day_df.index[bar_pos + 1]
        nxt = day_df.iloc[bar_pos + 1]
        ref = float(nxt["Open"])
        stop = ref - stop_atr_mult * av
        tp = ref + 1.5 * av   # +1.5 ATR
        return [TradePlan(
            day=nxt_ts.date(),
            entry_ts=nxt_ts,
            side="long",
            order_type="MKT",
            limit_price=ref,
            stop_price=stop,
            tp_price=tp,
            max_hold_bars=max_hold_bars,
            note=f"VWAP_MR rsi={rv:.1f} dist={dist_atr:.1f}ATR",
        )]
    return []


# ═══════════════════════════════════════════════════════════════════════
# 策略 5: EMA 9/21 Cross + VWAP + ADX 过滤
# ═══════════════════════════════════════════════════════════════════════
def strat_ema_cross_vwap(day_df: pd.DataFrame, full_df: pd.DataFrame,
                         ema_fast: int = 9, ema_slow: int = 21,
                         adx_thresh: float = 20.0,
                         atr_period: int = 14,
                         stop_atr_mult: float = 1.5,
                         tp_r: float = 1.5,
                         max_hold_bars: int = 30,
                         **kw) -> list[TradePlan]:
    """
    5m EMA9 上穿 EMA21 + close>VWAP + ADX(14)>20 → MKT BUY 下根 open
    """
    if len(day_df) < max(ema_slow + 5, 16):
        return []
    e_fast = ema(day_df["Close"], ema_fast)
    e_slow = ema(day_df["Close"], ema_slow)
    vwap = vwap_intraday(day_df)
    a = adx(day_df["High"], day_df["Low"], day_df["Close"], 14)
    atr_v = atr(day_df["High"], day_df["Low"], day_df["Close"], atr_period)

    post = day_df.iloc[ema_slow + 1:-1]
    for i, (ts, bar) in enumerate(post.iterrows()):
        bar_pos = day_df.index.get_loc(ts)
        if bar_pos < 1:
            continue
        prev_ts = day_df.index[bar_pos - 1]
        # cross: 前根 fast<=slow, 当根 fast>slow
        if not (e_fast.loc[prev_ts] <= e_slow.loc[prev_ts] and
                e_fast.loc[ts] > e_slow.loc[ts]):
            continue
        if bar["Close"] <= vwap.loc[ts]:
            continue
        if pd.isna(a.loc[ts]) or a.loc[ts] < adx_thresh:
            continue
        if bar_pos + 1 >= len(day_df):
            continue
        nxt_ts = day_df.index[bar_pos + 1]
        nxt = day_df.iloc[bar_pos + 1]
        ref = float(nxt["Open"])
        av = atr_v.loc[ts]
        if pd.isna(av):
            continue
        stop = ref - stop_atr_mult * av
        stop_dist = ref - stop
        tp = ref + tp_r * stop_dist
        return [TradePlan(
            day=nxt_ts.date(),
            entry_ts=nxt_ts,
            side="long",
            order_type="MKT",
            limit_price=ref,
            stop_price=stop,
            tp_price=tp,
            max_hold_bars=max_hold_bars,
            note=f"EMAX adx={a.loc[ts]:.0f}",
        )]
    return []


# ═══════════════════════════════════════════════════════════════════════
# 策略 6: Donchian 20 Breakout
# ═══════════════════════════════════════════════════════════════════════
def strat_donchian_breakout(day_df: pd.DataFrame, full_df: pd.DataFrame,
                            channel_period: int = 20,
                            ema_filter_period: int = 50,
                            atr_period: int = 14,
                            stop_atr_mult: float = 2.0,
                            tp_r: float = 2.0,
                            max_hold_bars: int = 30,
                            **kw) -> list[TradePlan]:
    """
    close > 前 channel_period 根最高 + close > EMA50 → LMT @ breakout
    止损 ATR×2; 止盈 2R
    """
    # BUG-AB 修复: warmup 取 max 而非 sum.
    # 旧版 iloc[20+50:]=iloc[70:] 一天才 78 bars, post 只剩 8 根 → DC20 只看尾盘 40min!
    # donchian.shift(1).rolling(20) 从 iloc[20] 起有效, ema(50) 从 iloc[50] 起收敛,
    # 取 max(20,50)=50 作为 warmup 即可.
    warmup = max(channel_period, ema_filter_period)
    if len(day_df) < warmup + 5:
        return []
    upper, _, _ = donchian(day_df["High"], day_df["Low"], channel_period)
    e50 = ema(day_df["Close"], ema_filter_period)
    atr_v = atr(day_df["High"], day_df["Low"], day_df["Close"], atr_period)

    post = day_df.iloc[warmup:]
    for ts, bar in post.iterrows():
        u = upper.loc[ts]
        if pd.isna(u):
            continue
        if bar["High"] > u and bar["Close"] > u and bar["Close"] > e50.loc[ts]:
            av = atr_v.loc[ts]
            if pd.isna(av):
                continue
            limit = float(u)
            stop = limit - stop_atr_mult * av
            stop_dist = limit - stop
            tp = limit + tp_r * stop_dist
            # BUG-AT 修复: 突破信号 bar.Close>upper 在 bar 收盘后才确认,
            # 实盘只能在 *下一根 bar* 才有 fill 机会.
            bar_pos = day_df.index.get_loc(ts)
            if bar_pos + 1 >= len(day_df):
                return []
            next_ts = day_df.index[bar_pos + 1]
            return [TradePlan(
                day=next_ts.date(),
                entry_ts=next_ts,
                side="long",
                order_type="LMT",
                limit_price=limit,
                stop_price=stop,
                tp_price=tp,
                max_hold_bars=max_hold_bars,
                note=f"DC{channel_period} brk@{u:.2f}",
            )]
    return []


# ═══════════════════════════════════════════════════════════════════════
# 策略 7: TTM Squeeze Pro
# ═══════════════════════════════════════════════════════════════════════
def strat_ttm_squeeze(day_df: pd.DataFrame, full_df: pd.DataFrame,
                       period: int = 20,
                       bb_mult: float = 2.0,
                       kc_mult: float = 1.5,
                       atr_period: int = 14,
                       stop_atr_mult: float = 1.5,
                       tp_r: float = 2.0,
                       max_hold_bars: int = 30,
                       skip_open_min: int = 30,
                       **kw) -> list[TradePlan]:
    """
    BB(20,2) 嵌入 KC(20,1.5×ATR) = squeeze 状态.
    BB 重新钻出 KC = 释放. 释放那根动量 (close - mid) > 0 → MKT BUY 下根.
    跳过开盘 30 分钟 (避免噪声).
    """
    if len(day_df) < period + 10:
        return []
    mid_bb, bb_up, bb_dn = bollinger(day_df["Close"], period, bb_mult)
    mid_kc, kc_up, kc_dn = keltner(day_df["High"], day_df["Low"], day_df["Close"],
                                    period, kc_mult)
    atr_v = atr(day_df["High"], day_df["Low"], day_df["Close"], atr_period)

    in_squeeze = (bb_up < kc_up) & (bb_dn > kc_dn)

    post_open_cutoff = day_df.index[0] + pd.Timedelta(minutes=skip_open_min)
    post = day_df[day_df.index >= post_open_cutoff]

    for i in range(1, len(post)):
        ts = post.index[i]
        prev_ts = post.index[i - 1]
        if pd.isna(in_squeeze.loc[prev_ts]) or pd.isna(in_squeeze.loc[ts]):
            continue
        # squeeze 释放: 前根在 squeeze, 当根不在
        if not (in_squeeze.loc[prev_ts] and not in_squeeze.loc[ts]):
            continue
        # 动量方向 (close vs mid)
        bar = post.iloc[i]
        if bar["Close"] <= mid_bb.loc[ts]:
            continue
        bar_pos = day_df.index.get_loc(ts)
        if bar_pos + 1 >= len(day_df):
            continue
        nxt_ts = day_df.index[bar_pos + 1]
        nxt = day_df.iloc[bar_pos + 1]
        ref = float(nxt["Open"])
        av = atr_v.loc[ts]
        if pd.isna(av):
            continue
        stop = ref - stop_atr_mult * av
        stop_dist = ref - stop
        tp = ref + tp_r * stop_dist
        return [TradePlan(
            day=nxt_ts.date(),
            entry_ts=nxt_ts,
            side="long",
            order_type="MKT",
            limit_price=ref,
            stop_price=stop,
            tp_price=tp,
            max_hold_bars=max_hold_bars,
            note=f"TTM_SQ release",
        )]
    return []


# ═══════════════════════════════════════════════════════════════════════
# 策略 8: Supertrend(10,3) + EMA200(daily) 过滤
# ═══════════════════════════════════════════════════════════════════════
def strat_supertrend(day_df: pd.DataFrame, full_df: pd.DataFrame,
                     st_period: int = 10, st_mult: float = 3.0,
                     atr_period: int = 14,
                     stop_atr_mult: float = 1.5,
                     tp_r: float = 2.0,
                     max_hold_bars: int = 30,
                     **kw) -> list[TradePlan]:
    """
    日内 5m Supertrend 翻多 + close > 日线 EMA200 (大方向过滤) → MKT BUY 下根
    """
    if len(day_df) < st_period + 5:
        return []
    st_v, st_dir = supertrend(day_df["High"], day_df["Low"], day_df["Close"],
                              st_period, st_mult)
    atr_v = atr(day_df["High"], day_df["Low"], day_df["Close"], atr_period)
    daily_ema200 = daily_ema_from_5m(full_df, 200)

    for i in range(1, len(day_df) - 1):
        ts = day_df.index[i]
        prev_ts = day_df.index[i - 1]
        # 翻多: 前 -1, 当 +1
        if not (st_dir.loc[prev_ts] == -1 and st_dir.loc[ts] == 1):
            continue
        # daily EMA200 过滤
        d_ema = daily_ema200.loc[ts] if ts in daily_ema200.index else np.nan
        if pd.isna(d_ema):
            continue
        if day_df.loc[ts, "Close"] <= d_ema:
            continue
        nxt_ts = day_df.index[i + 1]
        nxt = day_df.iloc[i + 1]
        ref = float(nxt["Open"])
        av = atr_v.loc[ts]
        if pd.isna(av):
            continue
        stop = ref - stop_atr_mult * av
        stop_dist = ref - stop
        tp = ref + tp_r * stop_dist
        return [TradePlan(
            day=nxt_ts.date(),
            entry_ts=nxt_ts,
            side="long",
            order_type="MKT",
            limit_price=ref,
            stop_price=stop,
            tp_price=tp,
            max_hold_bars=max_hold_bars,
            note=f"ST flip+",
        )]
    return []


# ═══════════════════════════════════════════════════════════════════════
# 策略注册表
# ═══════════════════════════════════════════════════════════════════════
STRATEGIES = {
    "ORB5_Z":       strat_orb_5min_zarattini,
    "ORB15_VWAP":   strat_orb_15min_vwap,
    "VWAP_PB":      strat_vwap_pullback,
    "VWAP_MR":      strat_vwap_meanrev_rsi2,
    "EMA_X":        strat_ema_cross_vwap,
    "DC20":         strat_donchian_breakout,
    "TTM_SQ":       strat_ttm_squeeze,
    "ST_10_3":      strat_supertrend,
}
