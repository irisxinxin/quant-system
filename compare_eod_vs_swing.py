"""
compare_eod_vs_swing.py — 对比"日末平仓" vs "跨日持仓" (修正版, 当日只交易首次突破)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from signals.orb_strategy import (
    list_available_symbols, load_5m_data, TICKER_CONFIG, ORB_PARAMS
)


def backtest(symbol: str, eod_exit: bool = True, max_hold_days: int = 10) -> dict:
    """
    eod_exit=True : 当日 15:55 ET 强平 (现有策略)
    eod_exit=False: 跨日持仓直到 SL/TP 触发 (最多 max_hold_days 天)
    每日最多 1 笔交易 (首次突破)
    """
    df = load_5m_data(symbol)
    if df.empty: return None

    cfg = TICKER_CONFIG.get(symbol, {})
    p = ORB_PARAMS
    df["date"] = df.index.date
    days = sorted(df["date"].unique())

    trades = []
    state = "FLAT"
    entry_price = stop_price = tp_price = None
    entry_day = None
    traded_today = False

    for day in days:
        day_df = df[df["date"] == day]
        if len(day_df) < p["or_bars"] + 5: continue
        traded_today = False  # 每日重置

        # 计算当日 OR
        first_n = day_df.head(p["or_bars"])
        or_high = float(first_n["High"].max())
        or_low = float(first_n["Low"].min())
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2
        or_range_pct = or_range / or_mid

        for ts, bar in day_df.iterrows():
            day_bar_idx = day_df.index.get_loc(ts)

            # 持仓中: 检查止损/止盈/EOD
            if state == "IN_TRADE":
                if bar["Low"] <= stop_price:
                    exit_p = stop_price * (1 - 0.0005)
                    pnl = (exit_p - entry_price) / entry_price * 100
                    trades.append({"day": str(entry_day), "exit_day": str(day),
                                   "result": "STOP", "pnl_pct": round(pnl, 3),
                                   "days_held": (day - entry_day).days})
                    state = "FLAT"
                    continue
                if bar["High"] >= tp_price:
                    exit_p = tp_price * (1 - 0.0005)
                    pnl = (exit_p - entry_price) / entry_price * 100
                    trades.append({"day": str(entry_day), "exit_day": str(day),
                                   "result": "TP", "pnl_pct": round(pnl, 3),
                                   "days_held": (day - entry_day).days})
                    state = "FLAT"
                    continue
                # EOD 平仓
                if eod_exit:
                    bars_left = len(day_df) - day_bar_idx - 1
                    if bars_left <= p["eod_exit_bars"]:
                        exit_p = float(bar["Close"]) * (1 - 0.0005)
                        pnl = (exit_p - entry_price) / entry_price * 100
                        trades.append({"day": str(entry_day), "exit_day": str(day),
                                       "result": "EOD", "pnl_pct": round(pnl, 3),
                                       "days_held": (day - entry_day).days})
                        state = "FLAT"
                        continue
                else:
                    # Swing 模式: 检查最大持仓天数
                    if (day - entry_day).days >= max_hold_days:
                        exit_p = float(bar["Close"]) * (1 - 0.0005)
                        pnl = (exit_p - entry_price) / entry_price * 100
                        trades.append({"day": str(entry_day), "exit_day": str(day),
                                       "result": "MAX_HOLD", "pnl_pct": round(pnl, 3),
                                       "days_held": (day - entry_day).days})
                        state = "FLAT"
                        continue
                continue

            # 空仓 + 当日还没交易过: 检测 OR 后突破
            if state == "FLAT" and not traded_today:
                # 还在 OR 形成期, 跳过
                if day_bar_idx < p["or_bars"]: continue
                # OR 范围异常, 跳过当日
                if or_range_pct < p["min_or_range_pct"] or or_range_pct > 0.05: continue

                if bar["High"] > or_high and bar["Close"] > or_high:
                    bar_pos = day_bar_idx
                    lookback_start = max(0, bar_pos - p["rvol_lookback"])
                    avg_v = day_df.iloc[lookback_start:bar_pos]["Volume"].mean() or 1
                    rvol = bar["Volume"] / avg_v
                    if rvol >= p["rvol_threshold"]:
                        entry_price = or_high * (1 + cfg.get("entry_slip", 0.001))
                        stop_price = or_low
                        tp_price = or_high + p["target_r"] * or_range
                        entry_day = day
                        state = "IN_TRADE"
                        traded_today = True

    # 数据末尾仍持仓 → 强平
    if state == "IN_TRADE":
        last_bar = df.iloc[-1]
        exit_p = float(last_bar["Close"]) * (1 - 0.0005)
        pnl = (exit_p - entry_price) / entry_price * 100
        trades.append({"day": str(entry_day), "exit_day": str(df.index[-1].date()),
                       "result": "DATA_END", "pnl_pct": round(pnl, 3),
                       "days_held": (df.index[-1].date() - entry_day).days})

    if not trades:
        return {"trades": 0}
    td = pd.DataFrame(trades)
    n = len(td)
    pos = (td["pnl_pct"] > 0).sum()
    cum = (1 + td["pnl_pct"] / 100).prod() - 1
    avg_win = td[td["pnl_pct"] > 0]["pnl_pct"].mean() if pos > 0 else 0
    avg_loss = td[td["pnl_pct"] < 0]["pnl_pct"].mean() if (n - pos) > 0 else 0
    pf = (td[td["pnl_pct"] > 0]["pnl_pct"].sum() /
          abs(td[td["pnl_pct"] < 0]["pnl_pct"].sum())) if (n - pos) > 0 else float("inf")
    equity = (1 + td["pnl_pct"] / 100).cumprod()
    dd = (equity / equity.cummax() - 1).min() * 100

    # 持仓时间分布
    avg_hold = td["days_held"].mean()
    pct_overnight = (td["days_held"] >= 1).sum() / n * 100

    return {
        "trades": int(n),
        "positive_pct": round(pos / n * 100, 1),
        "wins_tp": int((td["result"] == "TP").sum()),
        "losses_stop": int((td["result"] == "STOP").sum()),
        "cumulative_return_pct": round(cum * 100, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "profit_factor": round(pf, 2) if pf != float("inf") else "Inf",
        "max_drawdown_pct": round(dd, 2),
        "avg_hold_days": round(avg_hold, 1),
        "pct_overnight": round(pct_overnight, 1),
    }


if __name__ == "__main__":
    tickers = ["AMZN.US", "PLTR.US", "RKLB.US", "IREN.US", "INTC.US",
               "TSLL.US", "SOXL.US", "OKLO.US", "HOOD.US", "AMD.US"]

    print("="*135)
    print("EOD 平仓 vs 跨日持仓 (Swing) — 18 月数据, 每日只交易首次突破")
    print("="*135)
    print(f"{'标的':<10} {'模式':<8} {'交易':>5} {'胜率':>7} {'累计':>10} {'PF':>6} {'均盈':>7} {'均亏':>7} {'回撤':>8} {'均持有':>7} {'过夜%':>6}")
    print("-"*135)

    for sym in tickers:
        for mode_name, eod in [("EOD", True), ("Swing", False)]:
            r = backtest(sym, eod_exit=eod)
            if not r or r.get("trades", 0) == 0: continue
            print(f"{sym:<10} {mode_name:<8} {r['trades']:>5} {r['positive_pct']:>6.1f}% "
                  f"{r['cumulative_return_pct']:>+9.1f}% {str(r['profit_factor']):>6} "
                  f"{r['avg_win_pct']:>+6.2f}% {r['avg_loss_pct']:>+6.2f}% "
                  f"{r['max_drawdown_pct']:>+7.2f}% {r['avg_hold_days']:>5.1f}天 {r['pct_overnight']:>5.0f}%")
        print()
