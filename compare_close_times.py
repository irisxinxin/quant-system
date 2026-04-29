"""
compare_close_times.py — 测试不同"日内强平时间"的策略表现

测试的 cutoff (ET 时间):
  10:30 — 早盘结束 (黄金时段后)
  11:30 — 午前
  12:30 — 午餐前
  13:00 — 用户睡觉时间 (= 01:00 SGT)
  14:00 — 下午中段
  15:00 — 收盘前 1 小时
  15:55 — 当前默认 EOD (3:55 SGT)

不测试的 (因为之前已知):
  - Swing (跨日持仓): 回撤翻倍, 不推荐
"""
import sys
from datetime import time as dtime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from signals.orb_strategy import load_5m_data, TICKER_CONFIG, ORB_PARAMS


def backtest_with_cutoff(symbol: str, cutoff_et: dtime) -> dict:
    """
    回测: 收盘时间设为 cutoff_et (ET 时间), 之后强制市价平仓
    每日只交易首次突破
    """
    df = load_5m_data(symbol)
    if df.empty: return None

    cfg = TICKER_CONFIG.get(symbol, {})
    p = ORB_PARAMS
    df["date"] = df.index.date
    days = sorted(df["date"].unique())

    trades = []
    state = "FLAT"
    entry_price = stop_price = tp_price = entry_day = None
    traded_today = False

    for day in days:
        day_df = df[df["date"] == day]
        if len(day_df) < p["or_bars"] + 5: continue
        traded_today = False

        first_n = day_df.head(p["or_bars"])
        or_high = float(first_n["High"].max())
        or_low = float(first_n["Low"].min())
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2
        or_range_pct = or_range / or_mid

        for ts, bar in day_df.iterrows():
            day_bar_idx = day_df.index.get_loc(ts)
            bar_time = ts.time()

            if state == "IN_TRADE":
                if bar["Low"] <= stop_price:
                    exit_p = stop_price * (1 - 0.0005)
                    pnl = (exit_p - entry_price) / entry_price * 100
                    trades.append({"day": str(day), "result": "STOP",
                                   "pnl_pct": round(pnl, 3),
                                   "exit_time": bar_time.strftime("%H:%M")})
                    state = "FLAT"
                    continue
                if bar["High"] >= tp_price:
                    exit_p = tp_price * (1 - 0.0005)
                    pnl = (exit_p - entry_price) / entry_price * 100
                    trades.append({"day": str(day), "result": "TP",
                                   "pnl_pct": round(pnl, 3),
                                   "exit_time": bar_time.strftime("%H:%M")})
                    state = "FLAT"
                    continue
                # 提前 cutoff 平仓
                if bar_time >= cutoff_et:
                    exit_p = float(bar["Close"]) * (1 - 0.0005)
                    pnl = (exit_p - entry_price) / entry_price * 100
                    trades.append({"day": str(day), "result": "CUTOFF",
                                   "pnl_pct": round(pnl, 3),
                                   "exit_time": bar_time.strftime("%H:%M")})
                    state = "FLAT"
                    continue

            if state == "FLAT" and not traded_today:
                if day_bar_idx < p["or_bars"]: continue
                if or_range_pct < p["min_or_range_pct"] or or_range_pct > 0.05: continue
                if bar_time >= cutoff_et: continue   # 已过 cutoff 不开新仓

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
        trades.append({"day": str(entry_day), "result": "DATA_END", "pnl_pct": round(pnl, 3),
                       "exit_time": "EOD"})

    if not trades: return {"trades": 0}
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

    n_tp = (td["result"] == "TP").sum()
    n_stop = (td["result"] == "STOP").sum()
    n_cutoff = (td["result"] == "CUTOFF").sum()

    return {
        "trades": int(n),
        "positive_pct": round(pos / n * 100, 1),
        "n_tp": int(n_tp),
        "n_stop": int(n_stop),
        "n_cutoff": int(n_cutoff),
        "cumulative_return_pct": round(cum * 100, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "profit_factor": round(pf, 2) if pf != float("inf") else "Inf",
        "max_drawdown_pct": round(dd, 2),
    }


if __name__ == "__main__":
    cutoffs = [
        ("10:30", dtime(10, 30)),
        ("11:30", dtime(11, 30)),
        ("12:30", dtime(12, 30)),
        ("13:00", dtime(13, 0)),    # 用户睡觉时间
        ("14:00", dtime(14, 0)),
        ("15:00", dtime(15, 0)),
        ("15:55", dtime(15, 55)),   # 当前默认
    ]

    tickers = ["AMZN.US", "PLTR.US", "RKLB.US", "IREN.US", "INTC.US",
               "TSLL.US", "SOXL.US", "OKLO.US", "HOOD.US", "AMD.US"]

    print("="*150)
    print("不同收盘时间对比 — ORB 策略")
    print("="*150)

    for sym in tickers:
        print(f"\n━━━ {sym} ━━━")
        print(f"{'Cutoff':<8} {'交易':>5} {'胜率':>7} {'累计':>10} {'PF':>6} "
              f"{'TP':>4} {'SL':>4} {'CUT':>4} {'均盈':>7} {'均亏':>7} {'回撤':>8}")
        for label, ct in cutoffs:
            r = backtest_with_cutoff(sym, ct)
            if not r or r.get("trades", 0) == 0: continue
            print(f"{label:<8} {r['trades']:>5} {r['positive_pct']:>6.1f}% "
                  f"{r['cumulative_return_pct']:>+9.1f}% {str(r['profit_factor']):>6} "
                  f"{r['n_tp']:>4} {r['n_stop']:>4} {r['n_cutoff']:>4} "
                  f"{r['avg_win_pct']:>+6.2f}% {r['avg_loss_pct']:>+6.2f}% "
                  f"{r['max_drawdown_pct']:>+7.2f}%")
