"""
diagnose_fill_bias.py — 诊断旧回测的"幽灵成交"偏差

问题:
  原回测在突破 bar 开高于 OR_High (gap up) 时, 仍假设按 OR_High 成交,
  但实盘限价单根本不会 fill (价格从未回到 OR_High).
  这会高估盈利信号, 让回测看起来比实际好.

本脚本对每只标的, 同样的信号序列下对比三种成交模型:
  A. OLD_BIASED   — 旧回测: 全部按 OR_High*(1+slip) 入场 (含幽灵)
  B. LIMIT_REAL   — 限价单(实盘 LIT): bar.Low <= OR_High 才 fill, 否则 skip
  C. MARKET_REAL  — 市价单(保守): gap 按 bar.Open*(1+slip) 入场

出场也修正:
  - 止损/止盈遇 gap-through 按 bar.Open 成交, 不再假设精确 stop/tp 价

用法:
  python3 diagnose_fill_bias.py
  python3 diagnose_fill_bias.py --cutoff 13:00       # 自定义 cutoff
  python3 diagnose_fill_bias.py --tickers TOP9       # TOP9 / SUITABLE / ALL

输出:
  - 控制台: per-ticker 三模型对比表 + gap 率
  - output/fill_bias_report.csv: 详细数据
  - output/fill_bias_trades.csv: 每笔交易明细 (3 模型并列)
"""
import argparse
import sys
from datetime import time as dtime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from signals.orb_strategy import load_5m_data, TICKER_CONFIG, ORB_PARAMS, filter_signal


# ─── TOP9 (用户当前实盘组合) ─────────────────────────────────────
TOP9_PORTFOLIO = [
    "CRWV.US", "RKLB.US", "OKLO.US", "INTC.US", "IREN.US",
    "TSLL.US", "PLTR.US", "NBIS.US", "AMZN.US",
]

DEFAULT_CUTOFF = dtime(15, 55)


def scan_signals(symbol: str) -> list:
    """
    扫描所有历史信号, 返回每个突破点的原始信息 (含 bar.Open/Low/High/Close).
    不在这一步算盈亏 — 让下游的三个 fill 模型分别处理.
    """
    df = load_5m_data(symbol)
    if df.empty:
        return []

    p = ORB_PARAMS
    df["date"] = df.index.date
    days = sorted(df["date"].unique())

    signals = []
    for day in days:
        day_df = df[df["date"] == day]
        if len(day_df) < p["or_bars"] + 5:
            continue

        first_n = day_df.head(p["or_bars"])
        or_high = float(first_n["High"].max())
        or_low = float(first_n["Low"].min())
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2
        or_range_pct = or_range / or_mid

        if or_range_pct < p["min_or_range_pct"] or or_range_pct > 0.05:
            continue

        # 应用 filter (反向 ETF / OR 异动) — 与实盘一致
        ok, _ = filter_signal(symbol, or_range_pct)
        if not ok:
            continue

        post_or = day_df.iloc[p["or_bars"]:]
        if len(post_or) == 0:
            continue

        # 找首次突破 (含 RVOL 过滤)
        for ts, bar in post_or.iterrows():
            if bar["High"] > or_high and bar["Close"] > or_high:
                idx_pos = day_df.index.get_loc(ts)
                lookback_start = max(0, idx_pos - p["rvol_lookback"])
                avg_vol = day_df.iloc[lookback_start:idx_pos]["Volume"].mean() or 1
                rvol = bar["Volume"] / avg_vol
                if rvol >= p["rvol_threshold"]:
                    signals.append({
                        "day": day,
                        "entry_ts": ts,
                        "entry_bar_open": float(bar["Open"]),
                        "entry_bar_high": float(bar["High"]),
                        "entry_bar_low": float(bar["Low"]),
                        "entry_bar_close": float(bar["Close"]),
                        "or_high": or_high,
                        "or_low": or_low,
                        "or_range": or_range,
                        "or_range_pct": or_range_pct,
                        "rvol": rvol,
                        # gap_up = 整根 bar 的 low 都在 OR_High 之上 → 限价单 miss
                        "is_gap_up": float(bar["Low"]) > or_high,
                    })
                    break  # 当日只交易首次突破
    return signals


def simulate_exit(day_df: pd.DataFrame, entry_ts, entry_price: float,
                  stop_price: float, tp_price: float, cutoff_et: dtime,
                  stop_slip: float = 0.0005, tp_slip: float = 0.0005) -> dict:
    """
    模拟出场, 修复 gap-through 偏差:
      - 止损: 如果 bar.Open <= stop, 按 bar.Open 成交 (gap-down through stop = 滑点更深)
      - 止盈: 如果 bar.Open >= tp, 按 bar.Open 成交 (gap-up through tp = 滑点更浅)
      - cutoff/EOD 强平: 按 bar.Close
    """
    bars_after = day_df.loc[entry_ts:].iloc[1:]   # entry bar 不参与出场判定 (实盘 entry 当根可能尾部 fill)
    for ts, bar in bars_after.iterrows():
        bar_time = ts.time()

        # 1. 止损先于止盈 (保守: 同根 bar 假设先到 stop)
        if bar["Low"] <= stop_price:
            # gap-down through stop
            exit_p = min(float(bar["Open"]), stop_price) * (1 - stop_slip)
            return {"exit_ts": ts, "exit_p": exit_p, "result": "STOP"}

        # 2. 止盈
        if bar["High"] >= tp_price:
            exit_p = max(float(bar["Open"]), tp_price) * (1 - tp_slip)
            return {"exit_ts": ts, "exit_p": exit_p, "result": "TP"}

        # 3. cutoff 强平
        if bar_time >= cutoff_et:
            exit_p = float(bar["Close"]) * (1 - 0.0005)
            return {"exit_ts": ts, "exit_p": exit_p, "result": "CUTOFF"}

    # 数据末尾仍持仓
    last_bar = day_df.iloc[-1]
    exit_p = float(last_bar["Close"]) * (1 - 0.0005)
    return {"exit_ts": day_df.index[-1], "exit_p": exit_p, "result": "EOD"}


def backtest_three_models(symbol: str, cutoff_et: dtime) -> dict:
    """对单只标的跑 3 个 fill 模型, 返回 dict."""
    df = load_5m_data(symbol)
    if df.empty:
        return None

    cfg = TICKER_CONFIG.get(symbol, {})
    entry_slip = cfg.get("entry_slip", 0.001)
    df["date"] = df.index.date

    signals = scan_signals(symbol)
    if not signals:
        return {"trades_a": 0, "trades_b": 0, "trades_c": 0}

    trades_a, trades_b, trades_c = [], [], []
    n_gap = sum(1 for s in signals if s["is_gap_up"])

    for sig in signals:
        day_df = df[df["date"] == sig["day"]]
        or_high = sig["or_high"]
        or_low = sig["or_low"]
        or_range = sig["or_range"]
        tp_price = or_high + ORB_PARAMS["target_r"] * or_range
        stop_price = or_low

        # ── Model A: OLD_BIASED (无视 gap, 全按 OR_High*1+slip 入场)
        entry_a = or_high * (1 + entry_slip)
        ex_a = simulate_exit(day_df, sig["entry_ts"], entry_a, stop_price, tp_price, cutoff_et)
        pnl_a = (ex_a["exit_p"] - entry_a) / entry_a * 100
        trades_a.append({"day": str(sig["day"]), "is_gap": sig["is_gap_up"],
                         "entry": entry_a, "exit": ex_a["exit_p"],
                         "result": ex_a["result"], "pnl_pct": pnl_a})

        # ── Model B: LIMIT_REAL (gap up = 不成交, skip)
        if not sig["is_gap_up"]:
            entry_b = or_high   # 限价单按 OR_High 成交, 无 slip (你挂的就是 OR_High)
            ex_b = simulate_exit(day_df, sig["entry_ts"], entry_b, stop_price, tp_price, cutoff_et)
            pnl_b = (ex_b["exit_p"] - entry_b) / entry_b * 100
            trades_b.append({"day": str(sig["day"]), "is_gap": False,
                             "entry": entry_b, "exit": ex_b["exit_p"],
                             "result": ex_b["result"], "pnl_pct": pnl_b})
        # else: 限价单 miss, 没有交易

        # ── Model C: MARKET_REAL (gap up = 按 bar.Open 入场)
        if sig["is_gap_up"]:
            entry_c = sig["entry_bar_open"] * (1 + entry_slip)
        else:
            entry_c = or_high * (1 + entry_slip)
        ex_c = simulate_exit(day_df, sig["entry_ts"], entry_c, stop_price, tp_price, cutoff_et)
        pnl_c = (ex_c["exit_p"] - entry_c) / entry_c * 100
        trades_c.append({"day": str(sig["day"]), "is_gap": sig["is_gap_up"],
                         "entry": entry_c, "exit": ex_c["exit_p"],
                         "result": ex_c["result"], "pnl_pct": pnl_c})

    def summary(trades, label):
        if not trades:
            return {"label": label, "trades": 0}
        td = pd.DataFrame(trades)
        n = len(td)
        pos = (td["pnl_pct"] > 0).sum()
        cum = (1 + td["pnl_pct"] / 100).prod() - 1
        win_pnl = td[td["pnl_pct"] > 0]["pnl_pct"].sum()
        loss_pnl = abs(td[td["pnl_pct"] < 0]["pnl_pct"].sum())
        pf = win_pnl / loss_pnl if loss_pnl > 0 else float("inf")
        equity = (1 + td["pnl_pct"] / 100).cumprod()
        dd = (equity / equity.cummax() - 1).min() * 100
        avg = td["pnl_pct"].mean()
        return {
            "label": label,
            "trades": n,
            "win_rate": round(pos / n * 100, 1),
            "cum_pct": round(cum * 100, 2),
            "pf": round(pf, 2) if pf != float("inf") else float("inf"),
            "avg_pnl": round(avg, 3),
            "max_dd": round(dd, 2),
        }

    return {
        "symbol": symbol,
        "n_signals": len(signals),
        "n_gap_up": n_gap,
        "gap_rate_pct": round(n_gap / len(signals) * 100, 1) if signals else 0,
        "model_a": summary(trades_a, "OLD_BIASED"),
        "model_b": summary(trades_b, "LIMIT_REAL"),
        "model_c": summary(trades_c, "MARKET_REAL"),
        "trades_a": trades_a,
        "trades_b": trades_b,
        "trades_c": trades_c,
    }


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="15:55", help="cutoff time HH:MM (ET)")
    ap.add_argument("--tickers", default="TOP9", choices=["TOP9", "SUITABLE", "ALL"])
    return ap.parse_args()


def get_ticker_list(mode: str) -> list:
    if mode == "TOP9":
        return TOP9_PORTFOLIO
    if mode == "SUITABLE":
        return [s for s, c in TICKER_CONFIG.items() if c.get("suitable_intraday")]
    return list(TICKER_CONFIG.keys())


def main():
    args = parse_args()
    h, m = map(int, args.cutoff.split(":"))
    cutoff_et = dtime(h, m)
    tickers = get_ticker_list(args.tickers)

    print("=" * 130)
    print(f"📊 ORB 回测 fill 模型对比诊断   cutoff={args.cutoff} ET   组合={args.tickers} ({len(tickers)} 只)")
    print("=" * 130)
    print()
    print(f"  A. OLD_BIASED  — 旧回测: 全部按 OR_High*(1+slip) 入场 (含 gap-up 幽灵成交)")
    print(f"  B. LIMIT_REAL  — 你实盘 LIT: bar.Low > OR_High 时不成交, 跳过")
    print(f"  C. MARKET_REAL — 保守市价: gap-up 按 bar.Open*(1+slip) 入场")
    print()

    header = f"{'标的':<10} {'信号':>5} {'gap':>4} {'gap率':>6} | "
    header += f"{'A_笔':>5} {'A_胜':>6} {'A_累计':>9} {'A_PF':>5} | "
    header += f"{'B_笔':>5} {'B_胜':>6} {'B_累计':>9} {'B_PF':>5} | "
    header += f"{'C_笔':>5} {'C_胜':>6} {'C_累计':>9} {'C_PF':>5} | "
    header += f"{'A→B 缩水':>10}"
    print(header)
    print("-" * len(header))

    rows_summary = []
    rows_trades = []
    agg = {"a": [], "b": [], "c": []}

    for sym in tickers:
        r = backtest_three_models(sym, cutoff_et)
        if r is None:
            print(f"{sym:<10}  数据缺失")
            continue
        if r["n_signals"] == 0:
            print(f"{sym:<10}  无信号")
            continue

        a, b, c = r["model_a"], r["model_b"], r["model_c"]
        a_cum = a.get("cum_pct", 0)
        b_cum = b.get("cum_pct", 0)
        shrink_pct = ((b_cum - a_cum) / abs(a_cum) * 100) if a_cum != 0 else 0

        line = f"{sym:<10} {r['n_signals']:>5} {r['n_gap_up']:>4} {r['gap_rate_pct']:>5.1f}% | "
        line += f"{a['trades']:>5} {a.get('win_rate', 0):>5.1f}% {a_cum:>+8.1f}% {str(a.get('pf', '-')):>5} | "
        line += f"{b['trades']:>5} {b.get('win_rate', 0):>5.1f}% {b_cum:>+8.1f}% {str(b.get('pf', '-')):>5} | "
        line += f"{c['trades']:>5} {c.get('win_rate', 0):>5.1f}% {c.get('cum_pct', 0):>+8.1f}% {str(c.get('pf', '-')):>5} | "
        line += f"{shrink_pct:>+9.1f}%"
        print(line)

        rows_summary.append({
            "symbol": sym,
            "signals": r["n_signals"],
            "n_gap_up": r["n_gap_up"],
            "gap_rate_pct": r["gap_rate_pct"],
            "A_trades": a["trades"], "A_win_rate": a.get("win_rate"), "A_cum_pct": a_cum, "A_pf": a.get("pf"), "A_max_dd": a.get("max_dd"),
            "B_trades": b["trades"], "B_win_rate": b.get("win_rate"), "B_cum_pct": b_cum, "B_pf": b.get("pf"), "B_max_dd": b.get("max_dd"),
            "C_trades": c["trades"], "C_win_rate": c.get("win_rate"), "C_cum_pct": c.get("cum_pct"), "C_pf": c.get("pf"), "C_max_dd": c.get("max_dd"),
            "A_to_B_shrink_pct": round(shrink_pct, 2),
        })

        # 收集每笔交易明细
        for t in r["trades_a"]:
            rows_trades.append({"symbol": sym, "model": "A_OLD", **t})
        for t in r["trades_b"]:
            rows_trades.append({"symbol": sym, "model": "B_LIMIT", **t})
        for t in r["trades_c"]:
            rows_trades.append({"symbol": sym, "model": "C_MARKET", **t})

        agg["a"] += r["trades_a"]
        agg["b"] += r["trades_b"]
        agg["c"] += r["trades_c"]

    # ─── 组合汇总 (等权 — 单笔 pnl_pct 简单平均) ───
    print("-" * len(header))

    def agg_summary(trades, label):
        if not trades:
            return f"  {label}: 无交易"
        td = pd.DataFrame(trades)
        n = len(td)
        pos = (td["pnl_pct"] > 0).sum()
        win_pnl = td[td["pnl_pct"] > 0]["pnl_pct"].sum()
        loss_pnl = abs(td[td["pnl_pct"] < 0]["pnl_pct"].sum())
        pf = win_pnl / loss_pnl if loss_pnl > 0 else float("inf")
        avg = td["pnl_pct"].mean()
        cum = (1 + td["pnl_pct"] / 100).prod() - 1
        return (f"  {label:<13} 总交易={n:>4}  胜率={pos/n*100:>5.1f}%  "
                f"PF={pf:>5.2f}  均笔={avg:>+5.2f}%  "
                f"等权累计={cum*100:>+8.1f}%")

    print()
    print("📈 全组合等权汇总:")
    print(agg_summary(agg["a"], "A_OLD_BIASED"))
    print(agg_summary(agg["b"], "B_LIMIT_REAL"))
    print(agg_summary(agg["c"], "C_MARKET_REAL"))

    # ── 关键结论计算 ──
    if agg["a"] and agg["b"]:
        a_cum = (1 + pd.DataFrame(agg["a"])["pnl_pct"] / 100).prod() - 1
        b_cum = (1 + pd.DataFrame(agg["b"])["pnl_pct"] / 100).prod() - 1
        n_phantom = len(agg["a"]) - len(agg["b"])
        print()
        print(f"🔑 核心发现:")
        print(f"   - 旧模型(A) 总信号 {len(agg['a'])} 笔, 其中 {n_phantom} 笔是 gap-up '幽灵成交' ({n_phantom/len(agg['a'])*100:.1f}%)")
        print(f"   - 这些幽灵成交在实盘限价单根本不会触发, 应该全部从盈亏中扣除")
        print(f"   - 累计收益: A={a_cum*100:+.1f}%  →  B={b_cum*100:+.1f}%   缩水 {(b_cum-a_cum)/abs(a_cum)*100:+.1f}%")

    # ─── 保存 CSV ───
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    if rows_summary:
        df_sum = pd.DataFrame(rows_summary)
        f1 = out_dir / "fill_bias_report.csv"
        df_sum.to_csv(f1, index=False)
        print(f"\n💾 汇总已保存: {f1}")

    if rows_trades:
        df_t = pd.DataFrame(rows_trades)
        f2 = out_dir / "fill_bias_trades.csv"
        df_t.to_csv(f2, index=False)
        print(f"💾 每笔交易已保存: {f2}")


if __name__ == "__main__":
    main()
