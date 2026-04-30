"""
run_strategy_matrix.py — 跑 20 标的 × 8 策略 矩阵, 找每只股票最优策略

筛选标准 (用户要求"近期权重最重"):
  - filled >= MIN_TRADES (样本足够)
  - recency_pf >= MIN_PF (近期至少不亏)
  - max_dd >= MIN_DD (回撤可控)
  通过过滤的按综合分数降序排, 每只股票取最优.

综合分数公式:
  score = 0.5 * recency_avg_pnl + 0.3 * last_180d_avg + 0.2 * full_avg
  (近期权重 50%, 半年权重 30%, 全周期 20%)

输出:
  output/strategy_matrix.csv         — 全部 160 组合的指标
  output/best_strategy_per_ticker.csv — 每只股票的最优策略
  控制台: 排名表 + sanity 红旗汇总
"""
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from backtest_engine import (
    run_backtest, compute_metrics, check_sanity, TICKERS_FULL, load_5m_data,
)
from signals.strategies.intraday_pool import STRATEGIES


# ─── 筛选门槛 ───────────────────────────────────────
MIN_TRADES = 20         # 至少 20 笔成交
MIN_RECENCY_PF = 1.1    # 近期 PF (放宽到 1.1, 否则筛掉太多)
MIN_LAST_180D_PF = 1.0  # 近 6 月至少不亏
MAX_DRAWDOWN = -50      # 最大回撤不超过 -50%

# 综合分权重
W_RECENCY = 0.5
W_180D = 0.3
W_FULL = 0.2


def composite_score(m: dict) -> float:
    """综合分数 — 越高越好"""
    if m.get("filled", 0) == 0:
        return -999
    rec_avg = m.get("recency_avg_pnl", 0) or 0
    half_avg = 0
    if m.get("last_180d_n", 0) > 0:
        # avg pnl from last_180d_cum_pct? 不准, 用 cum/n 近似
        half_avg = m.get("last_180d_cum_pct", 0) / max(m.get("last_180d_n", 1), 1)
    full_avg = m.get("avg_pnl", 0) or 0
    return W_RECENCY * rec_avg + W_180D * half_avg + W_FULL * full_avg


def passes_filter(m: dict) -> tuple:
    """返回 (是否通过, 不通过原因列表)"""
    reasons = []
    if m.get("filled", 0) < MIN_TRADES:
        reasons.append(f"成交<{MIN_TRADES}")
    rec_pf = m.get("recency_pf")
    if rec_pf == float("inf") or rec_pf is None:
        rec_pf = 999 if rec_pf == float("inf") else 0
    if isinstance(rec_pf, (int, float)) and rec_pf < MIN_RECENCY_PF:
        reasons.append(f"近期PF<{MIN_RECENCY_PF}")
    last180_pf = m.get("last_180d_pf")
    if last180_pf == float("inf"):
        last180_pf = 999
    if last180_pf is not None and isinstance(last180_pf, (int, float)) and last180_pf < MIN_LAST_180D_PF:
        reasons.append(f"近半年PF<{MIN_LAST_180D_PF}")
    if m.get("max_dd", 0) < MAX_DRAWDOWN:
        reasons.append(f"DD<{MAX_DRAWDOWN}%")
    return (len(reasons) == 0, reasons)


def fmt(v, default="-"):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    if v == float("inf"):
        return "Inf"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main():
    rows = []
    sanity_flags = []

    print("=" * 130)
    print(f"📊 策略矩阵跑分: {len(TICKERS_FULL)} 标的 × {len(STRATEGIES)} 策略 = "
          f"{len(TICKERS_FULL) * len(STRATEGIES)} 组合")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   近期权重 half-life=90d, 综合分=0.5×rec + 0.3×180d + 0.2×full")
    print("=" * 130)
    print()

    for sym_idx, sym in enumerate(TICKERS_FULL, 1):
        df = load_5m_data(sym)
        if df.empty:
            print(f"[{sym_idx:>2}/{len(TICKERS_FULL)}] {sym:<12} ❌ 数据缺失")
            continue
        n_days = len(df["date"].unique())
        print(f"[{sym_idx:>2}/{len(TICKERS_FULL)}] {sym:<12} ({len(df):>5} bars, {n_days} 天)")

        for strat_name, strat_fn in STRATEGIES.items():
            try:
                recs = run_backtest(sym, strat_fn, strat_name)
                m = compute_metrics(recs)
            except Exception as e:
                print(f"   {strat_name:<12} ERROR: {type(e).__name__}: {e}")
                continue

            score = composite_score(m)
            pass_, why = passes_filter(m)
            row = {
                "symbol": sym,
                "strategy": strat_name,
                "filled": m.get("filled", 0),
                "phantom_rate_pct": m.get("phantom_rate_pct", 0),
                "win_rate": m.get("win_rate"),
                "pf": m.get("pf"),
                "cum_pct": m.get("cum_pct"),
                "max_dd": m.get("max_dd"),
                "avg_pnl": m.get("avg_pnl"),
                "recency_avg_pnl": m.get("recency_avg_pnl"),
                "recency_win_rate": m.get("recency_win_rate"),
                "recency_pf": m.get("recency_pf"),
                "last_90d_n": m.get("last_90d_n", 0),
                "last_90d_win": m.get("last_90d_win"),
                "last_90d_pf": m.get("last_90d_pf"),
                "last_90d_cum_pct": m.get("last_90d_cum_pct"),
                "last_180d_n": m.get("last_180d_n", 0),
                "last_180d_win": m.get("last_180d_win"),
                "last_180d_pf": m.get("last_180d_pf"),
                "last_180d_cum_pct": m.get("last_180d_cum_pct"),
                "n_tp": m.get("n_tp", 0),
                "n_stop": m.get("n_stop", 0),
                "composite_score": round(score, 3),
                "passes_filter": pass_,
                "filter_fail_reasons": "|".join(why) if why else "",
            }
            rows.append(row)

            # sanity check
            flags = check_sanity(m, sym, strat_name)
            sanity_flags.extend(flags)

    df_all = pd.DataFrame(rows)
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    f1 = out_dir / "strategy_matrix.csv"
    df_all.to_csv(f1, index=False)
    print(f"\n💾 全矩阵已保存: {f1}")

    # ─── 每只股票最优策略 ───
    print("\n" + "=" * 130)
    print("🏆 每只股票最优策略 (通过筛选 + 综合分最高)")
    print("=" * 130)

    best_rows = []
    print(f"\n{'标的':<10} {'最优策略':<12} {'成交':>5} {'phantom':>7} "
          f"{'胜率':>5} {'近期胜':>6} {'近期PF':>7} {'近90累计':>9} {'近180累计':>10} {'18m累计':>9} {'综合分':>7}")
    print("-" * 130)

    for sym in TICKERS_FULL:
        sub = df_all[(df_all["symbol"] == sym) & (df_all["passes_filter"])]
        if len(sub) == 0:
            print(f"{sym:<10}  ❌ 全部 8 策略未通过筛选")
            # 即便没通过, 也记录"次优"作为参考
            sub_all = df_all[df_all["symbol"] == sym].sort_values("composite_score", ascending=False)
            if len(sub_all) > 0:
                top = sub_all.iloc[0]
                best_rows.append({**top.to_dict(), "verdict": "FAIL_FILTER"})
            continue
        top = sub.sort_values("composite_score", ascending=False).iloc[0]
        best_rows.append({**top.to_dict(), "verdict": "OK"})
        print(f"{sym:<10} {top['strategy']:<12} {top['filled']:>5} "
              f"{fmt(top['phantom_rate_pct'])+'%':>7} "
              f"{fmt(top['win_rate'])+'%':>5} "
              f"{fmt(top['recency_win_rate'])+'%':>6} "
              f"{fmt(top['recency_pf']):>7} "
              f"{fmt(top['last_90d_cum_pct'])+'%':>9} "
              f"{fmt(top['last_180d_cum_pct'])+'%':>10} "
              f"{fmt(top['cum_pct'])+'%':>9} "
              f"{fmt(top['composite_score']):>7}")

    df_best = pd.DataFrame(best_rows)
    f2 = out_dir / "best_strategy_per_ticker.csv"
    df_best.to_csv(f2, index=False)
    print(f"\n💾 最优策略已保存: {f2}")

    # ─── Sanity Flag 汇总 ───
    if sanity_flags:
        print("\n" + "=" * 130)
        print(f"⚠️ Sanity Check 红旗 ({len(sanity_flags)} 条) — 这些组合可能仍有问题:")
        print("=" * 130)
        for flag in sanity_flags[:30]:
            print(flag)
        if len(sanity_flags) > 30:
            print(f"  ... 还有 {len(sanity_flags) - 30} 条 (查 CSV)")

    # ─── 汇总统计 ───
    print("\n" + "=" * 130)
    print("📈 矩阵汇总")
    print("=" * 130)
    by_strat = df_all.groupby("strategy").agg(
        通过数=("passes_filter", "sum"),
        平均成交=("filled", "mean"),
        平均近期PF=("recency_pf", lambda x: x[x != float("inf")].mean()),
        平均最大回撤=("max_dd", "mean"),
    ).round(2)
    print("\n各策略通过情况:")
    print(by_strat.to_string())

    pass_count = df_best[df_best["verdict"] == "OK"].shape[0] if not df_best.empty else 0
    print(f"\n通过筛选的标的: {pass_count}/{len(TICKERS_FULL)}")


if __name__ == "__main__":
    main()
