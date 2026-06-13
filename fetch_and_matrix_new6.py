"""
fetch_and_matrix_new6.py — 给 AAOI/ARM/MU/MRVL/DELL/ALAB 抓5m + 跑8策略矩阵找最优。

复用现有基础设施: fetch_history(分页拉LongPort 5m) + convert_one(转JSON) +
run_backtest/compute_metrics(fill-realistic回测) + STRATEGIES(8策略) + 评分/过滤。
输出 output/new6_strategy_matrix.csv(全48组合) + 控制台最优表。不动 best_strategy_per_ticker.csv(实盘池)。

用法: python3 fetch_and_matrix_new6.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

from longport.openapi import Config, QuoteContext
from longport_history_backtest import fetch_history
from data.save_intraday_data import convert_one, CACHE_DIR as CONV_CACHE
from backtest_engine import run_backtest, compute_metrics, load_5m_data, KLINES_5M_DIR
from signals.strategies.intraday_pool import STRATEGIES
from run_strategy_matrix import composite_score, passes_filter

NEW = ["AAOI.US", "ARM.US", "MU.US", "MRVL.US", "DELL.US", "ALAB.US"]
HIST_CACHE = Path(__file__).parent / "cache" / "longport_history"


def fetch_and_save(ctx):
    print("📥 抓 5m 数据 + 转 JSON")
    for sym in NEW:
        if load_5m_data(sym).shape[0] > 0:
            print(f"   {sym:<10} 已有JSON, 跳过抓取")
            continue
        df = fetch_history(ctx, sym)
        if df is None or df.empty:
            print(f"   {sym:<10} ❌ 无数据")
            continue
        pkl = HIST_CACHE / f"{sym.replace('.', '_')}_5m.pkl"
        payload = convert_one(pkl)
        if payload is None:
            print(f"   {sym:<10} ❌ 转换空")
            continue
        out = KLINES_5M_DIR / f"{sym}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        print(f"   {sym:<10} → {out.name} ({payload['bars']} 根, {payload['first'][:10]}→{payload['last'][:10]})")


def run_matrix():
    print("\n📊 跑 8 策略矩阵")
    rows = []
    for sym in NEW:
        df = load_5m_data(sym)
        if df.empty:
            print(f"   {sym:<10} ❌ 无数据, 跳过")
            continue
        data_end = df.index.max().date()
        ndays = len(df["date"].unique())
        print(f"   {sym:<10} ({len(df)} bars, {ndays} 天) ...")
        for sname, sfn in STRATEGIES.items():
            try:
                recs = run_backtest(sym, sfn, sname)
                m = compute_metrics(recs, data_end_date=data_end)
            except Exception as e:
                print(f"      {sname:<12} ERR {type(e).__name__}: {e}")
                continue
            pass_, why = passes_filter(m)
            rows.append({
                "symbol": sym, "strategy": sname,
                "filled": m.get("filled", 0), "phantom_rate_pct": m.get("phantom_rate_pct", 0),
                "win_rate": m.get("win_rate"), "pf": m.get("pf"), "cum_pct": m.get("cum_pct"),
                "max_dd": m.get("max_dd"), "avg_pnl": m.get("avg_pnl"),
                "recency_avg_pnl": m.get("recency_avg_pnl"), "recency_win_rate": m.get("recency_win_rate"),
                "recency_pf": m.get("recency_pf"),
                "last_90d_n": m.get("last_90d_n", 0), "last_90d_pf": m.get("last_90d_pf"),
                "last_90d_cum_pct": m.get("last_90d_cum_pct"),
                "last_180d_n": m.get("last_180d_n", 0), "last_180d_pf": m.get("last_180d_pf"),
                "last_180d_cum_pct": m.get("last_180d_cum_pct"),
                "n_tp": m.get("n_tp", 0), "n_stop": m.get("n_stop", 0),
                "composite_score": round(composite_score(m), 3),
                "passes_filter": pass_, "filter_fail_reasons": "|".join(why) if why else "",
            })
    df_all = pd.DataFrame(rows)
    out = Path(__file__).parent / "output" / "new6_strategy_matrix.csv"
    df_all.to_csv(out, index=False)
    print(f"\n💾 全矩阵 → {out}")

    print("\n" + "=" * 110)
    print("🏆 每只最优策略 (通过筛选 + 综合分最高)")
    print("=" * 110)
    print(f"{'票':9}{'最优策略':14}{'成交':>5}{'phantom':>8}{'胜率':>6}{'PF':>6}{'近期PF':>7}"
          f"{'近90累计':>9}{'18m累计':>9}{'综合分':>7}  判定")
    print("-" * 110)
    best_rows = []
    for sym in NEW:
        sub = df_all[df_all["symbol"] == sym]
        if sub.empty:
            print(f"{sym:9} 无数据"); continue
        ok = sub[sub["passes_filter"]]
        if len(ok):
            top = ok.sort_values("composite_score", ascending=False).iloc[0]
            verdict = "OK"
        else:
            top = sub.sort_values("composite_score", ascending=False).iloc[0]
            verdict = "FAIL_FILTER(全8策略未过筛)"
        best_rows.append({**top.to_dict(), "verdict": verdict})
        def f(v):
            return f"{v:.2f}" if isinstance(v, float) else str(v)
        print(f"{sym:9}{top['strategy']:14}{int(top['filled']):>5}{f(top['phantom_rate_pct'])+'%':>8}"
              f"{f(top['win_rate'])+'%':>6}{f(top['pf']):>6}{f(top['recency_pf']):>7}"
              f"{f(top['last_90d_cum_pct'])+'%':>9}{f(top['cum_pct'])+'%':>9}{f(top['composite_score']):>7}  {verdict}")
    pd.DataFrame(best_rows).to_csv(Path(__file__).parent / "output" / "new6_best_strategy.csv", index=False)
    print(f"\n💾 最优 → output/new6_best_strategy.csv")


def main():
    ctx = QuoteContext(Config.from_env())
    fetch_and_save(ctx)
    run_matrix()


if __name__ == "__main__":
    main()
