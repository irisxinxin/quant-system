"""
backtest_leverage_compare.py — 验证"杠杆ETF不适合裸K"假设
对比同样参数在 SOXL (3x) / SOXX (1x底层) / NVDA (真龙头股) 上的表现
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_soxl_5m import fetch_intraday, backtest_intraday_pa

tickers = ["SOXL", "SOXX", "NVDA", "SMH"]
results = {}

print("=" * 70)
print("裸K战法 5分钟 — 杠杆ETF vs 底层指数 vs 龙头股 对比")
print("=" * 70)

for t in tickers:
    df = fetch_intraday(t, period="60d", interval="5m")
    if df.empty:
        print(f"\n{t}: 无数据")
        continue
    result = backtest_intraday_pa(
        df.copy(),
        long_only=True, min_rr=2.0, max_wait_bars=3,
    )
    results[t] = result
    m = result["metrics"]
    print(f"\n── {t} ({len(df)} 根5m) ──")
    for k in ["区间", "总收益率", "年化收益", "最大回撤", "Calmar",
              "交易次数", "胜率", "平均盈利", "平均亏损", "盈亏比"]:
        print(f"  {k:10} {m[k]}")

# 汇总表
print("\n" + "=" * 70)
print("汇总（多头only, R:R≥2, 最长等待3根）")
print("=" * 70)
print(f"{'Ticker':<8} {'总收益':>10} {'胜率':>8} {'盈亏比':>8} {'交易数':>8} {'Calmar':>8}")
print("-" * 70)
for t, r in results.items():
    m = r["metrics"]
    print(f"{t:<8} {m['总收益率']:>10} {m['胜率']:>8} {m['盈亏比']:>8} {m['交易次数']:>8} {m['Calmar']:>8}")
