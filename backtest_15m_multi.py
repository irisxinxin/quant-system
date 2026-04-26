"""
backtest_15m_multi.py — 裸K战法在 15分钟 级别的多标的测试
对比用户原始股票池：OKLO/IREN/HOOD/NVDL/TSLL/CONL + 参照物 SOXL/NVDA
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_soxl_5m import fetch_intraday, backtest_intraday_pa

# 用户原始想做的股票 + 参照物
tickers = [
    # 用户列表
    "OKLO", "IREN", "HOOD", "NVDL", "TSLL", "CONL",
    # 参照物
    "SOXL", "NVDA",
]

print("=" * 80)
print("裸K战法 15分钟 级别多标的回测（R:R≥2, 多头only, 最长等待3根）")
print("=" * 80)

results = {}
for t in tickers:
    try:
        df = fetch_intraday(t, period="60d", interval="15m")
    except Exception as e:
        print(f"\n{t}: 下载失败 {e}")
        continue
    if df.empty or len(df) < 200:
        print(f"\n{t}: 数据不足 ({len(df)} 根)")
        continue

    result = backtest_intraday_pa(
        df.copy(),
        long_only=True,
        min_rr=2.0,
        max_wait_bars=3,        # 15m × 3 = 45分钟等待
        swing_lookback=20,       # 20根 = 5小时 ≈ 1.3 交易日
        skip_bars_open=2,        # 开盘前30分钟跳过
        skip_bars_close=2,       # 收盘前30分钟跳过
        no_overnight=True,
    )
    results[t] = result
    m = result["metrics"]
    print(f"\n── {t} ({len(df)} 根15m) ──")
    for k in ["区间", "总收益率", "年化收益", "最大回撤", "Calmar",
              "交易次数", "止盈次数", "止损次数", "日末平仓",
              "胜率", "平均盈利", "平均亏损", "盈亏比"]:
        print(f"  {k:10} {m[k]}")

# 汇总
print("\n" + "=" * 80)
print("汇总（15m, 多头only, R:R≥2）")
print("=" * 80)
print(f"{'Ticker':<8} {'总收益':>10} {'胜率':>8} {'盈亏比':>8} {'交易数':>8} {'Calmar':>8} {'回撤':>8}")
print("-" * 80)
# 按总收益降序
sorted_items = sorted(
    results.items(),
    key=lambda x: float(x[1]['metrics']['总收益率'].rstrip('%')),
    reverse=True,
)
for t, r in sorted_items:
    m = r["metrics"]
    print(f"{t:<8} {m['总收益率']:>10} {m['胜率']:>8} {m['盈亏比']:>8} "
          f"{m['交易次数']:>8} {m['Calmar']:>8} {m['最大回撤']:>8}")

# 找到盈利的标的，打印交易样本
print("\n" + "=" * 80)
profitable = [(t, r) for t, r in results.items()
              if float(r['metrics']['总收益率'].rstrip('%')) > 0]
if profitable:
    print(f"盈利标的: {[t for t, _ in profitable]}")
    for t, r in profitable:
        print(f"\n── {t} 前10笔交易 ──")
        if not r["trades"].empty:
            print(r["trades"].head(10).to_string())
else:
    print("⚠️ 所有标的均为亏损")
    # 打印交易数最多的样本
    best = max(results.items(),
               key=lambda x: x[1]['metrics'].get('交易次数', 0) or 0)
    print(f"\n交易数最多的样本 ({best[0]}) 前10笔:")
    if not best[1]["trades"].empty:
        print(best[1]["trades"].head(10).to_string())
