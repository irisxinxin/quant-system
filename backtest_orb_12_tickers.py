"""
backtest_orb_12_tickers.py — 12 只股票完整测试（Long-only + 真实滑点）
股票池: OKLO INTC IREN CIFR PLTR NBIS SOXL AMD AMZN TSLL NVDL EOSE
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_soxl_5m import fetch_intraday
from backtest_orb_validated import orb_backtest_v2, buy_and_hold_return

# 每只的滑点（按市值/流动性分档）
TICKERS = [
    # (ticker, entry_slip, stop_slip, 说明)
    ("OKLO", 0.0015, 0.0030, "核能小盘"),
    ("INTC", 0.0003, 0.0008, "半导体大盘(巨量流动)"),
    ("IREN", 0.0015, 0.0030, "BTC挖矿小盘"),
    ("CIFR", 0.0015, 0.0030, "BTC挖矿小盘"),
    ("PLTR", 0.0003, 0.0008, "AI大盘(日成交50M+)"),
    ("NBIS", 0.0010, 0.0020, "AI基建中盘"),
    ("SOXL", 0.0003, 0.0008, "3x半导体ETF"),
    ("AMD",  0.0003, 0.0008, "半导体大盘"),
    ("AMZN", 0.0002, 0.0005, "超大盘巨量"),
    ("TSLL", 0.0005, 0.0010, "2x特斯拉ETF"),
    ("NVDL", 0.0005, 0.0010, "2x英伟达ETF"),
    ("EOSE", 0.0025, 0.0050, "储能超小盘(低流动)"),
]

base_params = {
    "or_bars": 3,
    "target_r": 2.0,
    "use_volume_filter": True,
    "rvol_threshold": 1.5,
    "commission": 0.0,
    "normal_slip": 0.0005,
    "long_only": True,  # 用户不能做空
}

print("=" * 95)
print("ORB 只做多回测 | 12 标的 | 真实滑点 (entry+stop按流动性分档) | 60天5m")
print("=" * 95)

results = {}
failed = []
for t, e_slip, s_slip, note in TICKERS:
    try:
        df = fetch_intraday(t, period="60d", interval="5m")
        if df.empty or len(df) < 500:
            failed.append((t, f"数据不足 {len(df)} 根"))
            continue
        res = orb_backtest_v2(df.copy(), entry_slip=e_slip, stop_slip=s_slip, **base_params)
        bh = buy_and_hold_return(df) * 100
        results[t] = {"res": res, "bh": bh, "note": note, "bars": len(df)}
    except Exception as e:
        failed.append((t, str(e)[:60]))

# 按总收益降序打印
print(f"\n{'排名':<4} {'Ticker':<7} {'类型':<20} {'策略收益':>9} {'B&H':>8} "
      f"{'差值':>9} {'交易':>5} {'胜率':>7} {'盈亏比':>7} {'Calmar':>7}")
print("-" * 95)

sorted_results = sorted(results.items(),
                         key=lambda x: float(x[1]["res"]["metrics"]["总收益率"].rstrip('%')),
                         reverse=True)

for rank, (t, data) in enumerate(sorted_results, 1):
    m = data["res"]["metrics"]
    strat_ret = float(m["总收益率"].rstrip('%'))
    bh = data["bh"]
    diff = strat_ret - bh
    diff_sign = "+" if diff >= 0 else ""
    note = data["note"]

    print(f"{rank:<4} {t:<7} {note:<20} {strat_ret:>+8.2f}% {bh:>+7.2f}% "
          f"{diff_sign}{diff:>7.2f}% {m['交易次数']:>5} {m['真实胜率']:>7} "
          f"{m['盈亏比']:>7} {m['Calmar']:>7}")

if failed:
    print("\n⚠️ 失败/数据不足:")
    for t, e in failed:
        print(f"   {t}: {e}")

# 分组总结
print(f"\n{'=' * 95}")
print("【分组总结】")
print("=" * 95)

top = [(t, d) for t, d in sorted_results
       if float(d["res"]["metrics"]["总收益率"].rstrip('%')) > 30
       and float(d["res"]["metrics"]["总收益率"].rstrip('%')) > d["bh"]]
ok = [(t, d) for t, d in sorted_results
      if 10 <= float(d["res"]["metrics"]["总收益率"].rstrip('%')) <= 30
      and float(d["res"]["metrics"]["总收益率"].rstrip('%')) > d["bh"]]
bad = [(t, d) for t, d in sorted_results
       if float(d["res"]["metrics"]["总收益率"].rstrip('%')) < 10
       or float(d["res"]["metrics"]["总收益率"].rstrip('%')) <= d["bh"]]

print(f"\n🎯 强推 (>30% 且胜 B&H): {[t for t, _ in top]}")
print(f"✅ 可做 (10-30% 且胜 B&H): {[t for t, _ in ok]}")
print(f"❌ 不建议 (<10% 或输 B&H): {[t for t, _ in bad]}")
