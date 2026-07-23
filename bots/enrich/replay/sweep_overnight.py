#!/usr/bin/env python3
"""replay/sweep_overnight.py — 用回放定"runner 该留多少过夜"。
未落袋满仓已一律平(reduced守卫), F_EOD_CLOSE_BELOW 就是"已落袋runner留过夜的门槛"这根轴。
5档: 全不留 → 只留强 → F现行 → 留所有盈利 → 全留不砍。引擎=bot真实代码。
用法: DISCORD_BOT_TOKEN=x /usr/local/bin/python3 bots/enrich/replay/sweep_overnight.py oracle_260201_260731.pkl 2026-02-01 2026-07-31
"""
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import Replay

oracle = sys.argv[1] if len(sys.argv) > 1 else "oracle_260201_260731.pkl"
lo = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 2, 1)
hi = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2026, 7, 31)
op = str(Path(__file__).resolve().parent / oracle)

BASE = dict(MECH_BE=False, F_EOD_CLOSE_UNREDUCED=True)   # 共同: 未落袋满仓一律平
# 只调 runner 过夜门槛 (CLOSE_BELOW) 和砍半门槛 (TRIM_ABOVE)
CONFIGS = {
    "B 全不留过夜":        dict(F_EOD_CLOSE_BELOW=9.99,  F_EOD_TRIM_ABOVE=9.99),   # runner也收盘平
    "只留强runner(≥50)":  dict(F_EOD_CLOSE_BELOW=0.50,  F_EOD_TRIM_ABOVE=0.50),   # <50平, ≥50砍半留
    "F现行(30-50留/≥50砍": dict(F_EOD_CLOSE_BELOW=0.30,  F_EOD_TRIM_ABOVE=0.50),   # 线上
    "留所有盈利(≥0)":      dict(F_EOD_CLOSE_BELOW=0.0,   F_EOD_TRIM_ABOVE=0.50),   # 只砍亏损runner
    "全留不砍(reduced全过夜)": dict(F_EOD_CLOSE_BELOW=-9.99, F_EOD_TRIM_ABOVE=9.99),  # runner一律扛过夜
}

rows = []
for name, extra in CONFIGS.items():
    cfg = dict(BASE); cfg.update(extra)
    p = Replay(op, config=cfg).run(lo, hi).portfolio()
    rows.append((name, p))

print(f"\n{'='*80}\n该留多少过夜: 同一真实时间线({lo}~{hi}, 6个月) × runner过夜门槛 (引擎=bot真实代码)\n{'='*80}")
print(f"{'配置':22}{'n':>4}{'均值':>7}{'中位':>7}{'胜率':>7}{'过夜':>6}{'最差':>7}")
for name, p in rows:
    a = p["全部"]
    print(f"{name:22}{a['n']:>4}{a['均值']:>+6}%{a['中位']:>+6}%{a['胜率']:>6}%{a['过夜']:>6}{a['最差']:>+6}%")
print(f"\n真实K子集(仅7月18单, 幅度可信):")
for name, p in rows:
    rk = p.get("真实K", {})
    if rk:
        print(f"  {name:22} n={rk['n']} 均值{rk['均值']:+}% 中位{rk['中位']:+}% 胜率{rk['胜率']}% 过夜{rk['过夜']} 最差{rk['最差']:+}%")
print(f"\n⚠ 6个月共{rows[0][1]['全部']['n']}单但真实K仅7月18单; 5-6月及以前全BS(幅度失真, 只看排序);")
print("  5分K无sub-5min针尖。结论=相对排序可信, 绝对幅度存疑。")
