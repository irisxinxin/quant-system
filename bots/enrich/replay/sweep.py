#!/usr/bin/env python3
"""replay/sweep.py — 用回放当回测引擎: 同一真实时间线 × 不同策略配置对比。
引擎=bot真实代码(不是另写引擎), 每个配置只patch bot常量 → 出场逻辑始终是bot本身, 无分叉。
用法: DISCORD_BOT_TOKEN=x /usr/local/bin/python3 bots/enrich/replay/sweep.py oracle_260401_260731.pkl 2026-04-01 2026-07-31
"""
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import Replay

oracle = sys.argv[1] if len(sys.argv) > 1 else "oracle_260401_260731.pkl"
lo = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 4, 1)
hi = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2026, 7, 31)
op = str(Path(__file__).resolve().parent / oracle)

# 三个配置只改 bot 常量, 出场逻辑始终跑 bot 真实代码
CONFIGS = {
    "F阶梯(现行线上)":  dict(MECH_BE=False, F_EOD_CLOSE_UNREDUCED=True),
    "保本旧(07-21定案)": dict(MECH_BE=True,  F_EOD_CLOSE_UNREDUCED=False),
    "B全平不过夜":      dict(MECH_BE=False, F_EOD_CLOSE_UNREDUCED=True, F_EOD_CLOSE_BELOW=9.99),
}

rows = []
for name, cfg in CONFIGS.items():
    r = Replay(op, config=cfg).run(lo, hi)
    p = r.portfolio()
    rows.append((name, p))

print(f"\n{'='*78}")
print(f"回放当回测: 同一真实时间线({lo}~{hi}) × 3配置 (引擎=bot真实代码)")
print(f"{'='*78}")
hdr = f"{'配置':18}{'n':>4}{'均值':>7}{'中位':>7}{'胜率':>7}{'过夜':>6}{'止损':>6}{'最差':>7}{'违规':>6}{'异常':>6}"
print(hdr)
for name, p in rows:
    a = p["全部"]
    print(f"{name:18}{a['n']:>4}{a['均值']:>+6}%{a['中位']:>+6}%{a['胜率']:>6}%{a['过夜']:>6}{a['止损']:>6}{a['最差']:>+6}%{p['违规']:>6}{p['异常']:>6}")
print(f"\n真实K子集(仅7月18单, 幅度可信):")
for name, p in rows:
    rk = p.get("真实K", {})
    if rk:
        print(f"  {name:18} n={rk['n']} 均值{rk['均值']:+}% 中位{rk['中位']:+}% 胜率{rk['胜率']}% 过夜{rk['过夜']}")
print(f"\n⚠ 5-6月大部分是BS(幅度失真, 只看相对排序/结构); 真实K仅7月18单; 5分K抓不到sub-5min针尖。")
print("  引擎=bot本身 → 出场逻辑零分叉(vs bt_be_grid另写引擎缺武装门控那种坑)。")
