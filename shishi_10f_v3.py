#!/usr/bin/env python3
"""shishi_10f_v3.py — 诗诗 CONL 10F 法【锚定版真规则】回测 (2026-03-02 ~ 08-31)。

构造逆向定案 (2026-09-01, 三处独立线值读数拟合):
  图 = 10分钟 RTH-only K线 (盘前盘后不进均线, 线值隔夜/周末冻结)
  线 = SMA(N) 收盘价简单均线, N=18~20 不可分 (平台默认MA20最可能)
  读数核验: 8/27 13:20破线卖 线值6.58 (sma18=6.585/sma20=6.575);
            8/31 09:37 "剛剛5.56" 周末冻结线值 (sma18=5.578/sma20=5.577);
            8/25 09:58 "10F空頭" 状态✓; 8/28 12:22 "沒有金叉"✓
规则 (按他原话镜像):
  买: bar实体完全站上线 (min(O,C)>L) → 次bar开盘买入   ("要X之上的實體")
  卖: 盘中价格跌破线 → 以线价成交, 缺口按开盘价        ("10f在6.58賣出", 不等收盘)
      止损线用上一完成bar的线值 (无未来函数)
  隔夜持仓允许 (他8/26金叉持到8/27破线)
成本 0.1%/边。对照: 买入持有 + 逐月分解 + 8月下旬信号 vs 他实盘喊单。
"""
import csv
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from zoneinfo import ZoneInfo
import shishi_10f_backtest as S
from shishi_anchor_scan import load_bars, rth_only

ET = ZoneInfo("America/New_York")
COST = 0.001


def sma(v, n):
    out, s = [], 0.0
    for i, x in enumerate(v):
        s += x
        if i >= n:
            s -= v[i - n]
        out.append(s / min(i + 1, n))
    return out


def backtest(bars, n, warm=60):
    L = sma([b["c"] for b in bars], n)
    trades = []
    pos = None          # dict(entry_px, entry_ts)
    pending_buy = False
    for i in range(warm, len(bars)):
        b = bars[i]
        stop = L[i - 1]                       # 上一完成bar的线值 = 盘中止损线
        if pos is not None:
            if b["o"] < stop:                 # 缺口低开直接开盘卖
                ex = b["o"] * (1 - COST)
                trades.append(dict(ein=pos["ts"], eout=b["ts"], epx=pos["px"], xpx=ex,
                                   pct=(ex / pos["px"] - 1) * 100))
                pos = None
            elif b["l"] < stop:               # 盘中破线, 按线价成交
                ex = stop * (1 - COST)
                trades.append(dict(ein=pos["ts"], eout=b["ts"], epx=pos["px"], xpx=ex,
                                   pct=(ex / pos["px"] - 1) * 100))
                pos = None
        if pending_buy:
            if pos is None:
                pos = dict(px=b["o"] * (1 + COST), ts=b["ts"])
            pending_buy = False
        # bar收盘后的信号判定 (次bar执行)
        body_lo = min(b["o"], b["c"])
        if pos is None and not pending_buy and body_lo > L[i]:
            pending_buy = True
    if pos is not None:
        ex = bars[-1]["c"] * (1 - COST)
        trades.append(dict(ein=pos["ts"], eout=bars[-1]["ts"], epx=pos["px"], xpx=ex,
                           pct=(ex / pos["px"] - 1) * 100))
    return trades, L


def stats(trades):
    if not trades:
        return "无交易"
    vals = [t["pct"] for t in trades]
    w = sum(1 for v in vals if v > 0)
    eq, peak, mdd = 1.0, 1.0, 0.0
    for v in vals:
        eq *= 1 + v / 100
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    monthly = {}
    for t in trades:
        m = t["eout"].astimezone(ET).strftime("%m")
        monthly.setdefault(m, 0.0)
        monthly[m] += t["pct"]
    ms = " ".join(f"{m}月{v:+.0f}%" for m, v in sorted(monthly.items()))
    return (f"{len(vals)}笔 胜{w}({w / len(vals) * 100:.0f}%) 均{sum(vals) / len(vals):+.2f}%/笔 "
            f"复利{(eq - 1) * 100:+.0f}% 最大回撤{mdd * 100:.0f}% | {ms}")


def main():
    bars = rth_only(load_bars())
    print(f"RTH 10分K {len(bars)}根 ({bars[0]['ts'].astimezone(ET):%m-%d} ~ {bars[-1]['ts'].astimezone(ET):%m-%d})")
    bh = (bars[-1]["c"] / bars[0]["o"] - 1) * 100
    print(f"买入持有: {bh:+.1f}%\n")
    for n in (18, 20):
        trades, L = backtest(bars, n)
        print(f"SMA({n}) 实体买/破线卖:\n  {stats(trades)}")
    print("\n== SMA(20) 8/20-8/31 交易明细 (对照他的实盘喊单) ==")
    trades, L = backtest(bars, 20)
    for t in trades:
        te = t["eout"].astimezone(ET)
        if te >= datetime(2026, 8, 20, tzinfo=ET):
            print(f"  入 {t['ein'].astimezone(ET):%m-%d %H:%M} @{t['epx']:.2f} → "
                  f"出 {te:%m-%d %H:%M} @{t['xpx']:.2f}  {t['pct']:+.1f}%")


if __name__ == "__main__":
    main()
