#!/usr/bin/env python3
"""shishi_10f_fullstack.py — 诗诗完整体系回测: 10F线(SMA20-RTH) + 四色柱(RSI6黃柱提前止盈)。

截图锚定成果 (2026-09-01, 用户提供9张富途截图):
  ① MA90表头读数两时点均以0.1-0.3分误差命中 RTH-only 口径 → 指标线不含盘前盘后, 实锤
  ② 四色柱 = RSI(6) TDX口径 80/20 穿越事件 (黃=下穿80警告, 藍=上穿20底部):
     8/31 13:40黃柱 vs 他13:41"四色柱要先賣出" (差1分钟)
     8/21 10:10黃柱 vs 他10:08"CONL 6.4先賣了" (此前唯一未解释的喊单)
     8/27 11:20黃柱@6.57 vs 截图1(11:29拍摄)顶部黄警
     8/19 12:30黃柱@4.92 vs 截图5顶部变色K线
  ③ 買/賣/G/S/預備買入/移動止盈 = David WM套装自动标记; TEST3變盤線源码缺, 无法建模
⚠️ 修正(09-01深夜): 四色定案=趋势(EMA8/20)×阴阳2×2(用户8/31实盘图逐根校准),
本文件的RSI黃柱止盈规则及其结论(-32%)随旧映射作废。
正确蓝柱语义(多头段阴线)机械止盈重测: CONL -41%→-70%(即卖)/-53%(2蓝);
SOXL 15分 -2%→-62%/-52% — 机械化"見藍就賣"全面更差, 该信号只在他
临场裁量(大涨后防回吐日)才有价值。机械跟单判决维持: 否。
"""
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from zoneinfo import ZoneInfo
from shishi_anchor_scan import load_bars, rth_only
from shishi_10f_v3 import sma, stats, backtest, COST

ET = ZoneInfo("America/New_York")


def rsi_tdx(closes, n=6):
    out, up, dn = [50.0], None, None
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        u, d = max(ch, 0), abs(min(ch, 0))
        up = u if up is None else (u + (n - 1) * up) / n
        dn = d if dn is None else (d + (n - 1) * dn) / n
        out.append(100.0 if dn == 0 else 100 * up / (up + dn))
    return out


def backtest_full(bars, n=20, rsin=6, need_pullback=True, warm=60):
    c = [b["c"] for b in bars]
    L = sma(c, n)
    R = rsi_tdx(c, rsin)
    trades, pos, pending_buy = [], None, False
    armed = True
    for i in range(warm, len(bars)):
        b = bars[i]
        stop = L[i - 1]
        if pos is not None:
            if b["o"] < stop:
                trades.append(dict(ein=pos["ts"], eout=b["ts"],
                                   pct=(b["o"] * (1 - COST) / pos["px"] - 1) * 100, why="缺口破线"))
                pos = None
            elif b["l"] < stop:
                trades.append(dict(ein=pos["ts"], eout=b["ts"],
                                   pct=(stop * (1 - COST) / pos["px"] - 1) * 100, why="盘中破线"))
                pos = None
        if pending_buy:
            if pos is None:
                pos = dict(px=b["o"] * (1 + COST), ts=b["ts"])
            pending_buy = False
        if pos is not None and R[i - 1] >= 80 > R[i] and i + 1 < len(bars):
            nb = bars[i + 1]
            trades.append(dict(ein=pos["ts"], eout=nb["ts"],
                               pct=(nb["o"] * (1 - COST) / pos["px"] - 1) * 100, why="黃柱止盈"))
            pos = None
            if need_pullback:
                armed = False
        if not armed and (b["l"] <= L[i] or max(b["o"], b["c"]) < L[i]):
            armed = True
        if pos is None and not pending_buy and armed and min(b["o"], b["c"]) > L[i]:
            pending_buy = True
    if pos is not None:
        trades.append(dict(ein=pos["ts"], eout=bars[-1]["ts"],
                           pct=(bars[-1]["c"] * (1 - COST) / pos["px"] - 1) * 100, why="期末"))
    return trades


def main():
    bars = rth_only(load_bars())
    base, _ = backtest(bars, 20)
    print(f"基线 纯10F(SMA20): {stats(base)}")
    for label, kw in [("O1 黃柱止盈+等回踩重入", dict(need_pullback=True)),
                      ("O1b 黃柱止盈+立即重入", dict(need_pullback=False))]:
        print(f"{label}: {stats(backtest_full(bars, **kw))}")


if __name__ == "__main__":
    main()
