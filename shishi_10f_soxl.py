#!/usr/bin/env python3
"""shishi_10f_soxl.py — 诗诗10F体系在SOXL上的回测 + MA90趋势闸门 (2026-03-02~08-31)。

结论(2026-09-01):
  SOXL买入持有+94.1%(半导体单边牛), 任何日内进出版本都大幅跑输;
  15分图优于10分图(与诗诗"SOXL用15f"的选择一致);
  最优组合 15分纯10F+MA90闸门 = +27%/MDD38%, +黃柱版 +10%/MDD28%;
  MA90闸门在CONL上 -46%→-11%, 方向对但依旧转不正。
  "只在大趋势入"判决: 闸门显著少亏(震荡月亏损全线收窄)但不产生alpha,
  大肉月(4月)本来就是闸门全开的月份。
"""
import csv, sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from zoneinfo import ZoneInfo
from shishi_anchor_scan import rth_only
from shishi_10f_v3 import sma, stats, COST
from shishi_10f_fullstack import rsi_tdx

ET = ZoneInfo("America/New_York")


def load_csv(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(ts=datetime.fromisoformat(r["ts"]), o=float(r["o"]), h=float(r["h"]),
                             l=float(r["l"]), c=float(r["c"])))
    return sorted(rows, key=lambda x: x["ts"])


def to_nmin(b5, nmin):
    out = {}
    for b in b5:
        t = b["ts"].astimezone(ET)
        key = t.replace(minute=(t.minute // nmin) * nmin, second=0, microsecond=0)
        if key not in out:
            out[key] = dict(ts=key, o=b["o"], h=b["h"], l=b["l"], c=b["c"])
        else:
            x = out[key]
            x["h"] = max(x["h"], b["h"]); x["l"] = min(x["l"], b["l"]); x["c"] = b["c"]
    return [out[k] for k in sorted(out)]


def backtest_gate(bars, n=20, rsin=6, yellow=True, gate_n=None, warm=100):
    c = [b["c"] for b in bars]
    L = sma(c, n)
    G = sma(c, gate_n) if gate_n else None
    R = rsi_tdx(c, rsin)
    trades, pos, pend, armed = [], None, False, True
    for i in range(warm, len(bars)):
        b = bars[i]
        stop = L[i - 1]
        if pos is not None:
            if b["o"] < stop:
                trades.append(dict(ein=pos["ts"], eout=b["ts"], pct=(b["o"] * (1 - COST) / pos["px"] - 1) * 100))
                pos = None
            elif b["l"] < stop:
                trades.append(dict(ein=pos["ts"], eout=b["ts"], pct=(stop * (1 - COST) / pos["px"] - 1) * 100))
                pos = None
        if pend:
            if pos is None:
                pos = dict(px=b["o"] * (1 + COST), ts=b["ts"])
            pend = False
        if yellow and pos is not None and R[i - 1] >= 80 > R[i] and i + 1 < len(bars):
            nb = bars[i + 1]
            trades.append(dict(ein=pos["ts"], eout=nb["ts"], pct=(nb["o"] * (1 - COST) / pos["px"] - 1) * 100))
            pos = None; armed = False
        if not armed and (b["l"] <= L[i] or max(b["o"], b["c"]) < L[i]):
            armed = True
        ok_gate = True if G is None else (b["c"] > G[i])
        if pos is None and not pend and armed and ok_gate and min(b["o"], b["c"]) > L[i]:
            pend = True
    if pos is not None:
        trades.append(dict(ein=pos["ts"], eout=bars[-1]["ts"], pct=(bars[-1]["c"] * (1 - COST) / pos["px"] - 1) * 100))
    return trades


def main():
    b5 = load_csv("data/shishi_bars/SOXL_5m_ALL.csv")
    for nmin in (10, 15):
        bars = rth_only(to_nmin(b5, nmin))
        bh = (bars[-1]["c"] / bars[0]["o"] - 1) * 100
        print(f"== SOXL {nmin}分 RTH (买入持有 {bh:+.1f}%) ==")
        print(f"  纯10F(SMA20):    {stats(backtest_gate(bars, yellow=False))}")
        print(f"  +黃柱止盈:       {stats(backtest_gate(bars, yellow=True))}")
        print(f"  纯10F+MA90闸门:  {stats(backtest_gate(bars, yellow=False, gate_n=90))}")
        print(f"  +黃柱+MA90闸门:  {stats(backtest_gate(bars, yellow=True, gate_n=90))}")


if __name__ == "__main__":
    main()
