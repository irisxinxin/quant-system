#!/usr/bin/env python3
"""
shishi_10f_v2.py — 诗诗 CONL 10F 法【真规则】回测 (2026-03-02 ~ 08-28, 全时段10分K)。

v1(7月)测的是裸MA→全参数亏。本版按逆向出的完整规则:
  线 = 弘历迭代线: L(N) = FORCAST( [EMA(TP,N)+EMA(TP,2N)+EMA(TP,4N)]/3 , 6 ), TP=(O+H+L+C)/4
      快线 N=8, 慢线 N=24 (诗诗自述"設定是8跟24, 迭代公式來自David飄帶")
  V1 金死叉: 快上穿慢→次bar开盘买; 快下穿慢→次bar开盘卖
  V2 = V1 + 实体确认: 金叉时收盘价须在快线上方("要X之上的實體")
  V3 梯子版: NX结构 EMA(H/L,8) vs EMA(H/L,24), 快带下轨>慢带上轨=金叉
  V4 = V2 + RSI(14)黄警减仓: 持仓中RSI下穿80→先卖一半(顶部黄K警戒)
成本0.1%/边, 满进满出(等权%口径)。
校验环节: 对照他8月实盘喊单(8/21卖6.4, 8/25压力6.1-6.2空头, 8/28卖出信号6.58, 8/31买5.56)
逐一检查本重构指标是否在相同时点/价位出信号 — 对上=逆向成功。
"""
import sys
from datetime import date, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from zoneinfo import ZoneInfo
import shishi_10f_backtest as S

ET = ZoneInfo("America/New_York")
COST = 0.001


def ema(vals, n):
    a = 2 / (n + 1)
    out, e = [], None
    for v in vals:
        e = v if e is None else v * a + e * (1 - a)
        out.append(e)
    return out


def forcast(vals, n=6):
    """TDX FORCAST: 最近n点线性回归在当前bar的预测值。"""
    out = []
    for i in range(len(vals)):
        if i < n - 1:
            out.append(vals[i]); continue
        ys = vals[i - n + 1:i + 1]
        xm = (n - 1) / 2
        ym = sum(ys) / n
        num = sum((j - xm) * (ys[j] - ym) for j in range(n))
        den = sum((j - xm) ** 2 for j in range(n))
        b = num / den
        out.append(ym + b * ((n - 1) - xm))
    return out


def hongli_line(bars, n):
    tp = [(b["o"] + b["h"] + b["l"] + b["c"]) / 4 for b in bars]
    avg = [(x + y + z) / 3 for x, y, z in zip(ema(tp, n), ema(tp, 2 * n), ema(tp, 4 * n))]
    return forcast(avg, 6)


def rsi(bars, n=14):
    closes = [b["c"] for b in bars]
    out, up, dn = [50.0], None, None
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        u, d = max(ch, 0), abs(min(ch, 0))
        up = u if up is None else (u + (n - 1) * up) / n     # SMA(x,n,1) 口径
        dn = d if dn is None else (d + (n - 1) * dn) / n
        out.append(100.0 if dn == 0 else 100 * up / (up + dn))
    return out


def run(bars, variant):
    fast = hongli_line(bars, 8)
    slow = hongli_line(bars, 24)
    if variant == "V3":
        fh, fl = ema([b["h"] for b in bars], 8), ema([b["l"] for b in bars], 8)
        sh, sl = ema([b["h"] for b in bars], 24), ema([b["l"] for b in bars], 24)
    R = rsi(bars) if variant == "V4" else None
    pos, trades, entry_ts = None, [], None
    half_sold = False
    warm = 96 * 2
    for i in range(warm, len(bars) - 1):
        b, nxt = bars[i], bars[i + 1]
        if variant == "V3":
            gold = fl[i] > sh[i] and fl[i - 1] <= sh[i - 1]
            death = fh[i] < sl[i] and fh[i - 1] >= sl[i - 1]
        else:
            gold = fast[i] > slow[i] and fast[i - 1] <= slow[i - 1]
            death = fast[i] < slow[i] and fast[i - 1] >= slow[i - 1]
        if pos is None:
            ok = gold
            if variant in ("V2", "V4") and gold:
                ok = b["c"] > fast[i]                    # 实体确认
            if ok:
                pos = nxt["o"] * (1 + COST)
                entry_ts = nxt["ts"]; half_sold = False
        else:
            if variant == "V4" and not half_sold and R[i] < 80 <= R[i - 1]:
                ex = nxt["o"] * (1 - COST)               # 黄警减半
                trades.append(dict(ein=entry_ts, eout=nxt["ts"], pct=(ex / pos - 1) * 100 * 0.5, part=True))
                half_sold = True
                continue
            if death:
                ex = nxt["o"] * (1 - COST)
                w = 0.5 if (variant == "V4" and half_sold) else 1.0
                trades.append(dict(ein=entry_ts, eout=nxt["ts"], pct=(ex / pos - 1) * 100 * w, part=False))
                pos = None
    if pos is not None:
        ex = bars[-1]["c"] * (1 - COST)
        w = 0.5 if (variant == "V4" and half_sold) else 1.0
        trades.append(dict(ein=entry_ts, eout=bars[-1]["ts"], pct=(ex / pos - 1) * 100 * w, part=False))
    return trades, fast, slow


def stats(trades):
    if not trades:
        return "无交易"
    full = {}
    for t in trades:                    # V4半仓腿并回整笔
        full.setdefault(t["ein"], 0.0)
        full[t["ein"]] += t["pct"]
    vals = list(full.values())
    w = sum(1 for v in vals if v > 0)
    eq = 1.0
    for v in vals:
        eq *= 1 + v / 100
    peak, mdd = 1.0, 0.0
    e = 1.0
    for v in vals:
        e *= 1 + v / 100
        peak = max(peak, e); mdd = max(mdd, 1 - e / peak)
    monthly = {}
    for ein, v in full.items():
        monthly.setdefault(ein.astimezone(ET).strftime("%m"), []).append(v)
    ms = " ".join(f"{m}月{sum(vs):+.0f}%" for m, vs in sorted(monthly.items()))
    return (f"{len(vals)}笔 胜{w}({w/len(vals)*100:.0f}%) 均{sum(vals)/len(vals):+.2f}%/笔 "
            f"复利累计{(eq-1)*100:+.0f}% 最大回撤{mdd*100:.0f}% | {ms}")


def main():
    S.START, S.END = date(2026, 3, 2), date(2026, 8, 29)
    b5 = S.load_cached_fallback() if False else None
    import csv
    rows = []
    with open(S.CACHE) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(ts=datetime.fromisoformat(r["ts"]), o=float(r["o"]), h=float(r["h"]),
                             l=float(r["l"]), c=float(r["c"])))
    b5 = sorted(rows, key=lambda x: x["ts"])
    bars = S.to_10m(b5)
    print(f"10分K {len(bars)}根 ({bars[0]['ts'].astimezone(ET):%m-%d} ~ {bars[-1]['ts'].astimezone(ET):%m-%d})")
    bh = (bars[-1]["c"] / bars[0]["o"] - 1) * 100
    print(f"买入持有: {bh:+.1f}%\n")
    for v, name in [("V1", "金死叉(迭代8/24)"), ("V2", "V1+实体确认"), ("V3", "NX梯子带(8/24)"), ("V4", "V2+RSI黄警减半")]:
        trades, fast, slow = run(bars, v)
        print(f"{v} {name}:\n  {stats(trades)}")
    # ── 对照他的实盘喊单: V2信号在8月下旬的触发点 ──
    trades, fast, slow = run(bars, "V2")
    print("\n8月20日后 V2 全部信号(次bar开盘执行价):")
    for t in trades:
        if t["eout"].astimezone(ET) >= datetime(2026, 8, 20, tzinfo=ET):
            print(f"  入 {t['ein'].astimezone(ET):%m-%d %H:%M} → 出 {t['eout'].astimezone(ET):%m-%d %H:%M}  {t['pct']:+.1f}%")


if __name__ == "__main__":
    main()
