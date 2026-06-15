"""
swing_backtest.py — 全票波段择时回测。每只算买入持有 vs 波段(10E/20E/50S),
选风险调整最优(Calmar=收益/|最大回撤|)的跟踪线, 判定波段是否可做。输出 output/swing_ranked.csv。

波段规则(long-only, 日线, mark-to-market): 收盘>MA(且>200S)持有, 收盘<MA×0.99出场, 重新站上再进。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from longport.openapi import Config, QuoteContext, Period, AdjustType

# 全票池(美股 from all_tickers_ranked + 港股2x + 额外)
US = ["AAOI", "MXL", "SNDK", "ALAB", "DELL", "INTC", "CRWV", "CRCL", "MSFL", "ORCL", "EOSE", "SOXL", "MU", "PLTR",
      "OKLO", "IREN", "LITE", "TSLZ", "RKLB", "ARM", "MSFU", "NVDL", "HOOD", "AMD", "CIFR", "TSLL", "NBIS",
      "AMZN", "MRVL", "NVDS", "SOXS", "SOXX", "TWLO", "BMNR", "CONL"]
HK = ["7747", "7709"]
NAMES = {"7747.HK": "三星2x", "7709.HK": "海力士2x", "SOXL.US": "半导体3x", "NVDL.US": "英伟达2x",
         "MSFL.US": "微软2x", "MSFU.US": "微软2x", "TSLL.US": "特斯拉2x", "TSLZ.US": "特斯拉反", "NVDS.US": "英伟达反",
         "SOXS.US": "半导体反", "CONL.US": "Coinbase2x", "BMNR.US": "BitMine", "SOXX.US": "半导体ETF"}


def pull(ctx, sym, n=750):
    try:
        adj = AdjustType.ForwardAdjust if sym.endswith(".US") else AdjustType.NoAdjust
        b = list(ctx.candlesticks(sym, Period.Day, n, adj))
        return pd.DataFrame({"C": [float(x.close) for x in b]}, index=[x.timestamp for x in b]).sort_index().C
    except Exception:
        return None


def maxdd(eq):
    return float((eq / np.maximum.accumulate(eq) - 1).min()) * 100


COST = 0.001   # 单边成本 (手续费+滑点+点差 ~0.1%), 进出各扣一次

def swing(c, ma, s200, lo, hi, cost=COST):
    """
    在 [lo, hi) 区间评估波段 (MA/s200 在全序列上算好传入, 保证 200SMA 有效).
    决策用 bar i 收盘 (已收盘), 仓位作用于 [i→i+1]; 进出各扣 cost.
    强制 close>200SMA 才进 (200SMA 必须有效, 不再 NaN 静默放行). 返回 (ret%, dd%, n_trades).
    """
    eq = 1.0; equity = [1.0]; pos = 0; ntr = 0
    for i in range(lo, hi):
        if pos == 1:
            eq *= c.iloc[i] / c.iloc[i - 1]
        equity.append(eq)
        if pos == 0:
            if c.iloc[i] > ma.iloc[i] and (not pd.isna(s200.iloc[i])) and c.iloc[i] > s200.iloc[i]:
                pos = 1; eq *= (1 - cost); ntr += 1
        else:
            if c.iloc[i] < ma.iloc[i] * 0.99:
                pos = 0; eq *= (1 - cost)
    equity = np.array(equity)
    return (equity[-1] - 1) * 100, maxdd(equity), ntr


def _calmar(r, d):
    return r / (abs(d) + 1e-9)


def main():
    ctx = QuoteContext(Config.from_env())
    syms = [f"{s}.US" for s in US] + [f"{s}.HK" for s in HK]
    rows = []
    for sym in syms:
        c = pull(ctx, sym)
        if c is None or len(c) < 460:    # 需够长: 200SMA 暖机(200) + 样本内外各 ≥130
            continue
        bh = (c.iloc[-1] / c.iloc[0] - 1) * 100; bh_dd = maxdd((c / c.iloc[0]).values)
        # MA 在全序列上算 (保证两段都有有效 200SMA), 区间评估
        e10 = c.ewm(span=10).mean(); e20 = c.ewm(span=20).mean()
        s50 = c.rolling(50).mean(); s200 = c.rolling(200).mean()
        mas = {"10E": e10, "20E": e20, "50S": s50}
        # 样本内从 200 起 (避开 200SMA 暖机), 到中点; 样本外 = 中点到末尾
        n = len(c); mid = max(220, n // 2)
        # ── Walk-forward: 仅用样本内选线, 样本外 (诚实) 评估上报 ──
        is_v = {k: swing(c, ma, s200, 200, mid) for k, ma in mas.items()}
        best = max(is_v, key=lambda k: _calmar(is_v[k][0], is_v[k][1]))
        oos_v = {k: swing(c, ma, s200, mid, n) for k, ma in mas.items()}
        br, bd, ntr = oos_v[best]        # 样本外结果上报
        calmar = _calmar(br, bd)
        verdict = ("✅波段好" if calmar >= 2 and br >= 15 else "⚠️一般" if br > 0 else "❌不适合波段")
        rows.append(dict(symbol=sym, name=NAMES.get(sym, sym.replace(".US", "").replace(".HK", "")),
                         days=len(c), buyhold=round(bh), bh_dd=round(bh_dd),
                         best_ma=best, swing_ret=round(br), swing_dd=round(bd), calmar=round(calmar, 1),
                         n_trades=ntr, verdict=verdict,
                         r10=round(oos_v["10E"][0]), r20=round(oos_v["20E"][0]), r50=round(oos_v["50S"][0])))
    df = pd.DataFrame(rows).sort_values("calmar", ascending=False)
    df.to_csv("output/swing_ranked.csv", index=False)
    print(f"{'票':9}{'名称':10}{'买持':>7}{'最优线':>6}{'波段%':>7}{'回撤':>6}{'Calmar':>7}  判定")
    print("-" * 70)
    for _, x in df.iterrows():
        print(f"{x.symbol:9}{x['name'][:9]:10}{x.buyhold:>+6}%{x.best_ma:>6}{x.swing_ret:>+6}%{x.swing_dd:>5}%{x.calmar:>7}  {x.verdict}")
    n_ok = (df.verdict == "✅波段好").sum()
    print(f"\n✅波段好 {n_ok} | ⚠️一般 {(df.verdict=='⚠️一般').sum()} | ❌不适合 {(df.verdict=='❌不适合波段').sum()}")
    print("存 → output/swing_ranked.csv  (Calmar=波段收益/|最大回撤|, 越高越是好波段标的)")


if __name__ == "__main__":
    main()
