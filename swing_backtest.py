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


def pull(ctx, sym, n=300):
    try:
        adj = AdjustType.ForwardAdjust if sym.endswith(".US") else AdjustType.NoAdjust
        b = list(ctx.candlesticks(sym, Period.Day, n, adj))
        return pd.DataFrame({"C": [float(x.close) for x in b]}, index=[x.timestamp for x in b]).sort_index().C
    except Exception:
        return None


def maxdd(eq):
    return float((eq / np.maximum.accumulate(eq) - 1).min()) * 100


def swing(c, ma, s200):
    eq = [1.0]; inpos = False
    for i in range(1, len(c)):
        if inpos:
            eq.append(eq[-1] * c.iloc[i] / c.iloc[i - 1])
            if c.iloc[i] < ma.iloc[i] * 0.99:
                inpos = False
        else:
            eq.append(eq[-1])
            if c.iloc[i] > ma.iloc[i] and (pd.isna(s200.iloc[i]) or c.iloc[i] > s200.iloc[i]):
                inpos = True
    eq = np.array(eq)
    return (eq[-1] - 1) * 100, maxdd(eq)


def main():
    ctx = QuoteContext(Config.from_env())
    syms = [f"{s}.US" for s in US] + [f"{s}.HK" for s in HK]
    rows = []
    for sym in syms:
        c = pull(ctx, sym)
        if c is None or len(c) < 60:
            continue
        e10 = c.ewm(span=10).mean(); e20 = c.ewm(span=20).mean(); s50 = c.rolling(50).mean(); s200 = c.rolling(200).mean()
        bh = (c.iloc[-1] / c.iloc[0] - 1) * 100; bh_dd = maxdd((c / c.iloc[0]).values)
        variants = {"10E": swing(c, e10, s200), "20E": swing(c, e20, s200), "50S": swing(c, s50, s200)}
        # 选 Calmar(收益/|回撤|) 最高的跟踪线
        best = max(variants, key=lambda k: variants[k][0] / (abs(variants[k][1]) + 1e-9))
        br, bd = variants[best]
        calmar = br / (abs(bd) + 1e-9)
        verdict = ("✅波段好" if calmar >= 4 and br >= 50 else "⚠️一般" if br >= 20 and br > 0 else "❌不适合波段")
        rows.append(dict(symbol=sym, name=NAMES.get(sym, sym.replace(".US", "").replace(".HK", "")),
                         days=len(c), buyhold=round(bh), bh_dd=round(bh_dd),
                         best_ma=best, swing_ret=round(br), swing_dd=round(bd), calmar=round(calmar, 1),
                         verdict=verdict, r10=round(variants["10E"][0]), r20=round(variants["20E"][0]), r50=round(variants["50S"][0])))
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
