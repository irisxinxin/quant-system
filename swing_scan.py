"""
swing_scan.py — 扫聚焦动量龙头池(~80, 含大盘科技), 找"当前新鲜波段买点(未延伸)"。
每只: 测10/20/50跟踪线选Calmar最优 → 只留✅波段好(Calmar≥4) → 看当前状态:
  🟢新鲜买点 = 站上最优线 且 未延伸(距21E<1.5ATR) 且 上升趋势  ← 现在进还来得及
  🔵已持有/延伸 = 涨过头, 别追;  ⚪观望 = 还在线下方
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from longport.openapi import Config, QuoteContext, Period, AdjustType

MEGA = ["GOOG", "GOOGL", "AAPL", "MSFT", "META", "NVDA", "TSLA", "NFLX", "AMZN"]
SEMI = ["AMD", "INTC", "MU", "MXL", "ARM", "AVGO", "MRVL", "TSM", "ASML", "LRCX", "AMAT", "KLAC", "ON", "MCHP", "QCOM", "TXN", "MPWR", "ALAB", "CRDO", "NXPI",
        "UMC", "ASX", "ENTG", "TER", "COHR", "AEHR", "SITM", "VECO"]   # +台湾ADR(UMC/ASX) +半导体设备/材料/封测(VECO)
AI = ["NBIS", "CRWV", "ORCL", "PLTR", "SMCI", "VRT", "DELL", "ANET", "SNOW", "NOW", "PANW", "CRWD", "DDOG", "NET", "APP"]
STOR = ["SNDK", "STX", "WDC"]
NEW = ["IREN", "CIFR", "RKLB", "OKLO", "SMR", "CCJ", "AAOI", "LITE", "OUST", "HOOD", "COIN", "MSTR", "HIMS", "SOFI", "NOK", "EOSE", "RDW"]
ETF = ["SOXL", "NVDL", "TSLL", "MSFL", "SOXX"]
HK = ["7747", "7709", "0522", "3076"]   # 三星2x/海力士2x/ASMPT/港股3076
UNIVERSE = [f"{s}.US" for s in MEGA + SEMI + AI + STOR + NEW + ETF] + [f"{s}.HK" for s in HK]


def pull(ctx, sym, n=300):
    try:
        adj = AdjustType.ForwardAdjust if sym.endswith(".US") else AdjustType.NoAdjust
        b = list(ctx.candlesticks(sym, Period.Day, n, adj))
        return pd.DataFrame({"H": [float(x.high) for x in b], "L": [float(x.low) for x in b],
                             "C": [float(x.close) for x in b]}, index=[x.timestamp for x in b]).sort_index()
    except Exception:
        return None


def maxdd(eq): return float((eq / np.maximum.accumulate(eq) - 1).min()) * 100


def swing_eq(c, ma, s200):
    eq = [1.0]; inpos = False
    for i in range(1, len(c)):
        if inpos:
            eq.append(eq[-1] * c.iloc[i] / c.iloc[i - 1])
            if c.iloc[i] < ma.iloc[i] * 0.99: inpos = False
        else:
            eq.append(eq[-1])
            if c.iloc[i] > ma.iloc[i] and (pd.isna(s200.iloc[i]) or c.iloc[i] > s200.iloc[i]): inpos = True
    return np.array(eq)


def main():
    ctx = QuoteContext(Config.from_env())
    fresh, hold, watch = [], [], []
    n_ok = 0
    for sym in UNIVERSE:
        df = pull(ctx, sym)
        if df is None or len(df) < 120:
            continue
        c, h, l = df.C, df.H, df.L
        mas = {"10E": c.ewm(span=10).mean(), "20E": c.ewm(span=20).mean(), "50S": c.rolling(50).mean()}
        s200 = c.rolling(200).mean(); e21 = c.ewm(span=21).mean()
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1); atr = tr.rolling(14).mean()
        # 选Calmar最优线
        best, bestcal, bestret, bestdd = None, -9, 0, 0
        for k, ma in mas.items():
            eq = swing_eq(c, ma, s200); r = (eq[-1] - 1) * 100; dd = maxdd(eq); cal = r / (abs(dd) + 1e-9)
            if cal > bestcal: best, bestcal, bestret, bestdd = k, cal, r, dd
        if bestcal < 3 or bestret < 50:   # 不适合波段, 跳过 (Calmar≥3, 含高收益高波动名如ALAB/VECO)
            continue
        n_ok += 1
        eq = swing_eq(c, mas[best], s200); w = min(63, len(eq) - 1); sw90 = (eq[-1] / eq[-1 - w] - 1) * 100
        px = c.iloc[-1]; man = mas[best].iloc[-1]; dev = (px - e21.iloc[-1]) / atr.iloc[-1]
        uptrend = (px > mas["50S"].iloc[-1]) and (pd.isna(s200.iloc[-1]) or px > s200.iloc[-1])
        in_trend = px > man * 0.99; dist = (px / man - 1) * 100
        rec = dict(sym=sym.replace(".US", "").replace(".HK", "hk"), ma=best, px=px, man=man, dev=dev,
                   dist=dist, cal=round(bestcal, 1), sw90=round(sw90), uptrend=uptrend)
        # 风控分类: 止损=最优线; 距线% = 现在进场的止损风险
        if not in_trend and -5 < dist <= 0 and uptrend:   # 🟢即将触发(线下5%内)
            rec["pending"] = True; fresh.append(rec)
        elif in_trend and uptrend and dev < 1.5 and dist <= 8:   # 🟢新鲜买点: 趋势中+未延伸+止损≤8%(风险可控)
            fresh.append(rec)
        elif in_trend:                                    # 趋势中但止损远(>8%)或已延伸 → 别追, 等回踩
            hold.append(rec)
        else:
            watch.append(rec)
    print(f"扫描 {len(UNIVERSE)} 只, 其中 {n_ok} 只是好波段标的(Calmar≥3)")
    print(f"\n🟢🟢 现在新鲜买点 (未延伸, 进还来得及) — {len(fresh)} 只 [按近90收益排]:")
    print(f"{'票':8}{'线':4}{'现价':>9}{'关键线':>9}{'距线%':>7}{'距21E':>7}{'近90%':>7}{'Calmar':>7}  操作")
    for r in sorted(fresh, key=lambda x: -x["sw90"]):
        if r.get("pending"):
            act = f"站上{r['man']:.2f}买入(buy-stop)"
        else:
            act = f"现价区可建仓 · 止损{r['man']*0.99:.2f}"
        print(f"{r['sym']:8}{r['ma']:4}{r['px']:>9.2f}{r['man']:>9.2f}{r['dist']:>+6.1f}%{r['dev']:>+6.1f}{r['sw90']:>+6}%{r['cal']:>7}  {act}")
    print(f"\n🔵 已持有/已延伸(别追, 等回踩): {' '.join(r['sym'] for r in sorted(hold,key=lambda x:-x['sw90']))}")
    print(f"⚪ 观望(线下方远): {' '.join(r['sym'] for r in watch)}")
    pd.DataFrame(fresh).to_csv("output/swing_fresh.csv", index=False)
    print(f"\n新鲜买点存 → output/swing_fresh.csv")


if __name__ == "__main__":
    main()
