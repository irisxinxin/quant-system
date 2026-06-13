"""
kova_score_dimensions.py — 穷尽测试 Score 残差的所有 Pine 可达维度。

当前模型: Score ≈ 71.6 + 27.6·站200S + 1.05·rsHi(vs QQQ). 残差还剩 IREN类-20。
顺"Kova是Pine指标→只能用TV可达数据"框架, 把 request.* 能拉的全测:
  · request.financial(FactSet): 营收/EPS增长、利润率、ROE、估值、负债、beta
  · 更多RS变体: vs SPX、不同窗口rsHi、RS斜率
  · 趋势/波动: ATR%、ADX、趋势年龄、距50S
对 Score残差 做 Spearman相关 + 前向选择(留一票CV), 只保留真正降残差的维度。

用法: python3 kova_score_dimensions.py
"""
import csv, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

FUND_KEYS = ["revenueGrowth", "earningsQuarterlyGrowth", "profitMargins", "grossMargins",
             "returnOnEquity", "forwardPE", "priceToSalesTrailing12Months", "pegRatio",
             "debtToEquity", "beta"]


def _ctx():
    from longport.openapi import Config, QuoteContext
    return QuoteContext(Config.from_env())


def _pull(ctx, tk, n=340):
    from longport.openapi import Period, AdjustType
    try:
        b = list(ctx.candlesticks(f"{tk}.US", Period.Day, n, AdjustType.ForwardAdjust))
        df = pd.DataFrame({"High": [float(x.high) for x in b], "Low": [float(x.low) for x in b],
                           "Close": [float(x.close) for x in b], "Volume": [float(x.volume) for x in b]},
                          index=[x.timestamp for x in b]).sort_index()
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        return df
    except Exception:
        return None


def _fund(tk):
    import yfinance as yf
    try:
        info = yf.Ticker(tk).info
        d = {k: info.get(k) for k in FUND_KEYS}
        pm = info.get("profitMargins")
        d["isProfitable"] = 1 if (pm is not None and pm > 0) else 0
        mc = info.get("marketCap")
        d["logMktCap"] = np.log10(mc) if mc else np.nan
        return d
    except Exception:
        return {k: None for k in FUND_KEYS}


def _atr(df, n=14):
    h, l, c = df.High, df.Low, df.Close
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1).rolling(n).mean()


def _adx(df, n=14):
    up = df.High.diff(); dn = -df.Low.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _atr(df, 1)
    atr = tr.rolling(n).mean()
    pdi = 100 * pd.Series(plus, index=df.index).rolling(n).mean() / (atr + 1e-9)
    mdi = 100 * pd.Series(minus, index=df.index).rolling(n).mean() / (atr + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.rolling(n).mean()


def asof(df, d):
    return df[df.index <= pd.Timestamp(d) + pd.Timedelta(days=1)]


def main():
    rows = [r for r in csv.DictReader(l for l in open("kova_calibration.csv", encoding="utf-8") if not l.startswith("#"))]
    rows = [r for r in rows if r["score"] not in ("", "NaN") and r["ticker"]]
    ctx = _ctx()
    spy = _pull(ctx, "SPY", 420); qqq = _pull(ctx, "QQQ", 420)
    cache, fcache, recs = {}, {}, []
    for r in rows:
        tk, d = r["ticker"], r["date"]
        if tk not in cache:
            cache[tk] = _pull(ctx, tk); fcache[tk] = _fund(tk)
        df = cache[tk]
        if df is None:
            continue
        s = asof(df, d)
        if len(s) < 210:
            continue
        c = s.Close; px = float(c.iloc[-1])
        s200 = c.rolling(200).mean().iloc[-1]; s50 = c.rolling(50).mean().iloc[-1]
        a200 = int(px > s200); dist50 = (px / s50 - 1) * 100
        atrp = float(_atr(s).iloc[-1]) / px * 100
        adx = float(_adx(s).iloc[-1])
        days200 = int((c > c.rolling(200).mean()).iloc[::-1].cumprod().sum())
        rsfeat = {}
        for bn, bdf in [("QQQ", qqq), ("SPY", spy)]:
            bs = asof(bdf, d).Close; idx = c.index.intersection(bs.index)
            ratio = (c.reindex(idx) / bs.reindex(idx)).dropna()
            if len(ratio) < 210:
                continue
            rl = float(ratio.iloc[-1])
            rsfeat[f"rsHi252_{bn}"] = (rl / float(ratio.tail(252).max()) - 1) * 100
            rsfeat[f"rsHi126_{bn}"] = (rl / float(ratio.tail(126).max()) - 1) * 100
            rsfeat[f"rsSlope21_{bn}"] = (rl / float(ratio.iloc[-22]) - 1) * 100 if len(ratio) > 22 else np.nan
        rsHi = rsfeat.get("rsHi252_QQQ", np.nan)
        pred = 71.6 + 27.6 * a200 + 1.05 * rsHi
        rec = dict(tk=tk, date=d, score=float(r["score"]), a200=a200, rsHi=rsHi,
                   pred=pred, resid=float(r["score"]) - pred,
                   dist50=dist50, atrp=atrp, adx=adx, days200=days200, **rsfeat)
        f = fcache[tk]
        for k, v in f.items():
            rec[k] = (float(v) if v is not None and not isinstance(v, str) else np.nan)
        recs.append(rec)
    df = pd.DataFrame(recs)
    print(f"样本 {len(df)}  (基本面=当前快照, 截图都在2026-05~06约1月内, 近似)")
    from scipy.stats import spearmanr
    feats = ["rsHi126_QQQ", "rsHi252_SPY", "rsHi126_SPY", "rsSlope21_QQQ", "rsSlope21_SPY",
             "dist50", "atrp", "adx", "days200"] + FUND_KEYS + ["isProfitable", "logMktCap"]
    print(f"\n各维度 vs Score残差 的 Spearman 相关 (|r|排序):")
    cors = []
    for f in feats:
        if f in df.columns:
            sub = df[["resid", f]].dropna()
            if len(sub) >= 30:
                rho = spearmanr(sub.resid, sub[f]).correlation
                cors.append((f, rho, len(sub)))
    for f, rho, n in sorted(cors, key=lambda x: -abs(x[1])):
        print(f"  {f:30} r={rho:+.3f}  (n={n})")
    # 前向选择: 残差能被哪个基本面再降
    print(f"\n增量检验 (留一票CV, 在 a200+rsHi 基础上加单个维度):")
    from sklearn.linear_model import Ridge
    base = ["a200", "rsHi"]
    y = df.score.values
    def locv(cols):
        d2 = df[cols + ["score", "tk"]].dropna()
        if len(d2) < 40:
            return None, len(d2)
        X = d2[cols].values; yy = d2.score.values; pred = np.full(len(yy), np.nan)
        for tk in d2.tk.unique():
            te = (d2.tk == tk).values; tr = ~te
            if tr.sum() < 5:
                continue
            m = Ridge(alpha=1.0).fit(X[tr], yy[tr]); pred[te] = m.predict(X[te])
        ok = ~np.isnan(pred)
        return 1 - ((yy[ok] - pred[ok]) ** 2).sum() / ((yy[ok] - yy[ok].mean()) ** 2).sum(), len(d2)
    b, bn = locv(base)
    print(f"  基线 a200+rsHi: R²={b:.3f} (n={bn})")
    addcands = ["rsHi126_QQQ", "rsSlope21_QQQ", "adx", "days200", "dist50", "atrp"] + FUND_KEYS + ["isProfitable", "logMktCap"]
    res = []
    for f in addcands:
        if f in df.columns:
            r2, n = locv(base + [f])
            if r2 is not None:
                res.append((f, r2, n))
    for f, r2, n in sorted(res, key=lambda x: -x[1]):
        print(f"  +{f:28} R²={r2:.3f} (Δ{r2-b:+.3f}, n={n})")
    # IREN 诊断
    ir = df[df.tk == "IREN"]
    if len(ir):
        print(f"\n🔍 IREN: 实际{ir.score.iloc[0]:.0f} 预测{ir.pred.iloc[0]:.0f} 残差{ir.resid.iloc[0]:+.0f}")
        for f in FUND_KEYS + ["isProfitable"]:
            if f in ir.columns and not pd.isna(ir[f].iloc[0]):
                print(f"   {f:30}={ir[f].iloc[0]:+.3f}")
    df.to_csv("kova_score_dimensions.csv", index=False)
    print(f"\n存档 → kova_score_dimensions.csv")


if __name__ == "__main__":
    main()
