"""
rs_optimize.py — 网格搜最优 RS 公式(回看周期+权重), 最大化与 Kova Score 的匹配。

用扩大后的广谱池(rs_scanner.load_universe). 对每个候选 RS 公式:
  · 算各校准样本(非指数/非NaN)的 RS 百分位 → 与 Kova Score 的 Spearman 相关
  · 最优公式再跑 Score 回归(MA特征+RS) → R² / LOO-CV
找出最匹配 Kova 的 RS 周期组合。跑: python3 rs_optimize.py
"""
import pandas as pd
import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
import rs_scanner

ETF = {"SPY", "QQQ", "SPMO", "SQQQ", "SMH", "IWM", "DIA"}

# 候选 RS 公式: 名字 -> {lookback_days: weight}
CANDIDATES = {
    "r21": {21: 1.0},
    "r63": {63: 1.0},
    "r126": {126: 1.0},
    "short(21+63)": {21: 0.6, 63: 0.4},
    "mid(63+126)": {63: 0.5, 126: 0.5},
    "midshort(21+63+126)": {21: 0.33, 63: 0.34, 126: 0.33},
    "ibd(63+126+189+252)": {63: 0.4, 126: 0.2, 189: 0.2, 252: 0.2},
    "3-6-9(63+126+189)": {63: 0.34, 126: 0.33, 189: 0.33},
    "recent-heavy(21+63+126)": {21: 0.5, 63: 0.3, 126: 0.2},
    "63heavy(63+126+189)": {63: 0.5, 126: 0.3, 189: 0.2},
}


def rs_raw(close, asof, weights):
    s = close[close.index <= pd.Timestamp(asof)]
    need = max(weights) + 5
    if len(s) < need:
        return np.nan
    p = float(s.iloc[-1])
    return sum(w * (p / float(s.iloc[-n]) - 1) for n, w in weights.items())


def rs_pct_asof(uni, date, weights):
    raws = {tk: rs_raw(s, date, weights) for tk, s in uni.items()}
    raws = {k: v for k, v in raws.items() if not pd.isna(v)}
    return (pd.Series(raws).rank(pct=True) * 98 + 1).round().astype(int).to_dict()


def ma_feats(close, asof, panel):
    s = close[close.index <= pd.Timestamp(asof) + pd.Timedelta(days=1)]
    if len(s) < 200:
        return None
    px = float(panel) if panel else float(s.iloc[-1])
    e20 = s.ewm(span=20).mean().iloc[-1]; s50 = s.rolling(50).mean().iloc[-1]
    s200 = s.rolling(200).mean().iloc[-1]; hi = s.tail(250).max()
    return [int(px > s200), int(px > e20), (px / hi - 1) * 100]


def main():
    print("载入广谱池…")
    uni = rs_scanner.load_universe()
    print(f"池规模: {len(uni)} 只")
    rows = [r for r in csv.DictReader(
        [l for l in open("kova_calibration.csv", encoding="utf-8") if not l.startswith("#")])]
    samples = [(r["ticker"], r["date"], int(r["score"]), r.get("panel_price", ""))
               for r in rows if r["score"] not in ("NaN", "") and r["ticker"] not in ETF]

    print("\n=== RS 公式网格搜 (vs Kova Score, 非指数) ===")
    print(f"{'公式':24}{'Spearman':>10}{'命中票数':>8}")
    best = None
    pct_cache = {}
    for name, w in CANDIDATES.items():
        key = tuple(sorted(w.items()))
        scs, rss = [], []
        for tk, dt, sc, pp in samples:
            ck = (dt, key)
            if ck not in pct_cache:
                pct_cache[ck] = rs_pct_asof(uni, dt, w)
            rs = pct_cache[ck].get(tk)
            if rs is not None:
                scs.append(sc); rss.append(rs)
        if len(scs) >= 10:
            rho = spearmanr(scs, rss).statistic
            print(f"{name:24}{rho:>+10.3f}{len(scs):>8}")
            if best is None or rho > best[1]:
                best = (name, rho, w)

    print(f"\n最优 RS 公式: {best[0]}  Spearman={best[1]:+.3f}")

    # 用最优 RS 跑 Score 回归
    w = best[2]
    X, Xr, y = [], [], []
    for tk, dt, sc, pp in samples:
        if tk not in uni:
            continue
        mf = ma_feats(uni[tk], dt, pp)
        if mf is None:
            continue
        ck = (dt, tuple(sorted(w.items())))
        if ck not in pct_cache:
            pct_cache[ck] = rs_pct_asof(uni, dt, w)
        rs = pct_cache[ck].get(tk)
        if rs is None:
            continue
        X.append(mf); Xr.append(mf + [rs]); y.append(sc)
    X, Xr, y = np.array(X), np.array(Xr), np.array(y)
    r2a = LinearRegression().fit(X, y).score(X, y)
    r2b = LinearRegression().fit(Xr, y).score(Xr, y)
    mae_a = -cross_val_score(LinearRegression(), X, y, cv=LeaveOneOut(), scoring="neg_mean_absolute_error").mean()
    mae_b = -cross_val_score(LinearRegression(), Xr, y, cv=LeaveOneOut(), scoring="neg_mean_absolute_error").mean()
    print(f"\nScore 回归 (n={len(y)}):")
    print(f"  仅MA:    R²={r2a:.3f}  LOO-MAE={mae_a:.2f}")
    print(f"  MA+最优RS: R²={r2b:.3f}  LOO-MAE={mae_b:.2f}")
    m = LinearRegression().fit(Xr, y)
    print(f"  权重: 站200S={m.coef_[0]:+.1f} 站20E={m.coef_[1]:+.1f} 距高={m.coef_[2]:+.2f} RS={m.coef_[3]:+.2f} 截距={m.intercept_:.1f}")


if __name__ == "__main__":
    main()
