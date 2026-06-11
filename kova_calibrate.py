"""
kova_calibrate.py — Kova 指标反推校准引擎。

读 kova_calibration.csv 的标签样本 → 按 (ticker, date) 拉真实日线 OHLCV →
计算特征(站位/距高/ATR乖离/RS) → 验证健康度公式 → 输出特征表供 Score 权重拟合。

用法:
    python3 kova_calibrate.py              # 全量重算 + 验证 + 打印特征表
    python3 kova_calibrate.py --fit        # 额外跑 Score 线性拟合(需 sklearn)

数据源: 长桥 QuoteContext(只读行情) 优先, 失败回退 yfinance。
每次新增样本只改 CSV, 重跑本脚本即可——这就是"反复回测"的基础设施。
"""
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

CSV = "kova_calibration.csv"
BENCH = "QQQ"   # RS 基准(待定 QQQ vs SPY, 先用 QQQ)


def _atr(df, n=14):
    h, l, c = df.High, df.Low, df.Close
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _make_ctx():
    try:
        from longport.openapi import Config, QuoteContext
        return QuoteContext(Config.from_env())
    except Exception:
        return None


def _pull_longport(ctx, tk, count=260):
    from longport.openapi import Period, AdjustType
    bars = list(ctx.candlesticks(f"{tk}.US", Period.Day, count, AdjustType.ForwardAdjust))
    if not bars:
        return None
    return pd.DataFrame({
        "Open": [float(b.open) for b in bars],
        "High": [float(b.high) for b in bars], "Low": [float(b.low) for b in bars],
        "Close": [float(b.close) for b in bars], "Volume": [float(b.volume) for b in bars],
    }, index=[b.timestamp for b in bars]).sort_index()


def _pull_yf(tk):
    import yfinance as yf
    df = yf.download(tk, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def _pull(ctx, tk):
    if ctx:
        try:
            df = _pull_longport(ctx, tk)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    try:
        return _pull_yf(tk)
    except Exception:
        return None


def _slice_to_date(df, date):
    """截到 date(含)当日, 模拟"截图当天"的状态。"""
    d = pd.Timestamp(date)
    idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df = df.copy(); df.index = idx
    return df[df.index <= d + pd.Timedelta(days=1)]


def features(df, panel_price=None, bench_df=None):
    """从截到当日的 df 算 Kova 特征。panel_price 优先(盘中面板价), 否则用结算收盘。"""
    c, o, h, l, v = df.Close, df.Open, df.High, df.Low, df.Volume
    n = len(df)
    last = float(c.iloc[-1])
    px = float(panel_price) if panel_price and not pd.isna(panel_price) else last
    e10 = float(c.ewm(span=10).mean().iloc[-1])
    e20 = float(c.ewm(span=20).mean().iloc[-1])
    e21 = float(c.ewm(span=21).mean().iloc[-1])
    s50 = float(c.rolling(50).mean().iloc[-1]) if n >= 50 else np.nan
    s200 = float(c.rolling(200).mean().iloc[-1]) if n >= 200 else np.nan
    atr = float(_atr(df).iloc[-1])
    hi = float(c.tail(250).max())
    rs = np.nan
    if bench_df is not None and len(bench_df) >= 60:
        bc = bench_df.Close
        stock_ret = last / float(c.iloc[-60]) - 1 if n >= 60 else np.nan
        bench_ret = float(bc.iloc[-1]) / float(bc.iloc[-60]) - 1 if len(bc) >= 60 else np.nan
        if not (pd.isna(stock_ret) or pd.isna(bench_ret)):
            rs = round((stock_ret - bench_ret) * 100, 1)
    # ── 动量代理 (反推动能震荡器 + 六色量柱) ──
    vol_ma = v.rolling(20).mean()
    cmf = float((((c - l) - (h - c)) / (h - l + 1e-9) * v).tail(21).sum() / (v.tail(21).sum() + 1e-9))
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    obv_sl = float((obv.iloc[-1] - obv.iloc[-6]) / (v.tail(60).mean() * 5 + 1e-9))   # 归一: 近5日净OBV / 5日均量
    dist_days = int((((c < c.shift()) & (v > 1.5 * vol_ma)).tail(25).sum()))
    vr = float(v.iloc[-1] / (vol_ma.iloc[-1] + 1e-9))
    last_up = bool(c.iloc[-1] > c.shift().iloc[-1])
    return dict(
        bars=n, px=round(px, 2),
        a10=int(px > e10), a20=int(px > e20),
        a50=int(px > s50) if not pd.isna(s50) else -1,
        a200=int(px > s200) if not pd.isna(s200) else -1,
        stack=int((e10 > e20) and (not pd.isna(s50) and e20 > s50) and (not pd.isna(s200) and s50 > s200)),
        d10=round((px / e10 - 1) * 100, 1), d20=round((px / e20 - 1) * 100, 1),
        dist_hi=round((px / hi - 1) * 100, 1),
        dev21_atr=round((px - e21) / atr, 2) if atr else np.nan,
        rs=rs, cmf=round(cmf, 3), obv_sl=round(obv_sl, 2),
        distd=dist_days, vr=round(vr, 2), lastbar=("涨" if last_up else "跌"),
    )


# WATCH 假设(n=1, QQQ 06/05, 待更多橙样本验证):
#   顶部预警 = 贴高 + 破10E + 贴20E + 带派发(放量下跌)。仅 MA 挤压不够(SPMO/MU 反例)。
WATCH_CUSHION = 3.0     # 距20E垫子 < 此值
WATCH_NEAR_HIGH = -6.0  # 距高 > 此值(贴高)
WATCH_DISTRIB = 1.5     # 末柱量比 ≥ 此值(放量派发)


def predict_health(f):
    """三档反解: 破20/50/200 = UNHEALTHY；在三线上方再分 HEALTHY / WATCH。
       WATCH(顶部预警) = 破10E 且 贴20E(垫子<3%) 且 贴高(>-6%) 且 放量派发(量比≥1.5)。"""
    if f["a200"] == -1:
        return "UNHEALTHY"
    if not (f["a20"] == 1 and f["a50"] == 1 and f["a200"] == 1):
        return "UNHEALTHY"
    if (f["a10"] == 0 and not pd.isna(f["d20"]) and 0 <= f["d20"] < WATCH_CUSHION
            and f["dist_hi"] > WATCH_NEAR_HIGH and f["vr"] >= WATCH_DISTRIB):
        return "WATCH"
    return "HEALTHY"


def predict_reduce(f, health):
    if health != "HEALTHY":
        return "-"
    return "REDUCE" if (not pd.isna(f["dev21_atr"]) and f["dev21_atr"] >= 2.3) else "Hold"


def main():
    do_fit = "--fit" in sys.argv
    df_lbl = pd.read_csv(CSV, comment="#")
    ctx = _make_ctx()
    bench_cache = {}

    def bench_for(date):
        if date not in bench_cache:
            bd = _pull(ctx, BENCH)
            bench_cache[date] = _slice_to_date(bd, date) if bd is not None else None
        return bench_cache[date]

    out = []
    h_ok = h_tot = r_ok = r_tot = 0
    for _, row in df_lbl.iterrows():
        tk, date = row["ticker"], row["date"]
        raw = _pull(ctx, tk)
        if raw is None or raw.empty:
            print(f"  ⚠ {tk} 拉取失败, 跳过")
            continue
        df = _slice_to_date(raw, date)
        if len(df) < 30:
            print(f"  ⚠ {tk} bar 不足, 跳过")
            continue
        f = features(df, row.get("panel_price"), bench_for(date))
        ph = predict_health(f)
        pr = predict_reduce(f, row["health"])
        h_match = (ph == row["health"]); h_ok += h_match; h_tot += 1
        if not (isinstance(row["reduce"], float) and pd.isna(row["reduce"])):
            r_match = (pr == str(row["reduce"]).replace("(重复确认)", "").strip()) or row["health"] != "HEALTHY"
            r_ok += int(r_match); r_tot += 1
        sc = row["score"]
        out.append(dict(
            date=date, tk=tk, score=sc, health=row["health"], reduce=row["reduce"],
            **f, pred_health=ph, H=("✓" if h_match else "✗"),
        ))

    res = pd.DataFrame(out)
    cols = ["date", "tk", "score", "health", "pred_health", "H", "reduce",
            "a10", "a20", "a50", "a200", "d20", "dist_hi", "dev21_atr",
            "cmf", "obv_sl", "distd", "vr", "rs"]
    print("\n=== Kova 校准特征表 (站位: 1=价在均线上方) ===")
    print(res[cols].to_string(index=False))
    print(f"\n健康度公式命中: {h_ok}/{h_tot} = {h_ok/max(h_tot,1)*100:.0f}%")
    print(f"Reduce 公式命中: {r_ok}/{r_tot} = {r_ok/max(r_tot,1)*100:.0f}%")

    if do_fit:
        fit_score(res)


def fit_score(res):
    """用站位/距高/stack/RS 拟合 Kova Score(剔除 NaN 样本)。"""
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("\n(--fit 需要 sklearn: pip install scikit-learn)")
        return
    d = res.copy()
    d = d[pd.to_numeric(d["score"], errors="coerce").notna()]
    d["score"] = d["score"].astype(float)
    X = d[["a200", "a50", "a20", "stack", "dist_hi", "rs"]].fillna(0).values
    y = d["score"].values
    m = LinearRegression().fit(X, y)
    print("\n=== Score 线性拟合(初步) ===")
    for name, coef in zip(["站上200S", "站上50S", "站上20E", "多头stack", "距高%", "RS"], m.coef_):
        print(f"  {name:10} 权重 = {coef:+7.2f}")
    print(f"  截距 = {m.intercept_:.1f}   R² = {m.score(X, y):.3f}   样本 n={len(d)}")


if __name__ == "__main__":
    main()
