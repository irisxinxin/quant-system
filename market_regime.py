"""
market_regime.py — Kova 式"大盘风险开关"(组合级核按钮)。

反推自 Kova 06/05 10am 全清的决策, 理由"index high volume selloff"。
这是整套指标体系的最高层: 大盘放量抛售时, 再强的个股也让路 → 减仓/清仓/不做多。

信号(对 QQQ + SPY 双确认):
  · 放量抛售日 = 指数当日跌 ≤ -1.5% 且 量比 ≥ 1.5×50日均量 (O'Neil 派发日加强版)
  · 派发日累积 = 近25日 (收阴 且 放量) 计数, ≥5 = O'Neil 见顶警告
  · 趋势闸 = 指数 < 10EMA → 不做多 (用户心得: QQQ<10EMA不做多)
综合 → regime: RISK_ON / CAUTION / RISK_OFF + 文字理由。

数据源: 长桥 QuoteContext(只读), 失败回退 yfinance。
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SELLOFF_DROP = -1.5     # 放量抛售: 当日跌幅阈值(%)
SELLOFF_VR = 1.5        # 放量抛售: 量比阈值(×50日均量)
DIST_DROP = -0.2        # 派发日: 收阴阈值(%)
DIST_CLUSTER = 5        # 近25日派发日 ≥ 此值 = 见顶警告


def _pull_longport(ctx, tk, count=300):
    from longport.openapi import Period, AdjustType
    bars = list(ctx.candlesticks(f"{tk}.US", Period.Day, count, AdjustType.ForwardAdjust))
    if not bars:
        return None
    return pd.DataFrame({
        "Open": [float(b.open) for b in bars], "High": [float(b.high) for b in bars],
        "Low": [float(b.low) for b in bars], "Close": [float(b.close) for b in bars],
        "Volume": [float(b.volume) for b in bars],
    }, index=[b.timestamp for b in bars]).sort_index()


def _pull_yf(tk):
    import yfinance as yf
    df = yf.download(tk, period="1y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def _make_ctx():
    try:
        from longport.openapi import Config, QuoteContext
        return QuoteContext(Config.from_env())
    except Exception:
        return None


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


def _analyze_index(df):
    c = df.Close
    chg = c.pct_change() * 100
    vol_ma = df.Volume.rolling(50).mean()
    vr = df.Volume / vol_ma
    e10 = c.ewm(span=10).mean()
    dist = (chg <= DIST_DROP) & (df.Volume > df.Volume.shift())
    selloff = (chg <= SELLOFF_DROP) & (vr >= SELLOFF_VR)
    return dict(
        date=str(df.index[-1].date()),
        close=round(float(c.iloc[-1]), 2),
        chg=round(float(chg.iloc[-1]), 2),
        vr=round(float(vr.iloc[-1]), 2),
        below_10e=bool(c.iloc[-1] < e10.iloc[-1]),
        selloff=bool(selloff.iloc[-1]),
        dist_days_25=int(dist.tail(25).sum()),
    )


def market_regime(ctx=None) -> dict:
    """返回大盘风险状态。RISK_OFF=放量抛售/见顶簇 → 清仓/不做多。"""
    ctx = ctx or _make_ctx()
    idx = {}
    for tk in ("QQQ", "SPY"):
        df = _pull(ctx, tk)
        if df is not None and not df.empty:
            idx[tk] = _analyze_index(df)
    if not idx:
        return {"regime": "UNKNOWN", "reason": "无法获取指数数据", "indexes": {}}

    any_selloff = any(v["selloff"] for v in idx.values())
    any_cluster = any(v["dist_days_25"] >= DIST_CLUSTER for v in idx.values())
    any_below10 = any(v["below_10e"] for v in idx.values())
    all_below10 = all(v["below_10e"] for v in idx.values())

    reasons = []
    if any_selloff:
        sl = [f"{k} {v['chg']:+.1f}%/量比{v['vr']:.1f}" for k, v in idx.items() if v["selloff"]]
        reasons.append("🚨 指数放量抛售: " + "; ".join(sl))
    if any_cluster:
        cl = [f"{k} {v['dist_days_25']}个" for k, v in idx.items() if v["dist_days_25"] >= DIST_CLUSTER]
        reasons.append("⚠ 派发日成簇(≥5,O'Neil见顶): " + "; ".join(cl))
    if all_below10:
        reasons.append("📉 QQQ+SPY 双双跌破 10EMA → 不做多")
    elif any_below10:
        reasons.append("📉 有指数跌破 10EMA")

    if any_selloff or (any_cluster and any_below10):
        regime = "RISK_OFF"      # 清仓/大幅减仓, 不开新多
    elif any_cluster or any_below10:
        regime = "CAUTION"       # 警觉, 收紧止损, 不加仓
    else:
        regime = "RISK_ON"       # 正常持有/可开仓
    if not reasons:
        reasons.append("✅ 指数站上 10EMA、无放量抛售、派发日未成簇")

    return {"regime": regime, "reason": "；".join(reasons), "indexes": idx,
            "as_of": idx.get("QQQ", list(idx.values())[0])["date"]}


if __name__ == "__main__":
    import json
    r = market_regime()
    print(f"\n大盘风险状态: {r['regime']}  (截至 {r.get('as_of','-')})")
    print(f"理由: {r['reason']}")
    print("\n各指数:")
    for k, v in r["indexes"].items():
        print(f"  {k}: 收{v['close']} 日{v['chg']:+.2f}% 量比{v['vr']} "
              f"{'破10E' if v['below_10e'] else '站10E'} 近25日派发{v['dist_days_25']} "
              f"{'🚨放量抛售' if v['selloff'] else ''}")
