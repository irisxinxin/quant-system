"""
stops_core.py — 持仓止损计算核心（按 Kova A–E 状态机）。
被 app.py(/api/stops 实时) 与 kova_stops.py(静态生成) 共用。
数据源：长桥 QuoteContext 优先，失败回退 yfinance（只读行情，不下单）。
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# 真实持仓: ticker -> 成本（取自长桥 App 持仓截图，可在此维护）
HOLDINGS = {
    "MXL": 98.273, "GFS": 81.663, "OKTA": 110.795, "GOOG": 359.850,
    "HUT": 115.678, "NOK": 14.788, "POWI": 84.776, "ASTS": 100.200,
    "NVTS": 32.255, "DRAM": 40.736, "COHR": 407.812, "DELL": 251.998,
    "SNDK": 1774.870, "NBIS": 210.885, "ARM": 260.377, "BE": 285.573,
    "ALAB": 310.780, "MRVL": 129.870,
}

_STATE_ORDER = {"E 抛物线": 0, "D 趋势": 1, "C 确认上涨": 2, "B 微利区": 3, "A 亏损区": 4}


def _atr(df, n=14):
    h, l, c = df.High, df.Low, df.Close
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ─── 数据源 ───
def _fetch_longport(ctx, tk, count=250):
    from longport.openapi import Period, AdjustType
    bars = list(ctx.candlesticks(f"{tk}.US", Period.Day, count, AdjustType.ForwardAdjust))
    if not bars:
        return None
    return pd.DataFrame({
        "High": [float(b.high) for b in bars], "Low": [float(b.low) for b in bars],
        "Close": [float(b.close) for b in bars], "Volume": [float(b.volume) for b in bars],
    }, index=[b.timestamp for b in bars]).sort_index()


def _fetch_yfinance(tk):
    import yfinance as yf
    df = yf.download(tk, period="1y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["High", "Low", "Close", "Volume"]].copy()


def _make_quote_ctx():
    """尝试创建长桥行情上下文；失败返回 None（回退 yfinance）。"""
    try:
        from longport.openapi import Config, QuoteContext
        return QuoteContext(Config.from_env())
    except Exception:
        return None


def _analyze(df, cost):
    c = df.Close
    last = float(c.iloc[-1])
    ema5 = float(c.ewm(span=5).mean().iloc[-1])
    ema10 = float(c.ewm(span=10).mean().iloc[-1])
    ema21 = float(c.ewm(span=21).mean().iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1])
    atr = float(_atr(df).iloc[-1])
    dist21 = (last / ema21 - 1) * 100
    gain = (last / cost - 1) * 100
    swing = float(df.Low.iloc[-10:].min())

    if gain < 0:
        state, sc, win = "A 亏损区", "#f87171", False
        follow, flabel = cost * 0.92, "初始硬止损 成本×0.92（可挂单）"
        action = "守初始止损，绝不下移；逼近即按纪律砍"
    elif gain < 3:
        state, sc, win = "B 微利区", "#9ba6b4", False
        follow, flabel = cost * 0.92, "初始硬止损 成本×0.92（可挂单）"
        action = "忍住，给突破至少 3 天，先别提保本"
    elif gain < 10:
        state, sc, win = "C 确认上涨", "#f59e0b", False
        follow, flabel = cost * 1.001, "保本 成本×1.001（可挂单）"
        action = "止损提到保本+0.1%（收盘上半区确认）"
    elif dist21 > 25:
        state, sc, win = "E 抛物线", "#39c5cf", True
        follow, flabel = ema5 * 0.99, "5EMA×0.99（收盘判断）"
        action = "📐 手动画趋势线确认；5EMA 紧跟随；主动减仓 20–50%"
    else:
        state, sc, win = "D 趋势", "#4ade80", True
        follow, flabel = ema10 * 0.99, "10EMA×0.99（收盘判断）"
        action = "EMA 追踪，让利润奔跑；收盘跌破才离场"

    disaster = round(swing * 0.98, 2) if win else None
    return dict(
        tk="", cost=round(cost, 2), last=round(last, 2), gain=round(gain, 1),
        state=state, sc=sc, ema5=round(ema5, 2), ema10=round(ema10, 2), ema21=round(ema21, 2),
        sma50=round(sma50, 2), atr=round(atr, 2), dist21=round(dist21, 1),
        follow=round(follow, 2), follow_label=flabel, disaster=disaster,
        swing=round(swing, 2), action=action,
        date=str(df.index[-1].date()),
    )


def compute_stops(holdings: dict | None = None) -> dict:
    """返回 {rows:[...], data_date, source}。供缓存层调用（无参数也可）。"""
    holdings = holdings or HOLDINGS
    ctx = _make_quote_ctx()
    source = "longport" if ctx else "yfinance"
    rows = []
    for tk, cost in holdings.items():
        df = None
        if ctx:
            try:
                df = _fetch_longport(ctx, tk)
            except Exception:
                df = None
        if df is None or df.empty:
            try:
                df = _fetch_yfinance(tk)
                if source == "longport":
                    source = "longport+yfinance"
            except Exception:
                df = None
        if df is None or df.empty or len(df) < 30:   # 30+ 根足够算 10/21EMA；SMA50 不足则留空
            continue
        r = _analyze(df, cost)
        r["tk"] = tk
        rows.append(r)

    rows.sort(key=lambda r: (_STATE_ORDER.get(r["state"], 9), -r["gain"]))
    data_date = rows[0]["date"] if rows else "-"
    return {"rows": rows, "data_date": data_date, "source": source}
