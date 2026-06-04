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
# None = 待补成本（EMA 止损照常算，但盈亏%/A·B·C 状态需补成本后才准）
HOLDINGS = {
    "MXL": 98.273, "GFS": 81.663, "OKTA": 110.795, "GOOG": 359.850,
    "HUT": 115.678, "NOK": 14.788, "POWI": 84.776, "ASTS": 100.200,
    "NVTS": 32.255, "DRAM": 40.736, "COHR": 407.812, "DELL": 251.998,
    "SNDK": 1774.870, "NBIS": 210.885, "ARM": 260.377, "BE": 285.573,
    "ALAB": 310.780, "MRVL": 129.870,
    "STRL": 970.0, "ORCL": 216.4, "PENG": 69.33, "ST": 51.675, "TSEM": 252.283,
    "MU": 758.39,
}

# ticker -> 分类标签
SECTORS = {
    "MXL": "半导体/模拟", "GFS": "半导体/代工", "OKTA": "软件/安全", "GOOG": "大盘/核心",
    "HUT": "加密/矿工", "NOK": "光子/通信", "POWI": "半导体/电源", "ASTS": "太空/卫星",
    "NVTS": "半导体/功率", "DRAM": "存储ETF", "COHR": "光子/高速连接", "DELL": "AI硬件/数据中心",
    "SNDK": "存储", "NBIS": "AI算力/基建", "ARM": "半导体/IP", "BE": "AI电力/燃料电池",
    "ALAB": "半导体/AI连接", "MRVL": "半导体/连接",
    "STRL": "工业/基建", "ORCL": "软件/AI数据", "PENG": "AI/HPC",
    "ST": "传感器/汽车", "TSEM": "半导体/代工", "MU": "存储",
}

_STATE_ORDER = {"E 抛物线": 0, "D 趋势": 1, "C 确认上涨": 2, "B 微利区": 3,
                "A 亏损区": 4, "— 待成本": 5}


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
    gain = (last / cost - 1) * 100 if cost is not None else None
    swing = float(df.Low.iloc[-10:].min())

    if gain is None:
        # 无成本：按价格结构给 trail，A/B/C 待补成本后才能定
        if dist21 > 25:
            state, sc, win = "E 抛物线", "#39c5cf", True
            follow, flabel = ema5 * 0.99, "5EMA×0.99（待补成本）"
            action = "⚠ 待补成本；抛物线：📐手动画趋势线 + 减仓 20–50%"
        elif last > ema21:
            state, sc, win = "D 趋势", "#4ade80", True
            follow, flabel = ema10 * 0.99, "10EMA×0.99（待补成本）"
            action = "⚠ 待补成本；EMA 追踪，收盘跌破才离场"
        else:
            state, sc, win = "— 待成本", "#9ba6b4", False
            follow, flabel = ema10 * 0.99, "10EMA×0.99（参考）"
            action = "⚠ 补成本后才能定 A/B/C 与盈亏"
    elif gain < 0:
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
        tk="", cost=round(cost, 2) if cost is not None else None,
        last=round(last, 2), gain=round(gain, 1) if gain is not None else None,
        state=state, sc=sc, ema5=round(ema5, 2), ema10=round(ema10, 2), ema21=round(ema21, 2),
        sma50=round(sma50, 2), atr=round(atr, 2), dist21=round(dist21, 1),
        follow=round(follow, 2), follow_label=flabel, disaster=disaster,
        swing=round(swing, 2), action=action,
        date=str(df.index[-1].date()),
    )


def trendline_points(df, win=12):
    """近 win 根的上升趋势线：连最早 pivot 低点→其后最高 pivot 低点，外推到今日。
    返回 lightweight-charts 用的两点 [{time,value},{time,value}]，无则 None。"""
    sub = df.iloc[-win:]
    lows = sub.Low.values
    idx = sub.index
    n = len(sub)
    pts = [(i, float(lows[i])) for i in range(1, n - 1)
           if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]]
    if len(pts) < 2:
        return None
    anchor = pts[0]
    after = [p for p in pts if p[0] > anchor[0] and p[1] > anchor[1]]
    if not after:
        return None
    b = max(after, key=lambda p: p[1])
    slope = (b[1] - anchor[1]) / (b[0] - anchor[0])
    if slope <= 0:
        return None
    tl_last = b[1] + slope * ((n - 1) - b[0])
    return [
        {"time": int(idx[anchor[0]].timestamp()), "value": round(anchor[1], 4)},
        {"time": int(idx[-1].timestamp()), "value": round(tl_last, 4)},
    ]


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
        r["sector"] = SECTORS.get(tk, "")
        rows.append(r)

    rows.sort(key=lambda r: (_STATE_ORDER.get(r["state"], 9),
                             -(r["gain"] if r["gain"] is not None else -1e9)))
    data_date = rows[0]["date"] if rows else "-"
    return {"rows": rows, "data_date": data_date, "source": source}
