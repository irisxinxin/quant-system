"""
kova_features.py — 为每个校准样本算丰富特征矩阵, 供深度优化分析。

读 kova_calibration.csv → 每个unique ticker拉一次OHLCV → 每个(ticker,date)算30+特征
→ 存 kova_features.csv。特征含: 站位/距高/ATR乖离/RS百分位 + 速度/加速度/趋势年龄/冲顶度。
目标: 攻 REDUCE第二因子(加速冲顶vs横住) + Score趋势成熟度(突破后爬分)。
"""
import pandas as pd
import numpy as np
import csv
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def _atr(df, n=14):
    h, l, c = df.High, df.Low, df.Close
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1).rolling(n).mean()


def _ctx():
    from longport.openapi import Config, QuoteContext
    return QuoteContext(Config.from_env())


def _pull(ctx, tk, count=400):
    from longport.openapi import Period, AdjustType
    bars = list(ctx.candlesticks(f"{tk}.US", Period.Day, count, AdjustType.ForwardAdjust))
    if not bars:
        return None
    df = pd.DataFrame({
        "Open": [float(b.open) for b in bars], "High": [float(b.high) for b in bars],
        "Low": [float(b.low) for b in bars], "Close": [float(b.close) for b in bars],
        "Volume": [float(b.volume) for b in bars],
    }, index=[b.timestamp for b in bars]).sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def _rs_pct_cache():
    """rs_scanner 广谱池: {date: {ticker: pct}}，惰性算。"""
    uni = pickle.loads((Path(__file__).parent / "cache" / "rs_universe.pkl").read_bytes())
    cache = {}
    def get(date):
        if date not in cache:
            asof = pd.Timestamp(date)
            raws = {}
            for tk, s in uni.items():
                ss = s[s.index <= asof]
                if len(ss) >= 22:
                    raws[tk] = float(ss.iloc[-1]) / float(ss.iloc[-21]) - 1   # r21 近1月
            cache[date] = (pd.Series(raws).rank(pct=True) * 98 + 1).round().to_dict() if raws else {}
        return cache[date]
    return get


def features(d):
    """d = 已截到当日的 df。算全特征。"""
    c, o, h, l, v = d.Close, d.Open, d.High, d.Low, d.Volume
    n = len(d)
    px = float(c.iloc[-1])
    e10 = c.ewm(span=10).mean(); e20 = c.ewm(span=20).mean(); e21 = c.ewm(span=21).mean()
    s50 = c.rolling(50).mean(); s200 = c.rolling(200).mean()
    atr = _atr(d); a = float(atr.iloc[-1])
    hi = float(c.tail(250).max())
    f = {}
    # 站位
    f["a10"] = int(px > e10.iloc[-1]); f["a20"] = int(px > e20.iloc[-1])
    f["a50"] = int(px > s50.iloc[-1]) if n >= 50 else -1
    f["a200"] = int(px > s200.iloc[-1]) if n >= 200 else -1
    f["d10"] = (px / e10.iloc[-1] - 1) * 100; f["d20"] = (px / e20.iloc[-1] - 1) * 100
    f["dist_hi"] = (px / hi - 1) * 100
    f["dev21_atr"] = (px - e21.iloc[-1]) / a
    f["dev10_atr"] = (px - e10.iloc[-1]) / a
    # 速度 / 加速度
    def ret(k):
        return (px / float(c.iloc[-1 - k]) - 1) * 100 if n > k else np.nan
    f["ret_1d"] = ret(1); f["ret_3d"] = ret(3); f["ret_5d"] = ret(5); f["ret_10d"] = ret(10); f["ret_21d"] = ret(21)
    f["accel_3d"] = (ret(3) - ((float(c.iloc[-4]) / float(c.iloc[-7]) - 1) * 100)) if n > 7 else np.nan  # 3日动量变化
    f["slope10_5d"] = (e10.iloc[-1] / e10.iloc[-6] - 1) * 100 if n > 6 else np.nan  # 10EMA 5日斜率%
    f["slope21_5d"] = (e21.iloc[-1] / e21.iloc[-6] - 1) * 100 if n > 6 else np.nan
    # 冲顶/延伸持续性
    band = e21 + 2 * atr
    above = (c > band)
    ext = 0
    for x in above.iloc[::-1]:
        if x: ext += 1
        else: break
    f["days_ext"] = ext   # 连续在21E+2ATR上方天数
    # 距上次新高天数(越小=越新)
    roll_max = c.rolling(60).max()
    is_high = (c >= roll_max - 1e-9)
    since_hi = 0
    for x in is_high.iloc[::-1]:
        if x: break
        since_hi += 1
    f["days_since_hi"] = since_hi
    # 连续上涨天数
    up = (c > c.shift())
    streak = 0
    for x in up.iloc[::-1]:
        if x: streak += 1
        else: break
    f["up_streak"] = streak
    # 今日K线: gap + 振幅(检测blow-off/spike)
    f["gap_pct"] = (float(o.iloc[-1]) / float(c.iloc[-2]) - 1) * 100 if n > 1 else 0
    f["range_pct"] = (float(h.iloc[-1]) - float(l.iloc[-1])) / px * 100
    # 趋势年龄(成熟度)
    def days_above(ser):
        ab = (c > ser); cnt = 0
        for x in ab.iloc[::-1]:
            if x: cnt += 1
            else: break
        return cnt
    f["days_above_10e"] = days_above(e10)
    f["days_above_20e"] = days_above(e20)
    f["days_above_50s"] = days_above(s50) if n >= 50 else -1
    f["days_above_200s"] = days_above(s200) if n >= 200 else -1
    # 量
    vol_ma = v.rolling(20).mean()
    f["vr"] = float(v.iloc[-1] / (vol_ma.iloc[-1] + 1e-9))
    f["bars"] = n
    return {k: (round(val, 3) if isinstance(val, float) and not pd.isna(val) else val) for k, val in f.items()}


def main():
    rows = [r for r in csv.DictReader(
        [l for l in open("kova_calibration.csv", encoding="utf-8") if not l.startswith("#")])]
    ctx = _ctx()
    rs_get = _rs_pct_cache()
    raw_cache = {}
    out = []
    for r in rows:
        tk, dt = r["ticker"], r["date"]
        if tk not in raw_cache:
            try:
                raw_cache[tk] = _pull(ctx, tk)
            except Exception:
                raw_cache[tk] = None
        raw = raw_cache[tk]
        if raw is None:
            continue
        d = raw[raw.index <= pd.Timestamp(dt) + pd.Timedelta(days=1)]
        if len(d) < 60:
            continue
        ff = features(d)
        rs = rs_get(dt).get(tk)
        rec = dict(date=dt, ticker=tk, score=r["score"], health=r["health"], reduce=r["reduce"],
                   panel_price=r.get("panel_price", ""), rs_pct=rs, **ff)
        out.append(rec)
    df = pd.DataFrame(out)
    df.to_csv("kova_features.csv", index=False)
    print(f"特征矩阵: {len(df)} 行 × {len(df.columns)} 列 → kova_features.csv")
    print("列:", list(df.columns))


if __name__ == "__main__":
    main()
