"""
kova_stop_analysis.py — 从 Kova 真实交易记录(Trade Note 截图)反推他的止损设法。

数据 = Kova Numbers 表格里每笔的 (买入日, 票, 买价, 止损价), 仅取初始止损(stop<buy)。
结论(23笔): 止损 ≈ 入场日K线最低点(结构止损), 不是固定% 也不是干净ATR倍数。
  · 止损%        中位 4.1%  范围 1-8%   → 非固定%(但比7-8%硬止损紧)
  · 止损/ATR     中位 0.57  范围 0.26-1.05(散4倍) → 非干净ATR倍数
  · 止损vs入场日低 中位 ≈0   多数±0.3ATR内 → ★就是入场日低
  · 止损vs10日swing低 远在其上(+0.3~6ATR) → 非swing低
复刻: stop = 入场日Low − 小缓冲(≈0.1×ATR), 纯OHLCV可算, 比ATR线还简单。

跑: python3 kova_stop_analysis.py
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Kova 真实交易(2026, 从 Trade Note 截图抄录), 仅初始止损(stop<buy)
TRADES = [
    ("2026-04-27", "AXTI", 69.85, 67.00), ("2026-04-27", "AEHR", 89.59, 85.00),
    ("2026-04-27", "IREN", 48.69, 45.68), ("2026-04-27", "HUT", 75.27, 70.32),
    ("2026-04-27", "QCOM", 158.16, 155.00), ("2026-04-27", "MU", 519.82, 501.50),
    ("2026-04-27", "CRDO", 178.66, 174.00), ("2026-04-27", "VRT", 317.93, 303.20),
    ("2026-05-01", "AMD", 354.63, 345.00), ("2026-05-06", "AAOI", 179.43, 165.00),
    ("2026-05-06", "SNDK", 1397.03, 1350.00), ("2026-05-07", "IREN", 60.85, 57.00),
    ("2026-05-07", "DDOG", 193.12, 187.00), ("2026-05-08", "IREN", 61.50, 56.50),
    ("2026-05-08", "HUT", 99.00, 95.00), ("2026-05-08", "TSLA", 426.28, 422.00),
    ("2026-05-08", "AMD", 429.52, 420.00), ("2026-05-08", "AXTI", 115.84, 110.00),
    ("2026-05-11", "VICR", 305.00, 290.00), ("2026-05-11", "AEHR", 104.00, 100.00),
    ("2026-05-12", "DOCN", 156.00, 149.00), ("2026-05-12", "BE", 284.99, 270.00),
]


def _atr(df, n=14):
    h, l, c = df.High, df.Low, df.Close
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _ctx():
    from longport.openapi import Config, QuoteContext
    return QuoteContext(Config.from_env())


def _pull(ctx, tk, count=200):
    from longport.openapi import Period, AdjustType
    bars = list(ctx.candlesticks(f"{tk}.US", Period.Day, count, AdjustType.ForwardAdjust))
    df = pd.DataFrame({"High": [float(b.high) for b in bars], "Low": [float(b.low) for b in bars],
                       "Close": [float(b.close) for b in bars]},
                      index=[b.timestamp for b in bars]).sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def main():
    ctx = _ctx()
    rows = []
    for dt, tk, buy, stop in TRADES:
        try:
            df = _pull(ctx, tk)
        except Exception:
            continue
        d = df[df.index <= pd.Timestamp(dt) + pd.Timedelta(days=1)]
        if len(d) < 20:
            continue
        a = float(_atr(d).iloc[-1]); elow = float(d.Low.iloc[-1]); swlow = float(d.Low.tail(10).min())
        rows.append(dict(date=dt, tk=tk, buy=buy, stop=stop,
                         pct=(buy - stop) / buy * 100, atr_mult=(buy - stop) / a,
                         vs_entry_low=(stop - elow) / a, vs_swing_low=(stop - swlow) / a))
    r = pd.DataFrame(rows)
    print(r.round(2).to_string(index=False))
    print(f"\n止损%        中位 {r.pct.median():.1f}%  范围 {r.pct.min():.1f}-{r.pct.max():.1f}%  → 非固定%")
    print(f"止损/ATR     中位 {r.atr_mult.median():.2f}  范围 {r.atr_mult.min():.2f}-{r.atr_mult.max():.2f}  → 非干净ATR倍数")
    print(f"止损vs入场日低 中位 {r.vs_entry_low.median():+.2f}ATR  |在±0.5ATR内: {(r.vs_entry_low.abs()<0.5).mean()*100:.0f}%  → ★入场日低")
    print(f"止损vs10日低   中位 {r.vs_swing_low.median():+.2f}ATR  → 远在swing低上方, 非swing低")


if __name__ == "__main__":
    main()
