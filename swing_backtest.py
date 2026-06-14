"""
swing_backtest.py — 验证波段择时(日线持有, 吃隔夜+趋势)能否胜过日内/接近买入持有。

波段规则(long-only, 日线, mark-to-market): 站上均线趋势线就持有, 收盘跌破 MA×0.99 出场, 重新站上再进。
对比: 买入持有 vs 波段(10E/20E/50S 三档) — 看总收益 + 最大回撤 + 在场时间。
目标: 证明波段吃到趋势大头(远胜日内+19%), 且回撤比裸持有小。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from longport.openapi import Config, QuoteContext, Period, AdjustType

UNIVERSE = {  # 杠杆ETF + 港股2x + 几只强趋势股做对照
    "7747.HK": "三星2x", "7709.HK": "海力士2x", "SOXL.US": "半导体3x", "NVDL.US": "英伟达2x",
    "MSFL.US": "微软2x?", "TSLL.US": "特斯拉2x", "MXL.US": "MaxLinear", "NBIS.US": "Nebius", "AAOI.US": "AAOI",
}


def pull(ctx, sym, n=300):
    try:
        b = list(ctx.candlesticks(sym, Period.Day, n, AdjustType.ForwardAdjust if sym.endswith('.US') else AdjustType.NoAdjust))
        c = pd.Series([float(x.close) for x in b], index=[x.timestamp for x in b]).sort_index()
        return c
    except Exception as e:
        print(f"  {sym} ❌ {e}"); return None


def maxdd(eq):
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min()) * 100


def swing(close, ma, s200, use200=True):
    """日线波段: 收盘>MA(且>200S)进; 收盘<MA×0.99出。返回(总收益%, 最大回撤%, 交易数, 在场比例%)。"""
    eq = [1.0]; inpos = False; trades = 0; bars_in = 0
    for i in range(1, len(close)):
        if inpos:
            eq.append(eq[-1] * close.iloc[i] / close.iloc[i - 1]); bars_in += 1
            if close.iloc[i] < ma.iloc[i] * 0.99:
                inpos = False
        else:
            eq.append(eq[-1])
            cond = close.iloc[i] > ma.iloc[i] and (not use200 or pd.isna(s200.iloc[i]) or close.iloc[i] > s200.iloc[i])
            if cond:
                inpos = True
    eq = np.array(eq)
    return (eq[-1] - 1) * 100, maxdd(eq), trades, bars_in / len(close) * 100


def main():
    ctx = QuoteContext(Config.from_env())
    print(f"{'票':9}{'名称':10}{'天数':>5}{'买入持有':>9}{'波段10E':>9}{'(回撤)':>8}{'波段20E':>9}{'(回撤)':>8}{'波段50S':>9}{'BH回撤':>8}")
    print("-" * 95)
    for sym, name in UNIVERSE.items():
        c = pull(ctx, sym)
        if c is None or len(c) < 60:
            continue
        e10 = c.ewm(span=10).mean(); e20 = c.ewm(span=20).mean(); s50 = c.rolling(50).mean(); s200 = c.rolling(200).mean()
        bh = (c.iloc[-1] / c.iloc[0] - 1) * 100
        bh_dd = maxdd((c / c.iloc[0]).values)
        r10, dd10, _, _ = swing(c, e10, s200)
        r20, dd20, _, t20 = swing(c, e20, s200)
        r50, dd50, _, _ = swing(c, s50, s200)
        print(f"{sym:9}{name:10}{len(c):>5}{bh:>+8.0f}%{r10:>+8.0f}%{dd10:>7.0f}%{r20:>+8.0f}%{dd20:>7.0f}%{r50:>+8.0f}%{bh_dd:>7.0f}%")
    print("\n说明: 波段=收盘站上均线持有(含隔夜)、跌破均线×0.99出场。对比日内策略只吃盘中(三星2x日内仅+19%)。")
    print("      看波段能否吃到趋势大头 + 回撤是否比买入持有(BH回撤)小。")


if __name__ == "__main__":
    main()
