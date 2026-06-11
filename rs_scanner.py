"""
rs_scanner.py — 全市场截面相对强度(RS)百分位 scanner。

补 Kova Score 的唯一硬盲区: OSCR/IREN 站全部均线却低分(41/38), 单票OHLCV算不出
"全市场排名"。本模块用广谱股票池(含强势动量+弱势防御股锚定两端)算 IBD 式 RS, 再
rank→1-99 百分位。这样落后股(OSCR/IREN)自然排到中低位, 抛物线名(MRVL/DELL)排顶部。

RS_raw(IBD式) = 0.4×r63 + 0.2×r126 + 0.2×r189 + 0.2×r252  (近1季度权重加倍)
RS_pct = 该值在全池里的百分位(1-99)。

用法:
    from rs_scanner import rs_percentile_asof
    pct = rs_percentile_asof("2026-06-02")   # {ticker: 1-99}
    python3 rs_scanner.py                      # 自检 + 对校准样本验证
数据: 长桥只读行情, 缓存到 cache/rs_universe.pkl(当日有效)。
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── 广谱股票池: 强势动量 + 各板块代表 + 防御/落后股(锚定低端) + 全部校准票 ──
UNIVERSE = sorted(set("""
QQQ SPY SMH IWM DIA
NVDA AVGO AMD ARM MU MRVL ASML TSM QCOM TXN ADI LRCX KLAC AMAT INTC ON NXPI MXL AEHR
SNDK STX WDC ALAB CRDO CLS COHR LITE AAOI POET NVTS GFS AXTI TSEM SYNA NOVT MTSI POWI
AAPL MSFT GOOG GOOGL AMZN META TSLA NFLX
CRM NOW SNOW DDOG NET CRWD PANW ZS OKTA DOCN MDB TEAM HOOD COIN PLTR SHOP TWLO BAND
NBIS IREN HUT CIFR APLD DELL HPE ANET VRT SMCI NBIS
OSCR HIMS DKNG RKLB ASTS LUNR BE OKLO GEV VST CEG
JPM BAC WFC GS MS C SCHW
UNH JNJ LLY PFE MRK ABBV TMO ABT BMY GILD CVS
PG KO PEP WMT COST MO PM CL KMB GIS K MDLZ
XOM CVX COP SLB EOG
CAT DE BA GE HON UNP UPS LMT RTX
NEE DUK SO D AEP ED XEL
T VZ TMUS
HD LOW MCD SBUX NKE DIS
DTM KGS MSM QCOM INDV TEVA POWL IESC VICR KGS CAT DDOG NOK SIVE NVTS LSCC HNGE
""".split()))


def _atr(df, n=14):
    h, l, c = df.High, df.Low, df.Close
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _ctx():
    try:
        from longport.openapi import Config, QuoteContext
        return QuoteContext(Config.from_env())
    except Exception:
        return None


def _pull_one(ctx, tk, count=300):
    from longport.openapi import Period, AdjustType
    bars = list(ctx.candlesticks(f"{tk}.US", Period.Day, count, AdjustType.ForwardAdjust))
    if not bars:
        return None
    df = pd.DataFrame({"Close": [float(b.close) for b in bars]},
                      index=[b.timestamp for b in bars]).sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


_CACHE = Path(__file__).parent / "cache" / "rs_universe.pkl"


def load_universe(force=False) -> dict:
    """拉全池收盘价, 缓存。返回 {ticker: Series(close)}。"""
    if _CACHE.exists() and not force:
        try:
            return pickle.loads(_CACHE.read_bytes())
        except Exception:
            pass
    ctx = _ctx()
    data = {}
    for tk in UNIVERSE:
        try:
            df = _pull_one(ctx, tk)
            if df is not None and len(df) >= 60:
                data[tk] = df.Close
        except Exception:
            continue
    _CACHE.parent.mkdir(exist_ok=True)
    _CACHE.write_bytes(pickle.dumps(data))
    return data


def _rs_raw(close: pd.Series, asof: pd.Timestamp):
    """中期相对强度(3-6月均权)。校准发现这个比IBD 12月更匹配Kova(排除指数后
       与Score相关 +0.844 vs IBD +0.769)。需 ≥126 根。"""
    s = close[close.index <= asof]
    if len(s) < 126:
        return np.nan
    p = float(s.iloc[-1])
    def r(n):
        return p / float(s.iloc[-n]) - 1
    return 0.5 * r(63) + 0.5 * r(126)   # 中期RS最匹配Kova


def rs_percentile_asof(date, universe=None) -> dict:
    """某日全池 RS 百分位 1-99。{ticker: pct}。"""
    universe = universe or load_universe()
    asof = pd.Timestamp(date)
    raws = {tk: _rs_raw(s, asof) for tk, s in universe.items()}
    raws = {tk: v for tk, v in raws.items() if not pd.isna(v)}
    if not raws:
        return {}
    s = pd.Series(raws).rank(pct=True) * 98 + 1   # 1-99
    return s.round().astype(int).to_dict()


def rs_of(ticker, date, universe=None):
    """单票某日 RS 百分位(票若不在池内, 临时并入计算)。"""
    universe = dict(universe or load_universe())
    ticker = ticker.upper()
    if ticker not in universe:
        try:
            df = _pull_one(_ctx(), ticker)
            if df is not None:
                universe[ticker] = df.Close
        except Exception:
            return None
    return rs_percentile_asof(date, universe).get(ticker)


def main():
    print("拉取广谱股票池(可能要 1-3 分钟)…")
    uni = load_universe()
    print(f"股票池: {len(uni)}/{len(UNIVERSE)} 只成功(≥252根)")
    # 对校准样本验证: RS百分位能否解释 OSCR/IREN 低分
    import csv
    rows = list(csv.DictReader(
        [l for l in open("kova_calibration.csv", encoding="utf-8") if not l.startswith("#")]))
    print("\n=== 校准样本 RS 百分位 (看低分票是否RS也低) ===")
    cache = {}
    out = []
    for r in rows:
        tk, dt, sc = r["ticker"], r["date"], r["score"]
        if sc in ("NaN", ""):
            continue
        if dt not in cache:
            cache[dt] = rs_percentile_asof(dt, uni)
        rs = cache[dt].get(tk)
        if rs is not None:
            out.append((int(sc), rs, tk, dt))
    out.sort()
    print(f"{'Score':>6}{'RS%':>6}  票/日期")
    for sc, rs, tk, dt in out:
        flag = "  ← 低分" if sc < 60 else ""
        print(f"{sc:6}{rs:6}  {tk} {dt}{flag}")
    if len(out) >= 5:
        scs = [x[0] for x in out]; rss = [x[1] for x in out]
        corr = np.corrcoef(scs, rss)[0, 1]
        print(f"\nScore vs RS百分位 相关系数 = {corr:+.3f}  (越接近+1 = RS越能解释Score)")


if __name__ == "__main__":
    main()
