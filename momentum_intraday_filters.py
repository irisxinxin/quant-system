"""
momentum_intraday_filters.py — 用经典动量交易体系(O'Neil/Minervini/Morales)的日线条件,
给日内突破入场做"质量闸门", 诚实回测哪个条件真能提升。

不是套Kova Score, 是套我们反推Kova时学到的体系本身:
  · 趋势模板(Minervini 8条) · 52周高点距离(O'Neil 15%) · Pocket Pivot(Morales)
  · 量缩VCP(Minervini FU) · 派发日(O'Neil distribution) · RS线新高 · 市场环境M(QQQ)
全部【前一日收盘】算(防未来函数), join到日内FILLED交易, 分桶比 PF/胜率/期望。
仅真股票(动量体系是给真股选龙头的, 杠杆/反向ETF不适用)。

用法: python3 momentum_intraday_filters.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

INV = {"TSLZ.US", "NVDS.US", "SOXS.US"}
LEV = {"TSLL.US", "SOXL.US", "NVDL.US", "MSFU.US", "MSFL.US", "CONL.US"}


def _ctx():
    from longport.openapi import Config, QuoteContext
    return QuoteContext(Config.from_env())


def _pull(ctx, tk, n=600):
    from longport.openapi import Period, AdjustType
    try:
        b = list(ctx.candlesticks(f"{tk}.US", Period.Day, n, AdjustType.ForwardAdjust))
        df = pd.DataFrame({"High": [float(x.high) for x in b], "Low": [float(x.low) for x in b],
                           "Open": [float(x.open) for x in b],
                           "Close": [float(x.close) for x in b], "Volume": [float(x.volume) for x in b]},
                          index=[x.timestamp for x in b]).sort_index()
        df.index = (df.index.tz_localize(None) if df.index.tz else df.index).normalize()
        return df
    except Exception:
        return None


def momentum_daily(df, qqq):
    """逐日动量体系特征(每行=截至当日收盘已知)。交易日D取<D最后一行=前日已知。"""
    c, h, l, o, v = df.Close, df.High, df.Low, df.Open, df.Volume
    s50 = c.rolling(50).mean(); s150 = c.rolling(150).mean(); s200 = c.rolling(200).mean()
    hi52 = c.rolling(252).max(); lo52 = c.rolling(252).min()
    volma = v.rolling(20).mean()
    # Minervini 趋势模板 (8条 → 计数)
    tt = (
        (c > s50).astype(int) + (c > s150).astype(int) + (c > s200).astype(int)
        + (s50 > s150).astype(int) + (s150 > s200).astype(int)
        + (s200 > s200.shift(20)).astype(int)             # 200SMA 上升
        + (c >= 1.25 * lo52).astype(int)                   # ≥52周低 25% 上方
        + (c >= 0.85 * hi52).astype(int)                   # 距52周高 15% 内
    )
    near_high = (c / hi52 - 1) * 100                        # 距52周高 %
    # Pocket Pivot: 今为阳线 且 量 > 过去10日"下跌日"最大量
    down_vol = v.where(c < c.shift(), 0)
    max_down10 = down_vol.rolling(10).max()
    pocket = ((c > c.shift()) & (v > max_down10)).astype(int)
    # 量缩 (VCP/FU): 量 < 0.7×20日均量 且 窄幅
    rng = (h - l) / c
    dryup = ((v < 0.7 * volma) & (rng < rng.rolling(20).mean())).astype(int)
    # 派发日数(近25日): 收阴≥0.2% 且 量>1.5×均量
    dist_day = ((c < c.shift() * 0.998) & (v > 1.5 * volma)).astype(int)
    dist_25 = dist_day.rolling(25).sum()
    # RS线 vs QQQ 新高(50日)
    idx = c.index.intersection(qqq.index)
    ratio = (c.reindex(idx) / qqq.Close.reindex(idx)).reindex(c.index)
    rs_newhi = (ratio >= ratio.rolling(50).max() - 1e-12).astype(int)
    return pd.DataFrame({"tt": tt, "near_high": near_high, "pocket": pocket,
                         "dryup": dryup, "dist25": dist_25, "rs_newhi": rs_newhi}, index=c.index)


def market_daily(qqq):
    """市场环境M (QQQ): 站10E/站21E/近25日派发数。"""
    c, v = qqq.Close, qqq.Volume
    e10 = c.ewm(span=10).mean(); e21 = c.ewm(span=21).mean(); volma = v.rolling(50).mean()
    dist_day = ((c < c.shift() * 0.998) & (v > 1.4 * volma)).astype(int)
    return pd.DataFrame({"mkt_a10": (c > e10).astype(int), "mkt_a21": (c > e21).astype(int),
                         "mkt_dist25": dist_day.rolling(25).sum()}, index=c.index)


def lookup(fdf, d):
    prior = fdf[fdf.index < pd.Timestamp(d)]
    return prior.iloc[-1] if len(prior) else None


def pf(p):
    g = p[p > 0].sum(); b = -p[p < 0].sum(); return g / b if b > 0 else 99.0


def main():
    tr = pd.read_csv("kova_intraday_gate_trades.csv")
    tr = tr[~tr.symbol.isin(INV | LEV)].copy()   # 仅真股票
    tr["day"] = pd.to_datetime(tr.day)
    ctx = _ctx()
    qqq = _pull(ctx, "QQQ", 700)
    mkt = market_daily(qqq)
    feats = {}
    rows = []
    for _, t in tr.iterrows():
        tk = t.symbol.replace(".US", "")
        if tk not in feats:
            dd = _pull(ctx, tk)
            feats[tk] = momentum_daily(dd, qqq) if dd is not None else None
        f = feats[tk]
        if f is None:
            continue
        kf = lookup(f, t.day); km = lookup(mkt, t.day)
        if kf is None or km is None or pd.isna(kf["tt"]):
            continue
        rows.append(dict(symbol=t.symbol, day=t.day, pnl=t.pnl, win=t.win,
                         tt=int(kf["tt"]), near_high=float(kf["near_high"]), pocket=int(kf["pocket"]),
                         dryup=int(kf["dryup"]), dist25=int(kf["dist25"]), rs_newhi=int(kf["rs_newhi"]),
                         mkt_a10=int(km["mkt_a10"]), mkt_a21=int(km["mkt_a21"]), mkt_dist25=int(km["mkt_dist25"])))
    d = pd.DataFrame(rows)
    base = pf(d.pnl)
    print(f"=== 真股票日内交易 n={len(d)}  基线 PF={base:.2f} 胜率={d.win.mean()*100:.1f}% 总={d.pnl.sum():+.0f}% ===")

    def rep(name, groups):
        print(f"\n── {name} ──  {'桶':22}{'n':>6}{'留存':>6}{'胜率':>7}{'PF':>7}{'平均%':>9}{'总%':>8}")
        for label, m in groups:
            s = d[m]
            if len(s) < 15:
                print(f"{'':24}{label:22}{len(s):>6}  样本不足"); continue
            print(f"{'':24}{label:22}{len(s):>6}{len(s)/len(d)*100:>5.0f}%{s.win.mean()*100:>6.1f}%{pf(s.pnl):>7.2f}{s.pnl.mean():>+9.3f}{s.pnl.sum():>+8.0f}")

    rep("趋势模板(Minervini条数)", [("tt≤5", d.tt <= 5), ("tt=6", d.tt == 6), ("tt=7", d.tt == 7), ("tt=8满分", d.tt == 8)])
    rep("距52周高(O'Neil)", [("高15%内", d.near_high > -15), ("高15-30%", (d.near_high <= -15) & (d.near_high > -30)), ("高>30%外", d.near_high <= -30)])
    rep("Pocket Pivot(前日)", [("有PP", d.pocket == 1), ("无PP", d.pocket == 0)])
    rep("量缩FU(前日)", [("量缩", d.dryup == 1), ("非量缩", d.dryup == 0)])
    rep("个股派发日(近25)", [("≤2清洁", d.dist25 <= 2), ("3-5", (d.dist25 >= 3) & (d.dist25 <= 5)), ("≥6沉重", d.dist25 >= 6)])
    rep("RS线新高(50日)", [("RS新高", d.rs_newhi == 1), ("非新高", d.rs_newhi == 0)])
    rep("市场环境M(QQQ)", [("站10E", d.mkt_a10 == 1), ("破10E", d.mkt_a10 == 0),
                          ("站21E+派发<5", (d.mkt_a21 == 1) & (d.mkt_dist25 < 5)), ("破21E或派发≥5", (d.mkt_a21 == 0) | (d.mkt_dist25 >= 5))])

    print(f"\n=== 组合闸门 (vs 基线 PF={base:.2f}, n={len(d)}) ===")
    gates = {
        "市场站10E": d.mkt_a10 == 1,
        "趋势模板≥7": d.tt >= 7,
        "趋势≥7 且 市场站10E": (d.tt >= 7) & (d.mkt_a10 == 1),
        "高15%内 且 市场站10E": (d.near_high > -15) & (d.mkt_a10 == 1),
        "Pocket 且 市场站10E": (d.pocket == 1) & (d.mkt_a10 == 1),
        "趋势≥7 且 高15内 且 市场站10E": (d.tt >= 7) & (d.near_high > -15) & (d.mkt_a10 == 1),
        "个股派发≤2 且 市场站10E": (d.dist25 <= 2) & (d.mkt_a10 == 1),
    }
    print(f"{'闸门':34}{'n':>6}{'留存':>6}{'胜率':>7}{'PF':>7}{'平均%':>9}{'总%':>8}")
    for nm, m in gates.items():
        s = d[m]
        if len(s) < 15:
            print(f"{nm:34}{len(s):>6}  样本不足"); continue
        print(f"{nm:34}{len(s):>6}{len(s)/len(d)*100:>5.0f}%{s.win.mean()*100:>6.1f}%{pf(s.pnl):>7.2f}{s.pnl.mean():>+9.3f}{s.pnl.sum():>+8.0f}")
    d.to_csv("momentum_intraday_filters.csv", index=False)
    print(f"\n存档 → momentum_intraday_filters.csv ({len(d)} 行)")


if __name__ == "__main__":
    main()
