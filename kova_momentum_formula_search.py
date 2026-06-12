"""
kova_momentum_formula_search.py — 逼近 Kova 动能震荡器的"确切资金流线"。

Kova 截图露出参数 9/21(EMA对已确认)。未知=被9/21平滑的那条累积资金流线是哪个变体。
对多种资金流线 × 归一化做网格搜索, 用 306 粗点(ticker,月级date,shade)做相对比较:
  同样的日期噪声对所有公式一视同仁 → 谁把七档shade分得最开(Spearman ρ高)谁更接近原版。
防过拟合: 留一图CV + 报3档/7档命中。最后用结构锚点(gmax/gmin)独立验证赢家。

用法: python3 kova_momentum_formula_search.py
"""
import json, re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

ORDER = ["深红", "亮红", "浅红", "浅", "浅蓝", "深蓝", "亮蓝"]
IDX = {t: i for i, t in enumerate(ORDER)}
COARSE = "/private/tmp/claude-501/-Users-xin-Documents-Claude-Projects-money-quant-system/5b11c008-9c36-4cdb-a160-45b1d8bab741/tasks/wkvce6iei.output"


def _ctx():
    from longport.openapi import Config, QuoteContext
    return QuoteContext(Config.from_env())


def _pull(ctx, tk, count=480):
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


# ---------- 资金流线变体 (返回一条累积/振荡序列, 之后统一 EMA9-EMA21 + 归一化) ----------
def line_AD(d):   # Chaikin AD: CLV×Vol 累积
    clv = ((d.Close - d.Low) - (d.High - d.Close)) / (d.High - d.Low + 1e-9)
    return (clv * d.Volume).cumsum()

def line_OBV(d):  # 方向×量 累积
    return (np.sign(d.Close.diff().fillna(0)) * d.Volume).cumsum()

def line_PVT(d):  # 价量趋势: ΔC/C × Vol 累积
    return ((d.Close.pct_change().fillna(0)) * d.Volume).cumsum()

def line_FI(d):   # 力度: ΔC × Vol 累积(Elder Force 的累积形)
    return ((d.Close.diff().fillna(0)) * d.Volume).cumsum()

def line_WAD(d):  # Williams AD
    pc = d.Close.shift()
    trh = pd.concat([d.High, pc], axis=1).max(axis=1)
    trl = pd.concat([d.Low, pc], axis=1).min(axis=1)
    move = np.where(d.Close > pc, d.Close - trl, np.where(d.Close < pc, d.Close - trh, 0.0))
    return pd.Series(move, index=d.index).fillna(0).cumsum()

def line_ADtyp(d):  # AD 但用真实区间(含跳空)的 CLV
    pc = d.Close.shift()
    trh = pd.concat([d.High, pc], axis=1).max(axis=1)
    trl = pd.concat([d.Low, pc], axis=1).min(axis=1)
    clv = ((d.Close - trl) - (trh - d.Close)) / (trh - trl + 1e-9)
    return (clv * d.Volume).cumsum()

def line_MFV(d):  # 货币流量: CLV × 典型价 × Vol 累积(成交额加权)
    clv = ((d.Close - d.Low) - (d.High - d.Close)) / (d.High - d.Low + 1e-9)
    tp = (d.High + d.Low + d.Close) / 3
    return (clv * tp * d.Volume).cumsum()

LINES = {"AD": line_AD, "OBV": line_OBV, "PVT": line_PVT, "FI": line_FI,
         "WAD": line_WAD, "ADtyp": line_ADtyp, "MFV": line_MFV}

# 已经是振荡器(不需 EMA 差)的变体: 直接归一化
def osc_CMF(d, win=21):  # Chaikin Money Flow (有界振荡)
    clv = ((d.Close - d.Low) - (d.High - d.Close)) / (d.High - d.Low + 1e-9)
    mfv = clv * d.Volume
    return mfv.rolling(win).sum() / (d.Volume.rolling(win).sum() + 1e-9)

def osc_ChaikinStd(d):  # 标准 Chaikin 振荡器 EMA3-EMA10(AD)
    ad = line_AD(d)
    return ad.ewm(span=3).mean() - ad.ewm(span=10).mean()


def make_osc(d, line_fn, fast=9, slow=21):
    line = line_fn(d)
    return line.ewm(span=fast).mean() - line.ewm(span=slow).mean()


def normalize(osc, kind="z120"):
    if kind.startswith("z"):
        w = int(kind[1:])
        return (osc - osc.rolling(w).mean()) / (osc.rolling(w).std() + 1e-9)
    if kind.startswith("pct"):
        w = int(kind[3:])
        return osc.rolling(w).apply(lambda x: (x.argsort().argsort()[-1] / (len(x) - 1) - 0.5) * 4, raw=False)
    if kind == "rawstd":   # 仅除以std(不去均值)
        return osc / (osc.rolling(120).std() + 1e-9)
    return osc


MON = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
def pdate(s):
    s = s.strip(); m = re.search(r'(20\d\d)', s); yr = int(m.group(1)) if m else None; mon = None
    for k, v in MON.items():
        if k in s.lower():
            mon = v; break
    if mon is None and m:
        mm = re.search(r'20\d\d-(\d\d)', s); mon = int(mm.group(1)) if mm else None
    if mon is None:
        return None
    if yr is None:
        yr = 2026 if mon <= 6 else 2025
    day = 15
    if 'early' in s.lower(): day = 5
    elif 'late' in s.lower(): day = 24
    else:
        dd = [int(x) for x in re.findall(r'-(\d\d?)', s) if 1 <= int(x) <= 31 and int(x) != mon]
        if dd: day = dd[-1]
    try:
        return pd.Timestamp(yr, mon, min(day, 28))
    except Exception:
        return None


def load_coarse():
    data = json.load(open(COARSE))["result"]["data"]
    pts = []
    for c in data:
        for p in c["points"]:
            dt = pdate(p["date"])
            if dt is not None and p["tier"] in ORDER:
                pts.append((c["ticker"], dt, p["tier"]))
    return pts


def fit_thresholds(df):
    meds = {t: df[df.shade == t].z.median() for t in ORDER if len(df[df.shade == t])}
    pres = [t for t in ORDER if t in meds]
    ths = [(meds[pres[i]] + meds[pres[i + 1]]) / 2 for i in range(len(pres) - 1)]
    return pres, ths, meds


def classify(z, pres, ths):
    for i, th in enumerate(ths):
        if z < th:
            return pres[i]
    return pres[-1]


def evaluate(vals_by_point):
    """vals_by_point: list of (ticker, shade, value). 返回 spearman, 3档CV, 7档CV, 单调."""
    df = pd.DataFrame([(t, s, v) for t, s, v in vals_by_point if v is not None and not np.isnan(v)],
                      columns=["ticker", "shade", "z"])
    if len(df) < 50:
        return None
    rho = spearmanr(df.z, df.shade.map(IDX)).correlation
    pres, ths, meds = fit_thresholds(df)
    mono = all(meds[pres[i]] <= meds[pres[i + 1]] for i in range(len(pres) - 1))
    to3 = lambda t: "红" if "红" in t else ("蓝" if "蓝" in t else "浅")
    # LOO by chart
    h3 = h7 = n = 0
    for tk in df.ticker.unique():
        tr, te = df[df.ticker != tk], df[df.ticker == tk]
        pr, th, _ = fit_thresholds(tr)
        if len(pr) < 2:
            continue
        for _, r in te.iterrows():
            p = classify(r.z, pr, th)
            h7 += int(p == r.shade); h3 += int(to3(p) == to3(r.shade)); n += 1
    return dict(rho=rho, cv3=h3 / max(n, 1), cv7=h7 / max(n, 1), mono=mono, n=len(df))


def main():
    pts = load_coarse()
    tickers = sorted(set(t for t, _, _ in pts))
    print(f"训练点: {len(pts)} (来自 {len(tickers)} 票)")
    ctx = _ctx()
    raw = {}
    for tk in tickers:
        try:
            raw[tk] = _pull(ctx, tk)
        except Exception:
            raw[tk] = None

    # 候选公式集合
    configs = []
    for ln in LINES:
        for norm in ["z120", "z90", "z150", "rawstd"]:
            configs.append(("line", ln, norm))
    configs.append(("osc", "CMF", "z120"))
    configs.append(("osc", "CMF", "raw"))
    configs.append(("osc", "ChaikinStd", "z120"))

    results = []
    for kind, name, norm in configs:
        # 预算每票的归一化序列
        series = {}
        for tk, d in raw.items():
            if d is None or len(d) < 130:
                continue
            if kind == "line":
                osc = make_osc(d, LINES[name])
            elif name == "CMF":
                osc = osc_CMF(d)
            else:
                osc = osc_ChaikinStd(d)
            series[tk] = normalize(osc, norm).dropna()
        vals = []
        for tk, dt, sh in pts:
            s = series.get(tk)
            if s is None:
                continue
            ss = s[s.index <= dt]
            if len(ss):
                vals.append((tk, sh, float(ss.iloc[-1])))
        ev = evaluate(vals)
        if ev:
            results.append((f"{name}/{norm}", ev))

    results.sort(key=lambda x: -x[1]["rho"])
    print(f"\n{'公式/归一':18}{'Spearρ':>8}{'CV三档':>8}{'CV七档':>8}{'单调':>6}{'n':>5}")
    for name, ev in results:
        print(f"{name:18}{ev['rho']:>8.3f}{ev['cv3']*100:>7.0f}%{ev['cv7']*100:>7.0f}%{str(ev['mono']):>6}{ev['n']:>5}")
    base = next((ev for n, ev in results if n == "AD/z120"), None)
    if base:
        print(f"\n基线 AD/z120: ρ={base['rho']:.3f} CV三档={base['cv3']*100:.0f}% CV七档={base['cv7']*100:.0f}%")
        best = results[0]
        print(f"最优 {best[0]}: ρ={best[1]['rho']:.3f} CV三档={best[1]['cv3']*100:.0f}% CV七档={best[1]['cv7']*100:.0f}%")
        d_rho = best[1]['rho'] - base['rho']
        print(f"提升: Δρ={d_rho:+.3f} ΔCV三档={(best[1]['cv3']-base['cv3'])*100:+.0f}pt")


if __name__ == "__main__":
    main()
