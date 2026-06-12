"""
kova_momentum_calib.py — 用"结构地标自动对齐"标定动能震荡器(AD速度波)七档颜色阈值。

输入: kova-momentum-structural-align workflow 的输出 JSON (charts[].extrema/global/recent_runs)。
方法(摆脱月级日期噪声):
  1. 我用 OHLCV 算每只票每根 bar 的 AD速度波 z(EMA9-EMA21, 120d z标准化)——确定性、全分辨率。
  2. recent_runs(从最右=截图日往左的同色游程): 按 bar 索引精确回数交易日 → 每根 bar 一对 (z, 颜色)。这是最可靠的精确信号。
  3. global_max/min: 我的 z 在可见窗口里的全局极值那根, 必然对应图上最高/最低柱 → (z_extreme, 颜色), 外层阈值铁锚。
  4. extrema(有序波峰波谷): scipy.find_peaks 在我的 z 上找显著极值, 按时间顺序贪心匹配 agent 的有序极值 → 中段锚点。
聚合全部 (z, 颜色) → 拟合七档阈值, 留一图交叉验证(leave-one-chart-out)防过拟合, 报告诚实命中率。

用法: python3 kova_momentum_calib.py [workflow输出json路径]
"""
import sys, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings("ignore")

ORDER = ["深红", "亮红", "浅红", "浅", "浅蓝", "深蓝", "亮蓝"]
IDX = {t: i for i, t in enumerate(ORDER)}
DEFAULT_OUT = "/private/tmp/claude-501/-Users-xin-Documents-Claude-Projects-money-quant-system/5b11c008-9c36-4cdb-a160-45b1d8bab741/tasks/wutis3k0l.output"
WINDOW_BARS = 150   # 估计 Kova 图可见窗口(找全局极值的范围); 多数图 3-9 个月
USE_RECENT = False  # recent_runs 经诊断不可靠(全图数不准游程), 默认弃用; 只信 gmax/gmin/extrema


def _ctx():
    from longport.openapi import Config, QuoteContext
    return QuoteContext(Config.from_env())


def _pull(ctx, tk, count=460):
    from longport.openapi import Period, AdjustType
    bars = list(ctx.candlesticks(f"{tk}.US", Period.Day, count, AdjustType.ForwardAdjust))
    if not bars:
        return None
    df = pd.DataFrame({
        "High": [float(b.high) for b in bars], "Low": [float(b.low) for b in bars],
        "Close": [float(b.close) for b in bars], "Volume": [float(b.volume) for b in bars],
    }, index=[b.timestamp for b in bars]).sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


def zseries(df):
    clv = ((df.Close - df.Low) - (df.High - df.Close)) / (df.High - df.Low + 1e-9)
    ad = (clv * df.Volume).cumsum()
    wave = ad.ewm(span=9).mean() - ad.ewm(span=21).mean()
    z = (wave - wave.rolling(120).mean()) / wave.rolling(120).std()
    return z.dropna()


def collect_anchors(charts, zcache):
    """返回 DataFrame: (ticker, src, shade, z)。src ∈ recent/gmax/gmin/extrema。"""
    rows = []
    for c in charts:
        if not c.get("readable"):
            continue
        tk, asof = c["ticker"], pd.Timestamp(c["asof"])
        z = zcache.get(tk)
        if z is None or len(z) < 130:
            continue
        zwin = z[z.index <= asof]
        if len(zwin) < 5:
            continue
        # 可见窗口(取最近 WINDOW_BARS 根)
        vis = zwin.tail(WINDOW_BARS)
        # --- global max/min ---
        if c.get("global_max_shade") in ORDER:
            rows.append((tk, "gmax", c["global_max_shade"], float(vis.max())))
        if c.get("global_min_shade") in ORDER:
            rows.append((tk, "gmin", c["global_min_shade"], float(vis.min())))
        # --- recent_runs: 已诊断不可靠, 默认关闭 ---
        #   原因: agent 在密集全图上数不准游程 bars, 往左累积漂移 → 颜色贴错 bar → z 随机
        #   (实测深红 z中位 +1.55, 完全乱; 而 gmax/gmin/extrema 靠结构对齐, 干净单调)
        #   只有放大震荡器面板上的 recent_runs 可信; 全图的丢弃。
        if USE_RECENT:
            pos = len(zwin) - 1
            for run in c.get("recent_runs", []):
                sh = run.get("shade"); nb = int(run.get("bars", 0) or 0)
                if sh not in ORDER:
                    pos -= nb; continue
                for _ in range(nb):
                    if pos < 0:
                        break
                    rows.append((tk, "recent", sh, float(zwin.iloc[pos])))
                    pos -= 1
        # --- extrema: find_peaks 在可见窗口 ---
        ag_ext = sorted(c.get("extrema", []), key=lambda e: e.get("seq", 0))
        if ag_ext and len(vis) >= 10:
            zv = vis.values
            pk, _ = find_peaks(zv, prominence=0.4, distance=4)
            tr, _ = find_peaks(-zv, prominence=0.4, distance=4)
            mine = sorted([(int(i), "peak") for i in pk] + [(int(i), "trough") for i in tr])
            # 贪心按顺序+类型匹配
            j = 0
            for (mi, mtype) in mine:
                while j < len(ag_ext) and ag_ext[j].get("type") != mtype:
                    j += 1
                if j >= len(ag_ext):
                    break
                sh = ag_ext[j].get("shade")
                if sh in ORDER:
                    rows.append((tk, "extrema", sh, float(zv[mi])))
                j += 1
    return pd.DataFrame(rows, columns=["ticker", "src", "shade", "z"])


def fit_thresholds(df):
    """相邻档 z 中位中点作为阈值。返回 6 个边界 + 各档统计。"""
    meds = {}
    for t in ORDER:
        g = df[df.shade == t]
        if len(g):
            meds[t] = g.z.median()
    present = [t for t in ORDER if t in meds]
    ths = []
    for a, b in zip(present[:-1], present[1:]):
        ths.append((meds[a] + meds[b]) / 2)
    return present, ths, meds


def classify(z, present, ths):
    for i, th in enumerate(ths):
        if z < th:
            return present[i]
    return present[-1]


def report(df):
    present, ths, meds = fit_thresholds(df)
    print(f"\n锚点总数: {len(df)}  来源分布: " + ", ".join(f"{k}={v}" for k, v in df.src.value_counts().items()))
    print(f"\n{'档位':6}{'n':>5}{'z中位':>8}{'z均值':>8}{'q25':>8}{'q75':>8}")
    for t in ORDER:
        g = df[df.shade == t]
        if len(g):
            print(f"{t:6}{len(g):>5}{g.z.median():>8.2f}{g.z.mean():>8.2f}{g.z.quantile(.25):>8.2f}{g.z.quantile(.75):>8.2f}")
    mono = all(meds[present[i]] <= meds[present[i + 1]] for i in range(len(present) - 1))
    print(f"\n中位单调上升: {mono}  中位序列: {[round(meds[t],2) for t in present]}")
    print(f"拟合阈值: " + " | ".join(f"{present[i]}|{present[i+1]}={ths[i]:+.2f}" for i in range(len(ths))))
    # 全样本命中
    df = df.copy()
    df["pred"] = df.z.apply(lambda z: classify(z, present, ths))
    acc7 = (df.shade == df.pred).mean()
    adj = df.apply(lambda r: abs(IDX[r.shade] - IDX[r.pred]) <= 1, axis=1).mean()
    to3 = lambda t: "红" if "红" in t else ("蓝" if "蓝" in t else "浅")
    acc3 = (df.shade.map(to3) == df.pred.map(to3)).mean()
    print(f"\n[全样本] 七档精确={acc7*100:.0f}%  ±1档内={adj*100:.0f}%  三档={acc3*100:.0f}%")
    # 只用 recent(bar精确)
    rec = df[df.src == "recent"]
    if len(rec):
        a7 = (rec.shade == rec.pred).mean()
        a3 = (rec.shade.map(to3) == rec.pred.map(to3)).mean()
        print(f"[仅recent精确bar n={len(rec)}] 七档={a7*100:.0f}%  三档={a3*100:.0f}%")
    # 留一图交叉验证
    cv_hit3 = cv_hit7 = cv_n = 0
    for tk in df.ticker.unique():
        tr = df[df.ticker != tk]; te = df[df.ticker == tk]
        pr, th, _ = fit_thresholds(tr)
        if len(pr) < 2:
            continue
        for _, r in te.iterrows():
            p = classify(r.z, pr, th)
            cv_hit7 += int(p == r.shade)
            cv_hit3 += int(to3(p) == to3(r.shade))
            cv_n += 1
    if cv_n:
        print(f"[留一图CV n={cv_n}] 七档={cv_hit7/cv_n*100:.0f}%  三档={cv_hit3/cv_n*100:.0f}%")
    return present, ths


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    out = json.load(open(path))
    charts = out.get("result", out).get("charts", [])
    print(f"读入 {len(charts)} 张图记录")
    ctx = _ctx()
    zcache = {}
    for c in charts:
        tk = c["ticker"]
        if tk not in zcache:
            try:
                zcache[tk] = zseries(_pull(ctx, tk))
            except Exception as e:
                print(f"  ⚠ {tk} 拉取失败: {e}")
                zcache[tk] = None
    df = collect_anchors(charts, zcache)
    present, ths = report(df)
    df.to_csv("kova_momentum_anchors.csv", index=False)
    print(f"\n锚点存档 → kova_momentum_anchors.csv ({len(df)} 行)")


if __name__ == "__main__":
    main()
