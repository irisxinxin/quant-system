"""
sector_watch.py — 热门板块全景追踪
追踪 CTA 趋势方向 + 主力资金流向，多时间维度对比

用法：
  python3 sector_watch.py            # 默认综合评分排序
  python3 sector_watch.py --sort 1W  # 按1周涨幅排序
  python3 sector_watch.py --sort cta # 按CTA信号排序
"""
import sys, warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data.downloader import get_prices, get_ohlcv
from config import CTA_LOOKBACKS, CTA_VOL_WIN

# ──────────────────────────────────────────────
# 监控的板块 ETF（可随时扩充）
# ──────────────────────────────────────────────
SECTORS = {
    # 科技 & AI
    "半导体":    "SMH",
    "AI软件":    "AIQ",
    "软件/SaaS": "IGV",
    "纳指100":   "QQQ",
    # 高增长主题
    "太空/国防":  "XAR",
    "核能":      "NLR",
    "机器人/AI": "ROBO",
    "加密":      "IBIT",
    "生物科技":   "IBB",
    # 传统板块
    "金融":      "XLF",
    "能源":      "XLE",
    "工业":      "XLI",
    "消费成长":   "XLY",
    "必选消费":   "XLP",
    "公用事业":   "XLU",
    "医疗":      "XLV",
    # 大盘 / 宏观
    "标普500":   "SPY",
    "黄金":      "GLD",
    "债券20Y":   "TLT",
}

# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _cta_series(px: pd.Series) -> pd.Series:
    r = px.pct_change().dropna()
    sigs = []
    for lk in CTA_LOOKBACKS:
        m = r.rolling(lk).mean() * 252
        v = r.rolling(CTA_VOL_WIN).std() * np.sqrt(252)
        sigs.append((m / v.replace(0, np.nan)).clip(-1, 1))
    return pd.concat(sigs, axis=1).mean(axis=1)


def _ret(px: pd.Series, n: int) -> float:
    if len(px) < n + 1:
        return float("nan")
    return round(float(px.pct_change(n).iloc[-1]) * 100, 1)


def _scan_one(name: str, ticker: str) -> dict | None:
    try:
        px    = get_prices(ticker)
        ohlcv = get_ohlcv(ticker)
        if px.empty or len(px) < 270:
            return None

        hi  = ohlcv["High"].reindex(px.index)
        lo  = ohlcv["Low"].reindex(px.index)
        vol = ohlcv["Volume"].reindex(px.index).fillna(0)

        # ── 收益率多周期 ──
        r3d  = _ret(px, 3)
        r1w  = _ret(px, 5)
        r2w  = _ret(px, 10)
        r1m  = _ret(px, 21)
        r3m  = _ret(px, 63)
        r6m  = _ret(px, 126)
        r1y  = _ret(px, 252)

        # ── CTA 趋势信号 ──
        cta = _cta_series(px).dropna()
        n   = len(cta)
        cta_now  = round(float(cta.iloc[-1]),  3) if n >= 1  else 0.0
        cta_5d   = round(float(cta.iloc[-1] - cta.iloc[-5]),  3) if n >= 5  else 0.0
        cta_10d  = round(float(cta.iloc[-1] - cta.iloc[-10]), 3) if n >= 10 else 0.0
        cta_20d  = round(float(cta.iloc[-1] - cta.iloc[-20]), 3) if n >= 20 else 0.0
        cta_60d  = round(float(cta.iloc[-1] - cta.iloc[-60]), 3) if n >= 60 else 0.0

        # CTA 方向标签
        if   cta_now >  0.5:  cta_dir = "强多 ▲▲"
        elif cta_now >  0.15: cta_dir = "多头 ▲ "
        elif cta_now > -0.15: cta_dir = "中性 ─ "
        elif cta_now > -0.5:  cta_dir = "空头 ▼ "
        else:                  cta_dir = "强空 ▼▼"

        # CTA 加速/减速（看 5D 变化）
        if   cta_5d >  0.12: cta_accel = "加速↑↑"
        elif cta_5d >  0.05: cta_accel = "升温↑ "
        elif cta_5d < -0.12: cta_accel = "加速↓↓"
        elif cta_5d < -0.05: cta_accel = "降温↓ "
        else:                 cta_accel = "持平─ "

        # CTA 是否在翻转（空转多 or 多转空）
        cta_prev20 = cta_now - cta_20d
        if cta_prev20 < 0 and cta_now > 0:
            cta_regime = "空→多✨"
        elif cta_prev20 > 0 and cta_now < 0:
            cta_regime = "多→空⚠"
        elif cta_now < 0 and cta_20d > 0.25:
            cta_regime = "回升中↗"
        elif cta_now > 0 and cta_20d < -0.25:
            cta_regime = "高位降温"
        else:
            cta_regime = "方向稳定"

        # ── OBV（主力净买入方向）──
        obv_dir = np.sign(px.diff().fillna(0))
        obv     = (vol * obv_dir).cumsum()
        obv_ma20 = obv.rolling(20).mean()
        obv_now  = float(obv.iloc[-1])
        obv_ma_v = float(obv_ma20.iloc[-1]) if not obv_ma20.dropna().empty else 0

        obv_5d_chg  = float(obv.iloc[-1] - obv.iloc[-5])  if len(obv) >= 5  else 0
        obv_20d_chg = float(obv.iloc[-1] - obv.iloc[-20]) if len(obv) >= 20 else 0

        # 短期（5D）+ 中期（20D）OBV 方向
        obv_5d_dir  = "买入▲" if obv_5d_chg  > 0 else "卖出▼"
        obv_20d_dir = "积累↑" if obv_20d_chg > 0 else "派发↓"

        # ── CMF（Chaikin Money Flow）──
        hl    = (hi - lo).replace(0, np.nan)
        mf_m  = ((px - lo) - (hi - px)) / hl
        cmf   = (mf_m * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)
        cmf_clean = cmf.dropna()
        nc = len(cmf_clean)
        cmf_now  = round(float(cmf_clean.iloc[-1]),       3) if nc >= 1  else 0.0
        cmf_5d   = round(float(cmf_clean.iloc[-1] - cmf_clean.iloc[-5]),  3) if nc >= 5  else 0.0
        cmf_20d  = round(float(cmf_clean.iloc[-1] - cmf_clean.iloc[-20]), 3) if nc >= 20 else 0.0

        cmf_status = "流入" if cmf_now > 0.05 else ("流出" if cmf_now < -0.05 else "中性")

        # ── MFI（量价RSI）──
        tp     = (hi + lo + px) / 3
        raw_mf = tp * vol
        pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
        mf_rat = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
        mfi    = (100 - 100 / (1 + mf_rat)).dropna()
        nm = len(mfi)
        mfi_now = round(float(mfi.iloc[-1]),                   1) if nm >= 1 else 50.0
        mfi_5d  = round(float(mfi.iloc[-1] - mfi.iloc[-5]),    1) if nm >= 5 else 0.0
        mfi_20d = round(float(mfi.iloc[-1] - mfi.iloc[-20]),   1) if nm >= 20 else 0.0

        # ── 量能 ──
        vol_avg20 = vol.rolling(20).mean().iloc[-1]
        vol_r = round(float(vol.iloc[-1] / vol_avg20), 2) if vol_avg20 > 0 else 1.0

        # 近5日平均量能
        vol_5d_avg = round(float(vol.iloc[-5:].mean() / vol_avg20), 2) if vol_avg20 > 0 else 1.0

        # ── 综合评分 ──
        mom_score  = (r1w / 10 + r1m / 20 + r2w / 15) / 3
        cta_score  = cta_now
        flow_score = (cmf_now + (1 if obv_now > obv_ma_v else -1) * 0.3) / 1.3
        composite  = round(0.40 * mom_score + 0.35 * cta_score + 0.25 * flow_score, 3)

        # ── 信号分类 ──
        if cta_now > 0.3 and cmf_now > 0.05:
            signal_type = "趋势延续🚀"
        elif cta_now < -0.1 and cta_20d > 0.3:
            signal_type = "CTA翻转↗"
        elif cta_now < 0 and r1w > 5 and cmf_now < 0:
            signal_type = "技术反弹⚡"
        elif cta_now > 0 and cta_20d < -0.3:
            signal_type = "趋势减弱⚠"
        elif cta_now > 0.3 and cmf_now < -0.05:
            signal_type = "资金分歧🔶"
        else:
            signal_type = "观望等待─"

        return {
            "板块":      name,
            "ETF":       ticker,
            # 价格动量
            "3天%":      r3d,
            "1周%":      r1w,
            "2周%":      r2w,
            "1月%":      r1m,
            "3月%":      r3m,
            "6月%":      r6m,
            "1年%":      r1y,
            # CTA
            "CTA":       cta_now,
            "CTA5D":     cta_5d,
            "CTA10D":    cta_10d,
            "CTA20D":    cta_20d,
            "CTA60D":    cta_60d,
            "CTA方向":   cta_dir,
            "CTA加速":   cta_accel,
            "CTA体制":   cta_regime,
            # OBV
            "OBV5D":     obv_5d_dir,
            "OBV20D":    obv_20d_dir,
            # CMF
            "CMF":       cmf_now,
            "CMF5D":     cmf_5d,
            "CMF20D":    cmf_20d,
            "CMF状态":   cmf_status,
            # MFI
            "MFI":       mfi_now,
            "MFI5D":     mfi_5d,
            "MFI20D":    mfi_20d,
            # 量能
            "量能":      vol_r,
            "量能5D":    vol_5d_avg,
            # 综合
            "综合评分":   composite,
            "信号类型":   signal_type,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────
# 主扫描函数
# ──────────────────────────────────────────────

def scan_sectors(sort_by: str = "综合评分") -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_scan_one, n, t): n for n, t in SECTORS.items()}
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    col_map = {
        "综合评分": "综合评分",
        "cta": "CTA",
        "1W": "1周%", "1w": "1周%",
        "1M": "1月%", "1m": "1月%",
        "3M": "3月%", "3m": "3月%",
        "1D": "3天%", "3D": "3天%",
        "cmf": "CMF",
        "mfi": "MFI",
    }
    sort_col = col_map.get(sort_by, sort_by)
    if sort_col not in df.columns:
        sort_col = "综合评分"
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# 格式化输出
# ──────────────────────────────────────────────

def _p(v, w=7) -> str:
    """格式化百分比"""
    if pd.isna(v):
        return " " * w
    s = f"{v:>+.1f}%"
    return s.rjust(w)


def _d(v, w=7) -> str:
    """格式化小数（CTA/CMF 变化量）"""
    if pd.isna(v):
        return " " * w
    return f"{v:>+.3f}".rjust(w)


def print_report(df: pd.DataFrame) -> None:
    from datetime import datetime
    W = 120
    print(f"\n{'='*W}")
    print(f"  热门板块全景追踪   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*W}")

    # ════════════════════════════════════════════
    # 表1：价格动量（多时间维度）
    # ════════════════════════════════════════════
    print(f"\n── 表一：价格动量 ─────────────────────────────────────────────────────────────────────────")
    print(f"  {'板块':8s}  {'ETF':5s}  {'3天':>7s}  {'1周':>7s}  {'2周':>7s}  {'1月':>7s}  {'3月':>7s}  {'6月':>7s}  {'1年':>7s}  {'量能(今)':>8s}  {'量能(5D)':>8s}  信号类型")
    print(f"  {'─'*8}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*10}")

    for _, r in df.iterrows():
        vm = "🔥" if r["量能"] >= 1.5 else ("📈" if r["量能"] >= 1.0 else "  ")
        print(
            f"  {r['板块']:8s}  {r['ETF']:5s}  "
            f"{_p(r['3天%'])}  {_p(r['1周%'])}  {_p(r['2周%'])}  "
            f"{_p(r['1月%'])}  {_p(r['3月%'])}  {_p(r['6月%'])}  {_p(r['1年%'])}  "
            f"  {r['量能']:>4.1f}x{vm}   {r['量能5D']:>4.1f}x  "
            f"  {r['信号类型']}"
        )

    # ════════════════════════════════════════════
    # 表2：CTA 趋势追踪
    # ════════════════════════════════════════════
    print(f"\n── 表二：CTA 趋势追踪（CTA5D/10D/20D/60D = 变化量）─────────────────────────────────────────")
    print(f"  {'板块':8s}  {'ETF':5s}  {'CTA值':>7s}  {'5D△':>7s}  {'10D△':>7s}  {'20D△':>7s}  {'60D△':>7s}  {'方向':10s}  {'5D动能':7s}  {'体制变化':10s}")
    print(f"  {'─'*8}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}  {'─'*7}  {'─'*10}")

    df_cta = df.sort_values("CTA", ascending=False)
    for _, r in df_cta.iterrows():
        mark = "🔥" if r["CTA"] > 0.5 else ("✅" if r["CTA"] > 0.15 else ("⚠️" if r["CTA"] < -0.15 else "  "))
        d5  = _d(r["CTA5D"])
        d10 = _d(r["CTA10D"])
        d20 = _d(r["CTA20D"])
        d60 = _d(r["CTA60D"])
        print(
            f"  {r['板块']:8s}  {r['ETF']:5s}  "
            f"{r['CTA']:>+7.3f}{mark}  {d5}  {d10}  {d20}  {d60}  "
            f"{r['CTA方向']:10s}  {r['CTA加速']:7s}  {r['CTA体制']:10s}"
        )

    # ════════════════════════════════════════════
    # 表3：主力资金流追踪
    # ════════════════════════════════════════════
    print(f"\n── 表三：主力资金流（CMF/MFI 5D/20D = 变化量）─────────────────────────────────────────────")
    print(f"  {'板块':8s}  {'ETF':5s}  {'CMF值':>7s}  {'CMF5D':>7s}  {'CMF20D':>7s}  {'状态':4s}  │  {'MFI值':>6s}  {'MFI5D':>6s}  {'MFI20D':>7s}  │  {'OBV5D':6s}  {'OBV20D':6s}")
    print(f"  {'─'*8}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*4}  │  {'─'*6}  {'─'*6}  {'─'*7}  │  {'─'*6}  {'─'*6}")

    df_cmf = df.sort_values("CMF", ascending=False)
    for _, r in df_cmf.iterrows():
        cm = "↑" if r["CMF"] > 0.05 else ("↓" if r["CMF"] < -0.05 else "─")
        mfi_mark = "超买⚠" if r["MFI"] > 75 else ("超卖✨" if r["MFI"] < 30 else "     ")
        print(
            f"  {r['板块']:8s}  {r['ETF']:5s}  "
            f"{r['CMF']:>+7.3f}{cm}  {_d(r['CMF5D'])}  {_d(r['CMF20D'])}  {r['CMF状态']:4s}  │  "
            f"{r['MFI']:>6.1f}  {r['MFI5D']:>+6.1f}  {r['MFI20D']:>+7.1f}  {mfi_mark}  │  "
            f"{r['OBV5D']:6s}  {r['OBV20D']:6s}"
        )

    print(f"\n{'─'*W}")

    # ════════════════════════════════════════════
    # 摘要：CTA 动向
    # ════════════════════════════════════════════
    print(f"\n📡 CTA 动向摘要：")

    # 加速建仓（5D AND 20D 均上升）
    accel = df[(df["CTA5D"] > 0.08) & (df["CTA20D"] > 0.1)].sort_values("CTA5D", ascending=False)
    if not accel.empty:
        print("  🚀 加速建仓（5D+20D 均上升）：")
        for _, r in accel.iterrows():
            print(f"    {r['板块']:8s} {r['ETF']:5s}  CTA={r['CTA']:>+.3f}  5D={r['CTA5D']:>+.3f}  20D={r['CTA20D']:>+.3f}  {r['CTA加速']}")

    # CTA 翻转信号（曾经为负，正在回升）
    turning = df[(df["CTA"] < 0.2) & (df["CTA20D"] > 0.25)].sort_values("CTA20D", ascending=False)
    if not turning.empty:
        print("  🔄 CTA 翻转进行中（从低位快速回升）：")
        for _, r in turning.iterrows():
            prev = r["CTA"] - r["CTA20D"]
            print(f"    {r['板块']:8s} {r['ETF']:5s}  CTA: {prev:>+.3f} → {r['CTA']:>+.3f}  20D变化={r['CTA20D']:>+.3f}  {r['CTA体制']}")

    # CTA 减速（高位回落）
    decel = df[(df["CTA5D"] < -0.08) & (df["CTA"] > 0.1)].sort_values("CTA5D")
    if not decel.empty:
        print("  ⚠️  减速离场（CTA高位回落）：")
        for _, r in decel.iterrows():
            print(f"    {r['板块']:8s} {r['ETF']:5s}  CTA={r['CTA']:>+.3f}  5D={r['CTA5D']:>+.3f}  20D={r['CTA20D']:>+.3f}")

    # ════════════════════════════════════════════
    # 摘要：主力资金
    # ════════════════════════════════════════════
    print(f"\n💰 主力资金摘要：")

    # 持续流入（CMF>5% 且 20D 趋势向上）
    strong_in = df[(df["CMF"] > 0.05) & (df["CMF20D"] > 0) & (df["OBV5D"] == "买入▲")].sort_values("CMF", ascending=False)
    if not strong_in.empty:
        print("  🟢 持续流入（CMF>5% + 20D改善 + OBV买入）：")
        for _, r in strong_in.head(6).iterrows():
            print(f"    {r['板块']:8s} {r['ETF']:5s}  CMF={r['CMF']:>+.3f}  20D△={r['CMF20D']:>+.3f}  MFI={r['MFI']:.0f}  1周={_p(r['1周%'])}")

    # 资金回流（CMF 负值但 5D+20D 均在改善）
    recovering = df[(df["CMF"] < 0.05) & (df["CMF5D"] > 0.02) & (df["CMF20D"] > 0.02)].sort_values("CMF20D", ascending=False)
    if not recovering.empty:
        print("  🔶 资金回流信号（CMF改善中）：")
        for _, r in recovering.head(4).iterrows():
            print(f"    {r['板块']:8s} {r['ETF']:5s}  CMF={r['CMF']:>+.3f}  5D△={r['CMF5D']:>+.3f}  20D△={r['CMF20D']:>+.3f}  1周={_p(r['1周%'])}")

    # 持续流出
    strong_out = df[(df["CMF"] < -0.05) & (df["CMF20D"] < 0)].sort_values("CMF")
    if not strong_out.empty:
        print("  🔴 持续流出（CMF<-5% + 20D仍在下降）：")
        for _, r in strong_out.head(4).iterrows():
            print(f"    {r['板块']:8s} {r['ETF']:5s}  CMF={r['CMF']:>+.3f}  20D△={r['CMF20D']:>+.3f}  MFI={r['MFI']:.0f}  1周={_p(r['1周%'])}")

    # ════════════════════════════════════════════
    # 摘要：信号分类
    # ════════════════════════════════════════════
    print(f"\n🎯 板块信号分类：")

    type_groups = {
        "趋势延续🚀": "CTA强多 + 资金持续流入，可以追涨",
        "CTA翻转↗":  "CTA从低位快速回升，可能是新趋势起点",
        "技术反弹⚡": "价格弹升但CTA/资金未确认，均值回归短线",
        "趋势减弱⚠":  "CTA高位快速下降，注意减仓",
        "资金分歧🔶": "CTA多但资金流出，需等资金确认",
        "观望等待─":  "方向不明，持观望",
    }

    for sig_type, desc in type_groups.items():
        sub = df[df["信号类型"] == sig_type]
        if not sub.empty:
            names = "  ".join(f"{r['板块']}({r['ETF']})" for _, r in sub.iterrows())
            print(f"  {sig_type}  →  {names}")
            print(f"             {desc}")

    print()


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    sort_by = "综合评分"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--sort" and i + 1 < len(sys.argv) - 1:
            sort_by = sys.argv[i + 2]

    print("📡 扫描板块数据...")
    df = scan_sectors(sort_by=sort_by)
    if df.empty:
        print("❌ 数据获取失败")
    else:
        print_report(df)
