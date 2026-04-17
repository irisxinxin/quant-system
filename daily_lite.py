"""
daily_lite.py — LITE 每日收盘分析
运行时机：美股收盘后（美东4pm，北京时间次日4-5am，或隔天早上运行）
输出：入场建议 / 等待 / 回避，以及止损位

用法：python daily_lite.py
"""
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data.downloader import get_prices, get_ohlcv
from signals.smart_money import smc_signal
from config import CTA_LOOKBACKS, CTA_VOL_WIN


def analyze_lite(ticker: str = "LITE"):
    prices = get_prices(ticker, start="2024-01-01")
    smh    = get_prices("SMH",  start="2024-01-01")
    ref    = get_prices("SPY",  start="2024-01-01")
    ohlcv  = get_ohlcv(ticker)

    if prices.empty:
        print("❌ 数据获取失败")
        return

    # ── 指标 ──
    e20 = prices.ewm(span=20, adjust=False).mean()
    e60 = prices.ewm(span=60, adjust=False).mean()
    hi  = ohlcv["High"].reindex(prices.index)
    lo  = ohlcv["Low"].reindex(prices.index)
    vol = ohlcv["Volume"].reindex(prices.index)

    # ATR
    tr  = pd.concat([hi-lo, (hi-prices.shift()).abs(), (lo-prices.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()

    # RSI(10)
    d    = prices.diff()
    gain = d.clip(lower=0).rolling(10).mean()
    loss = (-d.clip(upper=0)).rolling(10).mean()
    rsi  = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # 布林带(20)
    bb_m  = prices.rolling(20).mean()
    bb_s  = prices.rolling(20).std()
    bb_lo = bb_m - 2 * bb_s
    bb_hi = bb_m + 2 * bb_s
    bb_pct = ((prices - bb_lo) / (4 * bb_s) * 100)

    # DC20 突破
    dc20_high = hi.rolling(20).max().shift(1)
    dc20_break = prices > dc20_high

    # 量能
    vol_r = vol / vol.rolling(20).mean()

    # CTA
    def cta_s(px):
        r = px.pct_change().dropna()
        sigs = []
        for lk in CTA_LOOKBACKS:
            m = r.rolling(lk).mean() * 252
            v = r.rolling(CTA_VOL_WIN).std() * np.sqrt(252)
            sigs.append((m / v.replace(0, np.nan)).clip(-1, 1))
        return pd.concat(sigs, axis=1).mean(axis=1)

    cta_smh = cta_s(smh).reindex(prices.index).ffill()
    cta_mkt = cta_s(ref).reindex(prices.index).ffill()
    cta_val = float(((cta_smh + cta_mkt) / 2).iloc[-1])

    # 相对强度(20日)
    rs20 = float(prices.pct_change(20).iloc[-1]) - float(smh.pct_change(20).iloc[-1])

    # SMC
    try:
        smc_df  = smc_signal(ticker)
        smc_val = int(smc_df["signal"].iloc[-1])
    except Exception:
        smc_val = 0

    # ── 取最新值 ──
    px    = float(prices.iloc[-1])
    e20v  = float(e20.iloc[-1])
    e60v  = float(e60.iloc[-1])
    atrv  = float(atr.iloc[-1])
    rsiv  = float(rsi.iloc[-1])
    bbpct = float(bb_pct.iloc[-1])
    volr  = float(vol_r.iloc[-1])
    dc_ok = bool(dc20_break.iloc[-1])
    dc20h = float(dc20_high.iloc[-1])
    date  = prices.index[-1].date()

    # 连续涨跌天数
    rets = prices.pct_change().tail(10)
    streak = 0
    for r in reversed(rets.values):
        if np.isnan(r): break
        if r > 0:
            if streak >= 0: streak += 1
            else: break
        else:
            if streak <= 0: streak -= 1
            else: break

    # ── 评分系统 ──
    signals  = []
    warnings = []
    score    = 0

    # 1. EMA 趋势方向
    ema_ok = e20v > e60v
    if ema_ok:
        signals.append(f"✅ EMA金叉（趋势向上）")
        score += 1
    else:
        warnings.append(f"🔴 EMA死叉（趋势向下，不入场）")

    # 2. DC20 突破
    if dc_ok:
        signals.append(f"✅ DC20突破（收盘{px:.1f} > 20日高点{dc20h:.1f}）")
        score += 2  # 主要入场触发，权重更高
    else:
        dist_dc = (dc20h - px) / px * 100
        signals.append(f"   DC20未突破（距高点{dc20h:.1f}还差{dist_dc:.1f}%）")

    # 3. 位置：是否贴近 EMA20（回踩入场机会）
    ext_e20 = (px / e20v - 1) * 100
    ext_e60 = (px / e60v - 1) * 100
    if abs(ext_e20) <= 4:
        signals.append(f"✅ 贴近EMA20（偏离{ext_e20:+.1f}%，回踩入场机会）")
        score += 1
    elif ext_e20 > 15:
        warnings.append(f"⚠️  价格偏离EMA20 +{ext_e20:.0f}%，追高风险")
    else:
        signals.append(f"   距EMA20偏离{ext_e20:+.1f}%")

    # 4. RSI
    if rsiv < 40:
        signals.append(f"✅ RSI超卖={rsiv:.0f}（历史上反弹概率高）")
        score += 1
    elif rsiv > 78:
        warnings.append(f"⚠️  RSI超买={rsiv:.0f}，追高风险")
    else:
        signals.append(f"   RSI={rsiv:.0f}（健康区间）")

    # 5. 布林带位置
    if bbpct < 15:
        signals.append(f"✅ 布林下轨区（位置{bbpct:.0f}%，均值回归机会）")
        score += 1
    elif bbpct > 95:
        warnings.append(f"⚠️  布林上轨（位置{bbpct:.0f}%，短期过热）")
    else:
        signals.append(f"   布林位置{bbpct:.0f}%（中性）")

    # 6. 量能
    if volr >= 1.5:
        signals.append(f"✅ 放量{volr:.1f}x（资金确认）")
        score += 1
    elif volr < 0.7:
        warnings.append(f"⚠️  缩量{volr:.1f}x（无资金支撑）")
    else:
        signals.append(f"   量能{volr:.1f}x（正常）")

    # 7. SMC
    if smc_val >= 2:
        signals.append(f"✅ SMC={smc_val}（机构建仓信号，可加仓）")
        score += 1
    elif smc_val == 1:
        signals.append(f"   SMC=1（有建仓迹象）")

    # 8. 相对强度
    if rs20 > 5:
        signals.append(f"✅ 跑赢SMH板块 +{rs20*100:.0f}%（强势）")
    elif rs20 < -5:
        warnings.append(f"⚠️  弱于SMH板块 {rs20*100:.0f}%")

    # ── CTA 宏观 ──
    if cta_val < 0:
        warnings.append(f"🔴 板块CTA负向={cta_val:.2f}，宏观不利，禁止入场")
        ema_ok = False  # 强制不入场
    elif cta_val < 0.3:
        warnings.append(f"⚠️  板块CTA偏弱={cta_val:.2f}")

    # ── 止损计算 ──
    stop_atr    = px * (1 - atrv * 2 / px)   # 2×ATR硬止损
    stop_ema60  = e60v * 0.98                 # EMA60下方2%
    stop_pct10  = px * 0.90                   # 10%硬止损

    # ── 综合判断 ──
    if not ema_ok or cta_val < 0:
        verdict    = "🔴 回避  — EMA死叉或CTA负向，不入场"
        entry_zone = "无"
    elif dc_ok and score >= 3:
        verdict    = "🟢 强烈推荐入场  — DC20突破 + 多信号共振"
        entry_zone = f"当前价 {px:.1f} 附近即可，收盘确认突破后入场"
    elif dc_ok:
        verdict    = "🟡 可以入场  — DC20突破，但其他信号一般"
        entry_zone = f"当前价 {px:.1f}，小仓位试探"
    elif abs(ext_e20) <= 4 and score >= 3:
        verdict    = "🟢 推荐入场  — EMA20回踩 + 多信号共振"
        entry_zone = f"EMA20 附近 {e20v:.0f}~{e20v*1.02:.0f}"
    elif rsiv < 40 and bbpct < 20:
        verdict    = "🟡 可以入场  — 超卖回弹机会，风险较高"
        entry_zone = f"当前价 {px:.1f}，需要次日确认反弹"
    elif score >= 3:
        verdict    = "🟡 观望偏多  — 信号尚可但无明确触发点"
        entry_zone = f"等 DC20突破>{dc20h:.0f} 或 回踩EMA20≈{e20v:.0f}"
    else:
        verdict    = "⚪ 观望  — 条件不足，耐心等待"
        entry_zone = f"等 DC20突破>{dc20h:.0f} 或 回踩EMA20≈{e20v:.0f}"

    # ── 输出 ──
    print(f"\n{'='*60}")
    print(f"  {ticker} 每日收盘分析  {date}  收盘价 ${px:.1f}")
    print(f"{'='*60}")

    print(f"\n【今日信号】")
    for s in signals:
        print(f"  {s}")
    if warnings:
        print(f"\n【风险提示】")
        for w in warnings:
            print(f"  {w}")

    print(f"\n【综合判断】  {verdict}")
    print(f"【入场区间】  {entry_zone}")

    print(f"\n【止损参考】")
    print(f"  建议止损（2×ATR）: ${stop_atr:.1f}  （距现价 -{(1-stop_atr/px)*100:.1f}%）")
    print(f"  宽松止损（EMA60）: ${stop_ema60:.1f}  （距现价 -{(1-stop_ema60/px)*100:.1f}%）")
    print(f"  10%硬止损:         ${stop_pct10:.1f}  {'⚠️ 偏紧，日常波动可能触发' if atrv/px > 0.07 else ''}")

    print(f"\n【关键位置】")
    print(f"  EMA20={e20v:.1f}  EMA60={e60v:.1f}  ATR={atrv:.1f}({atrv/px*100:.1f}%)")
    print(f"  DC20突破线={dc20h:.1f}  CTA板块={cta_val:.2f}  SMC={smc_val}")
    streak_str = f"连涨{streak}天" if streak > 0 else f"连跌{abs(streak)}天"
    print(f"  {streak_str}  RSI={rsiv:.0f}  布林位={bbpct:.0f}%")
    print()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.basicConfig(level=logging.WARNING)

    ticker = sys.argv[1] if len(sys.argv) > 1 else "LITE"
    analyze_lite(ticker)
