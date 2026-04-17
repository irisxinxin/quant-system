"""
scanner.py — 批量信号扫描（Web 仪表盘后端）
"""
import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data.downloader import get_prices, get_ohlcv
from signals.smart_money import smc_signal
from signals.cta_monitor import cta_trend_signal, run_cta_dashboard
from signals.sector_flows import sector_heatmap, sector_flow_proxy
from config import CTA_LOOKBACKS, CTA_VOL_WIN

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Top3 策略缓存（从 CSV 加载，扫描时实时检查信号状态）
# ──────────────────────────────────────────────
_TOP3_CSV = Path(__file__).parent / "output" / "top3_strategies.csv"

def _load_top3() -> dict:
    """加载 top3_strategies.csv，返回 {ticker: [策略列表]} """
    if not _TOP3_CSV.exists():
        return {}
    try:
        df = pd.read_csv(_TOP3_CSV)
        result = {}
        for _, row in df.iterrows():
            t = row["ticker"]
            if t not in result:
                result[t] = []
            result[t].append({
                "rank":     int(row["rank"]),
                "entry":    row["entry"],
                "cta":      row["cta"],
                "exit":     row["exit"],
                "score":    float(row["score"]),
                "win_rate": float(row["win_rate"]),
                "ret_1m":   float(row["ret_1m"]),
                "ret_3m":   float(row["ret_3m"]),
                "ret_ytd":  float(row.get("ret_ytd", 0)),
                "cagr_1y":  float(row["cagr_1y"]),
            })
        return result
    except Exception as e:
        logger.warning(f"load_top3 failed: {e}")
        return {}


def _quick_strategy_states(
    ticker: str,
    strategies: list,
    prices: pd.Series,
    ohlcv: pd.DataFrame,
    smh_cta: pd.Series,
    spy_cta: pd.Series,
    qqq_cta: pd.Series,
    extra_ctas: dict,
) -> list:
    """
    对给定的策略列表，快速计算当前信号状态，不跑完整回测。
    返回: [{rank, entry, cta, exit, state, in_trade, win_rate, ret_3m, ...}, ...]
    state: 'in_trade' | 'entry_today' | 'exit_today' | 'waiting'
    """
    if prices.empty or len(prices) < 60:
        return []

    try:
        from optimize_stocks import _make_pos, CALMAR_CAP

        # ── 指标（和 optimize_ticker 保持一致）──
        hi  = ohlcv["High"].reindex(prices.index)
        lo  = ohlcv["Low"].reindex(prices.index)
        vol = ohlcv["Volume"].reindex(prices.index).fillna(0)

        e20  = prices.ewm(span=20, adjust=False).mean()
        e60  = prices.ewm(span=60, adjust=False).mean()
        ma50 = prices.rolling(50).mean()
        ma200= prices.rolling(200).mean()
        dc20h = hi.rolling(20).max().shift(1)

        bb_m  = prices.rolling(20).mean()
        bb_s  = prices.rolling(20).std()
        bb_lo = bb_m - 2 * bb_s
        bb_hi = bb_m + 2 * bb_s

        d    = prices.diff()
        gain = d.clip(lower=0).rolling(10).mean()
        loss = (-d.clip(upper=0)).rolling(10).mean()
        rsi  = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

        obv_dir  = np.sign(prices.diff().fillna(0))
        obv      = (vol * obv_dir).cumsum()
        obv_ma20 = obv.rolling(20).mean()

        hl     = (hi - lo).replace(0, np.nan)
        mf_mult= ((prices - lo) - (hi - prices)) / hl
        cmf    = (mf_mult * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)

        tp     = (hi + lo + prices) / 3
        raw_mf = tp * vol
        pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)
        mf_rat = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
        mfi    = (100 - 100 / (1 + mf_rat)).fillna(50)

        vol_ratio    = vol / vol.rolling(20).mean().replace(0, np.nan)
        vol_surge_up = (vol_ratio > 1.5) & (prices > e20)
        hi20_max     = hi.rolling(20).max()
        ema20_band   = (prices >= e20 * 0.97) & (prices <= e20 * 1.04) & (e20 > e60)

        combo_cta_s = (
            smh_cta.reindex(prices.index).ffill().fillna(0) +
            spy_cta.reindex(prices.index).ffill().fillna(0)
        ) / 2
        qqq_s = qqq_cta.reindex(prices.index).ffill().fillna(0) if qqq_cta is not None \
                else pd.Series(0.0, index=prices.index)

        # ── 入场条件 ──
        entries = {
            "ema2060":       (e20 > e60).fillna(False).astype(float),
            "dc20":          (prices > dc20h).fillna(False).astype(float),
            "dc20|ema":      ((prices > dc20h) | (e20 > e60)).fillna(False).astype(float),
            "ma5200":        (ma50 > ma200).fillna(False).astype(float),
            "dc20+obv":      ((prices > dc20h) & (obv > obv_ma20)).fillna(False).astype(float),
            "dc20+cmf":      ((prices > dc20h) & (cmf > 0.0)).fillna(False).astype(float),
            "vol+ema":       (vol_surge_up & (e20 > e60)).fillna(False).astype(float),
            "obv_up":        (obv > obv_ma20).fillna(False).astype(float),
            "cmf_pos":       (cmf > 0.05).fillna(False).astype(float),
            "mfi_os":        (mfi < 35).fillna(False).astype(float),
            "vol_surge":     vol_surge_up.fillna(False).astype(float),
            "ema20_dip":     ema20_band.fillna(False).astype(float),
            "ema20_dip+obv": (ema20_band & (obv > obv_ma20)).fillna(False).astype(float),
            "bb_lo":         (prices < bb_lo).fillna(False).astype(float),
            "rsi35":         (rsi < 35).fillna(False).astype(float),
            "rsi28":         (rsi < 28).fillna(False).astype(float),
        }

        # ── CTA 过滤 ──
        cta_gates = {
            "none":  pd.Series(1.0, index=prices.index),
            "combo": (combo_cta_s > 0).astype(float),
            "soft":  (combo_cta_s > -0.25).astype(float),
            "spy":   (spy_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
            "smh":   (smh_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
            "qqq":   (qqq_s > 0).astype(float),
        }
        for _cn, _cs in (extra_ctas or {}).items():
            if _cn not in cta_gates:
                cta_gates[_cn] = (_cs.reindex(prices.index).ffill().fillna(0) > 0).astype(float)

        # ── 出场条件 ──
        rsi_was_hot = rsi.rolling(5).max() > 70
        exits = {
            "ema_x":    (e20 < e60).fillna(False).astype(float),
            "ma_x":     (ma50 < ma200).fillna(False).astype(float),
            "rsi80":    (rsi > 80).astype(float),
            "rsi70":    (rsi > 70).astype(float),
            "obv_down": (obv < obv_ma20).fillna(False).astype(float),
            "cmf_neg":  (cmf < -0.05).fillna(False).astype(float),
            "trail_8":  (prices < hi20_max * 0.92).fillna(False).astype(float),
            "trail_12": (prices < hi20_max * 0.88).fillna(False).astype(float),
            "rsi_fade": ((rsi < 60) & rsi_was_hot).fillna(False).astype(float),
        }

        results = []
        for s in strategies:
            en, cn, xn = s["entry"], s["cta"], s["exit"]
            e_sig = entries.get(en)
            c_sig = cta_gates.get(cn)
            x_sig = exits.get(xn)
            if e_sig is None or c_sig is None or x_sig is None:
                continue

            if en in ("bb_lo", "rsi35", "rsi28"):
                base_exit = (prices > bb_hi).fillna(False) if en == "bb_lo" else (prices > e20).fillna(False)
                add_exit  = {
                    "ema_x": (e20 < e60).fillna(False), "ma_x": (ma50 < ma200).fillna(False),
                    "rsi80": (rsi > 80), "rsi70": (rsi > 70),
                    "obv_down": (obv < obv_ma20).fillna(False), "cmf_neg": (cmf < -0.05).fillna(False),
                    "trail_8": (prices < hi20_max * 0.92).fillna(False),
                    "trail_12": (prices < hi20_max * 0.88).fillna(False),
                }.get(xn, pd.Series(False, index=prices.index))
                eff_exit = (base_exit | add_exit).astype(float)
            else:
                eff_exit = x_sig

            pos_arr = _make_pos(e_sig.fillna(0).values, c_sig.fillna(0).values, eff_exit.fillna(0).values)
            pos     = pd.Series(pos_arr, index=prices.index)

            cur  = int(pos.iloc[-1])
            prev = int(pos.iloc[-2]) if len(pos) > 1 else 0
            if cur == 1 and prev == 0:
                state = "entry_today"
            elif cur == 0 and prev == 1:
                state = "exit_today"
            elif cur == 1:
                state = "in_trade"
            else:
                state = "waiting"

            # 当前关键指标值（用于出场/入场条件描述）
            px_now   = round(float(prices.iloc[-1]), 2)
            rsi_now  = round(float(rsi.dropna().iloc[-1]), 1) if not rsi.dropna().empty else 50.0
            e20_now  = round(float(e20.iloc[-1]), 2)
            e60_now  = round(float(e60.iloc[-1]), 2)
            ma50_now = round(float(ma50.dropna().iloc[-1]), 2) if not ma50.dropna().empty else px_now
            ma200_now= round(float(ma200.dropna().iloc[-1]), 2) if not ma200.dropna().empty else px_now
            hi20_now = round(float(hi20_max.dropna().iloc[-1]), 2) if not hi20_max.dropna().empty else px_now
            mfi_now  = round(float(mfi.dropna().iloc[-1]), 1) if not mfi.dropna().empty else 50.0
            dc20_now = round(float(dc20h.dropna().iloc[-1]), 2) if not dc20h.dropna().empty else px_now * 1.05

            # 出场条件说明
            exit_desc = {
                "rsi80":    f"RSI>80（当前{rsi_now}）",
                "rsi70":    f"RSI>70（当前{rsi_now}）",
                "rsi_fade": f"RSI曾超70后回落<60（当前{rsi_now}）",
                "trail_8":  f"20日高点回撤8%  止损≈${hi20_now*0.92:.2f}",
                "trail_12": f"20日高点回撤12% 止损≈${hi20_now*0.88:.2f}",
                "ema_x":    f"EMA20<EMA60（当前{e20_now}/{e60_now}）",
                "ma_x":     f"MA50<MA200（当前{ma50_now}/{ma200_now}）",
                "obv_down": "OBV跌破均线（资金出逃）",
                "cmf_neg":  "CMF<-0.05（主力净流出）",
            }.get(xn, xn)

            # 入场条件说明（waiting时用）
            entry_desc = {
                "rsi28":         f"RSI<28（当前{rsi_now}）",
                "rsi35":         f"RSI<35（当前{rsi_now}）",
                "mfi_os":        f"MFI<35（当前{mfi_now}）",
                "bb_lo":         f"价格跌破布林下轨",
                "dc20":          f"突破20日高点${dc20_now}",
                "dc20|ema":      f"突破${dc20_now} 或EMA金叉",
                "ema2060":       f"EMA20>{e20_now}>EMA60={e60_now}",
                "ma5200":        f"MA50>{ma50_now}>MA200={ma200_now}",
                "cmf_pos":       f"CMF>0.05（主力净流入）",
                "obv_up":        f"OBV站上20日均线",
                "ema20_dip":     f"回踩EMA20≈${e20_now}（趋势内）",
                "ema20_dip+obv": f"回踩EMA20≈${e20_now}+OBV确认",
                "vol+ema":       f"放量突破+EMA金叉",
                "vol_surge":     f"成交量>1.5x均量+价格站上EMA20",
                "dc20+obv":      f"突破${dc20_now}+OBV确认",
                "dc20+cmf":      f"突破${dc20_now}+CMF正值",
            }.get(en, en)

            results.append({
                **s,
                "state":       state,
                "in_trade":    cur == 1,
                "exit_desc":   exit_desc,
                "entry_desc":  entry_desc,
                "rsi_now":     rsi_now,
                "e20_now":     e20_now,
                "e60_now":     e60_now,
                "trail_stop":  round(hi20_now * 0.92, 2) if xn == "trail_8" else round(hi20_now * 0.88, 2) if xn == "trail_12" else None,
            })
    except Exception as e:
        logger.warning(f"_quick_strategy_states {ticker}: {e}")
        return []

    return results


# ──────────────────────────────────────────────
# Watchlist
# ──────────────────────────────────────────────
WATCHLIST = {
    "🔵 大盘/核心":       ["QQQ", "SPY", "SMH", "GOOG", "META", "TSLA", "AMZN"],
    "⚡ 半导体/AI算力":   ["NVDA", "ASML", "TSM", "AMD", "ARM", "AVGO", "AEHR", "TXN", "MRVL", "KLAC"],
    "💾 存储":            ["MU", "WDC", "STX", "SNDK"],
    "🏗 AI电力/数据中心": ["BE", "VRT", "ETN", "GEV", "PWR"],
    "🌐 光子/高速连接":   ["LITE", "COHR", "FN", "AAOI", "LWLG", "VIAV", "CLS", "CIEN", "GLW", "TSEM"],
    "🚚 物流/运输":       ["ODFL", "XPO", "JBHT", "PCAR", "CMI"],
    "🏭 工业/航天制造":   ["CAT", "DE", "HWM", "ITT", "EME", "AME"],
    "💰 金融":            ["MS", "CBOE", "TRV"],
    "🪙 加密/Fintech":   ["COIN", "MSTR", "IREN"],
    "🔋 电池/稀土":       ["MP", "ALB", "EOSE"],
    "🚀 太空/机器人":     ["LUNR", "PL", "TER", "RKLB"],
}

# 额外监控的板块 ETF（资金流向用）
FLOW_ETFS = {
    "半导体": "SMH", "科技/软件": "IGV", "AI概念": "AIQ",
    "纳指100": "QQQ", "标普500": "SPY",
    "金融": "XLF", "医疗": "XLV", "能源": "XLE",
    "工业": "XLI", "消费": "XLY", "公用事业": "XLU",
    "清洁能源": "ICLN", "加密": "IBIT", "创新ARK": "ARKK",
}


# ──────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────

def _cta_series_full(px: pd.Series) -> pd.Series:
    """返回完整 CTA 信号序列（和 optimize_stocks._cta_series 相同逻辑）"""
    r = px.pct_change().dropna()
    sigs = []
    for lk in CTA_LOOKBACKS:
        m = r.rolling(lk).mean() * 252
        v = r.rolling(CTA_VOL_WIN).std() * np.sqrt(252)
        sigs.append((m / v.replace(0, np.nan)).clip(-1, 1))
    return pd.concat(sigs, axis=1).mean(axis=1)


def _cta_combo_series(ref_px: pd.Series, target_prices: pd.Series) -> pd.Series:
    """返回对齐到 target_prices 索引的 CTA 序列"""
    return _cta_series_full(ref_px).reindex(target_prices.index).ffill().fillna(0)


def _cta_combo(smh_px: pd.Series, ref_px: pd.Series, prices: pd.Series) -> float:
    def _cta_s(px):
        r = px.pct_change().dropna()
        sigs = []
        for lk in CTA_LOOKBACKS:
            m = r.rolling(lk).mean() * 252
            v = r.rolling(CTA_VOL_WIN).std() * np.sqrt(252)
            sigs.append((m / v.replace(0, np.nan)).clip(-1, 1))
        return pd.concat(sigs, axis=1).mean(axis=1)

    cta_smh = _cta_s(smh_px).reindex(prices.index).ffill()
    cta_mkt = _cta_s(ref_px).reindex(prices.index).ffill()
    val = ((cta_smh + cta_mkt) / 2).dropna()
    return float(val.iloc[-1]) if not val.empty else 0.0


def _rsi(prices: pd.Series, period: int = 10) -> float:
    d    = prices.diff()
    gain = d.clip(lower=0).rolling(period).mean()
    loss = (-d.clip(upper=0)).rolling(period).mean()
    rs   = gain / loss.replace(0, np.nan)
    rsi  = 100 - 100 / (1 + rs)
    v    = rsi.dropna()
    return float(v.iloc[-1]) if not v.empty else 50.0


# ──────────────────────────────────────────────
# 单股扫描
# ──────────────────────────────────────────────

def scan_ticker(
    ticker: str,
    smh_px: pd.Series = None,
    ref_px: pd.Series = None,
    qqq_px: pd.Series = None,
    extra_ctas: dict = None,
    top3_map: dict = None,
) -> dict:
    """扫描单只股票，返回结构化分析结果"""
    try:
        prices = get_prices(ticker, start="2024-01-01")
        if smh_px is None:
            smh_px = get_prices("SMH", start="2024-01-01")
        if ref_px is None:
            ref_px = get_prices("SPY", start="2024-01-01")
        if qqq_px is None:
            qqq_px = get_prices("QQQ", start="2024-01-01")
        ohlcv = get_ohlcv(ticker)

        if prices.empty or len(prices) < 65:
            return {"ticker": ticker, "error": "数据不足", "verdict_code": "gray",
                    "price": 0, "price_chg": 0, "verdict": "无数据", "entry_zone": "-"}

        # ── 技术指标 ──
        e20 = prices.ewm(span=20, adjust=False).mean()
        e60 = prices.ewm(span=60, adjust=False).mean()
        hi  = ohlcv["High"].reindex(prices.index)
        lo  = ohlcv["Low"].reindex(prices.index)
        vol = ohlcv["Volume"].reindex(prices.index)

        tr  = pd.concat([hi-lo, (hi-prices.shift()).abs(), (lo-prices.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=14, adjust=False).mean()

        d    = prices.diff()
        gain = d.clip(lower=0).rolling(10).mean()
        loss = (-d.clip(upper=0)).rolling(10).mean()
        rsi  = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        bb_m   = prices.rolling(20).mean()
        bb_s   = prices.rolling(20).std()
        bb_lo  = bb_m - 2 * bb_s
        bb_pct = ((prices - bb_lo) / (4 * bb_s) * 100)

        dc20_high  = hi.rolling(20).max().shift(1)
        dc20_break = prices > dc20_high
        vol_r      = vol / vol.rolling(20).mean()

        rs20 = float(prices.pct_change(20).iloc[-1]) - float(
            smh_px.pct_change(20).reindex(prices.index).ffill().iloc[-1]
        )
        cta_macro = _cta_combo(smh_px, ref_px, prices)   # 宏观门控（SMH+SPY均值）
        cta_stock = float(_cta_series_full(prices).dropna().iloc[-1]) if len(prices) > 80 else 0.0  # 个股自身动量
        cta_val   = round(cta_stock, 2)                   # 表格展示用个股CTA

        try:
            smc_df  = smc_signal(ticker)
            smc_val = int(smc_df["signal"].iloc[-1])
        except Exception:
            smc_val = 0

        # ── 最新值 ──
        px    = float(prices.iloc[-1])
        e20v  = float(e20.iloc[-1])
        e60v  = float(e60.iloc[-1])
        atrv  = float(atr.iloc[-1])
        rsiv  = float(rsi.dropna().iloc[-1]) if not rsi.dropna().empty else 50.0
        bbpct = float(bb_pct.dropna().iloc[-1]) if not bb_pct.dropna().empty else 50.0
        volr  = float(vol_r.dropna().iloc[-1]) if not vol_r.dropna().empty else 1.0
        dc_ok = bool(dc20_break.iloc[-1])
        dc20h = float(dc20_high.dropna().iloc[-1]) if not dc20_high.dropna().empty else px * 1.1
        date  = prices.index[-1].date()

        px_chg = float(prices.pct_change().iloc[-1]) * 100

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

        # ── 评分 ──
        signals  = []
        warnings = []
        score    = 0
        ema_ok   = e20v > e60v

        if ema_ok:
            signals.append("EMA金叉（趋势向上）")
            score += 1
        else:
            warnings.append("EMA死叉（趋势向下）")

        if dc_ok:
            signals.append(f"DC20突破（{px:.1f}>{dc20h:.1f}）")
            score += 2
        else:
            dist_dc = (dc20h - px) / px * 100
            signals.append(f"DC20未突破（差{dist_dc:.1f}%）")

        ext_e20 = (px / e20v - 1) * 100
        if abs(ext_e20) <= 4:
            signals.append(f"贴近EMA20（{ext_e20:+.1f}%）")
            score += 1
        elif ext_e20 > 15:
            warnings.append(f"偏离EMA20 +{ext_e20:.0f}%，追高风险")
        else:
            signals.append(f"距EMA20 {ext_e20:+.1f}%")

        if rsiv < 40:
            signals.append(f"RSI超卖={rsiv:.0f}")
            score += 1
        elif rsiv > 78:
            warnings.append(f"RSI超买={rsiv:.0f}")
        else:
            signals.append(f"RSI={rsiv:.0f}")

        if bbpct < 15:
            signals.append(f"布林下轨（{bbpct:.0f}%）")
            score += 1
        elif bbpct > 95:
            warnings.append(f"布林上轨（{bbpct:.0f}%）")
        else:
            signals.append(f"布林位{bbpct:.0f}%")

        if volr >= 1.5:
            signals.append(f"放量{volr:.1f}x")
            score += 1
        elif volr < 0.7:
            warnings.append(f"缩量{volr:.1f}x")
        else:
            signals.append(f"量能{volr:.1f}x")

        if smc_val >= 2:
            signals.append(f"SMC={smc_val}（机构建仓）")
            score += 1
        elif smc_val == 1:
            signals.append("SMC=1（有建仓迹象）")

        if rs20 > 0.05:
            signals.append(f"跑赢SMH +{rs20*100:.1f}%")
        elif rs20 < -0.05:
            warnings.append(f"弱于SMH {rs20*100:.1f}%")

        if cta_macro < 0:
            warnings.append(f"板块CTA={cta_macro:.2f}，禁止入场")
            ema_ok = False
        elif cta_macro < 0.3:
            warnings.append(f"板块CTA偏弱={cta_macro:.2f}")

        # ── 止损 ──
        stop_atr   = round(px - atrv * 2, 1)
        stop_ema60 = round(e60v * 0.98, 1)
        stop_pct10 = round(px * 0.90, 1)

        # ── 最优策略实时信号状态（先算，再参与 verdict）──
        strat_states = []
        top3_list = (top3_map or {}).get(ticker, [])
        if top3_list:
            smh_cta_s = _cta_combo_series(smh_px, prices)
            spy_cta_s = _cta_combo_series(ref_px, prices)
            qqq_cta_s = _cta_combo_series(qqq_px, prices)
            strat_states = _quick_strategy_states(
                ticker, top3_list, prices, ohlcv,
                smh_cta_s, spy_cta_s, qqq_cta_s, extra_ctas or {},
            )

        # 策略状态统计（用于修正 verdict）
        n_entry   = sum(1 for s in strat_states if s["state"] == "entry_today")
        n_intrade = sum(1 for s in strat_states if s["state"] == "in_trade")
        n_exit    = sum(1 for s in strat_states if s["state"] == "exit_today")
        n_wait    = sum(1 for s in strat_states if s["state"] == "waiting")
        n_strats  = len(strat_states)

        # ── 综合判断（策略信号优先级最高）──
        if not ema_ok or cta_macro < 0:
            verdict, verdict_code = "回避", "red"
            entry_zone = "不入场"

        # 有策略今日出场：不管技术面多好，都不新入场
        elif n_exit > 0 and n_entry == 0:
            if n_exit == n_strats:
                verdict, verdict_code = "出场信号", "red"
                entry_zone = f"最优策略全部今日出场，暂勿追入"
                warnings.append(f"⚠ Top{n_strats}策略今日均出场")
            else:
                verdict, verdict_code = "观望", "gray"
                entry_zone = f"部分策略出场（{n_exit}/{n_strats}），等信号明朗"
                warnings.append(f"⚠ {n_exit}个策略今日出场")

        # 有策略今日入场：强信号
        elif n_entry > 0:
            if n_entry >= 2 or (n_entry == 1 and score >= 3):
                verdict, verdict_code = "强烈推荐", "green"
                entry_zone = f"当前价 {px:.1f}，{n_entry}个策略今日触发入场"
                signals.append(f"✓ {n_entry}个最优策略今日入场信号")
            else:
                verdict, verdict_code = "可以入场", "yellow"
                entry_zone = f"当前价 {px:.1f}，1个策略今日触发入场"
                signals.append("✓ 1个最优策略今日入场信号")

        # 策略全部在仓：趋势持续，但不是新买点
        elif n_intrade == n_strats and n_strats > 0:
            verdict, verdict_code = "持仓续持", "blue"
            entry_zone = f"入场点已过，追高有风险，等回踩EMA20≈{e20v:.0f}"
            signals.append(f"✓ Top{n_strats}策略均在仓中（非新买点）")

        # 策略全部等待：按技术面打分
        else:
            if dc_ok and score >= 3:
                verdict, verdict_code = "观望偏多", "yellow"
                entry_zone = f"技术面较强但策略未触发，等DC20>{dc20h:.0f}确认"
            elif rsiv < 40 and bbpct < 20:
                verdict, verdict_code = "关注低吸", "yellow"
                entry_zone = f"当前价 {px:.1f}，超卖区域，等策略入场信号"
            elif score >= 3:
                verdict, verdict_code = "观望偏多", "yellow"
                entry_zone = f"等DC20>{dc20h:.0f} 或回踩EMA20≈{e20v:.0f}"
            else:
                verdict, verdict_code = "观望", "gray"
                entry_zone = f"等DC20>{dc20h:.0f} 或回踩EMA20≈{e20v:.0f}"

        return {
            "ticker":        ticker,
            "date":          str(date),
            "price":         round(px, 2),
            "price_chg":     round(px_chg, 2),
            "verdict":       verdict,
            "verdict_code":  verdict_code,
            "entry_zone":    entry_zone,
            "score":         score,
            "signals":       signals,
            "warnings":      warnings,
            "stop_atr":      stop_atr,
            "stop_ema60":    stop_ema60,
            "stop_pct10":    stop_pct10,
            "ema20":         round(e20v, 1),
            "ema60":         round(e60v, 1),
            "atr":           round(atrv, 1),
            "atr_pct":       round(atrv / px * 100, 1),
            "dc20_high":     round(dc20h, 1),
            "cta_val":       round(cta_val, 2),
            "smc_val":       smc_val,
            "rsi":           round(rsiv, 1),
            "bb_pct":        round(bbpct, 1),
            "vol_r":         round(volr, 2),
            "streak":        streak,
            "rs20":          round(rs20 * 100, 1),
            "strategies":    strat_states,   # top3 策略当前状态
            "error":         None,
        }

    except Exception as e:
        logger.warning(f"scan_ticker {ticker}: {e}")
        return {
            "ticker": ticker, "error": str(e), "verdict_code": "gray",
            "price": 0, "price_chg": 0, "verdict": "错误", "entry_zone": "-",
            "score": 0, "signals": [], "warnings": [str(e)],
        }


# ──────────────────────────────────────────────
# 批量扫描
# ──────────────────────────────────────────────

def scan_all() -> dict:
    """并行扫描所有 watchlist 股票"""
    smh_px = get_prices("SMH", start="2024-01-01")
    ref_px = get_prices("SPY", start="2024-01-01")
    qqq_px = get_prices("QQQ", start="2024-01-01")

    # 行业 CTA（和 optimize_stocks 保持一致）
    _sector_etfs = {"soxx": "SOXX", "igv": "IGV", "xly": "XLY", "xar": "XAR", "ibit": "IBIT"}
    extra_ctas = {}
    for _name, _sym in _sector_etfs.items():
        try:
            extra_ctas[_name] = _cta_series_full(get_prices(_sym))
        except Exception:
            pass

    # 加载已保存的 top3 策略
    top3_map = _load_top3()

    group_results: dict[str, dict] = {g: {} for g in WATCHLIST}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(scan_ticker, ticker, smh_px, ref_px, qqq_px, extra_ctas, top3_map): (group, ticker)
            for group, tickers in WATCHLIST.items()
            for ticker in tickers
        }
        for future in as_completed(futures):
            group, ticker = futures[future]
            try:
                group_results[group][ticker] = future.result()
            except Exception as e:
                group_results[group][ticker] = {
                    "ticker": ticker, "error": str(e), "verdict_code": "gray",
                    "price": 0, "price_chg": 0, "verdict": "错误", "entry_zone": "-",
                    "score": 0, "signals": [], "warnings": [],
                }

    order_map = {"green": 0, "yellow": 1, "gray": 2, "red": 3}
    ordered = {}
    for group, tickers in WATCHLIST.items():
        lst = [group_results[group].get(t, {"ticker": t, "verdict_code": "gray", "price": 0,
               "price_chg": 0, "verdict": "无数据", "entry_zone": "-", "score": 0,
               "signals": [], "warnings": []}) for t in tickers]
        lst.sort(key=lambda x: order_map.get(x.get("verdict_code", "gray"), 2))
        ordered[group] = lst

    return ordered


# ──────────────────────────────────────────────
# 宏观数据
# ──────────────────────────────────────────────

def get_macro() -> dict:
    """VIX + 关键 CTA 信号"""
    import yfinance as yf

    result: dict = {"vix": None, "vix_status": "unknown", "vix_label": "VIX 获取失败", "ctas": {}}
    try:
        vix_data = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
        vix = float(vix_data["Close"].iloc[-1])
        result["vix"] = round(vix, 1)
        if vix < 20:
            result["vix_status"] = "low"
            result["vix_label"]  = f"VIX {vix:.1f}  低波动"
        elif vix < 30:
            result["vix_status"] = "normal"
            result["vix_label"]  = f"VIX {vix:.1f}  正常"
        elif vix < 40:
            result["vix_status"] = "high"
            result["vix_label"]  = f"VIX {vix:.1f}  高波动"
        else:
            result["vix_status"] = "extreme"
            result["vix_label"]  = f"VIX {vix:.1f}  极端"
    except Exception:
        pass

    for name, ticker in [("SPY", "SPY"), ("QQQ", "QQQ"), ("SMH", "SMH")]:
        try:
            val = cta_trend_signal(ticker)
            result["ctas"][name] = {
                "value":     round(val, 2),
                "direction": "多头" if val > 0.15 else ("空头" if val < -0.15 else "中性"),
                "ok":        val > 0,
            }
        except Exception:
            result["ctas"][name] = {"value": 0, "direction": "未知", "ok": False}

    return result


# ──────────────────────────────────────────────
# 资金流向
# ──────────────────────────────────────────────

def get_flows() -> dict:
    """板块资金流向 + 热门股票"""
    rows = []
    for name, ticker in FLOW_ETFS.items():
        try:
            prices = get_prices(ticker)
            ohlcv  = get_ohlcv(ticker)
            if prices.empty or len(prices) < 25:
                continue

            ret_1w = float(prices.pct_change(5).iloc[-1]) * 100
            ret_1m = float(prices.pct_change(21).iloc[-1]) * 100
            ret_3m = float(prices.pct_change(63).iloc[-1]) * 100 if len(prices) >= 63 else 0

            vol_today = float(ohlcv["Volume"].iloc[-1])
            vol_avg20 = float(ohlcv["Volume"].rolling(20).mean().iloc[-1])
            vol_r = round(vol_today / vol_avg20, 2) if vol_avg20 > 0 else 1.0

            # 简单趋势信号
            ret  = prices.pct_change().dropna()
            mom60 = float(ret.rolling(60).mean().iloc[-1]) * 252 if len(ret) >= 60 else 0
            vol60 = float(ret.rolling(60).std().iloc[-1]) * (252 ** 0.5) if len(ret) >= 60 else 1
            sig = float(np.clip(mom60 / vol60 if vol60 > 0 else 0, -1, 1))

            rows.append({
                "name":   name,
                "ticker": ticker,
                "ret_1w": round(ret_1w, 1),
                "ret_1m": round(ret_1m, 1),
                "ret_3m": round(ret_3m, 1),
                "vol_r":  vol_r,
                "sig":    round(sig, 2),
                "direction": "多头" if sig > 0.15 else ("空头" if sig < -0.15 else "中性"),
            })
        except Exception as e:
            logger.debug(f"flows {ticker}: {e}")

    rows.sort(key=lambda x: x["ret_1w"], reverse=True)

    # 热门个股（成交量放大 > 1.5x）
    hot_stocks = []
    all_tickers = [t for tickers in WATCHLIST.values() for t in tickers]
    for ticker in all_tickers:
        try:
            ohlcv = get_ohlcv(ticker)
            if ohlcv.empty or len(ohlcv) < 22:
                continue
            vol_r = float(ohlcv["Volume"].iloc[-1] / ohlcv["Volume"].rolling(20).mean().iloc[-1])
            prices = get_prices(ticker)
            chg = float(prices.pct_change().iloc[-1]) * 100 if not prices.empty else 0
            if vol_r >= 1.5:
                hot_stocks.append({
                    "ticker": ticker,
                    "vol_r":  round(vol_r, 1),
                    "chg":    round(chg, 1),
                })
        except Exception:
            pass

    hot_stocks.sort(key=lambda x: x["vol_r"], reverse=True)

    return {"sectors": rows, "hot_stocks": hot_stocks[:10], "error": None}


# ──────────────────────────────────────────────
# CTA 仪表盘
# ──────────────────────────────────────────────

def get_cta_dashboard() -> dict:
    """多资产 CTA 趋势信号"""
    try:
        df   = run_cta_dashboard()
        rows = df.to_dict(orient="records")
        return {"rows": rows, "error": None}
    except Exception as e:
        logger.error(f"get_cta_dashboard: {e}")
        return {"rows": [], "error": str(e)}


# ──────────────────────────────────────────────
# 板块全景追踪（sector_watch 完整数据）
# ──────────────────────────────────────────────

def get_sector_full() -> dict:
    """返回 sector_watch 完整板块数据（含 CTA + 资金流 + 多时间维度）"""
    try:
        from sector_watch import scan_sectors as _scan_sectors
        import math

        df = _scan_sectors()
        if df.empty:
            return {"rows": [], "signals": {}, "error": "数据获取失败"}

        def _safe(v):
            if v is None:
                return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        rows = []
        for rec in df.to_dict(orient="records"):
            rows.append({k: _safe(v) for k, v in rec.items()})

        # Signal type grouping
        signal_groups: dict = {}
        for r in rows:
            sig = r.get("信号类型", "观望等待─")
            if sig not in signal_groups:
                signal_groups[sig] = []
            signal_groups[sig].append({"板块": r["板块"], "ETF": r["ETF"]})

        return {"rows": rows, "signals": signal_groups, "error": None}
    except Exception as e:
        logger.error(f"get_sector_full: {e}")
        return {"rows": [], "signals": {}, "error": str(e)}


def get_bt_signals(ticker: str) -> dict:
    """运行最优策略回测，返回 K 线 + 买卖点 + 交易记录（供 /api/backtest/{ticker} 使用）"""
    try:
        from backtest_review import get_optimal_and_signals
        return get_optimal_and_signals(ticker.upper())
    except Exception as e:
        logger.error(f"get_bt_signals {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}
