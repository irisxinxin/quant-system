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
# Watchlist
# ──────────────────────────────────────────────
WATCHLIST = {
    "A类·高波动": [
        "LITE", "IREN", "COIN", "QUBT", "IONQ", "RGTI",
        "OKLO", "NNE", "SMR", "BWXT",
        "SOUN", "BBAI", "PLTR",
        "MSTR", "HOOD", "RIOT", "MARA",
    ],
    "B类·成长股": [
        "MRVL", "SNDK", "ANET", "SMCI",
        "VRT", "VST", "CEG", "ETN", "HUBB",
        "BE", "AXTI", "WOLF", "ENPH", "FSLR",
        "HIMS", "DKNG", "SOFI", "UPST", "AFRM",
        "UBER", "LYFT", "SNOW", "DDOG", "NET",
    ],
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

def scan_ticker(ticker: str, smh_px: pd.Series = None, ref_px: pd.Series = None) -> dict:
    """扫描单只股票，返回结构化分析结果"""
    try:
        prices = get_prices(ticker, start="2024-01-01")
        if smh_px is None:
            smh_px = get_prices("SMH", start="2024-01-01")
        if ref_px is None:
            ref_px = get_prices("SPY", start="2024-01-01")
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
        cta_val = _cta_combo(smh_px, ref_px, prices)

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

        if cta_val < 0:
            warnings.append(f"板块CTA={cta_val:.2f}，禁止入场")
            ema_ok = False
        elif cta_val < 0.3:
            warnings.append(f"板块CTA偏弱={cta_val:.2f}")

        # ── 止损 ──
        stop_atr   = round(px - atrv * 2, 1)
        stop_ema60 = round(e60v * 0.98, 1)
        stop_pct10 = round(px * 0.90, 1)

        # ── 综合判断 ──
        if not ema_ok or cta_val < 0:
            verdict, verdict_code = "回避", "red"
            entry_zone = "不入场"
        elif dc_ok and score >= 3:
            verdict, verdict_code = "强烈推荐", "green"
            entry_zone = f"当前价 {px:.1f}，收盘确认突破"
        elif dc_ok:
            verdict, verdict_code = "可以入场", "yellow"
            entry_zone = f"当前价 {px:.1f}，小仓位试探"
        elif abs(ext_e20) <= 4 and score >= 3:
            verdict, verdict_code = "推荐入场", "green"
            entry_zone = f"EMA20附近 {e20v:.0f}~{e20v*1.02:.0f}"
        elif rsiv < 40 and bbpct < 20:
            verdict, verdict_code = "可以入场", "yellow"
            entry_zone = f"当前价 {px:.1f}，需次日确认"
        elif score >= 3:
            verdict, verdict_code = "观望偏多", "yellow"
            entry_zone = f"等DC20>{dc20h:.0f} 或回踩EMA20≈{e20v:.0f}"
        else:
            verdict, verdict_code = "观望", "gray"
            entry_zone = f"等DC20>{dc20h:.0f} 或回踩EMA20≈{e20v:.0f}"

        return {
            "ticker":       ticker,
            "date":         str(date),
            "price":        round(px, 2),
            "price_chg":    round(px_chg, 2),
            "verdict":      verdict,
            "verdict_code": verdict_code,
            "entry_zone":   entry_zone,
            "score":        score,
            "signals":      signals,
            "warnings":     warnings,
            "stop_atr":     stop_atr,
            "stop_ema60":   stop_ema60,
            "stop_pct10":   stop_pct10,
            "ema20":        round(e20v, 1),
            "ema60":        round(e60v, 1),
            "atr":          round(atrv, 1),
            "atr_pct":      round(atrv / px * 100, 1),
            "dc20_high":    round(dc20h, 1),
            "cta_val":      round(cta_val, 2),
            "smc_val":      smc_val,
            "rsi":          round(rsiv, 1),
            "bb_pct":       round(bbpct, 1),
            "vol_r":        round(volr, 2),
            "streak":       streak,
            "rs20":         round(rs20 * 100, 1),
            "error":        None,
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

    group_results: dict[str, dict] = {g: {} for g in WATCHLIST}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(scan_ticker, ticker, smh_px, ref_px): (group, ticker)
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
