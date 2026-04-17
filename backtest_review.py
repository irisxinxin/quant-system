"""
backtest_review.py — 回测信号历史（Web 仪表盘展示用）
对单只股票运行已知最优策略，返回 K 线 + 买卖点 + 交易记录 + 绩效
"""
import sys
import math
import logging
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data.downloader import get_prices, get_ohlcv
from backtest.engine import backtest
from config import CTA_LOOKBACKS, CTA_VOL_WIN

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 复用 optimize_stocks 的核心函数
# ──────────────────────────────────────────────

def _cta_series(px: pd.Series) -> pd.Series:
    r = px.pct_change().dropna()
    sigs = []
    for lk in CTA_LOOKBACKS:
        m = r.rolling(lk).mean() * 252
        v = r.rolling(CTA_VOL_WIN).std() * np.sqrt(252)
        sigs.append((m / v.replace(0, np.nan)).clip(-1, 1))
    return pd.concat(sigs, axis=1).mean(axis=1)


def _make_pos(entry: np.ndarray, cta_ok: np.ndarray, exit_cond: np.ndarray) -> np.ndarray:
    pos = np.zeros(len(entry), dtype=float)
    in_trade = False
    for i in range(len(entry)):
        if not in_trade:
            if entry[i] and cta_ok[i]:
                in_trade = True
                pos[i] = 1.0
        else:
            if exit_cond[i]:
                in_trade = False
            else:
                pos[i] = 1.0
    return pos


def _build_signals(ticker: str, entry_name: str, cta_name: str, exit_name: str,
                   smh_cta: pd.Series, spy_cta: pd.Series, qqq_cta: pd.Series | None = None,
                   extra_ctas: dict | None = None):
    """
    用指定参数重建仓位信号序列。
    返回 (prices, ohlcv_aligned, signal_series)
    """
    prices = get_prices(ticker)
    ohlcv  = get_ohlcv(ticker)
    ohlcv  = ohlcv.reindex(prices.index)

    hi  = ohlcv["High"].fillna(prices)
    lo  = ohlcv["Low"].fillna(prices)
    vol = ohlcv["Volume"].fillna(0)

    e20   = prices.ewm(span=20, adjust=False).mean()
    e60   = prices.ewm(span=60, adjust=False).mean()
    ma50  = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()
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

    hl       = (hi - lo).replace(0, np.nan)
    mf_mult  = ((prices - lo) - (hi - prices)) / hl
    cmf      = (mf_mult * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)

    tp      = (hi + lo + prices) / 3
    raw_mf  = tp * vol
    pos_mf  = raw_mf.where(tp > tp.shift(1), 0.0)
    neg_mf  = raw_mf.where(tp < tp.shift(1), 0.0)
    mf_rat  = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    mfi     = (100 - 100 / (1 + mf_rat)).fillna(50)

    vol_ratio    = vol / vol.rolling(20).mean().replace(0, np.nan)
    vol_surge_up = (vol_ratio > 1.5) & (prices > e20)
    hi20_max     = hi.rolling(20).max()
    ema20_band   = (prices >= e20 * 0.97) & (prices <= e20 * 1.04) & (e20 > e60)
    rsi_deep_os  = rsi < 35
    rsi_ext_os   = rsi < 28

    combo_cta_s = (
        smh_cta.reindex(prices.index).ffill().fillna(0) +
        spy_cta.reindex(prices.index).ffill().fillna(0)
    ) / 2
    qqq_s = qqq_cta.reindex(prices.index).ffill().fillna(0) if qqq_cta is not None \
            else pd.Series(0.0, index=prices.index)

    entries = {
        "ema2060":       (e20 > e60).fillna(False).astype(float),
        "dc20":          (prices > dc20h).fillna(False).astype(float),
        "dc20|ema":      ((prices > dc20h) | (e20 > e60)).fillna(False).astype(float),
        "ma5200":        (ma50 > ma200).fillna(False).astype(float),
        "bb_lo":         (prices < bb_lo).fillna(False).astype(float),
        "obv_up":        (obv > obv_ma20).fillna(False).astype(float),
        "cmf_pos":       (cmf > 0.05).fillna(False).astype(float),
        "mfi_os":        (mfi < 35).fillna(False).astype(float),
        "vol_surge":     vol_surge_up.fillna(False).astype(float),
        "ema20_dip":     ema20_band.fillna(False).astype(float),
        "ema20_dip+obv": (ema20_band & (obv > obv_ma20)).fillna(False).astype(float),
        "dc20+obv":      ((prices > dc20h) & (obv > obv_ma20)).fillna(False).astype(float),
        "dc20+cmf":      ((prices > dc20h) & (cmf > 0.0)).fillna(False).astype(float),
        "vol+ema":       (vol_surge_up & (e20 > e60)).fillna(False).astype(float),
        "rsi35":         rsi_deep_os.fillna(False).astype(float),
        "rsi28":         rsi_ext_os.fillna(False).astype(float),
    }
    cta_gates = {
        "none":  pd.Series(1.0, index=prices.index),
        "combo": (combo_cta_s > 0).astype(float),
        "soft":  (combo_cta_s > -0.25).astype(float),
        "spy":   (spy_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
        "smh":   (smh_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
        "qqq":   (qqq_s > 0).astype(float),
    }
    if extra_ctas:
        for _cn, _cs in extra_ctas.items():
            if _cn not in cta_gates:
                cta_gates[_cn] = (_cs.reindex(prices.index).ffill().fillna(0) > 0).astype(float)
    rsi_was_hot  = rsi.rolling(5).max() > 70
    exits = {
        "ema_x":    (e20 < e60).fillna(False).astype(float),
        "ma_x":     (ma50 < ma200).fillna(False).astype(float),
        "rsi80":    (rsi > 80).astype(float),
        "obv_down": (obv < obv_ma20).fillna(False).astype(float),
        "cmf_neg":  (cmf < -0.05).fillna(False).astype(float),
        "trail_8":  (prices < hi20_max * 0.92).fillna(False).astype(float),
        "trail_12": (prices < hi20_max * 0.88).fillna(False).astype(float),
        "rsi_fade": ((rsi < 60) & rsi_was_hot).fillna(False).astype(float),
        "rsi70":    (rsi > 70).astype(float),
    }

    e_sig = entries.get(entry_name, entries["dc20"])
    c_sig = cta_gates.get(cta_name, cta_gates["none"])
    x_sig = exits.get(exit_name, exits["ema_x"])

    if entry_name in ("bb_lo", "rsi35", "rsi28"):
        if entry_name == "bb_lo":
            base_exit = (prices > bb_hi).fillna(False)
        else:
            base_exit = (prices > e20).fillna(False)
        all_exits = {
            "ema_x":    (e20 < e60).fillna(False),
            "ma_x":     (ma50 < ma200).fillna(False),
            "rsi80":    (rsi > 80),
            "rsi70":    (rsi > 70),
            "obv_down": (obv < obv_ma20).fillna(False),
            "cmf_neg":  (cmf < -0.05).fillna(False),
            "trail_8":  (prices < hi20_max * 0.92).fillna(False),
            "trail_12": (prices < hi20_max * 0.88).fillna(False),
        }
        add_x = all_exits.get(exit_name, pd.Series(False, index=prices.index))
        x_sig = (base_exit | add_x).astype(float)

    pos_arr = _make_pos(
        e_sig.fillna(0).values,
        c_sig.fillna(0).values,
        x_sig.fillna(0).values,
    )
    return prices, ohlcv, pd.Series(pos_arr, index=prices.index)


# ──────────────────────────────────────────────
# 主函数：优化 + 信号提取
# ──────────────────────────────────────────────

def _signals_for_combo(ticker, entry_name, cta_name, exit_name, smh_cta, spy_cta, qqq_cta=None, extra_ctas=None):
    """重建指定策略，返回 candles/markers/trades/metrics dict"""
    prices, ohlcv, signal = _build_signals(ticker, entry_name, cta_name, exit_name, smh_cta, spy_cta, qqq_cta, extra_ctas)
    res       = backtest(prices, signal)
    trades_df = res["trades"]

    if not trades_df.empty:
        n_win    = int((trades_df["pnl_pct"] > 0).sum())
        win_rate = round(n_win / len(trades_df) * 100, 1)
        avg_hold = round(float(trades_df["days_held"].mean()), 1)
    else:
        win_rate = 0.0
        avg_hold = 0.0

    def _s(v):
        if v is None:
            return None
        try:
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
        except Exception:
            return None

    # ── K 线（近 6 个月）——用 iloc 位置索引避免 tz / dtype 不匹配 ──
    cutoff     = prices.index[-1] - pd.Timedelta(days=185)
    mask       = prices.index >= cutoff
    recent_prices = prices[mask]       # Series
    recent_ohlcv  = ohlcv[mask]        # DataFrame（已 reindex 对齐）

    candles = []
    for i in range(len(recent_prices)):
        try:
            c = _s(float(recent_prices.iloc[i]))
            if c is None:
                continue
            dt_str = recent_prices.index[i].strftime("%Y-%m-%d")
            try:
                orow = recent_ohlcv.iloc[i]
                o = _s(orow["Open"])
                h = _s(orow["High"])
                l = _s(orow["Low"])
                vol_raw = orow["Volume"]
                v = int(float(vol_raw)) if not pd.isna(vol_raw) else 0
            except Exception:
                o = h = l = None
                v = 0
            candles.append({
                "time":   dt_str,
                "open":   o if o is not None else c,
                "high":   h if h is not None else c,
                "low":    l if l is not None else c,
                "close":  c,
                "volume": v,
            })
        except Exception:
            continue

    # ── 买持基准线（归一化到 1.0）──
    bah = []
    if len(recent_prices) > 0:
        try:
            base0 = float(recent_prices.iloc[0])
            for i in range(len(recent_prices)):
                try:
                    val = float(recent_prices.iloc[i]) / base0
                    bah.append({"time": recent_prices.index[i].strftime("%Y-%m-%d"),
                                 "value": round(val, 4)})
                except Exception:
                    continue
        except Exception:
            pass

    # ── 买卖点 markers ──
    markers = []
    for _, tr in trades_df.iterrows():
        try:
            ep  = _s(tr["entry_px"])
            markers.append({"time": tr["entry_date"].strftime("%Y-%m-%d"), "type": "buy",
                             "price": ep, "text": f"B ${ep:.2f}" if ep else "B"})
        except Exception:
            pass
        try:
            exit_dt = tr.get("exit_date") if hasattr(tr, "get") else tr["exit_date"]
            if pd.notna(exit_dt):
                xp  = _s(tr["exit_px"]); pnl = _s(tr["pnl_pct"])
                markers.append({"time": exit_dt.strftime("%Y-%m-%d"), "type": "sell",
                                 "price": xp, "pnl": pnl,
                                 "text": f"S ${xp:.2f} ({pnl:+.1f}%)" if (xp and pnl is not None) else "S"})
        except Exception:
            pass

    # ── 交易记录（近 3 年，最新在前）──
    cutoff_trades = prices.index[-1] - pd.Timedelta(days=1095)
    trades_list   = []
    for _, tr in trades_df.iterrows():
        try:
            if tr["exit_date"] < cutoff_trades:
                continue
            trades_list.append({
                "entry_date": tr["entry_date"].strftime("%Y-%m-%d"),
                "exit_date":  tr["exit_date"].strftime("%Y-%m-%d"),
                "entry_px":   _s(tr["entry_px"]),
                "exit_px":    _s(tr["exit_px"]),
                "pnl_pct":    _s(tr["pnl_pct"]),
                "days_held":  int(tr["days_held"]),
            })
        except Exception:
            continue
    trades_list = list(reversed(trades_list))

    # ── 分段绩效 ──
    m    = res["metrics"]
    mets = res.get("returns", pd.Series(dtype=float))

    def _pstat(ret):
        if ret.empty or len(ret) < 2:
            return {"ret": 0.0, "calmar": 0.0, "dd": 0.0}
        try:
            cum  = (1 + ret).cumprod()
            tot  = float((cum.iloc[-1] - 1) * 100)
            dd   = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)
            ny   = len(ret) / 252
            cagr = float(((cum.iloc[-1] ** (1 / ny)) - 1) * 100) if ny > 0.05 else tot
            cal  = abs(cagr / dd) if dd < 0 else 0.0
            return {"ret": round(tot, 1), "calmar": round(cal, 2), "dd": round(dd, 1)}
        except Exception:
            return {"ret": 0.0, "calmar": 0.0, "dd": 0.0}

    def _metric(key, default=0):
        v = m.get(key, default)
        if isinstance(v, str):
            v = v.replace("%", "").replace("×", "").strip()
            try:
                v = float(v)
            except Exception:
                return default
        return _s(v)

    p1m = _pstat(mets.iloc[-21:]  if len(mets) >= 21  else mets)
    p3m = _pstat(mets.iloc[-63:]  if len(mets) >= 63  else mets)
    p1y = _pstat(mets.iloc[-252:] if len(mets) >= 252 else mets)

    return {
        "candles":  candles,
        "bah":      bah,
        "markers":  markers,
        "trades":   trades_list,
        "metrics": {
            "calmar":    _metric("Calmar比率"),
            "sharpe":    _metric("Sharpe比率"),
            "n_trades":  len(trades_df),
            "win_rate":  win_rate,
            "avg_hold":  avg_hold,
            "in_market": _s(float(signal.mean()) * 100),
            "ret_1m":    p1m["ret"], "ret_3m":    p3m["ret"],
            "calmar_1y": p1y["calmar"], "dd_1y":  p1y["dd"],
        },
    }


def get_optimal_and_signals(ticker: str) -> dict:
    """
    1. 跑批量优化找到 Top 3 策略
    2. 对每个策略重建信号 + 提取买卖点
    返回 dict 可直接序列化为 JSON
    """
    try:
        smh_px  = get_prices("SMH")
        spy_px  = get_prices("SPY")
        qqq_px  = get_prices("QQQ")
        smh_cta = _cta_series(smh_px)
        spy_cta = _cta_series(spy_px)
        qqq_cta = _cta_series(qqq_px)
        _sector_etfs = {"soxx": "SOXX", "igv": "IGV", "xly": "XLY", "xar": "XAR", "ibit": "IBIT"}
        extra_ctas = {}
        for _name, _sym in _sector_etfs.items():
            try:
                extra_ctas[_name] = _cta_series(get_prices(_sym))
            except Exception:
                pass

        # ── 找 Top 3 最优策略 ──
        from optimize_stocks import optimize_ticker
        opt = optimize_ticker(ticker, smh_cta, spy_cta, qqq_cta, extra_ctas)
        if opt.get("error"):
            return {"ticker": ticker, "error": opt["error"]}

        # Top 3 策略
        top3 = opt.get("top3", [])[:3]
        if not top3:
            return {"ticker": ticker, "error": "无有效策略"}

        strategies_out = []
        for rank, strat in enumerate(top3):
            en, cn, xn = strat["entry"], strat["cta"], strat["exit"]
            try:
                sig_data = _signals_for_combo(ticker, en, cn, xn, smh_cta, spy_cta, qqq_cta, extra_ctas=extra_ctas)
            except Exception as e:
                logger.warning(f"  #{rank+1} {en}+{cn}+{xn} failed: {e}")
                continue
            strategies_out.append({
                "rank":     rank + 1,
                "entry":    en,
                "cta":      cn,
                "exit":     xn,
                "label":    f"{en} | {cn} | {xn}",
                "score":    round(float(strat.get("score", 0)), 2),
                **sig_data,
            })

        if not strategies_out:
            return {"ticker": ticker, "error": "策略信号重建失败"}

        return {
            "ticker":     ticker,
            "strategies": strategies_out,
            "active":     0,   # 默认展示第1个
            "error":      None,
        }

    except Exception as e:
        logger.error(f"get_optimal_and_signals {ticker}: {e}", exc_info=True)
        return {"ticker": ticker, "error": str(e)}
