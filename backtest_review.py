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
                   smh_cta: pd.Series, spy_cta: pd.Series):
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

    combo_cta = (
        smh_cta.reindex(prices.index).ffill().fillna(0) +
        spy_cta.reindex(prices.index).ffill().fillna(0)
    ) / 2

    entries = {
        "ema2060":  (e20 > e60).fillna(False).astype(float),
        "dc20":     (prices > dc20h).fillna(False).astype(float),
        "dc20|ema": ((prices > dc20h) | (e20 > e60)).fillna(False).astype(float),
        "ma5200":   (ma50 > ma200).fillna(False).astype(float),
        "bb_lo":    (prices < bb_lo).fillna(False).astype(float),
        "obv_up":   (obv > obv_ma20).fillna(False).astype(float),
        "cmf_pos":  (cmf > 0.05).fillna(False).astype(float),
        "mfi_os":   (mfi < 35).fillna(False).astype(float),
        "vol_surge": vol_surge_up.fillna(False).astype(float),
        "dc20+obv": ((prices > dc20h) & (obv > obv_ma20)).fillna(False).astype(float),
        "dc20+cmf": ((prices > dc20h) & (cmf > 0.0)).fillna(False).astype(float),
        "vol+ema":  (vol_surge_up & (e20 > e60)).fillna(False).astype(float),
    }
    cta_gates = {
        "none":  pd.Series(1.0, index=prices.index),
        "combo": (combo_cta > 0).astype(float),
        "spy":   (spy_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
        "smh":   (smh_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
    }
    exits = {
        "ema_x":    (e20 < e60).fillna(False).astype(float),
        "ma_x":     (ma50 < ma200).fillna(False).astype(float),
        "rsi80":    (rsi > 80).astype(float),
        "obv_down": (obv < obv_ma20).fillna(False).astype(float),
        "cmf_neg":  (cmf < -0.05).fillna(False).astype(float),
    }

    e_sig = entries.get(entry_name, entries["dc20"])
    c_sig = cta_gates.get(cta_name, cta_gates["none"])
    x_sig = exits.get(exit_name, exits["ema_x"])

    if entry_name == "bb_lo":
        bb_exit_base = (prices > bb_hi).fillna(False)
        add_x = exits.get(exit_name, pd.Series(False, index=prices.index))
        if hasattr(add_x, "fillna"):
            add_x = add_x.fillna(False)
        x_sig = (bb_exit_base | add_x).astype(float)

    pos_arr = _make_pos(
        e_sig.fillna(0).values,
        c_sig.fillna(0).values,
        x_sig.fillna(0).values,
    )
    return prices, ohlcv, pd.Series(pos_arr, index=prices.index)


# ──────────────────────────────────────────────
# 主函数：优化 + 信号提取
# ──────────────────────────────────────────────

def get_optimal_and_signals(ticker: str) -> dict:
    """
    1. 跑批量优化找到最优策略
    2. 重建该策略的信号序列
    3. 提取买卖点 + K 线 + 交易记录
    返回 dict 可直接序列化为 JSON
    """
    try:
        smh_px  = get_prices("SMH")
        spy_px  = get_prices("SPY")
        smh_cta = _cta_series(smh_px)
        spy_cta = _cta_series(spy_px)

        # ── 找最优策略 ──
        from optimize_stocks import optimize_ticker
        opt = optimize_ticker(ticker, smh_cta, spy_cta)
        if opt.get("error"):
            return {"ticker": ticker, "error": opt["error"]}

        entry_name = opt["entry"]
        cta_name   = opt["cta"]
        exit_name  = opt["exit"]

        # ── 重建信号 ──
        prices, ohlcv, signal = _build_signals(
            ticker, entry_name, cta_name, exit_name, smh_cta, spy_cta
        )

        # ── 完整回测（含交易记录）──
        res      = backtest(prices, signal)
        trades_df = res["trades"]

        # 胜率 / 平均持仓
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

        # ── K 线（近 6 个月）──
        cutoff     = prices.index[-1] - pd.Timedelta(days=185)
        recent_idx = prices.index[prices.index >= cutoff]

        candles = []
        for dt in recent_idx:
            try:
                row = ohlcv.loc[dt]
                o = _s(row["Open"])
                h = _s(row["High"])
                l = _s(row["Low"])
                v = int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
            except Exception:
                o = h = l = _s(prices.get(dt))
                v = 0
            c = _s(prices.get(dt))
            if c is None:
                continue
            candles.append({
                "time":   dt.strftime("%Y-%m-%d"),
                "open":   o or c,
                "high":   h or c,
                "low":    l or c,
                "close":  c,
                "volume": v,
            })

        # ── 买卖点 markers（全历史，图表只展示可视范围内的）──
        markers = []
        for _, tr in trades_df.iterrows():
            ep  = _s(tr["entry_px"])
            xp  = _s(tr["exit_px"])
            pnl = _s(tr["pnl_pct"])
            markers.append({
                "time":  tr["entry_date"].strftime("%Y-%m-%d"),
                "type":  "buy",
                "price": ep,
                "text":  f"B ${ep:.2f}" if ep else "B",
            })
            markers.append({
                "time":  tr["exit_date"].strftime("%Y-%m-%d"),
                "type":  "sell",
                "price": xp,
                "pnl":   pnl,
                "text":  f"S ${xp:.2f} ({pnl:+.1f}%)" if (xp and pnl is not None) else "S",
            })

        # ── 交易记录（最近在前，只取近3年）──
        cutoff_trades = prices.index[-1] - pd.Timedelta(days=1095)
        trades_list   = []
        for _, tr in trades_df.iterrows():
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
        trades_list = list(reversed(trades_list))

        # ── 绩效摘要 ──
        p3m = opt["periods"]["3M"]
        p1m = opt["periods"]["1M"]
        p1y = opt["periods"]["1Y"]

        m = res["metrics"]
        def _metric(key, default=0):
            v = m.get(key, default)
            if isinstance(v, str):
                v = v.replace("%", "").replace("×", "").strip()
                try:
                    v = float(v)
                except Exception:
                    return default
            return _s(v)

        return {
            "ticker":   ticker,
            "strategy": {
                "entry": entry_name,
                "cta":   cta_name,
                "exit":  exit_name,
                "label": f"{entry_name} | {cta_name} | {exit_name}",
            },
            "candles":  candles,
            "markers":  markers,
            "trades":   trades_list,
            "metrics": {
                "calmar":    _metric("Calmar比率"),
                "sharpe":    _metric("Sharpe比率"),
                "n_trades":  len(trades_df),
                "win_rate":  win_rate,
                "avg_hold":  avg_hold,
                "in_market": _s(float(signal.mean()) * 100),
                "ret_1m":    _s(p1m["ret"]),
                "ret_3m":    _s(p3m["ret"]),
                "calmar_1y": _s(p1y["calmar"]),
                "dd_1y":     _s(p1y["dd"]),
                "score":     _s(opt["score"]),
            },
            "error": None,
        }

    except Exception as e:
        logger.error(f"get_optimal_and_signals {ticker}: {e}", exc_info=True)
        return {"ticker": ticker, "error": str(e)}
