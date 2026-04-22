"""
data/downloader.py — 统一数据下载 + 本地缓存
优先使用 Tiingo（稳定，不封数据中心IP），fallback yfinance（本地开发）
所有模块调用 get_prices() / get_ohlcv() / get_multi()，不直接用 yfinance/requests
"""
import os
import json
import time
import pickle
import hashlib
import logging
import random
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests as _requests
import yfinance as yf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR, DATA_START, DATA_END, CACHE_EXPIRE_H

# 本地预存基线目录（已 commit 到 git，冷启动时用）
_KLINES_DIR = Path(__file__).parent.parent / "output" / "klines"

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Tiingo 配置
# ─────────────────────────────────────────────
_TIINGO_KEY = os.environ.get("TIINGO_API_KEY", "")

# 这些 ticker 不在 Tiingo 日线接口，走 yfinance
_TIINGO_SKIP = {"^VIX", "VIX"}

# 加密货币 ticker → Tiingo crypto 符号
_CRYPTO_MAP = {
    "BTC-USD":  "btcusd",
    "ETH-USD":  "ethusd",
    "SOL-USD":  "solusd",
    "BNB-USD":  "bnbusd",
}

# Tiingo 全局限速 + 电路熔断
_tiingo_sem          = threading.Semaphore(2)
_tiingo_last_time    = 0.0
_tiingo_time_lock    = threading.Lock()
_TIINGO_MIN_INTERVAL = 0.5          # 秒，正常间隔
_tiingo_broken_until = 0.0          # 熔断截止时间（epoch 秒）
_TIINGO_BREAK_SECS   = 300          # 触发 429 后熔断 5 分钟

# yfinance 全局限速（本地开发 fallback 用）
_yf_sem           = threading.Semaphore(2)
_yf_last_time     = 0.0
_yf_time_lock     = threading.Lock()
_YF_MIN_INTERVAL  = 0.6


# ─────────────────────────────────────────────
# 缓存工具
# ─────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{h}.pkl"


def _is_fresh(path: Path, hours: float = CACHE_EXPIRE_H) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < hours * 3600


def _last_trading_date() -> datetime.date:
    """返回最近一个交易日的日期（用于判断缓存是否是最新的）"""
    now = datetime.now(timezone.utc)
    # 美股收盘 20:00 UTC，收盘前用昨天
    d = now.date() if now.hour >= 20 else (now - timedelta(days=1)).date()
    # 跳过周末
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _save(path: Path, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_baseline(ticker: str, start: str) -> pd.DataFrame:
    """
    从 output/klines/{TICKER}.json 加载预存基线数据。
    冷启动时 pickle 缓存为空，先从 git-committed JSON 读取历史，再只拉增量。
    """
    p = _KLINES_DIR / f"{ticker.upper()}.json"
    if not p.exists():
        return pd.DataFrame()
    try:
        records = json.loads(p.read_text(encoding="utf-8"))
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # 只保留 start 之后的数据（与 pickle 缓存的 cache_key 保持一致）
        df = df[df.index >= pd.Timestamp(start)]
        return df
    except Exception as e:
        logger.warning(f"[klines baseline] {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Tiingo 下载（主路径）
# ─────────────────────────────────────────────

def _tiingo_ohlcv(ticker: str, start: str, end_str: str) -> pd.DataFrame:
    """
    从 Tiingo 下载 OHLCV（复权价，等效于 yfinance auto_adjust=True）
    带全局限速 semaphore + 429 指数退避重试
    返回 DataFrame[Open, High, Low, Close, Volume]，失败返回空 DataFrame
    """
    global _tiingo_last_time

    if not _TIINGO_KEY or ticker in _TIINGO_SKIP:
        return pd.DataFrame()

    # 熔断检查：最近触发过 429，直接跳过 Tiingo
    if time.time() < _tiingo_broken_until:
        return pd.DataFrame()

    headers = {"Authorization": f"Token {_TIINGO_KEY}", "Content-Type": "application/json"}

    def _do_request(url, params):
        """限速 + 单次请求，429 立即熔断不重试"""
        global _tiingo_last_time, _tiingo_broken_until
        with _tiingo_sem:
            with _tiingo_time_lock:
                gap = time.time() - _tiingo_last_time
                if gap < _TIINGO_MIN_INTERVAL:
                    time.sleep(_TIINGO_MIN_INTERVAL - gap)
                _tiingo_last_time = time.time()
            resp = _requests.get(url, headers=headers, params=params, timeout=30)

        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            _tiingo_broken_until = time.time() + _TIINGO_BREAK_SECS
            logger.warning(f"[tiingo 429] {ticker} — 熔断 {_TIINGO_BREAK_SECS//60} 分钟，切换 yfinance")
            return None
        logger.warning(f"[tiingo] {ticker} HTTP {resp.status_code}")
        return None

    try:
        # ── 加密货币走独立接口 ──
        if ticker in _CRYPTO_MAP:
            sym = _CRYPTO_MAP[ticker]
            resp = _do_request(
                "https://api.tiingo.com/tiingo/crypto/prices",
                {"tickers": sym, "startDate": start, "endDate": end_str, "resampleFreq": "1Day"},
            )
            if resp is None or not resp.json():
                return pd.DataFrame()
            raw = resp.json()[0].get("priceData", [])
            if not raw:
                return pd.DataFrame()
            df = pd.DataFrame(raw)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.set_index("date").sort_index()
            df = df.rename(columns={"open": "Open", "high": "High",
                                    "low": "Low", "close": "Close", "volume": "Volume"})
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            return df[cols].dropna(subset=["Close"])

        # ── 普通股票/ETF ──
        resp = _do_request(
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
            {"startDate": start, "endDate": end_str, "resampleFreq": "daily"},
        )
        if resp is None:
            return pd.DataFrame()

        data = resp.json()
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.set_index("date").sort_index()

        # 优先使用复权价（adjClose 等），与 yfinance auto_adjust=True 一致
        col_map = {
            "adjOpen":   "Open",
            "adjHigh":   "High",
            "adjLow":    "Low",
            "adjClose":  "Close",
            "adjVolume": "Volume",
        }
        df = df.rename(columns=col_map)
        # 若 Tiingo 没有 adjVolume，用原始 volume
        if "Volume" not in df.columns and "volume" in df.columns:
            df = df.rename(columns={"volume": "Volume"})

        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].dropna(subset=["Close"])

    except Exception as e:
        logger.warning(f"[tiingo] {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# yfinance 下载（fallback）
# ─────────────────────────────────────────────

def _yf_ohlcv(ticker: str, start: str, end_str: str) -> pd.DataFrame:
    """yfinance 下载，带全局限速 semaphore + 指数退避重试"""
    global _yf_last_time
    MAX_RETRIES = 4
    RETRY_BASE  = 2.0

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            with _yf_sem:
                with _yf_time_lock:
                    gap = time.time() - _yf_last_time
                    if gap < _YF_MIN_INTERVAL:
                        time.sleep(_YF_MIN_INTERVAL - gap + random.uniform(0, 0.3))
                    _yf_last_time = time.time()
                raw = yf.download(ticker, start=start, end=end_str,
                                  auto_adjust=True, progress=False)

            if raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            return df.dropna(subset=["Close"])

        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("rate limit", "too many requests", "401", "invalid crumb")):
                wait = RETRY_BASE ** attempt + random.uniform(0.5, 2.0)
                logger.warning(f"[yf rate limit] {ticker} attempt {attempt+1}/{MAX_RETRIES}, retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                break

    logger.error(f"[yf error] {ticker}: {last_exc}")
    return pd.DataFrame()


# ─────────────────────────────────────────────
# 统一路由
# ─────────────────────────────────────────────

def _download_ohlcv(ticker: str, start: str, end_str: str) -> pd.DataFrame:
    """
    主路径 yfinance（稳定、无配额限制）
    Tiingo 仅作备用（有 key 且 yfinance 失败时才用）
    """
    df = _yf_ohlcv(ticker, start, end_str)
    if not df.empty:
        return df

    # yfinance 失败时才尝试 Tiingo
    if _TIINGO_KEY and ticker not in _TIINGO_SKIP:
        logger.debug(f"[yf miss] {ticker}, trying tiingo")
        df = _tiingo_ohlcv(ticker, start, end_str)

    return df


# ─────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────

def get_ohlcv(
    ticker: str,
    start: str = DATA_START,
    end: Optional[str] = DATA_END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    增量缓存：首次全量下载，之后只追加新交易日数据。
    cache key 不含 end_str，同一 pkl 文件每天追加一行。
    """
    end_str = end or datetime.today().strftime("%Y-%m-%d")
    cache_key = f"{ticker}_{start}_OHLCV"          # 不含 end_str
    cpath     = _cache_path(cache_key)

    existing   = None
    fetch_from = start                              # 默认全量

    if not force_refresh and cpath.exists():
        try:
            existing = _load(cpath)
            if isinstance(existing, pd.DataFrame) and isinstance(existing.columns, pd.MultiIndex):
                existing.columns = existing.columns.get_level_values(0)
            if isinstance(existing, pd.DataFrame) and not existing.empty:
                last_date = existing.index[-1].date()
                if last_date >= _last_trading_date():
                    return existing                 # 已是最新，直接返回
                # 只拉增量（上次缓存最后一天之后）
                fetch_from = (existing.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
                logger.debug(f"[incremental] {ticker} {fetch_from} → {end_str}")
            else:
                existing = None
        except Exception as e:
            logger.warning(f"[cache load err] {ticker}: {e}")
            existing = None

    # pickle miss → 尝试本地预存 JSON baseline（git-committed，冷启动免全量下载）
    if existing is None:
        baseline = _load_baseline(ticker, start)
        if not baseline.empty:
            last_date = baseline.index[-1].date()
            if last_date >= _last_trading_date():
                # baseline 已是最新，直接提升为 pickle 缓存并返回
                _save(cpath, baseline)
                logger.debug(f"[baseline] {ticker} already fresh, promoted to cache")
                return baseline
            fetch_from = (baseline.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
            existing = baseline
            logger.debug(f"[baseline] {ticker} loaded from local JSON, delta from {fetch_from}")

    logger.info(f"[download OHLCV] {ticker} from {fetch_from}")
    new_df = _download_ohlcv(ticker, fetch_from, end_str)

    if new_df.empty:
        return existing if existing is not None else pd.DataFrame()

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df

    _save(cpath, combined)
    return combined


def get_prices(
    ticker: str,
    start: str = DATA_START,
    end: Optional[str] = DATA_END,
    field: str = "Close",
    force_refresh: bool = False,
) -> pd.Series:
    """从 get_ohlcv 提取单列，复用同一增量缓存。"""
    df = get_ohlcv(ticker, start=start, end=end, force_refresh=force_refresh)
    if df.empty or field not in df.columns:
        logger.warning(f"[empty] {ticker} field={field}")
        return pd.Series(name=ticker, dtype=float)
    return df[field].rename(ticker).dropna()


def get_multi(
    tickers: list,
    start: str = DATA_START,
    end: Optional[str] = DATA_END,
    field: str = "Close",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    批量下载多个 ticker 的价格，返回宽表。
    复用 get_prices() 的单股缓存（Tiingo/yfinance 路由在内部处理）。
    """
    end_str = end or datetime.today().strftime("%Y-%m-%d")
    cache_key = f"multi_{'_'.join(sorted(tickers))}_{start}_{end_str}_{field}"
    cpath = _cache_path(cache_key)

    if not force_refresh and _is_fresh(cpath):
        logger.debug(f"[cache hit] multi {len(tickers)} tickers")
        return _load(cpath)

    logger.info(f"[download multi] {len(tickers)} tickers")
    results = {}
    for ticker in tickers:
        try:
            s = get_prices(ticker, start=start, end=end_str, field=field, force_refresh=force_refresh)
            if not s.empty:
                results[ticker] = s
        except Exception as e:
            logger.error(f"[multi error] {ticker}: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.index = pd.to_datetime(df.index)
    _save(cpath, df)
    return df


def get_returns(
    ticker_or_df: Union[str, pd.DataFrame, pd.Series],
    start: str = DATA_START,
    end: Optional[str] = DATA_END,
    log_return: bool = False,
) -> pd.Series:
    """计算日收益率"""
    if isinstance(ticker_or_df, str):
        prices = get_prices(ticker_or_df, start=start, end=end)
    else:
        prices = ticker_or_df

    if log_return:
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def clear_cache(ticker: Optional[str] = None) -> None:
    """清除缓存，ticker=None 时清除全部"""
    if ticker is None:
        for f in CACHE_DIR.glob("*.pkl"):
            f.unlink()
        logger.info("[cache] 全部清除")
    else:
        removed = 0
        for f in CACHE_DIR.glob("*.pkl"):
            try:
                obj = _load(f)
                name = getattr(obj, "name", "") or ""
                if ticker in str(name):
                    f.unlink()
                    removed += 1
            except Exception:
                pass
        logger.info(f"[cache] 清除 {ticker} 相关缓存 {removed} 个")


# ─────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    src = "Tiingo" if _TIINGO_KEY else "yfinance"
    print(f"=== 数据源: {src} ===")

    spy = get_prices("SPY", start="2022-01-01")
    print(f"SPY: {len(spy)} 条, 最新价 {spy.iloc[-1]:.2f}")

    ohlcv = get_ohlcv("QQQ", start="2023-01-01")
    print(ohlcv.tail(3))

    multi = get_multi(["SPY", "QQQ", "TLT"], start="2023-01-01")
    print(multi.tail(3))

    ret = get_returns("SPY", start="2023-01-01")
    print(f"年化波动率: {ret.std() * (252**0.5) * 100:.1f}%")
    print("✅ downloader 测试通过")
