"""
data/downloader.py — 统一数据下载 + 本地缓存
所有模块调用 get_prices() / get_multi() 获取数据，不直接用 yfinance
"""
import os
import time
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import random
import pandas as pd
import yfinance as yf

# 下载重试参数
_MAX_RETRIES = 4
_RETRY_BASE  = 2.0   # 指数退避底数（秒）

# 引入全局配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR, DATA_START, DATA_END, CACHE_EXPIRE_H

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{h}.pkl"


def _is_fresh(path: Path, hours: float = CACHE_EXPIRE_H) -> bool:
    """缓存文件是否在有效期内"""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < hours * 3600


def _save(path: Path, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# 核心接口
# ─────────────────────────────────────────────

def get_prices(
    ticker: str,
    start: str = DATA_START,
    end: Optional[str] = DATA_END,
    field: str = "Close",
    force_refresh: bool = False,
) -> pd.Series:
    """
    下载单个 ticker 的日线价格，带本地缓存。

    Args:
        ticker:        股票/ETF 代码，如 'SPY'
        start:         起始日期，'YYYY-MM-DD'
        end:           结束日期，None = 今天
        field:         价格字段，'Close' / 'Open' / 'High' / 'Low' / 'Volume'
        force_refresh: 强制重新下载

    Returns:
        pd.Series，index 为日期，name 为 ticker
    """
    end_str = end or datetime.today().strftime("%Y-%m-%d")
    cache_key = f"{ticker}_{start}_{end_str}_{field}"
    cpath = _cache_path(cache_key)

    if not force_refresh and _is_fresh(cpath):
        logger.debug(f"[cache hit] {ticker} {field}")
        return _load(cpath)

    logger.info(f"[download] {ticker} {start} → {end_str}")
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            raw = yf.download(ticker, start=start, end=end_str,
                              auto_adjust=True, progress=False)
            if raw.empty:
                logger.warning(f"[empty] {ticker} returned no data")
                return pd.Series(name=ticker, dtype=float)

            # yfinance 有时返回 MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            series = raw[field].rename(ticker)
            series.index = pd.to_datetime(series.index)
            series = series.dropna()

            _save(cpath, series)
            return series

        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("rate limit", "too many requests", "401", "invalid crumb")):
                wait = _RETRY_BASE ** attempt + random.uniform(0.5, 2.0)
                logger.warning(f"[rate limit] {ticker} attempt {attempt+1}/{_MAX_RETRIES}, retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                break   # 非限速错误，不重试

    logger.error(f"[error] {ticker}: {last_exc}")
    return pd.Series(name=ticker, dtype=float)


def get_ohlcv(
    ticker: str,
    start: str = DATA_START,
    end: Optional[str] = DATA_END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    下载完整 OHLCV 日线数据（用于裸K、SMC等需要高低开收的模块）

    Returns:
        pd.DataFrame，columns = [Open, High, Low, Close, Volume]
    """
    end_str = end or datetime.today().strftime("%Y-%m-%d")
    cache_key = f"{ticker}_{start}_{end_str}_OHLCV"
    cpath = _cache_path(cache_key)

    if not force_refresh and _is_fresh(cpath):
        cached = _load(cpath)
        # 旧缓存可能保存了 MultiIndex 列，展平后返回
        if isinstance(cached, pd.DataFrame) and isinstance(cached.columns, pd.MultiIndex):
            cached.columns = cached.columns.get_level_values(0)
        return cached

    logger.info(f"[download OHLCV] {ticker}")
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            raw = yf.download(ticker, start=start, end=end_str,
                              auto_adjust=True, progress=False)
            if raw.empty:
                return pd.DataFrame()

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df = df.dropna()

            _save(cpath, df)
            return df

        except Exception as e:
            last_exc = e
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("rate limit", "too many requests", "401", "invalid crumb")):
                wait = _RETRY_BASE ** attempt + random.uniform(0.5, 2.0)
                logger.warning(f"[rate limit] {ticker} OHLCV attempt {attempt+1}/{_MAX_RETRIES}, retry in {wait:.1f}s")
                time.sleep(wait)
            else:
                break

    logger.error(f"[error] {ticker}: {last_exc}")
    return pd.DataFrame()


def get_multi(
    tickers: list,
    start: str = DATA_START,
    end: Optional[str] = DATA_END,
    field: str = "Close",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    批量下载多个 ticker 的价格，返回宽表。

    Returns:
        pd.DataFrame，index 为日期，columns 为 ticker
    """
    end_str = end or datetime.today().strftime("%Y-%m-%d")
    cache_key = f"multi_{'_'.join(sorted(tickers))}_{start}_{end_str}_{field}"
    cpath = _cache_path(cache_key)

    if not force_refresh and _is_fresh(cpath):
        logger.debug(f"[cache hit] multi {len(tickers)} tickers")
        return _load(cpath)

    logger.info(f"[download multi] {len(tickers)} tickers")
    results = {}
    # 分批下载，每批 20 个，批间随机延时防止限速
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                raw = yf.download(batch, start=start, end=end_str,
                                  auto_adjust=True, progress=False)
                if raw.empty:
                    break
                if isinstance(raw.columns, pd.MultiIndex):
                    price_df = raw[field]
                else:
                    price_df = raw[[field]].rename(columns={field: batch[0]})

                for t in price_df.columns:
                    results[t] = price_df[t].dropna()
                break  # 成功，退出重试

            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                if any(kw in err_str for kw in ("rate limit", "too many requests", "401", "invalid crumb")):
                    wait = _RETRY_BASE ** attempt + random.uniform(1.0, 3.0)
                    logger.warning(f"[rate limit] multi batch {i//batch_size+1} attempt {attempt+1}/{_MAX_RETRIES}, retry in {wait:.1f}s")
                    time.sleep(wait)
                else:
                    logger.error(f"[error] batch {batch}: {e}")
                    break
        if last_exc and attempt == _MAX_RETRIES - 1:
            logger.error(f"[error] batch {batch} failed after {_MAX_RETRIES} retries: {last_exc}")
        # 批间随机延时 1~3s，避免触发频率限制
        time.sleep(random.uniform(1.0, 3.0))

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
    """
    计算日收益率

    Args:
        ticker_or_df: ticker 字符串，或已有价格 Series/DataFrame
        log_return:   True = 对数收益率，False = 简单收益率
    """
    if isinstance(ticker_or_df, str):
        prices = get_prices(ticker_or_df, start=start, end=end)
    else:
        prices = ticker_or_df

    if log_return:
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna()
    else:
        return prices.pct_change().dropna()


def clear_cache(ticker: Optional[str] = None) -> None:
    """清除缓存，ticker=None 时清除全部"""
    if ticker is None:
        for f in CACHE_DIR.glob("*.pkl"):
            f.unlink()
        logger.info("[cache] 全部清除")
    else:
        # 找到该 ticker 相关的缓存
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

    print("=== 测试 get_prices ===")
    spy = get_prices("SPY", start="2022-01-01")
    print(f"SPY: {len(spy)} 条, 最新价 {spy.iloc[-1]:.2f}")

    print("\n=== 测试 get_ohlcv ===")
    ohlcv = get_ohlcv("QQQ", start="2023-01-01")
    print(ohlcv.tail(3))

    print("\n=== 测试 get_multi ===")
    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    multi = get_multi(tickers, start="2023-01-01")
    print(multi.tail(3))

    print("\n=== 测试 get_returns ===")
    ret = get_returns("SPY", start="2023-01-01")
    print(f"年化波动率: {ret.std() * (252**0.5) * 100:.1f}%")
    print("✅ downloader 测试通过")
