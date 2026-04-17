"""
data/universe.py — 股票池管理
提供 S&P 500 成分股列表、板块 ETF、自定义股票池
"""
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR, SECTOR_ETFS, CTA_UNIVERSE, BENCHMARK

logger = logging.getLogger(__name__)

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CACHE = CACHE_DIR / "sp500_tickers.csv"


def get_sp500_tickers(force_refresh: bool = False) -> list:
    """
    从 Wikipedia 获取最新 S&P 500 成分股列表

    Returns:
        list of ticker strings, e.g. ['AAPL', 'MSFT', ...]
    """
    if not force_refresh and SP500_CACHE.exists():
        df = pd.read_csv(SP500_CACHE)
        tickers = df["Symbol"].tolist()
        logger.info(f"[universe] S&P 500 from cache: {len(tickers)} tickers")
        return tickers

    logger.info("[universe] 下载 S&P 500 成分股...")
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        # 修正 ticker（BRK.B → BRK-B，yfinance 格式）
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        df.to_csv(SP500_CACHE, index=False)
        tickers = df["Symbol"].tolist()
        logger.info(f"[universe] 获取 {len(tickers)} 只 S&P 500 成分股")
        return tickers
    except Exception as e:
        logger.error(f"[universe] 下载失败: {e}")
        # 返回兜底的核心股票
        return _fallback_universe()


def get_sp500_with_sectors() -> pd.DataFrame:
    """
    返回 S&P 500 成分股 + GICS 板块信息

    Returns:
        DataFrame，columns = [Symbol, Security, GICS Sector, GICS Sub-Industry, ...]
    """
    if SP500_CACHE.exists():
        df = pd.read_csv(SP500_CACHE)
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return df

    # 重新下载
    get_sp500_tickers(force_refresh=True)
    return pd.read_csv(SP500_CACHE)


def get_sector_etfs() -> dict:
    """返回板块 ETF 映射字典 {板块名: ticker}"""
    return {k: v for k, v in SECTOR_ETFS.items()}


def get_sector_tickers(sector_name: str) -> list:
    """
    获取某个 GICS 板块在 S&P 500 内的所有成分股

    Args:
        sector_name: GICS 大类名称，如 'Information Technology', 'Energy'

    Returns:
        该板块内的 ticker list
    """
    df = get_sp500_with_sectors()
    col = "GICS Sector"
    if col not in df.columns:
        logger.warning(f"列 '{col}' 不存在，返回空列表")
        return []
    mask = df[col].str.contains(sector_name, case=False, na=False)
    return df.loc[mask, "Symbol"].tolist()


def get_cta_universe() -> dict:
    """返回 CTA 监控资产映射 {名称: ticker}"""
    return dict(CTA_UNIVERSE)


def get_benchmark() -> str:
    return BENCHMARK


def get_all_etfs() -> list:
    """返回所有板块 ETF 的 ticker 列表"""
    return list(SECTOR_ETFS.values())


def _fallback_universe() -> list:
    """Wikipedia 下载失败时的兜底股票池（核心大盘股）"""
    return [
        # 科技
        "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AMZN",
        "TSLA", "AVGO", "AMD", "INTC", "QCOM", "MU", "AMAT",
        # 金融
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "V", "MA",
        # 医疗
        "UNH", "JNJ", "PFE", "ABBV", "LLY", "MRK", "TMO",
        # 消费
        "AMZN", "HD", "MCD", "SBUX", "NKE", "TGT", "WMT", "COST",
        # 能源
        "XOM", "CVX", "COP", "SLB", "EOG", "PXD",
        # 工业
        "CAT", "HON", "UPS", "FDX", "BA", "GE", "LMT",
        # ETFs
        "SPY", "QQQ", "IWM", "TLT", "GLD", "USO",
    ]


# ─────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tickers = get_sp500_tickers()
    print(f"S&P 500 成分股数量: {len(tickers)}")
    print(f"前10只: {tickers[:10]}")

    df = get_sp500_with_sectors()
    print(f"\n板块分布:")
    print(df["GICS Sector"].value_counts())

    tech = get_sector_tickers("Information Technology")
    print(f"\n科技板块成分股 ({len(tech)}): {tech[:10]}")

    print(f"\n板块 ETF: {get_sector_etfs()}")
    print("✅ universe 测试通过")
