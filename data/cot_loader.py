"""
data/cot_loader.py — CFTC COT 数据下载与解析
每周五下载最新 TFF 报告，提取 Leveraged Funds 净头寸
"""
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR, CTA_COT_PCT_HI, CTA_COT_PCT_LO

logger = logging.getLogger(__name__)

COT_CACHE = CACHE_DIR / "cot_tff.pkl"
COT_EXPIRE_H = 24 * 5   # 5天内不重新下载

# CFTC TFF 报告中的关键合约名称（部分匹配）
COT_CONTRACTS = {
    "SPX":  "S&P 500 STOCK INDEX",
    "NDX":  "NASDAQ-100 STOCK INDEX",
    "DOW":  "DOW JONES U.S. REAL ESTATE",
    "ZB":   "U.S. TREASURY BONDS",
    "ZN":   "10-YEAR U.S. TREASURY NOTES",
    "GC":   "GOLD",
    "CL":   "CRUDE OIL, LIGHT SWEET",
    "DX":   "U.S. DOLLAR INDEX",
}


def _cache_fresh() -> bool:
    if not COT_CACHE.exists():
        return False
    age = time.time() - COT_CACHE.stat().st_mtime
    return age < COT_EXPIRE_H * 3600


def load_cot_data(years_back: int = 3, force_refresh: bool = False) -> pd.DataFrame:
    """
    下载并解析 CFTC TFF（Traders in Financial Futures）报告
    提取 Leveraged Funds 净头寸，最接近 CTA 仓位

    Returns:
        DataFrame，columns = [date, contract, lev_long, lev_short, lev_net,
                               oi_long_pct, oi_short_pct]
    """
    if not force_refresh and _cache_fresh():
        logger.debug("[COT] 使用缓存数据")
        with open(COT_CACHE, "rb") as f:
            return pickle.load(f)

    try:
        import cot_reports as cot
    except ImportError:
        logger.error("[COT] 请先安装: pip install cot_reports")
        return pd.DataFrame()

    current_year = datetime.now().year
    frames = []

    for yr in range(current_year - years_back, current_year + 1):
        logger.info(f"[COT] 下载 {yr} 年 TFF 数据...")
        try:
            df = cot.cot_year(year=yr, cot_report_type="traders_in_financial_futures_fut")
            frames.append(df)
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"[COT] {yr} 年下载失败: {e}")

    if not frames:
        logger.error("[COT] 所有年份下载失败")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # 解析日期
    date_col = "As of Date in Form YYYY-MM-DD"
    if date_col not in raw.columns:
        date_col = [c for c in raw.columns if "date" in c.lower()][0]
    raw["date"] = pd.to_datetime(raw[date_col])

    # 提取 Leveraged Funds 相关列（列名可能因版本略有差异）
    lev_long_col  = _find_col(raw, ["Lev Money Positions-Long All",  "Lev Money Positions-Long"])
    lev_short_col = _find_col(raw, ["Lev Money Positions-Short All", "Lev Money Positions-Short"])
    lev_spread_col = _find_col(raw, ["Lev Money Positions-Spreading All"], required=False)

    name_col = "Market and Exchange Names"

    results = []
    for contract_id, keyword in COT_CONTRACTS.items():
        mask = raw[name_col].str.contains(keyword, case=False, na=False)
        sub = raw[mask].copy()
        if sub.empty:
            logger.warning(f"[COT] 未找到合约: {keyword}")
            continue

        sub = sub.sort_values("date").reset_index(drop=True)
        sub["lev_long"]  = pd.to_numeric(sub[lev_long_col],  errors="coerce")
        sub["lev_short"] = pd.to_numeric(sub[lev_short_col], errors="coerce")
        sub["lev_net"]   = sub["lev_long"] - sub["lev_short"]
        sub["contract"]  = contract_id

        results.append(sub[["date", "contract", "lev_long", "lev_short", "lev_net"]])

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)

    with open(COT_CACHE, "wb") as f:
        pickle.dump(out, f)
    logger.info(f"[COT] 解析完成，{len(out)} 条记录，合约: {out['contract'].unique().tolist()}")
    return out


def _find_col(df: pd.DataFrame, candidates: list, required: bool = True) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # 模糊匹配
    for c in candidates:
        matches = [col for col in df.columns if c.lower().split("-")[0] in col.lower()]
        if matches:
            return matches[0]
    if required:
        raise KeyError(f"未找到列: {candidates}\n可用列: {list(df.columns)[:20]}")
    return ""


def get_cot_signals(lookback_weeks: int = 52) -> pd.DataFrame:
    """
    计算每个合约的 COT 信号指标

    Returns:
        DataFrame，每行为一个合约的最新信号
        columns = [contract, lev_net, percentile, zscore, direction, warning]
    """
    raw = load_cot_data()
    if raw.empty:
        return pd.DataFrame()

    results = []
    for contract, grp in raw.groupby("contract"):
        grp = grp.sort_values("date").tail(lookback_weeks * 2)

        lev_net = grp["lev_net"].dropna()
        if len(lev_net) < 10:
            continue

        latest   = lev_net.iloc[-1]
        rolling  = lev_net.tail(lookback_weeks)

        pct  = (rolling < latest).mean()           # 百分位 0~1
        mean = rolling.mean()
        std  = rolling.std()
        z    = (latest - mean) / std if std > 0 else 0.0
        prev = lev_net.iloc[-2] if len(lev_net) > 1 else latest
        delta = latest - prev

        # 方向判断
        if pct > CTA_COT_PCT_HI:
            direction = "极度多头🔴"
            warning   = f"仓位在历史 {pct:.0%} 百分位，追多风险大"
        elif pct > 0.60:
            direction = "偏多⬆"
            warning   = ""
        elif pct < CTA_COT_PCT_LO:
            direction = "极度空头🟢"
            warning   = f"仓位在历史 {pct:.0%} 百分位，追空风险大"
        elif pct < 0.40:
            direction = "偏空⬇"
            warning   = ""
        else:
            direction = "中性➡"
            warning   = ""

        results.append({
            "合约":      contract,
            "净多头":    int(latest),
            "百分位":    round(pct, 2),
            "Z-score":   round(z, 2),
            "周度变化":  int(delta),
            "方向":      direction,
            "警告":      warning,
            "更新日期":  grp["date"].iloc[-1].strftime("%Y-%m-%d"),
        })

    df_out = pd.DataFrame(results).set_index("合约")
    return df_out


# ─────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=== 下载 COT 数据 ===")
    raw = load_cot_data(years_back=2)
    if not raw.empty:
        print(f"总记录数: {len(raw)}")
        print(raw.tail(5))

        print("\n=== COT 信号 ===")
        signals = get_cot_signals()
        print(signals.to_string())
        print("✅ cot_loader 测试通过")
    else:
        print("⚠️  COT 数据为空，请检查网络或安装 cot_reports: pip install cot_reports")
