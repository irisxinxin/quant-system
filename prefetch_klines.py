"""
prefetch_klines.py — 预拉取所有 ticker 的历史 K 线，存为 output/klines/{TICKER}.json
运行一次，commit 这批 JSON 文件，Railway 冷启动就不用全量下载了。

用法:
    python3 prefetch_klines.py               # 拉所有 ticker（跳过已有且新鲜的）
    python3 prefetch_klines.py --force       # 强制重拉所有
    python3 prefetch_klines.py SPY QQQ SMH  # 只拉指定 ticker
"""
import sys
import json
import time
import logging
import argparse
from datetime import datetime, date
from pathlib import Path

import pandas as pd

# 确保能 import 项目模块
sys.path.insert(0, str(Path(__file__).parent))
from config import BASE_DIR, SECTOR_ETFS, CTA_UNIVERSE
from data.downloader import _yf_ohlcv, _last_trading_date

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
KLINES_DIR  = BASE_DIR / "output" / "klines"
FETCH_START = "2023-01-01"      # 覆盖 app.py 里最早的 start="2023-01-01"

# 额外系统 ticker（板块 ETF + 基准 + CTA）
_EXTRA_TICKERS = {
    "TLT",          # 债券基准
    "USO",          # 原油 CTA
    "UUP",          # 美元 CTA
    "^VIX",         # VIX
}
_EXTRA_TICKERS |= set(SECTOR_ETFS.values())          # SMH/IGV/AIQ/...
_EXTRA_TICKERS |= set(CTA_UNIVERSE.values())          # SPY/QQQ/TLT/GLD/...


def _ticker_to_filename(ticker: str) -> str:
    """^VIX → ^VIX.json，其余直接大写"""
    return f"{ticker.upper()}.json"


def _load_existing(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        records = json.loads(p.read_text(encoding="utf-8"))
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
    except Exception as e:
        logger.warning(f"[load] {p.name}: {e}")
        return pd.DataFrame()


def _save(p: Path, df: pd.DataFrame) -> None:
    out = df.reset_index()
    # 统一索引列名为 date
    for col in ["Date", "index"]:
        if col in out.columns:
            out = out.rename(columns={col: "date"})
            break
    out["date"] = out["date"].astype(str).str[:10]
    cols = [c for c in ["date", "Open", "High", "Low", "Close", "Volume"] if c in out.columns]
    p.write_text(json.dumps(out[cols].to_dict(orient="records"), indent=2), encoding="utf-8")


def _get_all_tickers() -> list[str]:
    """从 strategy_optimization.csv + 额外系统 ticker 汇总全部 ticker"""
    csv_path = BASE_DIR / "output" / "strategy_optimization.csv"
    tickers = set()
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        tickers |= set(df["ticker"].dropna().str.strip().tolist())
    tickers |= _EXTRA_TICKERS
    return sorted(tickers)


def prefetch(tickers: list[str], force: bool = False) -> None:
    KLINES_DIR.mkdir(parents=True, exist_ok=True)
    today = _last_trading_date()
    end_str = datetime.today().strftime("%Y-%m-%d")

    total = len(tickers)
    ok = skip = fail = 0

    for i, ticker in enumerate(tickers, 1):
        fname = _ticker_to_filename(ticker)
        p = KLINES_DIR / fname
        prefix = f"[{i}/{total}] {ticker}"

        # 检查已有文件是否够新
        if not force and p.exists():
            existing = _load_existing(p)
            if not existing.empty and existing.index[-1].date() >= today:
                logger.info(f"{prefix} — 已是最新，跳过")
                skip += 1
                continue

        # 下载
        logger.info(f"{prefix} — 下载 {FETCH_START} → {end_str}")
        try:
            df = _yf_ohlcv(ticker, FETCH_START, end_str)
        except Exception as e:
            logger.error(f"{prefix} — 下载失败: {e}")
            fail += 1
            continue

        if df.empty:
            logger.warning(f"{prefix} — 空数据，跳过")
            fail += 1
            continue

        # 如果已有旧文件，合并（不丢失已有数据）
        if not force and p.exists():
            existing = _load_existing(p)
            if not existing.empty:
                df = pd.concat([existing, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()

        _save(p, df)
        logger.info(f"{prefix} — 保存 {len(df)} 行 → {p.name}")
        ok += 1
        time.sleep(0.3)   # 礼貌性限速

    print(f"\n完成: 成功={ok}  跳过={skip}  失败={fail}  共={total}")
    if fail:
        print("失败的 ticker 可能已退市或不在 yfinance 支持列表")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="*", help="指定 ticker（不填则拉所有）")
    parser.add_argument("--force", action="store_true", help="强制重拉（忽略已有文件）")
    args = parser.parse_args()

    target = [t.upper() for t in args.tickers] if args.tickers else _get_all_tickers()
    print(f"准备拉取 {len(target)} 个 ticker，start={FETCH_START}")
    print("ticker 列表:", ", ".join(target[:10]), "..." if len(target) > 10 else "")
    prefetch(target, force=args.force)
