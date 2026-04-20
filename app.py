"""
app.py — 量化交易仪表盘 Web 服务
Railway 部署：uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import os
import time
import json
import hashlib
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import sys
sys.path.insert(0, str(Path(__file__).parent))

from scanner import scan_all, get_macro, get_flows, get_cta_dashboard, get_sector_full, get_bt_signals, WATCHLIST

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _bg_generate_charts():
    """
    启动后在后台线程里生成所有 K 线图（不阻塞服务器启动）。
    每个交易日只生成一次：检查图片修改日期，当天已有则跳过。
    """
    try:
        import warnings
        warnings.filterwarnings("ignore")
        from generate_charts import make_chart
        import pandas as pd
        from pathlib import Path as P
        from datetime import date

        csv = P(__file__).parent / "output" / "top3_strategies.csv"
        if not csv.exists():
            return

        tickers = sorted(pd.read_csv(csv)["ticker"].unique().tolist())
        charts_dir = P(__file__).parent / "output" / "charts"
        today = date.today()

        # 检查是否已经是今天生成的（取第一只票的图片日期）
        sample_png = charts_dir / f"{tickers[0]}.png"
        if sample_png.exists():
            mtime = date.fromtimestamp(sample_png.stat().st_mtime)
            if mtime >= today:
                logger.warning(f"[charts] 今日已生成，跳过 ({mtime})")
                return

        logger.warning(f"[charts] 开始后台生成 {len(tickers)} 张 K 线图...")
        ok = 0
        for t in tickers:
            try:
                make_chart(t, force=True)
                ok += 1
            except Exception as e:
                logger.warning(f"[charts] {t} 失败: {e}")
        logger.warning(f"[charts] 完成 {ok}/{len(tickers)}")
    except Exception as e:
        logger.warning(f"[charts] 后台生成异常: {e}")


@asynccontextmanager
async def lifespan(app):
    # 启动时在后台线程生成 K 线图（不阻塞请求）
    t = threading.Thread(target=_bg_generate_charts, daemon=True)
    t.start()
    yield


app = FastAPI(title="量化交易仪表盘", lifespan=lifespan)

_HTML_PATH   = Path(__file__).parent / "templates" / "index.html"
_CACHE_DIR   = Path(__file__).parent / "cache"
_CHARTS_DIR  = Path(__file__).parent / "output" / "charts"
_CACHE_DIR.mkdir(exist_ok=True)
_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# 挂载 K线图静态文件目录
app.mount("/charts", StaticFiles(directory=str(_CHARTS_DIR)), name="charts")

# ─── 代码逻辑版本号（信号逻辑变更时手动递增，自动使旧缓存失效）───
_CODE_VER = "v7"   # bump this whenever signal logic in scanner.py changes

# ─── Watchlist 版本哈希（watchlist 变动时自动使旧缓存失效）───
_wl_hash = hashlib.md5(
    (json.dumps({g: sorted(t) for g, t in WATCHLIST.items()}, sort_keys=True) + _CODE_VER).encode()
).hexdigest()[:8]

# ─── 内存缓存（进程内快速读取）───
_mem_cache: dict = {}

# ─── TTL 配置（秒）───
# 收盘价数据按交易日缓存；盘中实时数据用较短 TTL
CACHE_TTL = {
    "scan":    24 * 3600,   # 扫描结果：1天（收盘价固定后不变）
    "sectors": 24 * 3600,
    "cta":      2 * 3600,
    "flows":    2 * 3600,
    "macro":       30 * 60,  # 宏观/VIX：30分钟（盘中随时变）
    "bt":       48 * 3600,   # 回测：2天
}


def _trade_date_key() -> str:
    """
    返回当前最新的交易日期字符串（用于磁盘缓存文件名）。
    美股收盘 16:00 ET = 20:00 UTC。收盘后数据稳定，以当天为 key。
    收盘前（UTC 20:00 前）用前一个自然日，避免盘中缓存脏数据。
    周末直接用周五。
    """
    now_utc = datetime.now(timezone.utc)
    # 如果还没到 20:00 UTC（美股收盘后约 0 分钟）则用昨天
    if now_utc.hour < 20:
        now_utc = now_utc - timedelta(days=1)
    d = now_utc.date()
    # 周六→周五，周日→周五
    if d.weekday() == 5:
        d -= timedelta(days=1)
    elif d.weekday() == 6:
        d -= timedelta(days=2)
    return d.strftime("%Y-%m-%d")


def _disk_path(key: str, date_key: str) -> Path:
    # 所有缓存都带版本号，代码逻辑变更时自动失效
    # scan/sectors 额外带 watchlist hash（watchlist 变动时也失效）
    if key in ("scan", "sectors"):
        return _CACHE_DIR / f"{key}_{date_key}_{_wl_hash}.json"
    return _CACHE_DIR / f"{key}_{date_key}_{_CODE_VER}.json"


def _load_disk(key: str, date_key: str):
    p = _disk_path(key, date_key)
    if p.exists():
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            return payload["data"], payload["ts"]
        except Exception:
            pass
    return None, None


def _save_disk(key: str, date_key: str, data, ts: float):
    p = _disk_path(key, date_key)
    try:
        p.write_text(json.dumps({"data": data, "ts": ts}, ensure_ascii=False, default=str),
                     encoding="utf-8")
    except Exception as e:
        logger.warning(f"disk cache write failed {key}: {e}")


def _get_cached(key: str, fn, ttl: int):
    now = time.time()
    date_key = _trade_date_key()

    # 1. 内存命中
    mem = _mem_cache.get(key)
    if mem and mem.get("date_key") == date_key and (now - mem["ts"]) < ttl:
        return mem["data"], mem["ts"]

    # 2. 磁盘命中（跨重启有效）
    data, ts = _load_disk(key, date_key)
    if data is not None and ts is not None and (now - ts) < ttl:
        _mem_cache[key] = {"data": data, "ts": ts, "date_key": date_key}
        return data, ts

    # 3. 重新计算
    logger.info(f"[cache] computing {key} (date={date_key})")
    data = fn()
    ts   = now
    _mem_cache[key] = {"data": data, "ts": ts, "date_key": date_key}
    _save_disk(key, date_key, data, ts)
    return data, ts


def _fmt_age(ts: float | None) -> str:
    if ts is None:
        return "未加载"
    age = time.time() - ts
    if age < 60:
        return f"{int(age)}秒前"
    if age < 3600:
        return f"{int(age // 60)}分钟前"
    if age < 86400:
        return f"{int(age // 3600)}小时前"
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


# ─── 路由 ───

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/macro")
async def api_macro():
    data, ts = _get_cached("macro", get_macro, CACHE_TTL["macro"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/scan")
async def api_scan():
    data, ts = _get_cached("scan", scan_all, CACHE_TTL["scan"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/flows")
async def api_flows():
    data, ts = _get_cached("flows", get_flows, CACHE_TTL["flows"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/cta")
async def api_cta():
    data, ts = _get_cached("cta", get_cta_dashboard, CACHE_TTL["cta"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/sectors")
async def api_sectors():
    data, ts = _get_cached("sectors", get_sector_full, CACHE_TTL["sectors"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/backtest/{ticker}")
async def api_backtest(ticker: str):
    key = f"bt_{ticker.upper()}"
    data, ts = _get_cached(key, lambda: get_bt_signals(ticker), CACHE_TTL["bt"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.post("/api/refresh")
async def api_refresh():
    """清除内存缓存，下次请求重新计算并写盘"""
    _mem_cache.clear()
    # 删除所有 JSON 磁盘缓存（强制重拉最新数据）
    deleted = 0
    for p in _CACHE_DIR.glob("*.json"):
        try:
            p.unlink()
            deleted += 1
        except Exception:
            pass
    return JSONResponse({"status": "ok", "message": f"缓存已清除（删除 {deleted} 个文件）"})


@app.get("/health")
async def health():
    date_key = _trade_date_key()
    cached = [p.stem for p in _CACHE_DIR.glob(f"*_{date_key}.json")]
    return {"status": "ok", "trade_date": date_key, "cached_keys": cached,
            "time": datetime.now().isoformat()}
