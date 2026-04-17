"""
app.py — 量化交易仪表盘 Web 服务
Railway 部署：uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import os
import time
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

import sys
sys.path.insert(0, str(Path(__file__).parent))

from scanner import scan_all, get_macro, get_flows, get_cta_dashboard, get_sector_full

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="量化交易仪表盘")

_HTML_PATH = Path(__file__).parent / "templates" / "index.html"

# ─── 内存缓存 ───
_cache: dict = {}
CACHE_TTL_SCAN  = 4 * 3600   # 扫描结果 4 小时
CACHE_TTL_MACRO = 30 * 60    # 宏观数据 30 分钟
CACHE_TTL_FLOWS = 2 * 3600   # 资金流向 2 小时
CACHE_TTL_CTA   = 2 * 3600   # CTA 2 小时
CACHE_TTL_SECTORS = 2 * 3600   # 板块全景 2 小时


def _get_cached(key: str, fn, ttl: int):
    now = time.time()
    entry = _cache.get(key)
    if entry and (now - entry["ts"]) < ttl:
        return entry["data"], entry["ts"]
    data = fn()
    _cache[key] = {"data": data, "ts": now}
    return data, _cache[key]["ts"]


def _fmt_age(ts: float | None) -> str:
    if ts is None:
        return "未加载"
    age = time.time() - ts
    if age < 60:
        return f"{int(age)}秒前"
    if age < 3600:
        return f"{int(age // 60)}分钟前"
    return f"{int(age // 3600)}小时前"


# ─── 路由 ───

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/macro")
async def api_macro():
    data, ts = _get_cached("macro", get_macro, CACHE_TTL_MACRO)
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/scan")
async def api_scan():
    data, ts = _get_cached("scan", scan_all, CACHE_TTL_SCAN)
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/flows")
async def api_flows():
    data, ts = _get_cached("flows", get_flows, CACHE_TTL_FLOWS)
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/cta")
async def api_cta():
    data, ts = _get_cached("cta", get_cta_dashboard, CACHE_TTL_CTA)
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/sectors")
async def api_sectors():
    data, ts = _get_cached("sectors", get_sector_full, CACHE_TTL_SECTORS)
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.post("/api/refresh")
async def api_refresh():
    _cache.clear()
    return JSONResponse({"status": "ok", "message": "缓存已清除"})


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}
