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

import uuid
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

_HTML_PATH     = Path(__file__).parent / "templates" / "index.html"
_CACHE_DIR     = Path(__file__).parent / "cache"
_CHARTS_DIR    = Path(__file__).parent / "output" / "charts"
_KOL_NOTES_PATH = Path(__file__).parent / "output" / "kol_notes.json"
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
def api_macro():
    data, ts = _get_cached("macro", get_macro, CACHE_TTL["macro"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/scan")
def api_scan():
    data, ts = _get_cached("scan", scan_all, CACHE_TTL["scan"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/flows")
def api_flows():
    data, ts = _get_cached("flows", get_flows, CACHE_TTL["flows"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/cta")
def api_cta():
    data, ts = _get_cached("cta", get_cta_dashboard, CACHE_TTL["cta"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/sectors")
def api_sectors():
    data, ts = _get_cached("sectors", get_sector_full, CACHE_TTL["sectors"])
    return JSONResponse({"data": data, "cached_at": _fmt_age(ts)})


@app.get("/api/backtest/{ticker}")
def api_backtest(ticker: str):
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


# ─── 大V观点 CRUD ───

def _load_kol() -> list:
    if _KOL_NOTES_PATH.exists():
        try:
            return json.loads(_KOL_NOTES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_kol(notes: list) -> None:
    _KOL_NOTES_PATH.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

@app.get("/api/kol")
async def api_kol_list():
    notes = _load_kol()
    notes.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return JSONResponse({"notes": notes})

@app.post("/api/kol")
async def api_kol_add(request: Request):
    body = await request.json()
    notes = _load_kol()
    note = {
        "id":         str(uuid.uuid4())[:8],
        "kol":        body.get("kol", "").strip(),
        "platform":   body.get("platform", "Discord"),
        "content":    body.get("content", "").strip(),
        "tickers":    [t.upper().strip() for t in body.get("tickers", []) if t.strip()],
        "sentiment":  body.get("sentiment", "neutral"),
        "date":       body.get("date", datetime.now().strftime("%Y-%m-%d")),
        "created_at": datetime.now().isoformat(),
        "reviewed":   False,
    }
    if not note["content"]:
        return JSONResponse({"ok": False, "error": "内容不能为空"}, status_code=400)
    notes.append(note)
    _save_kol(notes)
    return JSONResponse({"ok": True, "note": note})

@app.delete("/api/kol/{note_id}")
async def api_kol_delete(note_id: str):
    notes = [n for n in _load_kol() if n.get("id") != note_id]
    _save_kol(notes)
    return JSONResponse({"ok": True})

@app.patch("/api/kol/{note_id}/review")
async def api_kol_review(note_id: str):
    notes = _load_kol()
    for n in notes:
        if n.get("id") == note_id:
            n["reviewed"] = True
    _save_kol(notes)
    return JSONResponse({"ok": True})

@app.patch("/api/kol/review-all")
async def api_kol_review_all():
    notes = _load_kol()
    for n in notes:
        n["reviewed"] = True
    _save_kol(notes)
    return JSONResponse({"ok": True})


# ─── StockWhale 持仓跟踪 ───

# 从截图中提取的完整交易数据
_SW_TRADES = [
    {"ticker":"DHI",  "date":"2025-07-18","type":3,"entry":131.83,"stop":131.83,"half":None,  "target":221.28,"risk":2.0,"rr":"5:1","status":"in_trade","note":"half profits + raise stop"},
    {"ticker":"ACI",  "date":"2025-08-19","type":3,"entry":19.39, "stop":15.27, "half":None,  "target":43.98, "risk":1.0,"rr":"6:1","status":"in_trade","note":""},
    {"ticker":"COP",  "date":"2025-11-14","type":3,"entry":90.31, "stop":90.31, "half":112.09,"target":182.25,"risk":4.0,"rr":"2:1","status":"in_trade","note":"half profits + raise stop"},
    {"ticker":"TLT",  "date":"2026-01-19","type":2,"entry":90.00, "stop":75.36, "half":101.84,"target":145.36,"risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"AEVA", "date":"2026-01-21","type":2,"entry":16.35, "stop":8.69,  "half":25.44, "target":45.69, "risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"NVTS", "date":"2026-01-26","type":2,"entry":9.04,  "stop":6.03,  "half":13.23, "target":20.45, "risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"PFE",  "date":"2026-01-29","type":2,"entry":26.00, "stop":20.80, "half":30.41, "target":47.33, "risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"UPS",  "date":"2026-02-01","type":3,"entry":116.07,"stop":81.60, "half":140.14,"target":181.06,"risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"UPS",  "date":"2026-02-01","type":3,"entry":100.82,"stop":81.60, "half":140.14,"target":181.06,"risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"PEP",  "date":"2026-02-02","type":3,"entry":167.03,"stop":127.73,"half":182.01,"target":239.83,"risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"PEP",  "date":"2026-02-02","type":3,"entry":151.31,"stop":127.73,"half":172.60,"target":213.49,"risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"CVX",  "date":"2026-02-04","type":3,"entry":178.61,"stop":178.61,"half":200.49,"target":270.78,"risk":0.5,"rr":"2:1","status":"in_trade","note":"half profits + raise stop"},
    {"ticker":"SMCI", "date":"2026-02-11","type":3,"entry":33.81, "stop":16.84, "half":47.44, "target":78.03, "risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"GRAB", "date":"2026-02-12","type":3,"entry":3.51,  "stop":2.63,  "half":4.85,  "target":7.54,  "risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"ARTNA","date":"2026-02-22","type":3,"entry":34.00, "stop":29.04, "half":38.88, "target":51.24, "risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"MGM",  "date":"2026-02-23","type":3,"entry":37.00, "stop":25.25, "half":42.48, "target":57.62, "risk":0.5,"rr":"2:1","status":"in_trade","note":""},
    {"ticker":"LEN",  "date":"2026-02-25","type":3,"entry":117.00,"stop":82.35, "half":143.38,"target":224.64,"risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"VKTX", "date":"2026-02-27","type":2,"entry":34.00, "stop":18.12, "half":49.58, "target":118.25,"risk":0.5,"rr":"5:1","status":"in_trade","note":""},
    {"ticker":"AAP",  "date":"2026-02-27","type":3,"entry":53.00, "stop":21.08, "half":93.10, "target":155.42,"risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"SIRI", "date":"2026-02-27","type":3,"entry":22.00, "stop":12.68, "half":26.84, "target":44.11, "risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"WTRG", "date":"2026-02-27","type":3,"entry":39.18, "stop":30.27, "half":44.82, "target":60.22, "risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"INTU", "date":"2026-03-04","type":3,"entry":287.61,"stop":141.60,"half":504.84,"target":1005.46,"risk":0.5,"rr":"5:1","status":"waiting","note":"Waiting to be filled"},
    {"ticker":"ADBE", "date":"2026-03-04","type":3,"entry":213.88,"stop":81.22, "half":425.60,"target":1059.78,"risk":0.5,"rr":"6:1","status":"waiting","note":"Waiting to be filled"},
    {"ticker":"JOBY", "date":"2026-03-05","type":3,"entry":8.32,  "stop":3.45,  "half":14.09, "target":25.10, "risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"ACHR", "date":"2026-03-05","type":3,"entry":5.45,  "stop":2.29,  "half":9.66,  "target":17.70, "risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"META", "date":"2026-03-06","type":2,"entry":528.35,"stop":528.35,"half":636.60,"target":961.57, "risk":1.0,"rr":"2:1","status":"in_trade","note":"Stop-loss raised to entry"},
    {"ticker":"USAR", "date":"2026-03-10","type":2,"entry":18.72, "stop":11.02, "half":29.27, "target":52.98, "risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"RDDT", "date":"2026-03-11","type":2,"entry":123.08,"stop":51.38, "half":188.98,"target":340.93,"risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"CRML", "date":"2026-03-18","type":2,"entry":7.86,  "stop":1.98,  "half":20.40, "target":39.57, "risk":1.0,"rr":"6:1","status":"in_trade","note":""},
    {"ticker":"NCLH", "date":"2026-04-01","type":2,"entry":19.00, "stop":6.90,  "half":23.95, "target":40.96, "risk":0.5,"rr":"2:1","status":"in_trade","note":""},
    {"ticker":"RDW",  "date":"2026-04-06","type":2,"entry":10.00, "stop":6.90,  "half":12.22, "target":17.03, "risk":0.5,"rr":"2:1","status":"in_trade","note":""},
    {"ticker":"Q",    "date":"2026-04-06","type":2,"entry":117.00,"stop":92.59, "half":139.21,"target":173.00,"risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"MSFT", "date":"2026-04-14","type":2,"entry":387.07,"stop":306.80,"half":456.13,"target":608.47,"risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"LEU",  "date":"2026-04-15","type":2,"entry":195.00,"stop":80.93, "half":305.80,"target":563.93,"risk":1.0,"rr":"2:1","status":"in_trade","note":""},
    {"ticker":"BIRD", "date":"2026-04-16","type":1,"entry":10.40, "stop":4.83,  "half":15.34, "target":30.23, "risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"ASTS", "date":"2026-04-20","type":2,"entry":45.29, "stop":16.47, "half":74.64, "target":162.70,"risk":1.0,"rr":"4:1","status":"waiting","note":"Waiting to be filled"},
]

@app.get("/api/stockwhale")
def api_stockwhale_list():
    """返回 StockWhale 全部持仓 + 实时价格"""
    from data.downloader import get_prices
    result = []
    for t in _SW_TRADES:
        ticker = t["ticker"]
        try:
            px_s = get_prices(ticker, start="2026-01-01")
            cur  = float(px_s.iloc[-1]) if not px_s.empty else None
            chg  = float(px_s.pct_change().iloc[-1] * 100) if cur else None
        except Exception:
            cur, chg = None, None
        gain = round((cur - t["entry"]) / t["entry"] * 100, 1) if cur else None
        vs_target = round((t["target"] - cur) / cur * 100, 1) if cur else None
        result.append({**t, "cur": cur, "chg_1d": round(chg,1) if chg else None,
                       "gain": gain, "vs_target": vs_target})
    return JSONResponse({"trades": result})


@app.get("/api/stockwhale/chart/{ticker}")
def api_stockwhale_chart(ticker: str):
    """返回指定 ticker 的 OHLCV 数据（供 K 线图使用）"""
    import math
    from data.downloader import get_ohlcv
    ticker = ticker.upper()
    trades = [t for t in _SW_TRADES if t["ticker"] == ticker]
    if not trades:
        return JSONResponse({"error": "not found"}, status_code=404)
    # 从最早入场日前90天开始取数据
    earliest = min(t["date"] for t in trades)
    from datetime import date, timedelta
    start_dt = (datetime.strptime(earliest, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
    try:
        df = get_ohlcv(ticker, start=start_dt)
        if df.empty:
            return JSONResponse({"error": "no data"})
        candles = []
        for idx, row in df.iterrows():
            ts = int(idx.timestamp())
            o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
            if any(math.isnan(v) for v in [o,h,l,c]):
                continue
            candles.append({"time": ts, "open": o, "high": h, "low": l, "close": c})
        return JSONResponse({"ticker": ticker, "candles": candles, "trades": trades})
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/health")
async def health():
    date_key = _trade_date_key()
    cached = [p.stem for p in _CACHE_DIR.glob(f"*_{date_key}.json")]
    return {"status": "ok", "trade_date": date_key, "cached_keys": cached,
            "time": datetime.now().isoformat()}
