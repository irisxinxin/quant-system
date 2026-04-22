"""
app.py — 量化交易仪表盘 Web 服务
Railway 部署：uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import os
import math
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


def _bg_prewarm_caches():
    """启动时直接调计算函数填充缓存，无需 HTTP 自请求，无需等待"""
    # 注意：各 _compute_* 函数在模块末尾定义，但线程在所有模块代码执行后才启动，引用安全
    logger.warning("[prewarm] 开始预热缓存...")
    items = [
        ("sw",    lambda: _compute_sw(),    CACHE_TTL["sw"]),
        ("er",    lambda: _compute_er(),    CACHE_TTL["er"]),
        ("cm",    lambda: _compute_cm(),    CACHE_TTL["cm"]),
        ("cm_wl", lambda: _compute_cm_wl(), CACHE_TTL["cm_wl"]),
        ("cn",    lambda: _compute_cn(),    CACHE_TTL["cn"]),
    ]
    for key, fn, ttl in items:
        try:
            t0 = time.time()
            _get_cached(key, fn, ttl)
            logger.warning(f"[prewarm] {key} ✓ ({time.time()-t0:.1f}s)")
        except Exception as e:
            logger.warning(f"[prewarm] {key} ✗ {e}")


@asynccontextmanager
async def lifespan(app):
    # 启动时后台：生成 K 线图 + 预热监控页缓存
    threading.Thread(target=_bg_generate_charts, daemon=True).start()
    threading.Thread(target=_bg_prewarm_caches,  daemon=True).start()
    yield


app = FastAPI(title="量化交易仪表盘", lifespan=lifespan)

_HTML_PATH     = Path(__file__).parent / "templates" / "index.html"
_CACHE_DIR     = Path(__file__).parent / "cache"
_CHARTS_DIR    = Path(__file__).parent / "output" / "charts"
_KOL_NOTES_PATH    = Path(__file__).parent / "output" / "kol_notes.json"
_REVIEW_NOTES_PATH = Path(__file__).parent / "output" / "review_notes.json"
_SIG_NOTES_PATH    = Path(__file__).parent / "output" / "sig_notes.json"
_SIG_KLINES_DIR    = Path(__file__).parent / "output" / "sig_klines"
_CACHE_DIR.mkdir(exist_ok=True)
_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
_SIG_KLINES_DIR.mkdir(exist_ok=True)

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
    # 持仓监控页：日线数据当天不变，缓存整个交易日
    "sw":      24 * 3600,
    "er":      24 * 3600,
    "cm":      24 * 3600,
    "cm_wl":   24 * 3600,
    "cn":      24 * 3600,
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
    data = _clean_nan(fn())   # 清洗 NaN/Inf，防止 JSONResponse 序列化崩溃
    ts   = now
    _mem_cache[key] = {"data": data, "ts": ts, "date_key": date_key}
    _save_disk(key, date_key, data, ts)
    return data, ts


def _clean_nan(obj):
    """递归把 NaN / Inf 替换成 None，防止 JSONResponse 序列化崩溃"""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    return obj


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


# ─── 每日复盘 ───

def _load_review() -> list:
    if _REVIEW_NOTES_PATH.exists():
        try:
            return json.loads(_REVIEW_NOTES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_review(notes: list) -> None:
    _REVIEW_NOTES_PATH.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

@app.get("/api/review")
async def api_review_list():
    notes = _load_review()
    notes.sort(key=lambda n: n.get("date", ""), reverse=True)
    return JSONResponse({"notes": notes})

@app.post("/api/review")
async def api_review_add(request: Request):
    body = await request.json()
    import uuid
    note = {
        "id":      str(uuid.uuid4())[:8],
        "date":    body.get("date", datetime.now().strftime("%Y-%m-%d")),
        "content": body.get("content", "").strip(),
        "tags":    body.get("tags", []),
        "mood":    body.get("mood", "neutral"),   # correct / mistake / neutral
        "tickers": [t.upper() for t in body.get("tickers", []) if t.strip()],
    }
    if not note["content"]:
        return JSONResponse({"ok": False, "error": "内容不能为空"}, status_code=400)
    notes = _load_review()
    notes.insert(0, note)
    _save_review(notes)
    return JSONResponse({"ok": True, "note": note})

@app.delete("/api/review/{note_id}")
async def api_review_delete(note_id: str):
    notes = [n for n in _load_review() if n.get("id") != note_id]
    _save_review(notes)
    return JSONResponse({"ok": True})


# ─── 逸哥量化信号 ───

def _load_sig() -> list:
    if _SIG_NOTES_PATH.exists():
        try:
            return json.loads(_SIG_NOTES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_sig(notes: list) -> None:
    _SIG_NOTES_PATH.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def _load_klines_local(ticker: str):
    """从 output/sig_klines/{TICKER}.json 读本地 OHLCV，返回 DataFrame 或空 DF"""
    import pandas as pd
    p = _SIG_KLINES_DIR / f"{ticker.upper()}.json"
    if not p.exists():
        return pd.DataFrame()
    try:
        records = json.loads(p.read_text(encoding="utf-8"))
        if not records:
            return pd.DataFrame()
        import pandas as pd
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()
    except Exception as e:
        logger.warning(f"[sig_klines] load {ticker}: {e}")
        return pd.DataFrame()

def _save_klines_local(ticker: str, df) -> None:
    """将 OHLCV DataFrame 序列化到 output/sig_klines/{TICKER}.json"""
    p = _SIG_KLINES_DIR / f"{ticker.upper()}.json"
    out = df.reset_index()
    # 统一日期列名（index 可能是 'Date' 或 'date' 或 'index'）
    for c in ["Date", "index"]:
        if c in out.columns:
            out = out.rename(columns={c: "date"})
            break
    out["date"] = out["date"].astype(str).str[:10]   # 只保留 YYYY-MM-DD
    cols = [c for c in ["date","Open","High","Low","Close","Volume"] if c in out.columns]
    p.write_text(json.dumps(out[cols].to_dict(orient="records"), indent=2), encoding="utf-8")

@app.get("/api/sig")
async def api_sig_list():
    import math
    from data.downloader import get_ohlcv
    notes = _load_sig()
    notes.sort(key=lambda n: n.get("date", ""), reverse=True)
    result = []
    for n in notes:
        try:
            df = get_ohlcv(n["ticker"], start="2025-01-01")
            cur = float(df["Close"].iloc[-1]) if not df.empty else None
            chg = float(df["Close"].pct_change().iloc[-1] * 100) if cur else None
            entry = n.get("entry")
            gain_pct = round((cur - entry) / entry * 100, 1) if cur and entry else None
        except Exception:
            cur = chg = gain_pct = None
        result.append({**n, "cur": round(cur, 2) if cur else None,
                        "chg_1d": round(chg, 1) if chg else None,
                        "gain_pct": gain_pct})
    return JSONResponse({"signals": result})

@app.post("/api/sig")
async def api_sig_add(request: Request):
    body = await request.json()
    import uuid
    levels_raw = body.get("levels", [])
    if isinstance(levels_raw, str):
        import re
        levels_raw = [float(x) for x in re.split(r'[\s,/]+', levels_raw.strip()) if x]
    note = {
        "id":      str(uuid.uuid4())[:8],
        "ticker":  body.get("ticker", "").upper().strip(),
        "date":    body.get("date", datetime.now().strftime("%Y-%m-%d")),
        "entry":   float(body.get("entry", 0)),
        "sl":      float(body.get("sl", 0)),
        "levels":  [float(l) for l in levels_raw],
        "size":    body.get("size", "").strip(),
        "atr_pct": body.get("atr_pct", None),
        "note":    body.get("note", "").strip(),
        "status":  "open",  # open / hit / stopped
    }
    if not note["ticker"]:
        return JSONResponse({"ok": False, "error": "ticker不能为空"}, status_code=400)
    notes = _load_sig()
    notes.insert(0, note)
    _save_sig(notes)
    return JSONResponse({"ok": True, "note": note})

@app.delete("/api/sig/{note_id}")
async def api_sig_delete(note_id: str):
    notes = [n for n in _load_sig() if n.get("id") != note_id]
    _save_sig(notes)
    return JSONResponse({"ok": True})

@app.patch("/api/sig/{note_id}/status")
async def api_sig_status(note_id: str, request: Request):
    body = await request.json()
    notes = _load_sig()
    for n in notes:
        if n.get("id") == note_id:
            n["status"] = body.get("status", n["status"])
    _save_sig(notes)
    return JSONResponse({"ok": True})

@app.get("/api/sig/chart/{ticker}")
def api_sig_chart(ticker: str, id: str = ""):
    """量化信号 K 线 + EMA21/50/MA200 + 入场/止损/目标水平线
    优先读 output/sig_klines/{TICKER}.json（已提交到 git），只增量拉新交易日。
    """
    import math, pandas as pd
    from data.downloader import _yf_ohlcv, _last_trading_date
    ticker = ticker.upper()
    notes = _load_sig()
    info = next((n for n in notes if n.get("id") == id), None) or \
           next((n for n in notes if n["ticker"] == ticker), None)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        end_str = datetime.today().strftime("%Y-%m-%d")
        df = _load_klines_local(ticker)

        if df.empty:
            # 本地无文件，全量下载并存盘
            from data.downloader import get_ohlcv
            df = get_ohlcv(ticker, start="2025-07-01")
            if not df.empty:
                _save_klines_local(ticker, df)
        else:
            last_date = df.index[-1].date()
            if last_date < _last_trading_date():
                fetch_from = (df.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
                new_df = _yf_ohlcv(ticker, fetch_from, end_str)
                if not new_df.empty:
                    df = pd.concat([df, new_df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                    _save_klines_local(ticker, df)   # 更新本地文件

        if df.empty:
            return JSONResponse({"error": "no data"})
        close = df["Close"]
        ema21_s = close.ewm(span=21,  adjust=False).mean()
        ema50_s = close.ewm(span=50,  adjust=False).mean()
        ma200_s = close.rolling(200).mean()

        def to_line(series):
            out = []
            for idx, v in series.items():
                if math.isnan(v): continue
                out.append({"time": int(idx.timestamp()), "value": round(float(v), 4)})
            return out

        candles = []
        for idx, row in df.iterrows():
            o,h,l,c = float(row["Open"]),float(row["High"]),float(row["Low"]),float(row["Close"])
            if any(math.isnan(v) for v in [o,h,l,c]): continue
            candles.append({"time": int(idx.timestamp()), "open":o,"high":h,"low":l,"close":c})

        cur = float(close.iloc[-1])
        gain_pct = round((cur - info["entry"]) / info["entry"] * 100, 1) if info.get("entry") else None
        sl_pct   = round((info["sl"] - info["entry"]) / info["entry"] * 100, 1) if info.get("sl") and info.get("entry") else None

        return JSONResponse({
            "ticker":   ticker,
            "id":       info["id"],
            "date":     info.get("date"),
            "entry":    info.get("entry"),
            "sl":       info.get("sl"),
            "levels":   info.get("levels", []),
            "size":     info.get("size", ""),
            "atr_pct":  info.get("atr_pct"),
            "note":     info.get("note", ""),
            "status":   info.get("status", "open"),
            "cur":      round(cur, 2),
            "gain_pct": gain_pct,
            "sl_pct":   sl_pct,
            "candles":  candles,
            "ema21":    to_line(ema21_s),
            "ema50":    to_line(ema50_s),
            "ma200":    to_line(ma200_s),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ─── StockWhale 持仓跟踪 ───

# 从截图中提取的完整交易数据
_SW_TRADES = [
    {"ticker":"DHI",  "sector":"地产/建筑", "date":"2025-07-18","type":3,"entry":131.83,"stop":131.83,"half":None,  "target":221.28,"risk":2.0,"rr":"5:1","status":"in_trade","note":"half profits + raise stop"},
    {"ticker":"ACI",  "sector":"超市零售",  "date":"2025-08-19","type":3,"entry":19.39, "stop":15.27, "half":None,  "target":43.98, "risk":1.0,"rr":"6:1","status":"in_trade","note":""},
    {"ticker":"COP",  "sector":"石油能源",  "date":"2025-11-14","type":3,"entry":90.31, "stop":90.31, "half":112.09,"target":182.25,"risk":4.0,"rr":"2:1","status":"in_trade","note":"half profits + raise stop"},
    {"ticker":"TLT",  "sector":"国债ETF",   "date":"2026-01-19","type":2,"entry":90.00, "stop":75.36, "half":101.84,"target":145.36,"risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"AEVA", "sector":"自动驾驶",  "date":"2026-01-21","type":2,"entry":16.35, "stop":8.69,  "half":25.44, "target":45.69, "risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"NVTS", "sector":"半导体",    "date":"2026-01-26","type":2,"entry":9.04,  "stop":6.03,  "half":13.23, "target":20.45, "risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"PFE",  "sector":"制药",      "date":"2026-01-29","type":2,"entry":26.00, "stop":20.80, "half":30.41, "target":47.33, "risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"UPS",  "sector":"物流",      "date":"2026-02-01","type":3,"entry":116.07,"stop":81.60, "half":140.14,"target":181.06,"risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"UPS",  "sector":"物流",      "date":"2026-02-01","type":3,"entry":100.82,"stop":81.60, "half":140.14,"target":181.06,"risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"PEP",  "sector":"消费品",    "date":"2026-02-02","type":3,"entry":167.03,"stop":127.73,"half":182.01,"target":239.83,"risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"PEP",  "sector":"消费品",    "date":"2026-02-02","type":3,"entry":151.31,"stop":127.73,"half":172.60,"target":213.49,"risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"CVX",  "sector":"石油能源",  "date":"2026-02-04","type":3,"entry":178.61,"stop":178.61,"half":200.49,"target":270.78,"risk":0.5,"rr":"2:1","status":"in_trade","note":"half profits + raise stop"},
    {"ticker":"SMCI", "sector":"AI服务器",  "date":"2026-02-11","type":3,"entry":33.81, "stop":16.84, "half":47.44, "target":78.03, "risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"GRAB", "sector":"东南亚科技","date":"2026-02-12","type":3,"entry":3.51,  "stop":2.63,  "half":4.85,  "target":7.54,  "risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"ARTNA","sector":"公用事业",  "date":"2026-02-22","type":3,"entry":34.00, "stop":29.04, "half":38.88, "target":51.24, "risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"MGM",  "sector":"娱乐/赌场", "date":"2026-02-23","type":3,"entry":37.00, "stop":25.25, "half":42.48, "target":57.62, "risk":0.5,"rr":"2:1","status":"in_trade","note":""},
    {"ticker":"LEN",  "sector":"地产/建筑", "date":"2026-02-25","type":3,"entry":117.00,"stop":82.35, "half":143.38,"target":224.64,"risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"VKTX", "sector":"生物制药",  "date":"2026-02-27","type":2,"entry":34.00, "stop":18.12, "half":49.58, "target":118.25,"risk":0.5,"rr":"5:1","status":"in_trade","note":""},
    {"ticker":"AAP",  "sector":"汽车零部件","date":"2026-02-27","type":3,"entry":53.00, "stop":21.08, "half":93.10, "target":155.42,"risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"SIRI", "sector":"音频媒体",  "date":"2026-02-27","type":3,"entry":22.00, "stop":12.68, "half":26.84, "target":44.11, "risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"WTRG", "sector":"公用事业",  "date":"2026-02-27","type":3,"entry":39.18, "stop":30.27, "half":44.82, "target":60.22, "risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"INTU", "sector":"金融软件",  "date":"2026-03-04","type":3,"entry":287.61,"stop":141.60,"half":504.84,"target":1005.46,"risk":0.5,"rr":"5:1","status":"waiting","note":"Waiting to be filled"},
    {"ticker":"ADBE", "sector":"创意软件",  "date":"2026-03-04","type":3,"entry":213.88,"stop":81.22, "half":425.60,"target":1059.78,"risk":0.5,"rr":"6:1","status":"waiting","note":"Waiting to be filled"},
    {"ticker":"JOBY", "sector":"eVTOL",    "date":"2026-03-05","type":3,"entry":8.32,  "stop":3.45,  "half":14.09, "target":25.10, "risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"ACHR", "sector":"eVTOL",    "date":"2026-03-05","type":3,"entry":5.45,  "stop":2.29,  "half":9.66,  "target":17.70, "risk":0.5,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"META", "sector":"社交媒体",  "date":"2026-03-06","type":2,"entry":528.35,"stop":528.35,"half":636.60,"target":961.57, "risk":1.0,"rr":"2:1","status":"in_trade","note":"Stop-loss raised to entry"},
    {"ticker":"USAR", "sector":"国防无人机","date":"2026-03-10","type":2,"entry":18.72, "stop":11.02, "half":29.27, "target":52.98, "risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"RDDT", "sector":"社交媒体",  "date":"2026-03-11","type":2,"entry":123.08,"stop":51.38, "half":188.98,"target":340.93,"risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"CRML", "sector":"生物制药",  "date":"2026-03-18","type":2,"entry":7.86,  "stop":1.98,  "half":20.40, "target":39.57, "risk":1.0,"rr":"6:1","status":"in_trade","note":""},
    {"ticker":"NCLH", "sector":"邮轮旅游",  "date":"2026-04-01","type":2,"entry":19.00, "stop":6.90,  "half":23.95, "target":40.96, "risk":0.5,"rr":"2:1","status":"in_trade","note":""},
    {"ticker":"RDW",  "sector":"航天制造",  "date":"2026-04-06","type":2,"entry":10.00, "stop":6.90,  "half":12.22, "target":17.03, "risk":0.5,"rr":"2:1","status":"in_trade","note":""},
    {"ticker":"Q",    "sector":"金融数据",  "date":"2026-04-06","type":2,"entry":117.00,"stop":92.59, "half":139.21,"target":173.00,"risk":0.5,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"MSFT", "sector":"科技/软件", "date":"2026-04-14","type":2,"entry":387.07,"stop":306.80,"half":456.13,"target":608.47,"risk":1.0,"rr":"3:1","status":"in_trade","note":""},
    {"ticker":"LEU",  "sector":"核能",      "date":"2026-04-15","type":2,"entry":195.00,"stop":80.93, "half":305.80,"target":563.93,"risk":1.0,"rr":"2:1","status":"in_trade","note":""},
    {"ticker":"BIRD", "sector":"运动零售",  "date":"2026-04-16","type":1,"entry":10.40, "stop":4.83,  "half":15.34, "target":30.23, "risk":1.0,"rr":"4:1","status":"in_trade","note":""},
    {"ticker":"ASTS", "sector":"卫星通信",  "date":"2026-04-20","type":2,"entry":45.29, "stop":16.47, "half":74.64, "target":162.70,"risk":1.0,"rr":"4:1","status":"waiting","note":"Waiting to be filled"},
    # ── 二级研究员持仓（六棵士） ────────────────────────────────────────────────
    {"ticker":"VRT",  "sector":"AI电力/数据中心","date":"2026-04-20","type":2,"entry":307.34,"stop":238.00,"half":390.00, "target":480.00, "risk":1.5,"rr":"2.5:1","status":"in_trade","note":"2.92% | 策略ema20_dip+smh+rsi80 | AI数据中心电力散热，订单+109% YoY"},
    {"ticker":"FORM", "sector":"光学测试",       "date":"2026-04-20","type":2,"entry":137.21,"stop":105.00,"half":175.00, "target":215.00, "risk":1.5,"rr":"2.5:1","status":"in_trade","note":"2.93% | 策略bb_lo+soxx+cmf_neg | CPO探针台+光学对准，1Y_DD仅-4.1%"},
    {"ticker":"AEHR", "sector":"半导体测试",     "date":"2026-04-20","type":1,"entry":83.86, "stop":55.00, "half":130.00, "target":180.00, "risk":1.0,"rr":"3:1","status":"in_trade","note":"1.99% | 策略rsi28+combo+rsi70 | 光芯片测试设备，注意历史最大回撤-71.9%"},
    {"ticker":"ASX",  "sector":"半导体封测",     "date":"2026-04-20","type":2,"entry":28.59, "stop":21.50, "half":40.00,  "target":55.00,  "risk":2.0,"rr":"3:1","status":"in_trade","note":"2% | 策略ema2060+soxx+trail_12 | 日月光半导体，1Y +214%，Calmar 15.4"},
    {"ticker":"LMND", "sector":"金融科技/保险",  "date":"2026-04-20","type":2,"entry":70.94, "stop":50.00, "half":None,  "target":120.00, "risk":2.0,"rr":"2:1","status":"in_trade","note":"⚡止盈出场2026-04-20 | 模型不符，保险行业无供需错配"},
    {"ticker":"GOOG", "sector":"科技/广告",      "date":"2026-04-20","type":2,"entry":339.40,"stop":270.00,"half":None,  "target":500.00, "risk":1.0,"rr":"2:1","status":"in_trade","note":"⚡止盈出场2026-04-20 | 模型不符，策略ma5200+none+ma_x"},
    {"ticker":"TSLA", "sector":"电动车/AI",      "date":"2026-04-20","type":2,"entry":400.62,"stop":320.00,"half":None,  "target":600.00, "risk":1.0,"rr":"2:1","status":"in_trade","note":"⚡止盈出场2026-04-20 | 模型不符，策略rsi28+combo+rsi80"},
    # ── 二级研究员持仓 2026-04-20 ──────────────────────────────────────────────
    {"ticker":"LITE", "sector":"激光器/光模块","date":"2026-04-20","type":2,"entry":894.07,"stop":700.00,"half":1150.00,"target":1500.00,"risk":5.0,"rr":"3:1","status":"in_trade","note":"核心仓19% | 策略vol_surge+smh+trail_12 | EML拿单≥70%,订单至2027+"},
    {"ticker":"COHR", "sector":"激光器/光模块","date":"2026-04-20","type":2,"entry":345.02,"stop":270.00,"half":440.00, "target":540.00, "risk":5.0,"rr":"2.5:1","status":"in_trade","note":"核心仓19.7% | 策略bb_lo+soft+rsi70 | CPO激光源+SiC双赛道"},
    {"ticker":"ALAB", "sector":"CPU/互联芯片", "date":"2026-04-20","type":2,"entry":174.05,"stop":132.00,"half":220.00, "target":300.00, "risk":2.0,"rr":"3:1","status":"in_trade","note":"主仓5% | 策略rsi28+none+rsi70 | CPU供需失衡乘数效应最强,弹性>ARM>AMD"},
    {"ticker":"MRVL", "sector":"光子/DSP",     "date":"2026-04-20","type":2,"entry":139.69,"stop":108.00,"half":175.00, "target":220.00, "risk":2.0,"rr":"2.5:1","status":"in_trade","note":"4.5% | 策略bb_lo+combo+ma_x | 800G/1.6T光DSP核心,加仓至4.5%"},
    {"ticker":"DRAM", "sector":"存储",          "date":"2026-04-20","type":2,"entry":35.59, "stop":25.00, "half":50.00,  "target":70.00,  "risk":3.0,"rr":"3:1","status":"in_trade","note":"6% | 存储板块ETF | MU止盈后转入,长协重新定价催化"},
    {"ticker":"GFS",  "sector":"硅光代工",      "date":"2026-04-20","type":2,"entry":54.75, "stop":42.00, "half":70.00,  "target":90.00,  "risk":2.0,"rr":"2.5:1","status":"in_trade","note":"4% | 策略mfi_os+smh+rsi80 | 12英寸SiPh先发优势,24xPE低估值,起诉TSEM专利"},
    {"ticker":"ARM",  "sector":"半导体IP",      "date":"2026-04-20","type":2,"entry":166.73,"stop":132.00,"half":210.00, "target":260.00, "risk":1.0,"rr":"2.5:1","status":"in_trade","note":"2% | 策略ema20_dip+none+rsi80 | CPU链观察仓,预期差较小先建仓"},
    {"ticker":"AAOI", "sector":"光模块",        "date":"2026-04-20","type":1,"entry":159.42,"stop":118.00,"half":210.00, "target":300.00, "risk":2.0,"rr":"3:1","status":"in_trade","note":"4.5% | 策略ema20_dip+obv+soxx+rsi_fade | 27年营收指引若兑现250-300B市值弹性"},
    {"ticker":"POET", "sector":"光电集成",      "date":"2026-04-20","type":1,"entry":7.26,  "stop":5.20,  "half":10.50, "target":16.00,  "risk":1.0,"rr":"4:1","status":"in_trade","note":"1.8% | 策略bb_lo+none+cmf_neg | CPO Layer5光电集成期权"},
    {"ticker":"MU",   "sector":"存储/DRAM",     "date":"2026-04-20","type":2,"entry":455.07,"stop":380.00,"half":560.00, "target":650.00, "risk":2.0,"rr":"2:1","status":"in_trade","note":"⚡止盈出场2026-04-20 | 策略bb_lo+none+rsi80 | 转仓DRAM"},
    {"ticker":"TSEM", "sector":"硅光代工",      "date":"2026-04-20","type":2,"entry":226.45,"stop":185.00,"half":290.00, "target":360.00, "risk":2.0,"rr":"2.5:1","status":"in_trade","note":"⚡止盈出场2026-04-20 | 策略ma5200+soxx+trail_12 | GFS专利起诉降权"},
]

def _compute_sw():
    from data.downloader import get_prices
    BUY_ZONE_THRESH = {1: 0.05, 2: 0.08, 3: 0.12}
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
        display_status = t["status"]
        if display_status == "in_trade" and cur is not None:
            thresh = BUY_ZONE_THRESH.get(t.get("type", 2), 0.08)
            if cur <= t["entry"] * (1 + thresh):
                display_status = "buy_zone"
        result.append({**t, "cur": cur, "chg_1d": round(chg,1) if chg else None,
                       "gain": gain, "vs_target": vs_target,
                       "display_status": display_status})
    return result

@app.get("/api/stockwhale")
def api_stockwhale_list():
    """返回 StockWhale 全部持仓 + 实时价格（日线缓存）"""
    data, ts = _get_cached("sw", _compute_sw, CACHE_TTL["sw"])
    return JSONResponse({"trades": data, "cached_at": _fmt_age(ts)})


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


# ─── 大A 监控 ───

_CN_STOCKS = [
    {"ticker":"002384.SZ", "name":"东山精密",  "sector":"精密制造"},
    {"ticker":"001309.SZ", "name":"德明利",    "sector":"存储芯片"},
    {"ticker":"300548.SZ", "name":"长芯博创",  "sector":"光器件"},
    {"ticker":"688195.SS", "name":"腾景科技",  "sector":"光器件"},
    {"ticker":"603306.SS", "name":"华懋科技",  "sector":"PCB/特种材料"},
    {"ticker":"301377.SZ", "name":"鼎泰高科",  "sector":"精密制造"},
    {"ticker":"300394.SZ", "name":"天孚通信",  "sector":"光模块"},
    {"ticker":"300476.SZ", "name":"胜宏科技",  "sector":"PCB"},
    {"ticker":"300757.SZ", "name":"罗博特科",  "sector":"半导体设备"},
    {"ticker":"300620.SZ", "name":"光库科技",  "sector":"光器件"},
]


def _compute_cn():
    import math, logging
    from data.downloader import get_ohlcv
    import yfinance as yf
    result = []
    for s in _CN_STOCKS:
        ticker = s["ticker"]
        entry = {**s, "cur": None, "chg_1d": None, "vs_ema20": None, "vs_ma200": None,
                 "signals": [], "earnings_date": None, "earnings_days": None}
        try:
            df = get_ohlcv(ticker, start="2023-01-01")
            if not df.empty:
                close = df["Close"]
                cur   = float(close.iloc[-1])
                chg   = float(close.pct_change().iloc[-1] * 100)
                ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
                ma200_s = close.rolling(200).mean()
                ma200 = float(ma200_s.iloc[-1]) if len(close) >= 200 and not math.isnan(ma200_s.iloc[-1]) else None
                recent_high = float(close.iloc[-21:-1].max()) if len(close) > 21 else None
                vs_ema20 = (cur - ema20) / ema20 * 100
                vs_ma200 = (cur - ma200) / ma200 * 100 if ma200 else None
                signals = []
                if recent_high and cur >= recent_high * 0.97:
                    signals.append("🚀 接近突破")
                if abs(vs_ema20) <= 2.0 and cur >= ema20 * 0.98:
                    signals.append("↩️ 回踩EMA20")
                ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
                if abs((cur - ema10) / ema10 * 100) <= 1.5:
                    signals.append("↩️ 回踩EMA10")
                entry.update({
                    "cur": round(cur, 2), "chg_1d": round(chg, 1),
                    "vs_ema20": round(vs_ema20, 1),
                    "vs_ma200": round(vs_ma200, 1) if vs_ma200 is not None else None,
                    "signals": signals,
                    "ema20": round(ema20, 2),
                    "ma200": round(ma200, 2) if ma200 else None,
                })
        except Exception as e:
            logger.warning(f"[cn price] {ticker}: {e}")
        try:
            _yf_log = logging.getLogger("yfinance")
            _prev = _yf_log.level
            _yf_log.setLevel(logging.CRITICAL)
            try:
                t = yf.Ticker(ticker)
                cal = t.calendar
            finally:
                _yf_log.setLevel(_prev)
            if cal is not None:
                if isinstance(cal, dict):
                    ed = cal.get("Earnings Date") or cal.get("earningsDate")
                    if isinstance(ed, (list, tuple)) and ed:
                        ed = ed[0]
                elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    ed = cal.loc["Earnings Date"].iloc[0] if not cal.empty else None
                else:
                    ed = None
                if ed is not None:
                    import pandas as pd
                    ed_dt = pd.Timestamp(ed).date()
                    today = datetime.today().date()
                    days_left = (ed_dt - today).days
                    if days_left >= 0:
                        entry["earnings_date"] = str(ed_dt)
                        entry["earnings_days"] = days_left
                        if days_left <= 7:
                            entry["signals"].append(f"📢 财报还有{days_left}天")
                        elif days_left <= 14:
                            entry["signals"].append(f"📅 财报{days_left}天后")
        except Exception as e:
            logger.debug(f"[cn earnings] {ticker}: {e}")
        result.append(entry)
    return result

@app.get("/api/cn")
def api_cn_list():
    """大A股票监控：价格 + 技术信号 + 财报提醒（日线缓存）"""
    data, ts = _get_cached("cn", _compute_cn, CACHE_TTL["cn"])
    return JSONResponse({"stocks": data, "cached_at": _fmt_age(ts)})


@app.get("/api/cn/chart/{ticker}")
def api_cn_chart(ticker: str):
    """大A K线 + EMA10/20/MA200"""
    import math
    from data.downloader import get_ohlcv
    ticker = ticker.upper()
    # 支持带后缀（002384.SZ）直接传入
    info = next((s for s in _CN_STOCKS if s["ticker"].upper() == ticker), None)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        df = get_ohlcv(info["ticker"], start="2023-01-01")
        if df.empty:
            return JSONResponse({"error": "no data"})
        close = df["Close"]

        def to_line(series):
            out = []
            for idx, v in series.items():
                if math.isnan(v): continue
                out.append({"time": int(idx.timestamp()), "value": round(float(v), 4)})
            return out

        candles = []
        for idx, row in df.iterrows():
            o,h,l,c = float(row["Open"]),float(row["High"]),float(row["Low"]),float(row["Close"])
            if any(math.isnan(v) for v in [o,h,l,c]): continue
            candles.append({"time": int(idx.timestamp()), "open":o,"high":h,"low":l,"close":c})

        ema10_s = close.ewm(span=10, adjust=False).mean()
        ema20_s = close.ewm(span=20, adjust=False).mean()
        ma200_s = close.rolling(200).mean()
        recent_high = float(close.iloc[-21:-1].max()) if len(close) > 21 else None
        ma200_cur = float(ma200_s.iloc[-1]) if not math.isnan(ma200_s.iloc[-1]) else None

        return JSONResponse({
            "ticker": info["ticker"], "name": info["name"],
            "candles": candles,
            "ema10": to_line(ema10_s), "ema20": to_line(ema20_s), "ma200": to_line(ma200_s),
            "levels": {
                "recent_high": round(recent_high, 3) if recent_high else None,
                "ma200": round(ma200_cur, 3) if ma200_cur else None,
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ─── CM 选股 ───

_CM_STOCKS = [
    # 右侧强势股：买突破前高 or 回踩EMA10/20
    {"ticker":"AMZN","side":"right","sector":"电商/云","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"AAPL","side":"right","sector":"消费科技","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"GOOG","side":"right","sector":"广告/AI","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"NVDA","side":"right","sector":"AI算力","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"AMD", "side":"right","sector":"半导体","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"MU",  "side":"right","sector":"存储","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"DELL","side":"right","sector":"AI服务器","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"INTC","side":"right","sector":"半导体","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"GEV", "side":"right","sector":"AI电力","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"KLAC","side":"right","sector":"半导体设备","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"MRVL","side":"right","sector":"光子/连接","note":"突破前高 or 回踩10/20日线"},
    {"ticker":"AVGO","side":"right","sector":"AI芯片","note":"突破前高 or 回踩10/20日线"},
    # 左侧补涨股：分批左侧建仓 or 等MA200确认右侧
    {"ticker":"TSLA","side":"left","sector":"电动车","note":"左侧分批 or 站上MA200确认"},
    {"ticker":"META","side":"left","sector":"社交媒体","note":"左侧分批 or 站上MA200确认"},
    {"ticker":"MSFT","side":"left","sector":"科技/软件","note":"左侧分批 or 站上MA200确认"},
    {"ticker":"NFLX","side":"left","sector":"流媒体","note":"左侧分批 or 站上MA200确认"},
    # A股持仓（二级研究员）
    {"ticker":"603306.SS","side":"right","sector":"A股·光通信EMS","note":"华懋科技 24.27%"},
    {"ticker":"002384.SZ","side":"right","sector":"A股·光模块+PCB","note":"东山精密 29.07%"},
    {"ticker":"300476.SZ","side":"right","sector":"A股·AI服务器PCB","note":"胜宏科技 21.95%"},
    {"ticker":"300757.SZ","side":"right","sector":"A股·CPO精密制造","note":"罗博特科 23.49%"},
]


def _compute_cm():
    from data.downloader import get_ohlcv
    result = []
    for s in _CM_STOCKS:
        ticker = s["ticker"]
        try:
            df = get_ohlcv(ticker, start="2025-01-01")
            if df.empty:
                result.append({**s, "cur": None, "chg_1d": None, "vs_ema20": None, "vs_ma200": None, "signals": []})
                continue
            close = df["Close"]
            cur   = float(close.iloc[-1])
            chg   = float(close.pct_change().iloc[-1] * 100)
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
            recent_high = float(close.rolling(20).max().iloc[-2]) if len(close) > 20 else None
            signals = []
            vs_ema20 = (cur - ema20) / ema20 * 100
            vs_ma200 = (cur - ma200) / ma200 * 100 if ma200 else None
            if s["side"] == "right":
                if recent_high and cur >= recent_high * 0.97:
                    signals.append("🚀 接近突破前高")
                if abs(vs_ema20) <= 2.0 and cur > ema20 * 0.98:
                    signals.append("↩️ 回踩EMA20")
                ema10 = float(close.ewm(span=10, adjust=False).mean().iloc[-1])
                if abs((cur - ema10) / ema10 * 100) <= 1.5:
                    signals.append("↩️ 回踩EMA10")
            else:
                if ma200 and abs(vs_ma200) <= 3.0:
                    signals.append("⭐ 接近MA200确认位")
            result.append({**s, "cur": round(cur,2), "chg_1d": round(chg,1),
                            "vs_ema20": round(vs_ema20,1),
                            "vs_ma200": round(vs_ma200,1) if vs_ma200 is not None else None,
                            "recent_high": round(recent_high,2) if recent_high else None,
                            "ema20": round(ema20,2),
                            "ma200": round(ma200,2) if ma200 else None,
                            "signals": signals})
        except Exception:
            result.append({**s, "cur": None, "chg_1d": None, "vs_ema20": None, "vs_ma200": None, "signals": []})
    return result

@app.get("/api/cm")
def api_cm_list():
    """返回 CM 选股列表 + 实时价格 + 与关键均线距离（日线缓存）"""
    data, ts = _get_cached("cm", _compute_cm, CACHE_TTL["cm"])
    return JSONResponse({"stocks": data, "cached_at": _fmt_age(ts)})


# 市场回调后强势标的 watchlist
_CM_WATCHLIST = [
    # 中盘/大盘
    {"ticker":"AAOI", "cap":"mid","sector":"光子/连接"},
    {"ticker":"AEIS", "cap":"mid","sector":"半导体设备"},
    {"ticker":"INTC", "cap":"mid","sector":"半导体"},
    {"ticker":"AGX",  "cap":"mid","sector":"核能/工程"},
    {"ticker":"ARM",  "cap":"mid","sector":"半导体IP"},
    {"ticker":"AA",   "cap":"mid","sector":"铝/原材料"},
    {"ticker":"APLD", "cap":"mid","sector":"AI数据中心"},
    {"ticker":"BE",   "cap":"mid","sector":"AI电力"},
    {"ticker":"BWXT", "cap":"mid","sector":"核能"},
    {"ticker":"NBIS", "cap":"mid","sector":"半导体"},
    {"ticker":"CRDO", "cap":"mid","sector":"半导体/互连"},
    {"ticker":"CIFR", "cap":"mid","sector":"比特币挖矿"},
    {"ticker":"CIEN", "cap":"mid","sector":"光子/网络"},
    {"ticker":"CLS",  "cap":"mid","sector":"光子/连接"},
    {"ticker":"COHR", "cap":"mid","sector":"光子/激光"},
    {"ticker":"CORZ", "cap":"mid","sector":"比特币挖矿"},
    {"ticker":"DAVE", "cap":"mid","sector":"金融科技"},
    {"ticker":"DOCN", "cap":"mid","sector":"云计算"},
    {"ticker":"ECG",  "cap":"mid","sector":"工业"},
    {"ticker":"FIS",  "cap":"mid","sector":"金融科技"},
    {"ticker":"FLY",  "cap":"mid","sector":"飞机租赁"},
    {"ticker":"FN",   "cap":"mid","sector":"光子/精密制造"},
    {"ticker":"FORM", "cap":"mid","sector":"半导体设备"},
    {"ticker":"FTAI", "cap":"mid","sector":"航空基础设施"},
    {"ticker":"GLW",  "cap":"mid","sector":"光纤/材料"},
    {"ticker":"HUT",  "cap":"mid","sector":"比特币挖矿"},
    {"ticker":"LGN",  "cap":"mid","sector":"工业"},
    {"ticker":"JBL",  "cap":"mid","sector":"电子制造"},
    {"ticker":"LITE", "cap":"mid","sector":"光子/连接"},
    {"ticker":"LUMN", "cap":"mid","sector":"电信/光纤"},
    {"ticker":"LUNR", "cap":"mid","sector":"月球探索"},
    {"ticker":"MKSI", "cap":"mid","sector":"半导体设备"},
    {"ticker":"MOD",  "cap":"mid","sector":"热管理"},
    {"ticker":"MTZ",  "cap":"mid","sector":"电力基建"},
    {"ticker":"NOK",  "cap":"mid","sector":"电信设备"},
    {"ticker":"ONTO", "cap":"mid","sector":"半导体设备"},
    {"ticker":"PL",   "cap":"mid","sector":"卫星/遥感"},
    {"ticker":"POWL", "cap":"mid","sector":"AI电力"},
    {"ticker":"Q",    "cap":"mid","sector":"金融数据"},
    {"ticker":"SATS", "cap":"mid","sector":"卫星通信"},
    {"ticker":"SNDK", "cap":"mid","sector":"存储"},
    {"ticker":"SNEX", "cap":"mid","sector":"金融经纪"},
    {"ticker":"SQM",  "cap":"mid","sector":"锂矿"},
    {"ticker":"STX",  "cap":"mid","sector":"存储"},
    {"ticker":"TER",  "cap":"mid","sector":"半导体设备"},
    {"ticker":"TTMI", "cap":"mid","sector":"PCB"},
    {"ticker":"TWLO", "cap":"mid","sector":"通信软件"},
    {"ticker":"TSEM", "cap":"mid","sector":"半导体代工"},
    {"ticker":"UI",   "cap":"mid","sector":"网络设备"},
    {"ticker":"VIAV", "cap":"mid","sector":"光子/测试"},
    {"ticker":"VICR", "cap":"mid","sector":"电源转换"},
    {"ticker":"VRT",  "cap":"mid","sector":"AI数据中心"},
    {"ticker":"WDC",  "cap":"mid","sector":"存储"},
    {"ticker":"WULF", "cap":"mid","sector":"比特币挖矿"},
    {"ticker":"XPO",  "cap":"mid","sector":"物流"},
    {"ticker":"YOU",  "cap":"mid","sector":"身份安全"},
    # 小盘
    {"ticker":"AEHR", "cap":"small","sector":"半导体测试"},
    {"ticker":"ACLS", "cap":"small","sector":"半导体设备"},
    {"ticker":"AXTI", "cap":"small","sector":"半导体材料"},
    {"ticker":"FSLY", "cap":"small","sector":"CDN/边缘"},
    {"ticker":"POET", "cap":"small","sector":"光子集成"},
    {"ticker":"IRDM", "cap":"small","sector":"卫星通信"},
    {"ticker":"LWLG", "cap":"small","sector":"电光调制"},
    {"ticker":"UCTT", "cap":"small","sector":"半导体设备"},
    {"ticker":"YSS",  "cap":"small","sector":"休闲/消费"},
]

def _compute_stock_stats(s):
    """计算单只股票的价格 + 信号，供 watchlist 并行使用"""
    import math
    from data.downloader import get_ohlcv
    ticker = s["ticker"]
    try:
        df = get_ohlcv(ticker, start="2025-01-01")
        if df.empty:
            return {**s, "cur": None, "chg_1d": None, "vs_ema20": None, "vs_ma200": None, "signals": []}
        close = df["Close"]
        cur   = float(close.iloc[-1])
        chg   = float(close.pct_change().iloc[-1] * 100)
        ema10 = float(close.ewm(span=10,  adjust=False).mean().iloc[-1])
        ema20 = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
        ma200_s = close.rolling(200).mean()
        ma200 = float(ma200_s.iloc[-1]) if len(close) >= 200 and not math.isnan(ma200_s.iloc[-1]) else None
        recent_high = float(close.iloc[-21:-1].max()) if len(close) > 21 else None
        vs_ema20 = (cur - ema20) / ema20 * 100
        vs_ma200 = (cur - ma200) / ma200 * 100 if ma200 else None
        signals = []
        if recent_high and cur >= recent_high * 0.97:
            signals.append("🚀 接近突破")
        if abs(vs_ema20) <= 2.0 and cur >= ema20 * 0.98:
            signals.append("↩️ 回踩EMA20")
        if abs((cur - ema10) / ema10 * 100) <= 1.5:
            signals.append("↩️ 回踩EMA10")
        if ma200 and abs(vs_ma200) <= 3.0:
            signals.append("⭐ 近MA200")
        return {
            **s,
            "cur": round(cur, 2),
            "chg_1d": round(chg, 1),
            "vs_ema20": round(vs_ema20, 1),
            "vs_ma200": round(vs_ma200, 1) if vs_ma200 is not None else None,
            "signals": signals,
        }
    except Exception:
        return {**s, "cur": None, "chg_1d": None, "vs_ema20": None, "vs_ma200": None, "signals": []}


def _compute_cm_wl():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = [None] * len(_CM_WATCHLIST)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_compute_stock_stats, s): i for i, s in enumerate(_CM_WATCHLIST)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
    results = [r for r in results if r]
    results.sort(key=lambda x: (-len(x.get("signals") or []), x.get("ticker","")))
    return results

@app.get("/api/cm/watchlist")
def api_cm_watchlist():
    """并行拉取 watchlist 所有股票的价格 + 信号（日线缓存）"""
    data, ts = _get_cached("cm_wl", _compute_cm_wl, CACHE_TTL["cm_wl"])
    return JSONResponse({"stocks": data, "cached_at": _fmt_age(ts)})


@app.get("/api/cm/chart/{ticker}")
def api_cm_chart(ticker: str):
    """返回 CM 选股 K 线 + EMA10/20/MA200（支持龙头股 + watchlist）"""
    import math
    from data.downloader import get_ohlcv
    ticker = ticker.upper()
    all_stocks = _CM_STOCKS + _CM_WATCHLIST
    info = next((s for s in all_stocks if s["ticker"] == ticker), None)
    if not info:
        # 允许任意 ticker 查看图表
        info = {"ticker": ticker, "side": "watch", "note": ""}
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        df = get_ohlcv(ticker, start="2025-07-01")
        if df.empty:
            return JSONResponse({"error": "no data"})
        close = df["Close"]
        ema10_s  = close.ewm(span=10,  adjust=False).mean()
        ema20_s  = close.ewm(span=20,  adjust=False).mean()
        ma200_s  = close.rolling(200).mean()

        def to_line(series):
            out = []
            for idx, v in series.items():
                if math.isnan(v): continue
                out.append({"time": int(idx.timestamp()), "value": round(float(v), 4)})
            return out

        candles = []
        for idx, row in df.iterrows():
            o,h,l,c = float(row["Open"]),float(row["High"]),float(row["Low"]),float(row["Close"])
            if any(math.isnan(v) for v in [o,h,l,c]): continue
            candles.append({"time": int(idx.timestamp()), "open":o,"high":h,"low":l,"close":c})

        # 关键价格线
        cur = float(close.iloc[-1])
        ema20_cur = float(ema20_s.iloc[-1])
        ma200_cur = float(ma200_s.iloc[-1]) if not math.isnan(ma200_s.iloc[-1]) else None
        recent_high = float(close.iloc[-21:-1].max()) if len(close) > 21 else None  # 前20日高点

        return JSONResponse({
            "ticker": ticker,
            "side":   info.get("side", "watch"),
            "note":   info.get("note", ""),
            "candles": candles,
            "ema10":  to_line(ema10_s),
            "ema20":  to_line(ema20_s),
            "ma200":  to_line(ma200_s),
            "levels": {
                "ema20":       round(ema20_cur, 2),
                "ma200":       round(ma200_cur, 2) if ma200_cur else None,
                "recent_high": round(recent_high, 2) if recent_high else None,
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ─── 二级研究员持仓 ───

_ER_PORTFOLIO = [
    # tier: 核心 / 主仓 / 中仓 / 观察 / 出场
    {"ticker":"LITE", "tier":"核心", "size_pct":19.1, "sector":"激光器/光模块","entry":894.07,"strategy":"vol_surge+smh+trail_12",   "note":"EML拿单≥70%，订单至2027+"},
    {"ticker":"COHR", "tier":"核心", "size_pct":19.7, "sector":"激光器/光模块","entry":345.02,"strategy":"bb_lo+soft+rsi70",          "note":"CPO激光源+SiC双赛道"},
    {"ticker":"DRAM", "tier":"主仓", "size_pct":6.0,  "sector":"存储",         "entry":35.59, "strategy":"待优化(数据不足)",           "note":"MU止盈后转入，长协重新定价催化"},
    {"ticker":"ALAB", "tier":"主仓", "size_pct":5.0,  "sector":"CPU/互联芯片", "entry":174.05,"strategy":"rsi28+none+rsi70",          "note":"CPU供需失衡乘数效应最强，弹性>ARM>AMD"},
    {"ticker":"MRVL", "tier":"中仓", "size_pct":4.5,  "sector":"光子/DSP",     "entry":139.69,"strategy":"bb_lo+combo+ma_x",          "note":"800G/1.6T DSP核心供应商"},
    {"ticker":"AAOI", "tier":"中仓", "size_pct":4.5,  "sector":"光模块",       "entry":159.42,"strategy":"ema20_dip+obv+rsi_fade",    "note":"27年指引若兑现→250-300B市值弹性"},
    {"ticker":"GFS",  "tier":"中仓", "size_pct":4.0,  "sector":"硅光代工",     "entry":54.75, "strategy":"mfi_os+smh+rsi80",          "note":"12英寸SiPh先发2年，24x PE低估值"},
    {"ticker":"VRT",  "tier":"观察", "size_pct":2.9,  "sector":"AI电力",       "entry":307.34,"strategy":"ema20_dip+smh+rsi80",       "note":"订单+109% YoY，AI数据中心散热核心"},
    {"ticker":"FORM", "tier":"观察", "size_pct":2.9,  "sector":"光学测试",     "entry":137.21,"strategy":"bb_lo+soxx+cmf_neg",        "note":"CPO探针台，1Y_DD仅-4.1%"},
    {"ticker":"ARM",  "tier":"观察", "size_pct":2.0,  "sector":"半导体IP",     "entry":166.73,"strategy":"ema20_dip+none+rsi80",      "note":"CPU链预期差标的，先建仓后研究"},
    {"ticker":"AEHR", "tier":"观察", "size_pct":2.0,  "sector":"半导体测试",   "entry":83.86, "strategy":"rsi28+combo+rsi70",         "note":"光芯片测试设备，注意历史DD-71.9%"},
    {"ticker":"ASX",  "tier":"观察", "size_pct":2.0,  "sector":"半导体封测",   "entry":28.59, "strategy":"ema2060+soxx+trail_12",     "note":"日月光半导体，1Y策略+214%，Calmar 15"},
    {"ticker":"POET", "tier":"观察", "size_pct":1.8,  "sector":"光电集成",     "entry":7.26,  "strategy":"bb_lo+none+cmf_neg",        "note":"CPO Layer5光电集成期权"},
    # 止盈出场
    {"ticker":"MU",   "tier":"出场", "size_pct":0.0,  "sector":"存储/DRAM",    "entry":455.07,"strategy":"bb_lo+none+rsi80",          "note":"⚡止盈转DRAM 04-20"},
    {"ticker":"TSEM", "tier":"出场", "size_pct":0.0,  "sector":"硅光代工",     "entry":226.45,"strategy":"ma5200+soxx+trail_12",      "note":"⚡止盈 04-20，GFS专利起诉降权"},
    {"ticker":"LMND", "tier":"出场", "size_pct":0.0,  "sector":"金融科技",     "entry":70.94, "strategy":"—",                        "note":"⚡止盈 04-20，保险/模型不符"},
    {"ticker":"GOOG", "tier":"出场", "size_pct":0.0,  "sector":"科技/广告",    "entry":339.40,"strategy":"ma5200+none+ma_x",          "note":"⚡止盈 04-20，平台/模型不符"},
    {"ticker":"TSLA", "tier":"出场", "size_pct":0.0,  "sector":"电动车",       "entry":400.62,"strategy":"rsi28+combo+rsi80",         "note":"⚡止盈 04-20，汽车/模型不符"},
]


def _compute_er_stats(s):
    """计算持仓股票的实时价格 + 信号 + 盈亏"""
    import math
    from data.downloader import get_ohlcv
    ticker = s["ticker"]
    try:
        df = get_ohlcv(ticker, start="2025-01-01")
        if df.empty:
            return {**s, "cur": None, "chg_1d": None, "gain_pct": None,
                    "vs_ema20": None, "vs_ma200": None, "signals": []}
        close = df["Close"]
        cur   = float(close.iloc[-1])
        chg   = float(close.pct_change().iloc[-1] * 100)
        ema10 = float(close.ewm(span=10,  adjust=False).mean().iloc[-1])
        ema20 = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
        ma200_s = close.rolling(200).mean()
        ma200 = float(ma200_s.iloc[-1]) if len(close) >= 200 and not math.isnan(ma200_s.iloc[-1]) else None
        recent_high = float(close.iloc[-21:-1].max()) if len(close) > 21 else None
        vs_ema20 = (cur - ema20) / ema20 * 100
        vs_ma200 = (cur - ma200) / ma200 * 100 if ma200 else None
        gain_pct = (cur - s["entry"]) / s["entry"] * 100
        signals = []
        if s["tier"] != "出场":
            if recent_high and cur >= recent_high * 0.97:
                signals.append("🚀 接近突破")
            if abs(vs_ema20) <= 2.0 and cur >= ema20 * 0.98:
                signals.append("↩️ 回踩EMA20")
            if abs((cur - ema10) / ema10 * 100) <= 1.5:
                signals.append("↩️ 回踩EMA10")
            if ma200 and abs(vs_ma200) <= 3.0:
                signals.append("⭐ 近MA200")
        return {
            **s,
            "cur":      round(cur, 2),
            "chg_1d":   round(chg, 1),
            "gain_pct": round(gain_pct, 1),
            "vs_ema20": round(vs_ema20, 1),
            "vs_ma200": round(vs_ma200, 1) if vs_ma200 is not None else None,
            "signals":  signals,
        }
    except Exception:
        return {**s, "cur": None, "chg_1d": None, "gain_pct": None,
                "vs_ema20": None, "vs_ma200": None, "signals": []}


def _compute_er():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = [None] * len(_ER_PORTFOLIO)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_compute_er_stats, s): i for i, s in enumerate(_ER_PORTFOLIO)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()
    return [r for r in results if r]

@app.get("/api/er")
def api_er_list():
    """二级研究员持仓：价格 + 盈亏 + 信号（并行拉取，日线缓存）"""
    data, ts = _get_cached("er", _compute_er, CACHE_TTL["er"])
    return JSONResponse({"stocks": data, "cached_at": _fmt_age(ts)})


@app.get("/api/er/chart/{ticker}")
def api_er_chart(ticker: str):
    """二级研究员持仓 K 线 + EMA10/20/MA200 + 入场价水平线"""
    import math
    from data.downloader import get_ohlcv
    ticker = ticker.upper()
    info = next((s for s in _ER_PORTFOLIO if s["ticker"] == ticker), None)
    if not info:
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        df = get_ohlcv(ticker, start="2025-07-01")
        if df.empty:
            return JSONResponse({"error": "no data"})
        close = df["Close"]
        ema10_s = close.ewm(span=10,  adjust=False).mean()
        ema20_s = close.ewm(span=20,  adjust=False).mean()
        ma200_s = close.rolling(200).mean()

        def to_line(series):
            out = []
            for idx, v in series.items():
                if math.isnan(v): continue
                out.append({"time": int(idx.timestamp()), "value": round(float(v), 4)})
            return out

        candles = []
        for idx, row in df.iterrows():
            o,h,l,c = float(row["Open"]),float(row["High"]),float(row["Low"]),float(row["Close"])
            if any(math.isnan(v) for v in [o,h,l,c]): continue
            candles.append({"time": int(idx.timestamp()), "open":o,"high":h,"low":l,"close":c})

        cur = float(close.iloc[-1])
        ma200_cur = float(ma200_s.iloc[-1]) if not math.isnan(ma200_s.iloc[-1]) else None
        gain_pct = round((cur - info["entry"]) / info["entry"] * 100, 1)

        return JSONResponse({
            "ticker":   ticker,
            "tier":     info["tier"],
            "size_pct": info["size_pct"],
            "sector":   info["sector"],
            "entry":    info["entry"],
            "strategy": info["strategy"],
            "note":     info["note"],
            "gain_pct": gain_pct,
            "candles":  candles,
            "ema10":    to_line(ema10_s),
            "ema20":    to_line(ema20_s),
            "ma200":    to_line(ma200_s),
            "levels": {
                "entry":  info["entry"],
                "ema20":  round(float(ema20_s.iloc[-1]), 2),
                "ma200":  round(ma200_cur, 2) if ma200_cur else None,
            }
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/health")
async def health():
    date_key = _trade_date_key()
    cached = [p.stem for p in _CACHE_DIR.glob(f"*_{date_key}.json")]
    return {"status": "ok", "trade_date": date_key, "cached_keys": cached,
            "time": datetime.now().isoformat()}
