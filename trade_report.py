#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trade_report.py — enrich 期权跟单交易记录可读化报告

读取:
  output/enrich_journal.jsonl   bot 结构化事件流 (ts: 老行 naive=SGT本地, 新行带 +08:00)
  output/enrich_positions.json  当前持仓快照
  output/enrich_orders.csv      LongPort 券商订单流水 (submitted_at/updated_at 为 naive SGT)
  output/andy_tracked.json      andy 观察中合约 (ts 为 aware UTC)
  output/enrich_history.json / andy_history.json / zhaoge_history.json  Discord 消息 (UTC aware)
  data/enrich_bars/<OSI>.csv    期权 1m K线缓存 (UTC aware)

输出:
  ① 终端中文报告
  ② output/trade_report.html (自包含, 无外部 CDN, 手机可读)

时区约定: 报告一律 SGT 显示并标注; 关键交易行附 (ET xx:xx)。
转换用 zoneinfo, 不手写固定偏移 (美国有夏令时)。
"""
import csv
import html as _html
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
BARS_DIR = ROOT / "data" / "enrich_bars"
HTML_PATH = OUT / "trade_report.html"

SGT = ZoneInfo("Asia/Singapore")
ET = ZoneInfo("America/New_York")

MULT = 100  # 期权合约乘数

# ---------------------------------------------------------------- 时间处理

def parse_ts_any(s, assume_tz="Asia/Singapore"):
    """解析两类时间串:
    - naive  ("2026-07-15T03:55:29" / "2026-07-15 03:55:28") → 按 assume_tz 本地化
    - aware  ("...+00:00" / "...Z")                          → 直接转换
    返回 aware datetime; 解析失败返回 None。
    """
    if s is None:
        return None
    if isinstance(s, datetime):
        dt = s
    else:
        t = str(s).strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(t)
        except ValueError:
            dt = None
            for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(t, f)
                    break
                except ValueError:
                    continue
            if dt is None:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(assume_tz))
    return dt


def fmt_sgt(dt, with_date=True):
    if dt is None:
        return "?"
    d = dt.astimezone(SGT)
    return d.strftime("%m-%d %H:%M:%S") if with_date else d.strftime("%H:%M:%S")


def fmt_et(dt):
    if dt is None:
        return ""
    d = dt.astimezone(ET)
    return f"(ET {d.strftime('%m-%d %H:%M')})"


def fmt_dual(dt):
    """SGT 时间 + ET 括注, 用于关键交易行。"""
    if dt is None:
        return "?"
    return f"{fmt_sgt(dt)} SGT {fmt_et(dt)}"

# ---------------------------------------------------------------- 数据加载 (全部优雅降级)

def load_journal():
    events = []
    p = OUT / "enrich_journal.jsonl"
    if not p.exists():
        return events
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if not isinstance(d, dict):
                    continue
            except Exception:
                continue
            d["_dt"] = parse_ts_any(d.get("ts"), "Asia/Singapore")
            events.append(d)
    except Exception:
        pass
    events.sort(key=lambda e: e["_dt"] or datetime.min.replace(tzinfo=SGT))
    return events


def load_json(path, default):
    try:
        p = Path(path)
        if not p.exists():
            return default
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_orders():
    """返回 (rows, omap)。omap: order_id → row dict。submitted_at 为 naive SGT。"""
    rows, omap = [], {}
    p = OUT / "enrich_orders.csv"
    if not p.exists():
        return rows, omap
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["_sub_dt"] = parse_ts_any(r.get("submitted_at"), "Asia/Singapore")
                r["_upd_dt"] = parse_ts_any(r.get("updated_at"), "Asia/Singapore")
                rows.append(r)
                oid = (r.get("order_id") or "").strip()
                if oid:
                    omap[oid] = r
    except Exception:
        pass
    return rows, omap


def _f(x, default=None):
    """安全转 float。"""
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default

# ---------------------------------------------------------------- 回合重建

SELL_EVS = ("tp_fill", "tp_poll_sell", "stop_fill", "mirror_sell", "close_sell")

SELL_LABEL = {
    "tp_fill": "止盈成交",
    "tp_poll_sell": "轮询止盈卖出",
    "stop_fill": "止损成交",
    "mirror_sell": "镜像跟卖",
    "close_sell": "平仓卖出",
}


def sell_price(e, omap, last_stop_px):
    """卖出成交价: 券商 executed_price 真值优先 → journal px/last → 止损触发现价(估)。
    返回 (价 or None, 来源标签)。"""
    oid = str(e.get("order_id") or "")
    o = omap.get(oid)
    if o is not None:
        px = _f(o.get("executed_price"))
        if px is not None:
            return px, "券商成交价"
    for k in ("px", "last"):
        px = _f(e.get(k))
        if px is not None:
            src = "journal价" if k == "px" else "journal现价(近似)"
            return px, src
    if e.get("ev") == "close_sell" and last_stop_px is not None:
        return last_stop_px, "≈触发价(估)"
    return None, "未知"


def build_rounds(journal, omap):
    """按 OSI 把事件流串成回合。
    规则: entry_submit 开(或续)回合; 累计 fill / sell;
    当 filled>0 且 sold>=filled 时回合闭合。
    返回 (closed_rounds, open_rounds), 每个 round:
      {osi, ticker, entries[], sells[], filled, sold, cost, proceeds,
       px_unknown(bool), est(bool), first_dt, last_dt}
    """
    by_osi = defaultdict(list)
    for e in journal:
        osi = e.get("osi")
        if osi:
            by_osi[osi].append(e)

    closed, opened = [], []
    for osi, evs in by_osi.items():
        cur = None
        last_stop_px = None

        def new_round(e):
            return {
                "osi": osi,
                "ticker": e.get("ticker") or osi.split("2")[0],
                "entries": [], "sells": [],
                "filled": 0.0, "sold": 0.0,
                "cost": 0.0, "proceeds": 0.0,
                "px_unknown": False, "est": False,
                "first_dt": e.get("_dt"), "last_dt": e.get("_dt"),
            }

        for e in evs:
            ev = e.get("ev")
            if ev == "stop_trigger":
                last_stop_px = _f(e.get("last"))
            if cur is None and ev in ("entry_submit", "entry_fill") + SELL_EVS:
                cur = new_round(e)
            if cur is None:
                continue
            cur["last_dt"] = e.get("_dt") or cur["last_dt"]

            if ev == "entry_submit":
                cur["entries"].append({"dt": e.get("_dt"), "kind": "挂单",
                                       "qty": _f(e.get("qty"), 0), "px": _f(e.get("limit")),
                                       "sig": e.get("sig", "")})
            elif ev == "entry_fill":
                q = _f(e.get("qty"), 0) or 0
                avg = _f(e.get("avg"))
                cur["filled"] += q
                if avg is not None:
                    cur["cost"] += avg * MULT * q
                cur["entries"].append({"dt": e.get("_dt"), "kind": "成交",
                                       "qty": q, "px": avg, "sig": ""})
            elif ev in SELL_EVS:
                q = _f(e.get("qty"), 0) or 0
                px, src = sell_price(e, omap, last_stop_px)
                cur["sold"] += q
                if px is None:
                    cur["px_unknown"] = True
                else:
                    cur["proceeds"] += px * MULT * q
                    if src.startswith("≈") or "近似" in src:
                        cur["est"] = True
                cur["sells"].append({"dt": e.get("_dt"), "ev": ev,
                                     "label": SELL_LABEL.get(ev, ev), "qty": q,
                                     "px": px, "src": src,
                                     "reason": e.get("reason", "")})
                # 精确闭合: 卖出张数恰好等于成交张数才算回合结束。
                # (journal 若混入 dry-run 测试行导致数量对不上, 宁可留作
                #  "未闭合回合"提示, 也不拼出一个假盈亏。)
                if cur["filled"] > 0 and abs(cur["sold"] - cur["filled"]) < 1e-9:
                    closed.append(cur)
                    cur = None
        if cur is not None and (cur["entries"] or cur["sells"]):
            opened.append(cur)
    closed.sort(key=lambda r: r["last_dt"] or datetime.min.replace(tzinfo=SGT))
    return closed, opened

# ---------------------------------------------------------------- 事件流描述

def describe_event(e):
    ev = e.get("ev", "?")
    osi = e.get("osi", "")
    tk = e.get("ticker", "")
    q = e.get("qty", "")
    if ev == "entry_submit":
        return f"入场挂单 {osi} {q}张 @限价{e.get('limit')}", True
    if ev == "entry_fill":
        return f"入场成交 {osi} {q}张 @均价{e.get('avg')}", True
    if ev == "tp_place":
        return f"挂止盈单 {osi} {q}张 @{e.get('px')}", False
    if ev == "tp_fill":
        return f"止盈成交 {osi} {q}张 @{e.get('px')}", True
    if ev == "tp_poll_sell":
        return f"轮询止盈卖出 {osi} {q}张 (现价{e.get('last')})", True
    if ev == "stop_place":
        return f"挂止损单 {osi} {q}张 @触发{e.get('trigger')}", False
    if ev == "stop_fill":
        return f"止损成交 {osi} {q}张 @{e.get('px')}", True
    if ev == "stop_trigger":
        return f"触发止损 {osi} 现价{e.get('last')} / 成本{e.get('avg')} (×{e.get('mult')})", True
    if ev == "mirror_sell":
        return f"镜像跟卖 {osi} {q}张", True
    if ev == "close_sell":
        ok = "成功" if e.get("ok") else "失败"
        return f"平仓卖出 {osi} {q}张 ({e.get('reason', '')}) {ok}", True
    if ev == "exit_signal":
        return f"出场信号 {tk} [{e.get('level')}] 持仓{e.get('held')}", True
    if ev == "disambig":
        return f"歧义消歧 {tk or osi} {str(e.get('sig', ''))[:60]}", False
    if ev == "andy_entry":
        sub = "subset✓" if e.get("subset") else "仅观察"
        return f"[andy] 入场 {e.get('osi', tk)} 权利金{e.get('prem')} 止损{e.get('stop')} ({sub})", False
    if ev == "andy_exit":
        return f"[andy] 出场信号 {tk} [{e.get('level')}]", False
    # 未知事件类型: 原样降级展示
    extra = {k: v for k, v in e.items() if k not in ("ev", "ts", "_dt", "sig")}
    return f"{ev} {json.dumps(extra, ensure_ascii=False, default=str)[:80]}", False

# ---------------------------------------------------------------- 分区构建

def section_positions(positions, journal):
    """【enrich 当前持仓】→ list[dict]"""
    rows = []
    if not isinstance(positions, dict):
        return rows
    # journal 里该 OSI 最近一次 tp_place / stop_place 的价位
    last_tp, last_stop = {}, {}
    for e in journal:
        if e.get("ev") == "tp_place" and e.get("osi"):
            last_tp[e["osi"]] = _f(e.get("px"))
        if e.get("ev") == "stop_place" and e.get("osi"):
            last_stop[e["osi"]] = _f(e.get("trigger"))
    for osi, p in positions.items():
        if not isinstance(p, dict):
            continue
        status = p.get("status", "?")
        if status == "closed":
            continue
        filled = _f(p.get("filled"), 0) or 0
        sold = _f(p.get("sold"), 0) or 0
        hold = filled - sold
        avg = _f(p.get("avg"))
        tp_live = bool(p.get("tp_order_id"))
        stop_live = bool(p.get("stop_order_id"))
        tp_px = last_tp.get(osi)
        stop_px = last_stop.get(osi)
        # 无挂单价时按策略规则推算 (止盈+100% / 止损-30%), 标注"理论"
        tp_s = (f"{tp_px:g}" if tp_px is not None else
                (f"≈{avg * 2:g}(理论)" if avg else "?"))
        stop_s = (f"{stop_px:g}" if stop_px is not None else
                  (f"≈{avg * 0.7:g}(理论)" if avg else "?"))
        pend = []
        pend.append(("止盈挂单中" if tp_live else "止盈无挂单(轮询)") + f" @{tp_s}")
        pend.append(("止损挂单中" if stop_live else "止损无挂单(轮询)") + f" @{stop_s}")
        rows.append({
            "ticker": p.get("ticker", "?"), "osi": osi,
            "hold": hold, "filled": filled, "sold": sold,
            "avg": avg, "cost": (avg * MULT * hold) if avg else None,
            "status": {"pending": "待成交", "open": "持仓中"}.get(status, status),
            "expiry": p.get("expiry", "?"), "opened": p.get("opened", "?"),
            "pend": " / ".join(pend),
        })
    return rows


def section_andy(journal, tracked):
    """【andy 观察账本】subset=true 的 andy_entry + 匹配 exits; 未平仓标观察中。"""
    entries = [e for e in journal if e.get("ev") == "andy_entry" and e.get("subset")]
    exits = [e for e in journal if e.get("ev") == "andy_exit"]
    used = set()
    out = []
    for en in entries:
        tk = en.get("ticker")
        my_exits = []
        for i, ex in enumerate(exits):
            if i in used or ex.get("ticker") != tk:
                continue
            if (ex.get("_dt") and en.get("_dt")) and ex["_dt"] < en["_dt"]:
                continue
            my_exits.append(ex)
            used.add(i)
            if ex.get("level") in ("full", "stop", "be"):
                break
        still = isinstance(tracked, dict) and tk in tracked
        out.append({"src": "journal", "ticker": tk, "osi": en.get("osi", ""),
                    "dt": en.get("_dt"), "prem": en.get("prem"), "stop": en.get("stop"),
                    "exits": my_exits,
                    "state": "观察中" if (still or not my_exits) else "已出场"})
    # tracked 里有但 journal 没记到的 (降级兜底)
    seen_tk = {r["ticker"] for r in out}
    if isinstance(tracked, dict):
        for tk, t in tracked.items():
            if tk in seen_tk or not isinstance(t, dict):
                continue
            out.append({"src": "tracked", "ticker": tk, "osi": t.get("osi", ""),
                        "dt": parse_ts_any(t.get("ts"), "UTC"),
                        "prem": t.get("prem"), "stop": t.get("stop"),
                        "exits": [], "state": "观察中"})
    out.sort(key=lambda r: r["dt"] or datetime.min.replace(tzinfo=SGT))
    return out


def section_files():
    """【数据文件索引】"""
    def jsonl_lines(p):
        try:
            return sum(1 for line in open(p, encoding="utf-8") if line.strip())
        except Exception:
            return None

    def csv_rows(p):
        n = jsonl_lines(p)
        return None if n is None else max(0, n - 1)

    def json_len(p):
        d = load_json(p, None)
        return len(d) if hasattr(d, "__len__") else None

    items = [
        ("output/enrich_journal.jsonl", "enrich bot 结构化事件流 (JSONL, 回合重建原料)",
         jsonl_lines(OUT / "enrich_journal.jsonl"), "行"),
        ("output/enrich_positions.json", "当前持仓状态快照 (bot 自维护)",
         json_len(OUT / "enrich_positions.json"), "个合约"),
        ("output/enrich_orders.csv", "LongPort 券商订单流水 (成交价真值来源)",
         csv_rows(OUT / "enrich_orders.csv"), "笔"),
        ("output/andy_tracked.json", "andy 观察中合约 (subset 单)",
         json_len(OUT / "andy_tracked.json"), "个"),
        ("output/enrich_history.json", "enrich 频道 Discord 消息存档 (UTC)",
         json_len(OUT / "enrich_history.json"), "条"),
        ("output/andy_history.json", "andy 频道 Discord 消息存档 (UTC)",
         json_len(OUT / "andy_history.json"), "条"),
        ("output/zhaoge_history.json", "zhaoge 频道 Discord 消息存档 (UTC)",
         json_len(OUT / "zhaoge_history.json"), "条"),
        ("data/enrich_bars/", "期权 1m K线缓存 (每合约一个 csv, ts=UTC)",
         len(list(BARS_DIR.glob("*.csv"))) if BARS_DIR.is_dir() else None, "个合约"),
    ]
    out = []
    for path, desc, n, unit in items:
        cnt = "文件不存在" if n is None else f"{n} {unit}"
        out.append({"path": path, "desc": desc, "count": cnt})
    return out

# ---------------------------------------------------------------- 终端渲染

W = 78

def _hr(title):
    return f"\n{'═' * W}\n【{title}】\n{'─' * W}"


def money(x):
    if x is None:
        return "?"
    return f"{x:+,.0f}" if abs(x) >= 100 else f"{x:+,.2f}"


def render_terminal(pos_rows, closed, open_rounds, journal, andy_rows, file_rows):
    L = []
    now = datetime.now(SGT)
    L.append(f"enrich 交易记录报告  生成于 {now.strftime('%Y-%m-%d %H:%M:%S')} SGT {fmt_et(now)}")

    L.append(_hr("enrich 当前持仓"))
    if not pos_rows:
        L.append("  (无持仓)")
    for r in pos_rows:
        cost = f"${r['cost']:,.0f}" if r["cost"] is not None else "?"
        L.append(f"  {r['ticker']}  {r['osi']}")
        L.append(f"    状态: {r['status']}  持有 {r['hold']:g} 张 (成交{r['filled']:g}/已卖{r['sold']:g})"
                 f"  成本 @{r['avg'] if r['avg'] is not None else '?'} ≈{cost}")
        L.append(f"    到期 {r['expiry']}  信号日 {r['opened']}(UTC)")
        L.append(f"    {r['pend']}")

    L.append(_hr("enrich 已平仓回合"))
    if not closed:
        L.append("  (暂无已平仓回合)")
    total_pnl, pnl_ok = 0.0, True
    for i, r in enumerate(closed, 1):
        pnl = r["proceeds"] - r["cost"]
        pct = (pnl / r["cost"] * 100) if r["cost"] else None
        flag = ""
        if r["px_unknown"]:
            flag = " [部分卖价未知, 盈亏不完整]"
            pnl_ok = False
        elif r["est"]:
            flag = " [含估算价≈]"
        total_pnl += pnl
        pct_s = f"{pct:+.1f}%" if pct is not None else "?"
        L.append(f"  #{i} {r['ticker']}  {r['osi']}  盈亏 ${money(pnl)} ({pct_s}){flag}")
        for en in r["entries"]:
            px = f"@{en['px']:g}" if en["px"] is not None else ""
            L.append(f"      {fmt_dual(en['dt'])}  入场{en['kind']} {en['qty']:g}张 {px}")
        for s in r["sells"]:
            px = f"@{s['px']:g}" if s["px"] is not None else "@?"
            rsn = f" ({s['reason']})" if s["reason"] else ""
            L.append(f"      {fmt_dual(s['dt'])}  {s['label']} {s['qty']:g}张 {px} [{s['src']}]{rsn}")
    if closed:
        note = "" if pnl_ok else " (含卖价未知回合, 仅供参考)"
        L.append(f"  合计盈亏: ${money(total_pnl)}{note}")
    if open_rounds:
        osis = ", ".join(f"{r['osi']}(成交{r['filled']:g}/卖{r['sold']:g})" for r in open_rounds)
        L.append(f"  注: 进行中/未闭合回合 {len(open_rounds)} 个: {osis} — 以〈当前持仓〉为准")

    L.append(_hr("enrich 事件流 (最近 30 条, SGT)"))
    if not journal:
        L.append("  (journal 为空)")
    for e in journal[-30:]:
        desc, key = describe_event(e)
        t = fmt_sgt(e.get("_dt"))
        et = f"  {fmt_et(e.get('_dt'))}" if key else ""
        L.append(f"  {t}  {desc}{et}")

    L.append(_hr("andy 观察账本 (subset 单)"))
    if not andy_rows:
        L.append("  (暂无观察记录)")
    for r in andy_rows:
        L.append(f"  {r['ticker']}  {r['osi']}  权利金{r['prem']}  止损{r['stop']}"
                 f"  {fmt_dual(r['dt'])}  [{r['state']}]")
        for ex in r["exits"]:
            L.append(f"      {fmt_dual(ex.get('_dt'))}  出场信号 [{ex.get('level')}]")

    L.append(_hr("数据文件索引"))
    for r in file_rows:
        L.append(f"  {r['path']:<34} {r['count']:>12}   {r['desc']}")
    L.append("═" * W)
    L.append("时区说明: 报告统一 SGT (新加坡); (ET ...) 为美东时间, 用 zoneinfo 换算(含夏令时)。")
    return "\n".join(L)

# ---------------------------------------------------------------- HTML 渲染

def esc(x):
    return _html.escape(str(x if x is not None else ""))


def render_html(pos_rows, closed, open_rounds, journal, andy_rows, file_rows):
    now = datetime.now(SGT)
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
           background: #0f1419; color: #d8dee5; padding: 14px; font-size: 14px; }
    h1 { font-size: 18px; margin: 6px 0 2px; color: #fff; }
    .sub { color: #7d8590; font-size: 12px; margin-bottom: 14px; }
    h2 { font-size: 15px; margin: 20px 0 8px; color: #79b8ff;
         border-left: 3px solid #79b8ff; padding-left: 8px; }
    .wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    table { border-collapse: collapse; width: 100%; min-width: 480px; font-size: 13px; }
    th, td { border-bottom: 1px solid #263040; padding: 6px 8px; text-align: left;
             white-space: nowrap; }
    th { color: #7d8590; font-weight: 600; background: #161c26; position: sticky; top: 0; }
    td.wrapcell { white-space: normal; min-width: 160px; }
    .pos { color: #56d364; } .neg { color: #f85149; } .dim { color: #7d8590; }
    .tag { display: inline-block; padding: 1px 7px; border-radius: 9px; font-size: 11px; }
    .t-open { background: #1b3a2a; color: #56d364; }
    .t-watch { background: #3a2f1b; color: #e3b341; }
    .t-done { background: #26303c; color: #9aa7b5; }
    .card { background: #161c26; border: 1px solid #263040; border-radius: 8px;
            padding: 10px 12px; margin-bottom: 10px; }
    .note { color: #e3b341; font-size: 12px; margin: 6px 0; }
    .empty { color: #7d8590; padding: 8px 2px; }
    .foot { color: #566270; font-size: 11px; margin-top: 22px; line-height: 1.6; }
    @media (max-width: 480px) { body { padding: 8px; } table { font-size: 12px; } }
    """
    H = []
    H.append("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>")
    H.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    H.append("<title>enrich 交易记录报告</title>")
    H.append(f"<style>{css}</style></head><body>")
    H.append("<h1>enrich 交易记录报告</h1>")
    H.append(f"<div class='sub'>生成于 {esc(now.strftime('%Y-%m-%d %H:%M:%S'))} SGT "
             f"{esc(fmt_et(now))} · LongPort 模拟盘</div>")

    # 当前持仓
    H.append("<h2>enrich 当前持仓</h2>")
    if not pos_rows:
        H.append("<div class='empty'>无持仓</div>")
    else:
        H.append("<div class='wrap'><table><tr><th>票</th><th>合约</th><th>状态</th>"
                 "<th>持有</th><th>成本</th><th>市值成本</th><th>到期</th><th>挂单状态</th></tr>")
        for r in pos_rows:
            cost = f"${r['cost']:,.0f}" if r["cost"] is not None else "?"
            H.append(f"<tr><td><b>{esc(r['ticker'])}</b></td><td>{esc(r['osi'])}</td>"
                     f"<td><span class='tag t-open'>{esc(r['status'])}</span></td>"
                     f"<td>{r['hold']:g} 张</td><td>@{esc(r['avg'])}</td><td>{esc(cost)}</td>"
                     f"<td>{esc(r['expiry'])}</td><td class='wrapcell'>{esc(r['pend'])}</td></tr>")
        H.append("</table></div>")

    # 已平仓回合
    H.append("<h2>enrich 已平仓回合</h2>")
    if not closed:
        H.append("<div class='empty'>暂无已平仓回合</div>")
    total_pnl, pnl_ok = 0.0, True
    for i, r in enumerate(closed, 1):
        pnl = r["proceeds"] - r["cost"]
        total_pnl += pnl
        pct = (pnl / r["cost"] * 100) if r["cost"] else None
        cls = "pos" if pnl >= 0 else "neg"
        pct_s = f"{pct:+.1f}%" if pct is not None else "?"
        flag = ""
        if r["px_unknown"]:
            flag = " <span class='note'>[部分卖价未知, 盈亏不完整]</span>"
            pnl_ok = False
        elif r["est"]:
            flag = " <span class='note'>[含估算价≈]</span>"
        H.append(f"<div class='card'><b>#{i} {esc(r['ticker'])}</b> "
                 f"<span class='dim'>{esc(r['osi'])}</span> — "
                 f"<span class='{cls}'>${money(pnl)} ({pct_s})</span>{flag}")
        H.append("<div class='wrap'><table><tr><th>时间 (SGT)</th><th>ET</th>"
                 "<th>动作</th><th>张数</th><th>价格</th><th>价格来源</th></tr>")
        for en in r["entries"]:
            px = f"@{en['px']:g}" if en["px"] is not None else ""
            H.append(f"<tr><td>{esc(fmt_sgt(en['dt']))}</td>"
                     f"<td class='dim'>{esc(fmt_et(en['dt']))}</td>"
                     f"<td>入场{esc(en['kind'])}</td><td>{en['qty']:g}</td>"
                     f"<td>{esc(px)}</td><td class='dim'>—</td></tr>")
        for s in r["sells"]:
            px = f"@{s['px']:g}" if s["px"] is not None else "@?"
            rsn = f" ({s['reason']})" if s["reason"] else ""
            H.append(f"<tr><td>{esc(fmt_sgt(s['dt']))}</td>"
                     f"<td class='dim'>{esc(fmt_et(s['dt']))}</td>"
                     f"<td>{esc(s['label'] + rsn)}</td><td>{s['qty']:g}</td>"
                     f"<td>{esc(px)}</td><td class='dim'>{esc(s['src'])}</td></tr>")
        H.append("</table></div></div>")
    if closed:
        cls = "pos" if total_pnl >= 0 else "neg"
        note = "" if pnl_ok else " <span class='note'>(含卖价未知回合, 仅供参考)</span>"
        H.append(f"<div>合计盈亏: <b class='{cls}'>${money(total_pnl)}</b>{note}</div>")
    if open_rounds:
        osis = ", ".join(f"{esc(r['osi'])} (成交{r['filled']:g}/卖{r['sold']:g})"
                         for r in open_rounds)
        H.append(f"<div class='note'>进行中/未闭合回合 {len(open_rounds)} 个: {osis} — "
                 f"以〈当前持仓〉为准</div>")

    # 事件流
    H.append("<h2>enrich 事件流 (最近 30 条)</h2>")
    if not journal:
        H.append("<div class='empty'>journal 为空</div>")
    else:
        H.append("<div class='wrap'><table><tr><th>时间 (SGT)</th><th>事件</th><th>ET</th></tr>")
        for e in journal[-30:]:
            desc, key = describe_event(e)
            et = fmt_et(e.get("_dt")) if key else ""
            H.append(f"<tr><td>{esc(fmt_sgt(e.get('_dt')))}</td>"
                     f"<td class='wrapcell'>{esc(desc)}</td>"
                     f"<td class='dim'>{esc(et)}</td></tr>")
        H.append("</table></div>")

    # andy 观察账本
    H.append("<h2>andy 观察账本 (subset 单)</h2>")
    if not andy_rows:
        H.append("<div class='empty'>暂无观察记录</div>")
    else:
        H.append("<div class='wrap'><table><tr><th>票</th><th>合约</th><th>入场时间 (SGT)</th>"
                 "<th>权利金</th><th>止损</th><th>出场</th><th>状态</th></tr>")
        for r in andy_rows:
            exs = "; ".join(f"{fmt_sgt(x.get('_dt'))} [{x.get('level')}]"
                            for x in r["exits"]) or "—"
            tag = "t-watch" if r["state"] == "观察中" else "t-done"
            H.append(f"<tr><td><b>{esc(r['ticker'])}</b></td><td>{esc(r['osi'])}</td>"
                     f"<td>{esc(fmt_sgt(r['dt']))} <span class='dim'>{esc(fmt_et(r['dt']))}</span></td>"
                     f"<td>{esc(r['prem'])}</td><td>{esc(r['stop'])}</td>"
                     f"<td class='wrapcell'>{esc(exs)}</td>"
                     f"<td><span class='tag {tag}'>{esc(r['state'])}</span></td></tr>")
        H.append("</table></div>")

    # 数据文件索引
    H.append("<h2>数据文件索引</h2>")
    H.append("<div class='wrap'><table><tr><th>文件</th><th>规模</th><th>说明</th></tr>")
    for r in file_rows:
        H.append(f"<tr><td>{esc(r['path'])}</td><td>{esc(r['count'])}</td>"
                 f"<td class='wrapcell'>{esc(r['desc'])}</td></tr>")
    H.append("</table></div>")

    H.append("<div class='foot'>时区说明: 报告统一 SGT (Asia/Singapore); ET 为美东时间 "
             "(America/New_York, zoneinfo 换算, 自动处理夏令时)。<br>"
             "卖出成交价优先取 enrich_orders.csv 的 executed_price 真值; journal 缺价时回退, "
             "估算价以 ≈ 标注。所有交易均为 LongPort 模拟盘。</div>")
    H.append("</body></html>")
    return "\n".join(H)

# ---------------------------------------------------------------- main

def main():
    journal = load_journal()
    positions = load_json(OUT / "enrich_positions.json", {})
    orders, omap = load_orders()
    tracked = load_json(OUT / "andy_tracked.json", {})

    closed, open_rounds = build_rounds(journal, omap)
    pos_rows = section_positions(positions, journal)
    andy_rows = section_andy(journal, tracked)
    file_rows = section_files()

    print(render_terminal(pos_rows, closed, open_rounds, journal, andy_rows, file_rows))

    html_doc = render_html(pos_rows, closed, open_rounds, journal, andy_rows, file_rows)
    try:
        OUT.mkdir(exist_ok=True)
        HTML_PATH.write_text(html_doc, encoding="utf-8")
        print(f"\nHTML 报告已写入: {HTML_PATH}")
    except Exception as ex:
        print(f"\n[warn] HTML 写入失败: {ex}")


if __name__ == "__main__":
    main()
