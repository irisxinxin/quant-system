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


def fetch_marks(osis):
    """实时期权价 {osi: (价, 来源)}。OPRA实时优先; 失败回退本地K线最后收盘; 再失败缺省。"""
    marks = {}
    osis = [o for o in dict.fromkeys(osis) if o]
    if not osis:
        return marks
    try:
        from longport.openapi import Config, QuoteContext
        q = QuoteContext(Config.from_env())
        for i in range(0, len(osis), 20):
            for o in q.option_quote(osis[i:i + 20]):
                px = _f(o.last_done)
                if px:
                    marks[o.symbol] = (px, "实时")
    except Exception:
        pass
    for osi in osis:
        if osi in marks:
            continue
        f = BARS_DIR / f"{osi}.csv"
        try:
            last = None
            with open(f, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    last = r
            if last and _f(last.get("c")) is not None:
                marks[osi] = (_f(last["c"]), "K线收盘")
        except Exception:
            continue
    return marks

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
            "mark": None, "mark_src": "", "upnl": None, "upct": None,
        })
    return rows


def attach_marks(pos_rows, marks):
    """给持仓行挂实时价与浮动盈亏。"""
    for r in pos_rows:
        m = marks.get(r["osi"])
        if not m or not r.get("avg") or r["hold"] <= 0:
            continue
        px, src = m
        r["mark"], r["mark_src"] = px, src
        r["upnl"] = (px - r["avg"]) * MULT * r["hold"]
        r["upct"] = (px / r["avg"] - 1) * 100


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
        if r.get("mark") is not None:
            L.append(f"    现价 {r['mark']:g} ({r['mark_src']})  浮动盈亏 ${r['upnl']:+,.0f} ({r['upct']:+.1f}%)")
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


def pnl_html(v, pct=None, big=False):
    """盈亏着色 (长桥习惯: 红=盈利/绿=亏损), 带显式+/−号, 颜色从不单独承载含义。"""
    if v is None:
        return "<span class='dim'>—</span>"
    cls = "up" if v >= 0 else "dn"
    arrow = "▲" if v >= 0 else "▼"
    p = f" ({pct:+.1f}%)" if pct is not None else ""
    sz = " style='font-size:1.05em'" if big else ""
    return f"<span class='{cls} num'{sz}>{arrow} ${v:+,.0f}{p}</span>"


def render_html(pos_rows, closed, open_rounds, journal, andy_rows, file_rows, marks):
    now = datetime.now(SGT)
    css = """
    * { box-sizing: border-box; margin: 0; padding: 0; }
    :root { --bg:#0d0d0d; --surface:#1a1a19; --surface2:#222221; --ink:#ffffff;
            --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a;
            --border:rgba(255,255,255,0.10); --accent:#3987e5;
            --up:#e66767; --dn:#0ca30c; --warn:#fab219; }
    body { font-family: system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
           background: var(--bg); color: var(--ink2); padding: 16px; font-size: 14px;
           max-width: 860px; margin: 0 auto; }
    h1 { font-size: 19px; margin: 4px 0 2px; color: var(--ink); font-weight: 700; }
    .sub { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .legend { font-size: 12px; color: var(--muted); margin-bottom: 14px; }
    .legend .up, .legend .dn { font-weight: 700; }
    h2 { font-size: 15px; margin: 22px 0 10px; color: var(--ink); font-weight: 700; }
    h2 small { color: var(--muted); font-weight: 400; font-size: 12px; }
    .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
             gap: 10px; margin: 14px 0 4px; }
    .tile { background: var(--surface); border: 1px solid var(--border);
            border-radius: 10px; padding: 12px 14px; }
    .tile .k { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .tile .v { font-size: 22px; font-weight: 700; color: var(--ink); }
    .up { color: var(--up); } .dn { color: var(--dn); } .dim { color: var(--muted); }
    .num { font-variant-numeric: tabular-nums; }
    .card { background: var(--surface); border: 1px solid var(--border);
            border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; }
    .card .head { display: flex; justify-content: space-between; align-items: baseline;
                  flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
    .card .title { font-size: 15px; font-weight: 700; color: var(--ink); }
    .card .osi { color: var(--muted); font-size: 11px; }
    .kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
          gap: 6px 14px; font-size: 13px; }
    .kv .k { color: var(--muted); font-size: 11px; }
    .kv .v { color: var(--ink2); }
    .wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left;
             white-space: nowrap; }
    td.r, th.r { text-align: right; }
    th { color: var(--muted); font-weight: 600; font-size: 12px; }
    tr:last-child td { border-bottom: none; }
    td.wrapcell { white-space: normal; min-width: 150px; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px;
           border: 1px solid var(--border); }
    .t-open { color: var(--warn); }
    .t-watch { color: var(--accent); }
    .t-done { color: var(--muted); }
    .note { color: var(--warn); font-size: 12px; margin: 6px 0; }
    .empty { color: var(--muted); padding: 10px 2px; }
    .foot { color: var(--muted); font-size: 11px; margin-top: 26px; line-height: 1.7;
            border-top: 1px solid var(--line); padding-top: 10px; }
    @media (max-width: 480px) { body { padding: 10px; } .tile .v { font-size: 19px; } }
    """
    # 汇总数字
    realized = sum(r["proceeds"] - r["cost"] for r in closed) if closed else 0.0
    unreal_vals = [r["upnl"] for r in pos_rows if r.get("upnl") is not None]
    unreal = sum(unreal_vals) if unreal_vals else None
    n_watch = len(andy_rows)

    H = []
    H.append("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>")
    H.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    H.append("<title>交易战报</title>")
    H.append(f"<style>{css}</style></head><body>")
    H.append("<h1>📊 交易战报 · LongPort 模拟盘</h1>")
    H.append(f"<div class='sub'>生成于 {esc(now.strftime('%Y-%m-%d %H:%M:%S'))} SGT "
             f"{esc(fmt_et(now))} · 静态快照, 双击 战报.command 刷新</div>")
    H.append("<div class='legend'>配色: <span class='up'>▲红=盈利</span> / "
             "<span class='dn'>▼绿=亏损</span> (长桥习惯), 数字自带±号</div>")

    # ── 总览瓷砖 ──
    H.append("<div class='tiles'>")
    def tile(k, v_html):
        H.append(f"<div class='tile'><div class='k'>{k}</div><div class='v'>{v_html}</div></div>")
    tile("浮动盈亏 (持仓中)", pnl_html(unreal) if unreal is not None else "<span class='dim'>—</span>")
    tile("已实现盈亏", pnl_html(realized) if closed else "<span class='dim'>$0</span>")
    tile("持仓", f"{len(pos_rows)} 笔")
    tile("andy 观察", f"{n_watch} 笔")
    H.append("</div>")

    # ── 当前持仓卡片 ──
    H.append("<h2>💼 当前持仓 <small>enrich 实盘</small></h2>")
    if not pos_rows:
        H.append("<div class='empty'>无持仓</div>")
    for r in pos_rows:
        mark_s = (f"{r['mark']:g} <span class='dim'>({esc(r['mark_src'])})</span>"
                  if r.get("mark") is not None else "<span class='dim'>无行情</span>")
        H.append("<div class='card'>")
        H.append(f"<div class='head'><span class='title'>{esc(r['ticker'])} "
                 f"{esc(r['expiry'])} <span class='dim'>×{r['hold']:g}张</span></span>"
                 f"{pnl_html(r.get('upnl'), r.get('upct'), big=True)}</div>")
        H.append("<div class='kv'>"
                 f"<div><div class='k'>成本</div><div class='v num'>@{esc(r['avg'])} ≈${r['cost']:,.0f}</div></div>"
                 f"<div><div class='k'>现价</div><div class='v num'>{mark_s}</div></div>"
                 f"<div><div class='k'>止盈/止损</div><div class='v num'>{esc(r['pend'].replace('止盈无挂单(轮询) ','止盈 ').replace('止损无挂单(轮询) ','止损 ').replace('止盈挂单中 ','止盈✓ ').replace('止损挂单中 ','止损✓ '))}</div></div>"
                 f"<div><div class='k'>状态</div><div class='v'><span class='tag t-open'>{esc(r['status'])}</span></div></div>"
                 "</div>")
        H.append(f"<div class='osi' style='margin-top:6px'>{esc(r['osi'])} · 信号日 {esc(r['opened'])}</div>")
        H.append("</div>")

    # ── 已平仓回合 ──
    H.append("<h2>✅ 已平仓回合</h2>")
    if not closed:
        H.append("<div class='empty'>暂无 (第一笔平仓后自动出现)</div>")
    pnl_ok = True
    for i, r in enumerate(closed, 1):
        pnl = r["proceeds"] - r["cost"]
        pct = (pnl / r["cost"] * 100) if r["cost"] else None
        unit_cost = (r["cost"] / r["filled"] / MULT) if r["filled"] else None
        flag = ""
        if r["px_unknown"]:
            flag = " <span class='note'>[部分卖价未知]</span>"
            pnl_ok = False
        elif r["est"]:
            flag = " <span class='note'>[含估算≈]</span>"
        H.append(f"<div class='card'><div class='head'>"
                 f"<span class='title'>#{i} {esc(r['ticker'])}</span>"
                 f"{pnl_html(pnl, pct, big=True)}{flag}</div>")
        H.append("<div class='wrap'><table><tr><th>时间 SGT</th><th>动作</th>"
                 "<th class='r'>张</th><th class='r'>价格</th><th class='r'>本笔盈亏</th></tr>")
        for en in r["entries"]:
            px = f"{en['px']:g}" if en["px"] is not None else ""
            H.append(f"<tr><td class='num'>{esc(fmt_sgt(en['dt']))}</td>"
                     f"<td>🟦 入场{esc(en['kind'])}</td><td class='r num'>{en['qty']:g}</td>"
                     f"<td class='r num'>{esc(px)}</td><td class='r dim'>—</td></tr>")
        for s in r["sells"]:
            px = f"{s['px']:g}" if s["px"] is not None else "?"
            rsn = f" ({s['reason']})" if s["reason"] else ""
            leg = ((s["px"] - unit_cost) * MULT * s["qty"]
                   if (s["px"] is not None and unit_cost is not None) else None)
            H.append(f"<tr><td class='num'>{esc(fmt_sgt(s['dt']))}</td>"
                     f"<td>{esc(s['label'] + rsn)}</td><td class='r num'>{s['qty']:g}</td>"
                     f"<td class='r num'>{esc(px)}</td><td class='r'>{pnl_html(leg)}</td></tr>")
        H.append("</table></div></div>")
    if closed:
        note = "" if pnl_ok else " <span class='note'>(含卖价未知回合)</span>"
        H.append(f"<div style='margin:4px 0 2px'>合计已实现: {pnl_html(realized, big=True)}{note}</div>")
    if open_rounds:
        osis = ", ".join(f"{esc(r['osi'])}" for r in open_rounds)
        H.append(f"<div class='note'>进行中回合 {len(open_rounds)} 个 ({osis}) — 见〈当前持仓〉</div>")

    # ── 信号源历史成绩 (逐笔回测) ──
    hist = load_json(OUT / "signal_history.json", {})
    H.append("<h2>📜 信号源历史成绩 <small>按跟单规则逐笔模拟 · 真实K线</small></h2>")
    if not hist:
        H.append("<div class='empty'>暂无 (跑一次 python3 signal_history.py 生成)</div>")
    for src_key, src_name, rule in (
            ("enrich", "enrich (实盘同款规则: 2张·止盈2x·镜像·止损-30%)", ""),
            ("andy", "andy 波段+止损子集 (他的止损+BE·$1万/笔)", "")):
        rows_h = hist.get(src_key) or []
        if not rows_h:
            continue
        done = [r for r in rows_h if r.get("status") in ("closed", "open")]
        skipped_n = len(rows_h) - len(done)
        tot = sum(r.get("pnl", 0) for r in done)
        cost_sum = sum(r.get("cost", 0) for r in done) or 1
        wins = sum(1 for r in done if r.get("pnl", 0) > 0)
        H.append(f"<div class='card'><div class='head'><span class='title'>{esc(src_name)}</span>"
                 f"{pnl_html(tot, tot / cost_sum * 100, big=True)}</div>")
        H.append(f"<div class='sub'>可测 {len(done)} 笔 (胜 {wins} / 负 {len(done)-wins})"
                 f" · 无数据/未成交 {skipped_n} 笔 (详见折叠)"
                 f" · 数据截至 {esc(str(hist.get('generated', ''))[:16])}</div>")
        if done:
            H.append("<div class='wrap'><table><tr><th>日期 SGT</th><th>合约</th>"
                     "<th class='r'>入场</th><th>出场路径</th><th class='r'>盈亏</th></tr>")
            for r in sorted(done, key=lambda x: x.get("ts", "")):
                dt = parse_ts_any(r.get("ts"), "UTC")
                sells = r.get("sells") or []
                path = "; ".join(f"{s['qty']:g}@{s['px']:g}·{str(s['why'])[:6]}" for s in sells[:3])
                openmark = " <span class='tag t-watch'>持仓中</span>" if r["status"] == "open" else ""
                H.append(f"<tr><td class='num'>{esc(fmt_sgt(dt))}</td>"
                         f"<td><b>{esc(r.get('label', ''))}</b>{openmark}</td>"
                         f"<td class='r num'>{esc(r.get('entry', r.get('signal_px', '')))}</td>"
                         f"<td class='wrapcell dim'>{esc(path)}</td>"
                         f"<td class='r'>{pnl_html(r.get('pnl'), r.get('pct'))}</td></tr>")
            H.append("</table></div>")
        sk = [r for r in rows_h if r.get("status") not in ("closed", "open")]
        if sk:
            H.append("<details><summary class='dim' style='cursor:pointer;font-size:12px'>"
                     f"未成交/撤单/无K线数据 {len(sk)} 笔 (点开)</summary>"
                     "<div class='wrap' style='margin-top:6px'><table>")
            ST = {"no_fill": "限价未触及", "cancelled": "先出场撤单", "no_data": "K线已被长桥清除"}
            for r in sorted(sk, key=lambda x: x.get("ts", "")):
                dt = parse_ts_any(r.get("ts"), "UTC")
                H.append(f"<tr><td class='num'>{esc(fmt_sgt(dt))}</td>"
                         f"<td class='dim'>{esc(r.get('label', ''))}</td>"
                         f"<td class='dim'>{esc(ST.get(r.get('status'), r.get('status')))}</td></tr>")
            H.append("</table></div></details>")
        H.append("</div>")

    # ── andy 观察账本 ──
    H.append("<h2>📒 andy 观察账本 <small>波段+止损子集 · 只记录不下单</small></h2>")
    if not andy_rows:
        H.append("<div class='empty'>等他下一条合格信号 (波段+带止损), 自动出现在这里</div>")
    else:
        H.append("<div class='wrap'><table><tr><th>票</th><th>入场 SGT</th>"
                 "<th class='r'>喊单价</th><th class='r'>他的止损</th><th class='r'>现价</th>"
                 "<th class='r'>若跟涨跌</th><th>他的出场</th><th>状态</th></tr>")
        for r in andy_rows:
            exs = "; ".join(f"{fmt_sgt(x.get('_dt'), with_date=False)}[{x.get('level')}]"
                            for x in r["exits"]) or "—"
            tag = "t-watch" if r["state"] == "观察中" else "t-done"
            m = marks.get(r["osi"])
            prem = _f(r.get("prem"))
            mark_s, move = "<span class='dim'>—</span>", "<span class='dim'>—</span>"
            if m and prem:
                mark_s = f"{m[0]:g}"
                mv = (m[0] / prem - 1) * 100
                cls = "up" if mv >= 0 else "dn"
                move = f"<span class='{cls} num'>{mv:+.0f}%</span>"
            H.append(f"<tr><td><b>{esc(r['ticker'])}</b></td>"
                     f"<td class='num'>{esc(fmt_sgt(r['dt']))}</td>"
                     f"<td class='r num'>{esc(r['prem'])}</td><td class='r num'>{esc(r['stop'])}</td>"
                     f"<td class='r num'>{mark_s}</td><td class='r'>{move}</td>"
                     f"<td class='wrapcell'>{esc(exs)}</td>"
                     f"<td><span class='tag {tag}'>{esc(r['state'])}</span></td></tr>")
        H.append("</table></div>")

    # ── 事件流 ──
    H.append("<h2>🕒 事件流 <small>最近 30 条 · SGT</small></h2>")
    if not journal:
        H.append("<div class='empty'>journal 为空</div>")
    else:
        H.append("<div class='card'><div class='wrap'><table>")
        ICON = {"entry_submit": "🟦", "entry_fill": "🟦", "tp_place": "🎯", "tp_fill": "💰",
                "tp_poll_sell": "💰", "stop_place": "🛡️", "stop_fill": "🛑", "stop_trigger": "🛑",
                "mirror_sell": "🪞", "close_sell": "🔻", "exit_signal": "🟠", "disambig": "🔍",
                "andy_entry": "📒", "andy_exit": "📒"}
        for e in reversed(journal[-30:]):
            desc, key = describe_event(e)
            ic = ICON.get(e.get("ev", ""), "·")
            et = f" <span class='dim'>{esc(fmt_et(e.get('_dt')))}</span>" if key else ""
            H.append(f"<tr><td class='num' style='width:110px'>{esc(fmt_sgt(e.get('_dt')))}</td>"
                     f"<td class='wrapcell'>{ic} {esc(desc)}{et}</td></tr>")
        H.append("</table></div></div>")

    # ── 数据文件索引 (折叠) ──
    H.append("<details style='margin-top:18px'><summary class='dim' style='cursor:pointer'>"
             "📁 数据文件索引 (点开)</summary><div class='wrap' style='margin-top:8px'><table>")
    for r in file_rows:
        H.append(f"<tr><td>{esc(r['path'])}</td><td class='num'>{esc(r['count'])}</td>"
                 f"<td class='wrapcell dim'>{esc(r['desc'])}</td></tr>")
    H.append("</table></div></details>")

    H.append("<div class='foot'>配色遵循长桥习惯: 红=盈利/上涨, 绿=亏损/下跌, 且数字均带±号与▲▼。"
             "时间统一 SGT, (ET ...) 为美东, zoneinfo 换算含夏令时。<br>"
             "卖出价优先取券商 executed_price 真值; 现价来源标注: 实时=OPRA / K线收盘=归档数据。"
             "所有交易均为 LongPort 模拟盘。</div>")
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

    # 实时行情: 持仓 + andy观察合约 (失败回退K线收盘, 再失败显示无行情)
    marks = fetch_marks([r["osi"] for r in pos_rows] + [r["osi"] for r in andy_rows])
    attach_marks(pos_rows, marks)

    print(render_terminal(pos_rows, closed, open_rounds, journal, andy_rows, file_rows))

    html_doc = render_html(pos_rows, closed, open_rounds, journal, andy_rows, file_rows, marks)
    try:
        OUT.mkdir(exist_ok=True)
        HTML_PATH.write_text(html_doc, encoding="utf-8")
        print(f"\nHTML 报告已写入: {HTML_PATH}")
    except Exception as ex:
        print(f"\n[warn] HTML 写入失败: {ex}")


if __name__ == "__main__":
    main()
