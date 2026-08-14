#!/usr/bin/env python3
"""
elite_bot.py — elite(#elite-alert) 期权跟单 bot ⚠️LongPort 模拟盘 ONLY⚠️

规则 (backtest_elite_casey.py --stops 验证版, 2026-08-13):
  入场: BOUGHT警报 → 限价 ≤ 他报价×1.10, Day单, 20分钟未成交撤单
  出场: ① 跟他的SOLD腿按比例卖(1/2,1/4,1/3,ALL OUT)
        ② 自设硬止损 -60% (60s轮询 last_done)
        ③ 到期日 15:40 ET 无条件强平
        ④ 可选时间止损 TIME_STOP_H 小时无浮盈即撤 (默认关)
  仓位: 单笔预算 POSITION_USD=1000 (全损口径), 张数=预算//(权利金×100), ≥1
  只跟期权BOUGHT(带价), 无视股票行/建议型喊话/lotto字样不过滤(回测含lotto为正)
安全: 启动时模拟盘三重校验(JWT ac/ik + API channel = lb_papertrading), 不过即退出。
"""
import asyncio, json, os, re, socket, sys, threading, time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
_orig_init = aiohttp.TCPConnector.__init__
def _v4_init(self, *a, **kw):
    kw["family"] = socket.AF_INET          # 本机IPv6半残
    _orig_init(self, *a, **kw)
aiohttp.TCPConnector.__init__ = _v4_init
import discord

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
UTC, ET, SGT = timezone.utc, ZoneInfo("America/New_York"), ZoneInfo("Asia/Singapore")

CHANNEL_ID = 1530929738183610378           # #elite-alert
AUTHOR_ID = 1392020997393088542            # 站长转发#2054
LIVE = os.environ.get("ELITE_LIVE", "false").lower() == "true"
POSITION_USD = float(os.environ.get("POSITION_USD", "1000"))
STOP_PCT = float(os.environ.get("STOP_PCT", "0.60"))
LIMIT_MULT = 1.10
ENTRY_TTL_SEC = 1200
TIME_STOP_H = float(os.environ.get("TIME_STOP_H", "0"))    # 0=关
STALE_BUY_SEC = 300                        # 迟到>5分钟的BOUGHT只提醒不追(铁律: 不补迟到单)

STATE_F = ROOT / "output" / "elite_positions.json"
JOURNAL_F = ROOT / "output" / "elite_journal.jsonl"
LOG_F = ROOT / "output" / "elite_bot.log"

MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
B_RE = re.compile(r"\*\*BOUGHT\*\*\s*\|\s*([A-Z]+)\s+([A-Z]+)\s+(\d+)\s+(\d+(?:\.\d+)?)([CP])\s+\$(\d+(?:\.\d+)?)", re.I)
S_RE = re.compile(r"\*\*SOLD\*\*\s*\|\s*([A-Z]+)\s+([A-Z]+)\s+(\d+)\s+(\d+(?:\.\d+)?)([CP])\s+\$(\d+(?:\.\d+)?)\s*(.*)", re.I)


def log(msg):
    line = f"[{datetime.now(SGT):%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_F, "a") as f:
        f.write(line + "\n")


def journal(ev, **data):
    with open(JOURNAL_F, "a") as f:
        f.write(json.dumps(dict(ts=datetime.now(SGT).isoformat(timespec="seconds"), ev=ev, **data),
                           ensure_ascii=False) + "\n")


def paper_triple_check():
    import base64
    from longport.openapi import Config, TradeContext
    tok = os.environ["LONGPORT_ACCESS_TOKEN"]
    p = json.loads(base64.urlsafe_b64decode(tok.split(".")[1] + "=" * (-len(tok.split(".")[1]) % 4)))
    ok1 = p.get("ac") == "lb_papertrading"
    ok2 = str(p.get("ik", "")).startswith("lb_papertrading_")
    t = TradeContext(Config.from_env())
    chs = [c.account_channel for c in t.stock_positions().channels]
    ok3 = all(c == "lb_papertrading" for c in chs) and chs
    log(f"三重校验: ac={p.get('ac')} ik前缀={ok2} channels={chs}")
    if not (ok1 and ok2 and ok3):
        log("❌ 非模拟盘! 拒绝启动"); journal("safety_abort", ac=p.get("ac"), chs=chs)
        sys.exit(1)
    return t


def osi_of(tk, mon, day, strike, right, ref_date):
    mo = MON.get(mon[:3].upper())
    if mon.upper() == "JULY": mo = 7
    exp = date(ref_date.year, mo, int(day))
    if exp < ref_date:
        mo2 = mo + 1 if mo < 12 else 1
        exp = date(ref_date.year + (1 if mo2 == 1 else 0), mo2, int(day))
    return f"{tk}{exp:%y%m%d}{right.upper()}{int(float(strike)*1000):06d}.US", exp


class Book:
    """positions: osi -> {qty, entry, entry_ts, exp, label}"""
    def __init__(self):
        self.pos = {}
        if STATE_F.exists():
            try:
                self.pos = json.loads(STATE_F.read_text())
            except Exception:
                self.pos = {}

    def save(self):
        STATE_F.write_text(json.dumps(self.pos, ensure_ascii=False, default=str))


BOOK = Book()
_lock = threading.Lock()
_trade_ctx = None
_quote_ctx = None


def ctxs():
    global _trade_ctx, _quote_ctx
    if _trade_ctx is None:
        from longport.openapi import Config, TradeContext, QuoteContext
        cfg = Config.from_env()
        _trade_ctx, _quote_ctx = TradeContext(cfg), QuoteContext(cfg)
    return _trade_ctx, _quote_ctx


def submit_buy(osi, label, his_px, exp):
    from longport.openapi import OrderType, OrderSide, TimeInForceType
    t, q = ctxs()
    limit = round(his_px * LIMIT_MULT, 2)
    qty = max(1, int(POSITION_USD // (his_px * 100)))
    if not LIVE:
        log(f"[dry] 买 {osi} x{qty} 限价{limit}"); return
    r = t.submit_order(symbol=osi, order_type=OrderType.LO, side=OrderSide.Buy,
                       submitted_quantity=qty, submitted_price=limit,
                       time_in_force=TimeInForceType.Day, remark="elite-entry")
    journal("entry_submit", osi=osi, qty=qty, limit=limit, his_px=his_px, order_id=str(r.order_id))
    log(f"📥 跟入 {label} x{qty} 限价{limit} (他报{his_px})")
    threading.Timer(ENTRY_TTL_SEC, entry_ttl_check, args=(str(r.order_id), osi, label, his_px, exp, qty)).start()


def entry_ttl_check(order_id, osi, label, his_px, exp, qty):
    from longport.openapi import OrderStatus
    t, _ = ctxs()
    try:
        od = t.order_detail(order_id)
        if od.status in (OrderStatus.Filled, OrderStatus.PartialFilled):
            filled = int(od.executed_quantity or 0) or qty
            px = float(od.executed_price or his_px)
            with _lock:
                BOOK.pos[osi] = dict(qty=filled, entry=px, entry_ts=datetime.now(UTC).isoformat(),
                                     exp=str(exp), label=label)
                BOOK.save()
            journal("entry_fill", osi=osi, qty=filled, px=px)
            log(f"✅ 成交 {label} x{filled} @{px}")
        else:
            t.cancel_order(order_id)
            journal("entry_ttl_cancel", osi=osi)
            log(f"⏱️ 未成交撤单 {label}")
    except Exception as e:
        log(f"ttl检查异常 {osi}: {e}")


def sell_qty(osi, frac_word):
    with _lock:
        p = BOOK.pos.get(osi)
        if not p or p["qty"] <= 0:
            return 0
        q = p["qty"]
    if frac_word == "ALL":
        return q
    f = {"1/2": 0.5, "1/4": 0.25, "1/3": 1/3}.get(frac_word, 1.0)
    return max(1, int(q * f))


def submit_sell(osi, qty, why):
    from longport.openapi import OrderType, OrderSide, TimeInForceType
    t, _ = ctxs()
    if qty <= 0:
        return
    if not LIVE:
        log(f"[dry] 卖 {osi} x{qty} ({why})"); return
    r = t.submit_order(symbol=osi, order_type=OrderType.MO, side=OrderSide.Sell,
                       submitted_quantity=qty, time_in_force=TimeInForceType.Day, remark=why)
    with _lock:
        p = BOOK.pos.get(osi)
        if p:
            p["qty"] -= qty
            if p["qty"] <= 0:
                BOOK.pos.pop(osi, None)
            BOOK.save()
    journal("exit_submit", osi=osi, qty=qty, why=why, order_id=str(r.order_id))
    log(f"📤 卖出 {osi} x{qty} ({why})")


def handle(text, msg_ts):
    t = " ".join(text.split())
    age = (datetime.now(UTC) - msg_ts).total_seconds()
    mb, ms_ = B_RE.search(t), S_RE.search(t)
    if mb:
        tk, mon, day, st, r, px = mb.groups()
        if mon.upper() not in MON and mon.upper() != "JULY":
            return
        osi, exp = osi_of(tk, mon, day, st, r, msg_ts.date())
        label = f"{tk} {exp:%m/%d} {st}{r.upper()}"
        if age > STALE_BUY_SEC:
            journal("stale_buy_skipped", osi=osi, age=age)
            log(f"⏰ 迟到{age:.0f}s 只记不追: {label}"); return
        with _lock:
            if osi in BOOK.pos:
                log(f"已持仓 {label}, 跳过重复BOUGHT"); return
        submit_buy(osi, label, float(px), exp)
    elif ms_:
        tk, mon, day, st, r, px, tail = ms_.groups()
        if mon.upper() not in MON and mon.upper() != "JULY":
            return
        osi, _ = osi_of(tk, mon, day, st, r, msg_ts.date())
        up = tail.upper()
        frac = "ALL" if "ALL OUT" in up else ("1/2" if "1/2" in up else ("1/4" if "1/4" in up else ("1/3" if "1/3" in up else "ALL")))
        q = sell_qty(osi, frac)
        if q:
            submit_sell(osi, q, f"mirror-{frac}")
        else:
            journal("sold_no_position", osi=osi)


def manage_loop():
    """60s轮询: 止损-60% / 时间止损 / 到期强平。"""
    from longport.openapi import Config, QuoteContext
    while True:
        try:
            time.sleep(60)
            with _lock:
                snapshot = dict(BOOK.pos)
            if not snapshot:
                continue
            _, q = ctxs()
            now_et = datetime.now(ET)
            quotes = {r.symbol: float(r.last_done) for r in q.option_quote(list(snapshot))}
            for osi, p in snapshot.items():
                last = quotes.get(osi)
                if last is None:
                    continue
                entry = p["entry"]
                exp = date.fromisoformat(str(p["exp"]))
                if now_et.date() >= exp and now_et.time() >= datetime.strptime("15:40", "%H:%M").time():
                    submit_sell(osi, p["qty"], "expiry-force"); journal("expiry_force", osi=osi); continue
                if last <= entry * (1 - STOP_PCT):
                    submit_sell(osi, p["qty"], f"stop-{int(STOP_PCT*100)}pct")
                    journal("stop_trigger", osi=osi, last=last, entry=entry); continue
                if TIME_STOP_H > 0:
                    ets = datetime.fromisoformat(p["entry_ts"])
                    if (datetime.now(UTC) - ets).total_seconds() > TIME_STOP_H * 3600 and last < entry:
                        submit_sell(osi, p["qty"], f"timestop-{TIME_STOP_H}h")
                        journal("timestop_trigger", osi=osi, last=last, entry=entry)
        except Exception as e:
            log(f"manage异常: {e}")


def main():
    if "DISCORD_BOT_TOKEN" not in os.environ:
        log("无DISCORD_BOT_TOKEN"); sys.exit(1)
    paper_triple_check()
    log(f"启动 elite跟单bot LIVE={LIVE} 预算${POSITION_USD}/笔 止损-{STOP_PCT*100:.0f}% 时损={TIME_STOP_H or '关'}")
    threading.Thread(target=manage_loop, daemon=True).start()

    proxy = os.environ.get("HTTPS_PROXY")
    if not proxy:
        try:
            s = socket.create_connection(("127.0.0.1", 7897), timeout=1); s.close()
            proxy = "http://127.0.0.1:7897"
        except OSError:
            proxy = None
    intents = discord.Intents.default(); intents.message_content = True
    client = discord.Client(intents=intents, proxy=proxy)

    @client.event
    async def on_ready():
        log(f"✅ Discord已连 {client.user} | 监听 #elite-alert")

    @client.event
    async def on_message(m):
        if m.channel.id != CHANNEL_ID or m.author.id != AUTHOR_ID or not m.content:
            return
        ts = m.created_at if m.created_at.tzinfo else m.created_at.replace(tzinfo=UTC)
        await asyncio.to_thread(handle, m.content, ts)

    # SGT 04:10 自动退出 (美股收盘后), launchd 每晚重启
    async def curfew():
        while True:
            await asyncio.sleep(60)
            now = datetime.now(SGT)
            if now.hour == 4 and now.minute >= 10:
                log("收盘窗口, 退出"); await client.close(); return

    async def runner():
        asyncio.ensure_future(curfew())
        await client.start(os.environ["DISCORD_BOT_TOKEN"])

    asyncio.run(runner())


if __name__ == "__main__":
    main()
