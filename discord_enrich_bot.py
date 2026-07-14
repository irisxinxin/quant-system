#!/usr/bin/env python3
"""
discord_enrich_bot.py — 监听 Discord #期权-波段-enrich 信号 → 解析 → LongPort 模拟盘买期权 + 仓位管理。

链路: 站长转发(bot id锁死) 发英文原文 → enrich_parser 严格五要素解析
      → BUY: 限价买入 N 张 → 成交后自动挂止盈 → EXIT信号跟随平仓 → 到期日强平

🔒 安全护栏 (缺一不跑):
  1. 启动时 LongPort 模拟盘三重校验 (JWT.ac / JWT.ik / API channel 全=lb_papertrading), 不过即退出
  2. 只认 频道ID+作者ID 白名单 (昵称可仿冒, ID不可)
  3. 默认 DRY_RUN; ENRICH_LIVE=true 才真下(仍是模拟盘)
  4. 去重: 同一期权同一天只开一次; 消息ID去重
  5. 限价单 only, 单张权利金>MAX_PREMIUM 拒绝; 已到期信号跳过

仓位管理 (无期权行情权限 → 全部不依赖盯价):
  · 止盈: 入场成交 → 挂 GTC 限价卖 (TP_MULT×成本, 默认2.0=+100%), 卖出 半仓(2张卖1张; 1张全卖)
  · 出场跟随: 站长 EXIT 信号 (scaling out/all out/stopped...) → 撤止盈单 + 剩仓市价全平
  · 到期强平: 到期日 15:40 ET 剩仓市价全平, 防归零/行权
  · 天然风控: 买期权最大亏=权利金 (每笔 ≤ MAX_PREMIUM×100×张数)

用法:
  python3 discord_enrich_bot.py                                  # DRY_RUN
  ENRICH_LIVE=true OPTION_CONTRACTS=2 python3 discord_enrich_bot.py   # 模拟盘真下单
环境: DISCORD_BOT_TOKEN(必须) / OPTION_CONTRACTS(默认1) / MAX_PREMIUM(默认5.0) / TP_MULT(默认2.0)
      / DISCORD_WEBHOOK_URL(可选回报推送)
"""
import os, sys, json, base64
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")

import discord
from discord.ext import tasks
from enrich_parser import parse_signal, to_longport_symbol
from notify import push_discord

# ── 白名单 (2026-07-14 实测抓取, 锁ID) ──
CHANNEL_ID = 1392361900217602108          # #期权-波段-enrich
AUTHOR_ID  = 1392020997393088542          # 站长转发 (bot)

LIVE = os.environ.get("ENRICH_LIVE", "").lower() == "true"
CONTRACTS = int(os.environ.get("OPTION_CONTRACTS", "1"))
MAX_PREMIUM = float(os.environ.get("MAX_PREMIUM", "5.0"))
TP_MULT = float(os.environ.get("TP_MULT", "2.0"))     # 止盈倍数 (2.0 = +100%)
ET = ZoneInfo("America/New_York")

OUT = Path(__file__).parent / "output"
SEEN_JSON = OUT / "enrich_seen.json"
POS_JSON = OUT / "enrich_positions.json"
LOG = OUT / "enrich_bot.log"

_trade_ctx = None


def log(msg: str):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        OUT.mkdir(exist_ok=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save(p: Path, d: dict):
    try:
        OUT.mkdir(exist_ok=True)
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    except Exception:
        pass


def verify_paper_trading() -> bool:
    """铁律: 三处独立验证全是 lb_papertrading 才放行。"""
    try:
        tok = os.environ["LONGPORT_ACCESS_TOKEN"]
        part = tok.split(".")[1]
        p = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        ac, ik = p.get("ac"), p.get("ik", "")
        from longport.openapi import Config, TradeContext
        global _trade_ctx
        _trade_ctx = TradeContext(Config.from_env())
        chans = {c.account_channel for c in _trade_ctx.stock_positions().channels}
        ok = ac == "lb_papertrading" and ik.startswith("lb_papertrading_") and chans == {"lb_papertrading"}
        log(f"模拟盘校验: ac={ac} ik前缀={ik[:16]} channels={chans} → {'✅通过' if ok else '❌不通过'}")
        return ok
    except Exception as e:
        log(f"模拟盘校验异常: {e}")
        return False


# ── 下单原语 (全部模拟盘) ──

def _submit(osi: str, side_buy: bool, qty: int, price: float | None, tif_gtc=False, remark="enrich"):
    """限价(有price)/市价(None)。返回 (ok, order_id或错误)。"""
    from longport.openapi import OrderType, OrderSide, TimeInForceType
    try:
        kw = dict(symbol=osi,
                  side=OrderSide.Buy if side_buy else OrderSide.Sell,
                  submitted_quantity=Decimal(str(qty)),
                  time_in_force=TimeInForceType.GoodTilCanceled if tif_gtc else TimeInForceType.Day,
                  remark=remark)
        if price is not None:
            kw.update(order_type=OrderType.LO, submitted_price=Decimal(f"{price:.2f}"))
        else:
            kw.update(order_type=OrderType.MO)
        resp = _trade_ctx.submit_order(**kw)
        return True, resp.order_id
    except Exception as e:
        return False, str(e)


def _cancel(order_id: str):
    try:
        _trade_ctx.cancel_order(order_id)
        return True
    except Exception as e:
        log(f"   撤单失败 {order_id}: {e}")
        return False


def _order_state(order_id: str):
    """(状态名, 已成交张数, 成交均价) — 只读轮询, 不需要行情权限。"""
    try:
        od = _trade_ctx.order_detail(order_id)
        st = str(od.status).split(".")[-1]        # Filled/New/PartialFilled/Canceled/Expired/Rejected
        exq = int(od.executed_quantity or 0)
        exp = float(od.executed_price) if od.executed_price else 0.0
        return st, exq, exp
    except Exception as e:
        log(f"   查单失败 {order_id}: {e}")
        return None, 0, 0.0


# ── 仓位管理 ──

def close_position(positions: dict, osi: str, reason: str):
    """撤所有挂单 + 剩仓市价全平。"""
    p = positions.get(osi)
    if not p or p["status"] == "closed":
        return
    log(f"🔻 平仓 {osi} ({reason})")
    if p.get("tp_order_id"):
        _cancel(p["tp_order_id"])
        p["tp_order_id"] = None
    if p["status"] == "pending" and p.get("entry_order_id"):
        _cancel(p["entry_order_id"])   # 未成交部分撤掉
    remain = p.get("filled", 0) - p.get("sold", 0)
    if remain > 0:
        ok, r = _submit(osi, side_buy=False, qty=remain, price=None, remark=f"exit-{reason[:12]}")
        log(f"   市价卖 {remain} 张: {'✅' + str(r) if ok else '❌' + str(r)}")
        push_discord(f"🔻 enrich平仓 {osi} ×{remain}张 ({reason}) {'✅' if ok else '❌' + str(r)}")
    p["status"] = "closed"
    _save(POS_JSON, positions)


def manage_positions(positions: dict):
    """轮询: 入场单成交→挂止盈; 止盈成交→记账; 到期日强平。"""
    now_et = datetime.now(ET)
    for osi, p in list(positions.items()):
        if p["status"] == "closed":
            continue
        # ① 入场单状态
        if p["status"] == "pending":
            st, exq, exp = _order_state(p["entry_order_id"])
            if st is None:
                continue
            if exq > p.get("filled", 0):
                p["filled"], p["avg"] = exq, exp
                log(f"📥 {osi} 入场成交 {exq}张 @ ${exp}")
                push_discord(f"📥 enrich成交 {osi} ×{exq}张 @ ${exp} (成本${exp*100*exq:.0f})")
            if st == "Filled" or (st in ("Canceled", "Expired", "Rejected") and exq > 0):
                p["status"] = "open"
                tp_qty = max(1, p["filled"] // 2)
                tp_px = round(p["avg"] * TP_MULT, 2)
                ok, r = _submit(osi, side_buy=False, qty=tp_qty, price=tp_px, tif_gtc=True, remark="tp")
                if ok:
                    p["tp_order_id"], p["tp_qty"] = r, tp_qty
                    log(f"🎯 {osi} 挂止盈: 卖{tp_qty}张 @ ${tp_px} (+{(TP_MULT-1)*100:.0f}%)")
                    push_discord(f"🎯 enrich止盈单已挂 {osi} 卖{tp_qty}张 @ ${tp_px}")
                else:
                    log(f"⚠️ {osi} 止盈挂单失败: {r} (剩靠出场跟随+到期强平)")
            elif st in ("Canceled", "Expired", "Rejected"):
                log(f"🗑️ {osi} 入场未成交已失效 ({st})")
                p["status"] = "closed"
            _save(POS_JSON, positions)
        # ② 止盈单状态
        if p["status"] == "open" and p.get("tp_order_id"):
            st, exq, exp = _order_state(p["tp_order_id"])
            if st == "Filled":
                p["sold"] = p.get("sold", 0) + p.get("tp_qty", 0)
                p["tp_order_id"] = None
                log(f"💰 {osi} 止盈成交 {p.get('tp_qty')}张 @ ${exp}")
                push_discord(f"💰 enrich止盈成交 {osi} ×{p.get('tp_qty')}张 @ ${exp} — 剩{p['filled']-p['sold']}张跑趋势")
                if p["filled"] - p["sold"] <= 0:
                    p["status"] = "closed"
                _save(POS_JSON, positions)
        # ③ 到期日强平 (15:40 ET 后)
        if p["status"] in ("pending", "open"):
            try:
                exp_d = date.fromisoformat(p["expiry"])
                if now_et.date() >= exp_d and (now_et.hour, now_et.minute) >= (15, 40):
                    close_position(positions, osi, "到期强平")
            except Exception:
                pass


# ── 信号处理 ──

def handle(text: str, msg_date: date, msg_id: int, seen: dict, positions: dict):
    s = parse_signal(text, msg_date)
    if s.kind == "NOISE":
        return
    one = " ".join(text.split())[:100]

    if s.kind == "EXIT":
        held = [osi for osi, p in positions.items()
                if p["status"] in ("pending", "open") and p.get("ticker") == s.ticker]
        if held and LIVE:
            log(f"🟠 站长出场信号 [{s.ticker}] → 跟随平仓 {len(held)} 个持仓")
            push_discord(f"🟠 enrich出场 [{s.ticker}]: {one}")
            for osi in held:
                close_position(positions, osi, "站长出场")
        else:
            note = f"🟠 enrich出场提醒 [{s.ticker}] (无持仓/DRY_RUN): {one}"
            log(note)
            push_discord(note)
        return

    # BUY
    osi = to_longport_symbol(s)
    key = f"{osi}:{msg_date}"
    if str(msg_id) in seen or key in seen:
        log(f"↩️ 重复信号跳过: {osi}")
        return
    if s.expiry < msg_date:
        log(f"⏭️ 已到期跳过: {osi}")
        return
    if s.limit_price > MAX_PREMIUM:
        log(f"🚫 权利金${s.limit_price}>上限${MAX_PREMIUM}, 拒绝: {one}")
        return

    plan = (f"{'🚀模拟盘' if LIVE else '🧪DRY-RUN'} enrich买入\n"
            f"  {s.ticker} {s.expiry} ${s.strike} {'CALL' if s.right=='C' else 'PUT'}  ({osi})\n"
            f"  限价 ${s.limit_price} × {CONTRACTS}张 (≈${s.limit_price*100*CONTRACTS:.0f})"
            + (f"  [{s.size_tag}]" if s.size_tag else "") + f"\n  原文: {one}")
    log(plan)

    if LIVE:
        ok, r = _submit(osi, side_buy=True, qty=CONTRACTS, price=s.limit_price, remark="enrich-entry")
        if ok:
            positions[osi] = dict(ticker=s.ticker, entry_order_id=r, qty=CONTRACTS,
                                  limit=s.limit_price, expiry=s.expiry.isoformat(),
                                  filled=0, sold=0, avg=0.0, tp_order_id=None, tp_qty=0,
                                  status="pending", opened=str(msg_date))
            _save(POS_JSON, positions)
            plan += f"\n  ✅已提交 order_id={r} (成交后自动挂+{(TP_MULT-1)*100:.0f}%止盈)"
            log(f"  ✅已提交 {r}")
        else:
            plan += f"\n  ❌下单失败: {r}"
            log(f"  ❌下单失败: {r}")
    else:
        plan += "\n  (DRY_RUN 未下单)"
    push_discord(plan)

    seen[str(msg_id)] = key
    seen[key] = str(msg_id)
    _save(SEEN_JSON, seen)


def main():
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("缺 DISCORD_BOT_TOKEN"); sys.exit(1)

    if LIVE:
        if not verify_paper_trading():
            print("❌ 模拟盘三重校验不通过, 拒绝启动 LIVE"); sys.exit(1)
        log(f"🚀 LIVE(模拟盘): 每信号{CONTRACTS}张 | 权利金上限${MAX_PREMIUM} | 止盈+{(TP_MULT-1)*100:.0f}%卖半仓 | 出场跟随+到期强平")
    else:
        log("🧪 DRY_RUN: 只解析播报, 不下单")

    seen = _load(SEEN_JSON)
    positions = _load(POS_JSON)
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @tasks.loop(seconds=60)
    async def manager():
        if LIVE:
            try:
                manage_positions(positions)
            except Exception as e:
                log(f"仓位管理异常: {e}")

    @client.event
    async def on_ready():
        log(f"✅ 已连接 Discord: {client.user} | 监听频道 {CHANNEL_ID} 作者 {AUTHOR_ID}")
        if not manager.is_running():
            manager.start()

    @client.event
    async def on_message(msg):
        if msg.channel.id != CHANNEL_ID or msg.author.id != AUTHOR_ID or not msg.content:
            return
        try:
            handle(msg.content, msg.created_at.date(), msg.id, seen, positions)
        except Exception as e:
            log(f"处理消息异常: {e}")

    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
