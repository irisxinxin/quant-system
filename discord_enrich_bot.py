#!/usr/bin/env python3
"""
discord_enrich_bot.py — 监听 Discord #期权-波段-enrich 信号 → 解析 → LongPort 模拟盘买期权。

链路: 站长转发(bot id锁死) 发英文原文 → enrich_parser 严格五要素解析
      → BUY: 限价单买入 N 张 (LongPort paper) / EXIT: 仅提醒 / NOISE: 忽略

🔒 安全护栏 (缺一不跑):
  1. 启动时 LongPort 模拟盘三重校验 (JWT.ac / JWT.ik / API channel 全=lb_papertrading), 不过即退出
  2. 只认 频道ID+作者ID 白名单 (昵称可仿冒, ID不可), 其他人发一样格式也不动
  3. 默认 DRY_RUN: 只解析+播报"会下什么单", 不碰下单API; 环境变量 ENRICH_LIVE=true 才真下(仍是模拟盘)
  4. 去重: 同一期权代码同一天只下一次 (站长常重发同一信号); 消息ID也去重
  5. 限价单 only (权利金来自信号原文, 不依赖期权行情权限); 单张权利金>MAX_PREMIUM 拒绝
  6. 期权已到期(0DTE收盘后才看到)跳过

用法:
  python3 discord_enrich_bot.py            # DRY_RUN 监听 (推荐先跑几天)
  ENRICH_LIVE=true python3 discord_enrich_bot.py   # 真下模拟盘单
环境变量: DISCORD_BOT_TOKEN(必须) / OPTION_CONTRACTS(默认1张) / DISCORD_WEBHOOK_URL(可选,回报推送)
"""
import os, sys, json, base64
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")

import discord
from enrich_parser import parse_signal, to_longport_symbol
from notify import push_discord

# ── 白名单 (2026-07-14 实测抓取, 锁ID) ──
CHANNEL_ID = 1392361900217602108          # #期权-波段-enrich
AUTHOR_ID  = 1392020997393088542          # 站长转发 (bot)

LIVE = os.environ.get("ENRICH_LIVE", "").lower() == "true"
CONTRACTS = int(os.environ.get("OPTION_CONTRACTS", "1"))
MAX_PREMIUM = float(os.environ.get("MAX_PREMIUM", "5.0"))   # 单张权利金上限$, 防解析错高价单
STATE_JSON = Path(__file__).parent / "output" / "enrich_seen.json"
LOG = Path(__file__).parent / "output" / "enrich_bot.log"

_trade_ctx = None   # LIVE 时初始化


def log(msg: str):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
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


def load_seen() -> dict:
    try:
        return json.loads(STATE_JSON.read_text())
    except Exception:
        return {}


def save_seen(s: dict):
    try:
        STATE_JSON.parent.mkdir(exist_ok=True)
        STATE_JSON.write_text(json.dumps(s, ensure_ascii=False))
    except Exception:
        pass


def place_paper_order(osi: str, limit: float, qty: int) -> str:
    """LongPort 模拟盘限价买入期权。返回结果描述。"""
    from longport.openapi import OrderType, OrderSide, TimeInForceType
    try:
        resp = _trade_ctx.submit_order(
            symbol=osi, order_type=OrderType.LO, side=OrderSide.Buy,
            submitted_quantity=Decimal(str(qty)),
            time_in_force=TimeInForceType.Day,
            submitted_price=Decimal(f"{limit:.2f}"),
            remark="enrich-signal")
        return f"✅已提交 order_id={resp.order_id}"
    except Exception as e:
        return f"❌下单失败: {e}"


def handle(text: str, msg_date: date, msg_id: int, seen: dict):
    s = parse_signal(text, msg_date)
    if s.kind == "NOISE":
        return
    one = " ".join(text.split())[:100]

    if s.kind == "EXIT":
        note = f"🟠 enrich出场提醒 [{s.ticker}]: {one}\n(自然语言出场, 请手动管理仓位)"
        log(note)
        push_discord(note)
        return

    # BUY
    osi = to_longport_symbol(s)
    key = f"{osi}:{msg_date}"
    if str(msg_id) in seen or key in seen:
        log(f"↩️ 重复信号跳过: {osi} (站长重发)")
        return
    if s.expiry < msg_date:
        log(f"⏭️ 已到期跳过: {osi}")
        return
    if s.limit_price > MAX_PREMIUM:
        log(f"🚫 权利金${s.limit_price}>上限${MAX_PREMIUM}, 拒绝 (防解析错): {one}")
        return

    plan = (f"{'🧪DRY-RUN' if not LIVE else '🚀模拟盘'} enrich买入信号\n"
            f"  {s.ticker} {s.expiry} ${s.strike} {'CALL' if s.right=='C' else 'PUT'}\n"
            f"  期权代码 {osi}\n"
            f"  限价 ${s.limit_price} × {CONTRACTS}张 (≈${s.limit_price*100*CONTRACTS:.0f})"
            + (f"  [{s.size_tag}]" if s.size_tag else "") + f"\n  原文: {one}")
    log(plan)

    if LIVE:
        result = place_paper_order(osi, s.limit_price, CONTRACTS)
        log(f"  {result}")
        plan += f"\n  {result}"
    else:
        plan += "\n  (DRY_RUN 未下单; ENRICH_LIVE=true 才会真下模拟盘)"
    push_discord(plan)

    seen[str(msg_id)] = key
    seen[key] = str(msg_id)
    save_seen(seen)


def main():
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("缺 DISCORD_BOT_TOKEN"); sys.exit(1)

    if LIVE:
        if not verify_paper_trading():
            print("❌ 模拟盘三重校验不通过, 拒绝启动 LIVE"); sys.exit(1)
        log(f"🚀 LIVE模式(模拟盘): 每信号 {CONTRACTS} 张, 权利金上限 ${MAX_PREMIUM}")
    else:
        log("🧪 DRY_RUN模式: 只解析播报, 不下单")

    seen = load_seen()
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        log(f"✅ 已连接 Discord: {client.user} | 监听频道 {CHANNEL_ID} 作者 {AUTHOR_ID}")

    @client.event
    async def on_message(msg):
        if msg.channel.id != CHANNEL_ID:
            return
        if msg.author.id != AUTHOR_ID:      # 白名单: 只认站长转发, 昵称仿冒无效
            return
        if not msg.content:
            return
        try:
            handle(msg.content, msg.created_at.date(), msg.id, seen)
        except Exception as e:
            log(f"处理消息异常: {e}")

    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
