"""
auto_eod_close.py — 自动日末平仓 (LongPort 模拟账户)

每天 03:55 SGT (= 15:55 ET) 自动跑, 把模拟账户里所有持仓市价卖掉.

环境变量需求:
  LONGPORT_APP_KEY / LONGPORT_APP_SECRET / LONGPORT_ACCESS_TOKEN
  LONGPORT_PAPER_TRADE_PASSWORD  (模拟账户的 6 位交易密码)
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  (推送结果用)

安全设计:
  - 只用 TradeContext.submit_order(side=Sell, type=MO) — 只卖不买
  - 只操作模拟账户 (你 API 凭证就是 LBPT 账户的)
  - --dry-run 模式: 只列出持仓不真平 (测试用)

用法:
  # 测试 (列出持仓不平):
  python3 auto_eod_close.py --dry-run

  # 实际平仓:
  python3 auto_eod_close.py
"""
import os
import sys
import time
import urllib.request
import urllib.parse
from decimal import Decimal
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

try:
    from longport.openapi import (
        Config, TradeContext,
        OrderType, OrderSide, TimeInForceType, OutsideRTH,
    )
except ImportError:
    print("❌ longport 未安装: pip3 install --break-system-packages longport")
    sys.exit(1)


ET = ZoneInfo("US/Eastern")
SGT = ZoneInfo("Asia/Singapore")
DRY_RUN = "--dry-run" in sys.argv


def log(msg: str):
    now = datetime.now(SGT)
    print(f"[{now.strftime('%H:%M:%S SGT')}] {msg}", flush=True)


def telegram(title: str, body: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    try:
        msg = f"<b>{title}</b>\n\n<pre>{body}</pre>"
        data = urllib.parse.urlencode({
            "chat_id": chat_id, "text": msg, "parse_mode": "HTML",
            "disable_notification": "false",
        }).encode()
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=data, method="POST"),
            timeout=5,
        )
    except Exception as e:
        log(f"⚠️ Telegram 发送失败: {e}")


def main():
    log("=" * 60)
    log(f"自动日末平仓 {'(--dry-run 模式)' if DRY_RUN else ''}")
    log("=" * 60)

    # 1. 验证环境变量
    required = ["LONGPORT_APP_KEY", "LONGPORT_APP_SECRET", "LONGPORT_ACCESS_TOKEN",
                "LONGPORT_PAPER_TRADE_PASSWORD"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        log(f"❌ 缺少环境变量: {missing}")
        sys.exit(1)

    # 2. 连接 + 解锁 TradeContext
    log("🔌 连接 LongPort TradeContext...")
    try:
        config = Config.from_env()
        ctx = TradeContext(config)
    except Exception as e:
        log(f"❌ TradeContext 创建失败: {e}")
        telegram("⚠️ EOD 平仓失败", f"TradeContext 连接失败:\n{e}")
        sys.exit(1)

    log("🔓 解锁交易...")
    try:
        ctx.unlock_trade(os.environ["LONGPORT_PAPER_TRADE_PASSWORD"])
        log("✅ 解锁成功")
    except Exception as e:
        log(f"❌ 解锁失败: {e}")
        telegram("⚠️ EOD 平仓失败", f"交易密码错误或账户问题:\n{e}")
        sys.exit(1)

    # 3. 获取所有持仓
    log("📥 获取持仓列表...")
    try:
        positions_resp = ctx.stock_positions()
    except Exception as e:
        log(f"❌ 获取持仓失败: {e}")
        telegram("⚠️ EOD 平仓失败", f"获取持仓失败:\n{e}")
        sys.exit(1)

    # 解析持仓 (LongPort SDK 返回结构: channels → positions)
    all_positions = []
    for channel in positions_resp.channels:
        for pos in channel.positions:
            qty = int(pos.quantity)
            if qty > 0:  # 只关注多头持仓
                all_positions.append({
                    "symbol": pos.symbol,
                    "name": pos.symbol_name,
                    "qty": qty,
                    "cost": float(pos.cost_price) if pos.cost_price else 0,
                    "channel": channel.account_channel,
                })

    if not all_positions:
        log("📭 当前无持仓, 无需平仓")
        telegram("📊 EOD 平仓汇总", "今日无持仓, 无需操作 ✅")
        return

    log(f"📦 共 {len(all_positions)} 个持仓:")
    for p in all_positions:
        log(f"   {p['symbol']:<10} qty={p['qty']:>6}  cost=${p['cost']:.2f}  ({p['channel']})")

    # 4. 平仓 (除非 --dry-run)
    if DRY_RUN:
        log("\n🧪 DRY-RUN 模式: 不实际平仓, 退出")
        body = f"持仓数: {len(all_positions)}\n\n"
        for p in all_positions:
            body += f"{p['symbol']:<10} {p['qty']:>6}股  cost ${p['cost']:.2f}\n"
        telegram("🧪 EOD Dry-Run", body)
        return

    log("\n📤 开始市价平仓...")
    closed = []
    failed = []
    for p in all_positions:
        try:
            resp = ctx.submit_order(
                symbol=p["symbol"],
                order_type=OrderType.MO,           # 市价单
                side=OrderSide.Sell,                # 卖出
                submitted_quantity=Decimal(str(p["qty"])),
                time_in_force=TimeInForceType.Day,
                outside_rth=OutsideRTH.RTHOnly,
                remark="ORB-EOD-AUTO",
            )
            log(f"   ✅ {p['symbol']:<10} sent (order_id={resp.order_id})")
            closed.append(p)
            time.sleep(0.5)  # 避免 rate limit
        except Exception as e:
            log(f"   ❌ {p['symbol']:<10} 平仓失败: {e}")
            failed.append({**p, "error": str(e)})

    # 5. 发送 Telegram 汇总
    body = f"已下平仓单: {len(closed)} / {len(all_positions)}\n\n"
    if closed:
        body += "✅ 平仓:\n"
        for p in closed:
            body += f"  {p['symbol']:<10} {p['qty']:>6}股\n"
    if failed:
        body += "\n❌ 失败:\n"
        for p in failed:
            body += f"  {p['symbol']:<10} ({p['error'][:40]})\n"
    body += f"\n时间: {datetime.now(ET).strftime('%H:%M ET')}"
    body += "\n建议早上起床检查 longbridge App 确认成交价"

    telegram("📊 EOD 自动平仓汇总", body)
    log("\n✅ 自动平仓完成")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ 异常退出: {e}")
        telegram("⚠️ EOD 自动平仓异常", str(e))
        sys.exit(1)
