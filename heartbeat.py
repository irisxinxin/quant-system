"""
heartbeat.py — 每天发一条 Telegram 心跳, 确认 Mac Mini 在线 + 信号机运行

用法 (cron 每天中午跑一次):
    0 12 * * * cd /Users/xin/Documents/Claude/Projects/money/quant_system && \
        source ~/.longport_creds.env && python3 heartbeat.py
"""
import os
import sys
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LOG_FILE = Path(__file__).parent / "signals_live_longport.jsonl"


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("❌ Telegram 未配置")
        return

    # 检查 launchd 服务状态
    try:
        result = subprocess.run(
            ["launchctl", "list"],
            capture_output=True, text=True, timeout=5,
        )
        running = "com.xin.orbsignal" in result.stdout
    except Exception:
        running = False

    # 统计今日已收信号
    today = datetime.now(ZoneInfo("US/Eastern")).date().isoformat()
    today_signals = 0
    if LOG_FILE.exists():
        import json
        with open(LOG_FILE) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("event") == "breakout" and today in str(e.get("bar_time", "")):
                        today_signals += 1
                except json.JSONDecodeError:
                    continue

    sgt = datetime.now(ZoneInfo("Asia/Singapore"))
    et = datetime.now(ZoneInfo("US/Eastern"))

    status_emoji = "🟢" if running else "🔴"
    body = (
        f"Mac Mini 状态: {'运行中' if running else '⚠️ 服务未运行'}\n"
        f"SGT: {sgt.strftime('%Y-%m-%d %H:%M')}\n"
        f"ET:  {et.strftime('%Y-%m-%d %H:%M')}\n"
        f"今日已收信号: {today_signals} 条\n"
    )

    msg = f"<b>{status_emoji} ORB 心跳</b>\n\n<pre>{body}</pre>"

    try:
        data = urllib.parse.urlencode({
            "chat_id": chat_id, "text": msg, "parse_mode": "HTML",
            "disable_notification": "true",  # 静默, 不打扰
        }).encode()
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=data, method="POST",
            ),
            timeout=5,
        )
        print(f"✅ 心跳发送成功 (运行: {running}, 今日信号: {today_signals})")
    except Exception as e:
        print(f"❌ 心跳发送失败: {e}")


if __name__ == "__main__":
    main()
