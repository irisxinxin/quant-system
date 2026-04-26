"""
test_telegram.py — 测试 Telegram Bot 是否配置正确
运行:
    source ~/.longport_creds.env && python3 test_telegram.py

期望: 你的 Telegram bot 立刻收到一条测试消息.
若收不到, 看终端报错信息.
"""
import os
import sys
import urllib.request
import urllib.parse


def test():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token:
        print("❌ TELEGRAM_BOT_TOKEN 未设置")
        print("   去 ~/.longport_creds.env 加: export TELEGRAM_BOT_TOKEN=\"你的token\"")
        sys.exit(1)
    if not chat_id:
        print("❌ TELEGRAM_CHAT_ID 未设置")
        print("   去 https://api.telegram.org/bot<token>/getUpdates 找 chat id")
        sys.exit(1)

    print(f"📤 发送测试消息到 chat_id={chat_id}...")

    # 测试发一条强信号样本
    title = "🚀🚀 测试: RKLB 强突破"
    body = (
        "时间: 22:18:45 SGT (10:18:45 ET)\n"
        "入场: $80.45\n"
        "止损: $77.30 (-3.91%)\n"
        "止盈: $86.85 (+7.95%)\n"
        "RVOL: 3.42\n"
        "OR 范围: 3.85%\n"
        "R:R: 2.0\n\n"
        "⚠️ 03:55 SGT 前手动平仓\n\n"
        "(这只是测试消息, 不是真信号)"
    )
    msg = f"<b>{title}</b>\n\n<pre>{body}</pre>"

    try:
        data = urllib.parse.urlencode({
            "chat_id": chat_id, "text": msg, "parse_mode": "HTML",
            "disable_notification": "false",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data, method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=5)
        result = resp.read().decode()
        if '"ok":true' in result:
            print("✅ 发送成功! 检查你的 Telegram 是否收到了测试消息.")
            print("   如果手机没响, 检查:")
            print("     1. Telegram App 通知权限是否打开")
            print("     2. 这个 bot 的对话是否有 "静音" 设置")
        else:
            print(f"⚠️ API 返回异常: {result[:200]}")
    except Exception as e:
        print(f"❌ 发送失败: {type(e).__name__}: {e}")
        print("   可能原因:")
        print("     - Token 错了 → 去 @BotFather 重新拿")
        print("     - chat_id 错了 → 重新开浏览器拿一遍")
        print("     - 网络/防火墙 → 检查能否访问 api.telegram.org")


if __name__ == "__main__":
    test()
