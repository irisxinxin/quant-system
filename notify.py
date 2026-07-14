#!/usr/bin/env python3
"""
notify.py — 多路推送 (Telegram + Discord)。

朋友"跟着收信号"用法: 脚本把信号 POST 到一个 Discord 频道的 Webhook, 朋友加进那个频道即可
收到, 零密钥零配置 (他不需要你的任何 token/App ID)。想推给多个频道/多个朋友, 用 DISCORD_WEBHOOK_URLS
逗号分隔多个 URL 即可。

环境变量 (填进 ~/.longport_creds.env, 不入仓库):
  DISCORD_WEBHOOK_URL    单个 Discord 频道 Webhook (频道设置→整合→Webhooks→复制URL)
  DISCORD_WEBHOOK_URLS   可选: 逗号分隔多个 Webhook (一次推给多个频道)
"""
import os, json
import urllib.request


def _webhooks():
    urls = []
    u = os.environ.get("DISCORD_WEBHOOK_URL")
    if u:
        urls.append(u.strip())
    multi = os.environ.get("DISCORD_WEBHOOK_URLS")
    if multi:
        urls += [x.strip() for x in multi.split(",") if x.strip()]
    # 去重保序
    seen, out = set(), []
    for x in urls:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def push_discord(text: str) -> bool:
    """推到所有配置的 Discord Webhook。任一成功即 True; 未配置返回 False (调用方据此判断是否还有别的渠道)。"""
    urls = _webhooks()
    if not urls:
        return False
    ok = False
    payload = json.dumps({"content": text[:1900]}).encode()   # Discord 单条上限 2000 字, 留余量
    for url in urls:
        try:
            urllib.request.urlopen(urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}, method="POST"),
                timeout=10)
            ok = True
        except Exception as e:
            print(f"   ⚠️ Discord推送失败: {e}")
    if ok:
        print("   ✅ Discord已发")
    return ok
