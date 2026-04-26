"""
test_longbridge.py — 测试 LongPort OpenAPI 连接 + 美股实时数据

运行方式:
  1. 填入下面的 APP_KEY / APP_SECRET / ACCESS_TOKEN
  2. python3 test_longbridge.py
  3. 如果能看到 OKLO 的实时报价 + 最近 10 根 5m K 线, 说明成功

如果报错:
  - "Unauthorized" → 检查 ACCESS_TOKEN 是否过期（30天有效期）
  - "Permission denied" / "quote not available" → 美股 L1 行情权限未开通
  - "rate limited" → 暂停几秒再试
"""
import os

# ==============================================================
# 🔑 填入你的 LongPort OpenAPI 凭证
# ==============================================================
APP_KEY      = os.environ.get("LONGPORT_APP_KEY", "YOUR_APP_KEY_HERE")
APP_SECRET   = os.environ.get("LONGPORT_APP_SECRET", "YOUR_APP_SECRET_HERE")
ACCESS_TOKEN = os.environ.get("LONGPORT_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN_HERE")

# 安全建议: 不要把 key 写在代码里, 用环境变量:
#   export LONGPORT_APP_KEY="xxx"
#   export LONGPORT_APP_SECRET="xxx"
#   export LONGPORT_ACCESS_TOKEN="xxx"

# ==============================================================
# 测试代码
# ==============================================================
try:
    from longport.openapi import QuoteContext, Config, Period, AdjustType, SubType
except ImportError:
    print("❌ longport 包未安装，请运行: pip install longport")
    exit(1)

if "YOUR_" in APP_KEY or "YOUR_" in ACCESS_TOKEN:
    print("❌ 请先填入你的 APP_KEY / APP_SECRET / ACCESS_TOKEN")
    exit(1)

print("=" * 60)
print("🔌 LongPort OpenAPI 连接测试")
print("=" * 60)

# 建立连接
config = Config(app_key=APP_KEY, app_secret=APP_SECRET, access_token=ACCESS_TOKEN)
ctx = QuoteContext(config)
print("✅ 连接建立成功")

# 测试 1: 静态报价
print("\n━━━ 测试 1: 静态报价 ━━━")
tickers = ["OKLO.US", "IREN.US", "CIFR.US", "CRWV.US"]
quotes = ctx.quote(tickers)
for q in quotes:
    print(f"  {q.symbol:<10} 最新 ${q.last_done}  开盘 ${q.open}  "
          f"今日高 ${q.high}  今日低 ${q.low}  成交量 {q.volume:,}")

# 测试 2: 最近 10 根 5 分钟 K 线
print("\n━━━ 测试 2: OKLO 最近 10 根 5m K 线 ━━━")
candles = ctx.candlesticks("OKLO.US", Period.Min_5, 10, AdjustType.NoAdjust)
for c in candles:
    print(f"  {c.timestamp}  O={c.open}  H={c.high}  L={c.low}  "
          f"C={c.close}  V={c.volume:,}")

# 测试 3: 实时订阅（仅订阅，5 秒内打印推送的 quote）
print("\n━━━ 测试 3: 实时订阅 5 秒 ━━━")
import time

def on_quote(symbol, event):
    print(f"  📡 {symbol}  最新 ${event.last_done}  量 {event.volume:,}  "
          f"时间 {event.timestamp}")

ctx.set_on_quote(on_quote)
ctx.subscribe(tickers, [SubType.Quote])
print(f"  订阅 {tickers}, 等待 5 秒看推送...")
time.sleep(5)
print("\n✅ 所有测试完成")
