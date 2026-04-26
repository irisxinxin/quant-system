# 短线 ORB Tab 集成说明

新增的 ORB 短线策略页面，需要你手动改 **3 个文件，共 ~10 行**。

⭐ **新版**: K 线图改用 **TradingView Lightweight Charts** (开源 JS 库, 免费)
- 完全交互: 滚轮缩放 / 拖动平移 / 十字光标看 OHLC
- 自定义买卖点标记 (入场/止盈/止损/日末)
- 每日 OR 上下沿用线段标注
- Ticker 选择器: 单图模式 (省内存)
- **不再依赖 PNG 图片** (但 PNG 生成器保留, 你可选择用)

## 已生成的新文件 (无需改动)

```
output/klines_5m/             # 12 只股票的 18 个月 5m JSON 数据 (35 MB)
output/charts_orb/            # 12 张 K 线图 PNG (含 ORB 信号标记)
output/orb_signals.jsonl      # 信号日志 (将由 signal_live_longport 写入)

signals/orb_strategy.py       # 策略核心模块
generate_orb_charts.py        # 图表生成器 (每次更新数据后跑)
data/save_intraday_data.py    # cache pkl → 持久 JSON 转换器
dashboard/orb_routes.py       # FastAPI 路由
dashboard/__init__.py         # (空, 用于 import)
templates/orb_tab.html        # ORB tab HTML 内容 (含策略说明 + 图表)
```

---

## 你需要改的 3 个地方

### ① `app.py` — 注册 ORB 路由 (加 2 行)

**找位置**: 在 `from scanner import ...` 那行下面

```python
# 原来:
from scanner import scan_all, get_macro, get_flows, get_cta_dashboard, get_sector_full, get_bt_signals, WATCHLIST

# 加这两行:
from dashboard.orb_routes import register_orb_routes
```

**找位置**: 在创建 `app = FastAPI(...)` 那行之后

```python
# 创建 app 之后加:
register_orb_routes(app)
```

### ② `templates/index.html` — 加 tab 按钮 (加 1 行)

**找位置**: 第 153-154 行附近 (按钮列表)

```html
<!-- 现有: -->
<button class="tab-btn" id="tab-sig" onclick="sw('sig')">📡 逸哥信号</button>
<button class="tab-btn" id="tab-review" onclick="sw('review')">📓 每日复盘</button>

<!-- 在末尾加: -->
<button class="tab-btn" id="tab-orb" onclick="sw('orb')">🚀 短线 ORB</button>
```

### ③ `templates/index.html` — 嵌入 ORB tab 内容 (加 1 行)

**找位置**: 在某个 `<div id="...other tab..." class="tab-content"...>` 之后, 或 `</body>` 之前

需要先确认你的 Jinja2 模板支持 `{% include %}`。如果支持, 加这行:

```html
{% include 'orb_tab.html' %}
```

如果你的 FastAPI **没用 Jinja2 templates**, 而是直接 `return HTMLResponse(file.read())`, 那么在
`</body>` 标签前手动复制 `orb_tab.html` 的全部内容也可以, 但保持文件分离更清爽。

---

## 验证步骤

### 1. 跑 dashboard 看效果

```bash
# 假设你用 uvicorn 跑
cd /Users/xin/Documents/Claude/Projects/money/quant_system
uvicorn app:app --reload --port 8000
```

打开浏览器 http://localhost:8000

- 点 "🚀 短线 ORB" tab
- 应看到三大块:
  1. 策略说明 (入场规则 + 风险管理)
  2. 今日实时状态 (5 只核心标的)
  3. 12 张 K 线图 (含历史买卖点)

### 2. 测试 API 端点

```bash
curl http://localhost:8000/api/orb/manifest | python3 -m json.tool | head -30
curl http://localhost:8000/api/orb/today | python3 -m json.tool
curl http://localhost:8000/api/orb/symbol/OKLO.US | python3 -m json.tool | head -50
```

---

## 数据更新流程

每天美股收盘后 (大约 16:30 ET / 04:30 SGT 次日), 更新 5m 数据 + 重新生成图表:

```bash
# 1. 拉新数据 (LongPort)
source ~/.longport_creds.env && python3 longport_history_backtest.py

# 2. 转换为 JSON 持久化
python3 -m data.save_intraday_data

# 3. 重新生成 K 线图
python3 generate_orb_charts.py
```

可以加个 shell 脚本一次跑完, 或者用 cron 自动化。

---

## 实战流程 (周一开始)

### 21:15 SGT (开盘前 15 分钟)
```bash
source ~/.longport_creds.env && python3 signal_live_longport.py
```

### 21:30 - 01:00 SGT (盯盘窗口)
- 监控终端 + macOS 通知
- 信号触发 → 60 秒内手动到 longbridge / moomoo 模拟账户下单

### 03:55 SGT (日末平仓闹钟)
- 起来检查未触发止损/止盈的持仓
- 手动平掉 (或挂收盘市价单)

### 第二天复盘
- 看 dashboard 的 "🚀 短线 ORB" tab
- 对比信号 vs 实际成交差异 (滑点)
- 月底统计 paper 总收益, 决定是否上小资金实盘

---

## 已完成 vs 待完成

### ✅ 本次完成
- 18 个月 5m 历史数据持久化 (12 标的)
- 策略核心模块 (生成信号 / 历史回测)
- K 线图生成器 (含买卖点标记)
- Dashboard tab + API 路由
- 完整集成文档

### 📋 后续可选 (需要再说一声)
- 信号日志自动写入 (signal_live_longport.py 的简单改造)
- Paper 成交对比 (你输入实际成交价, 系统对比预期 vs 实际)
- Telegram bot 通知 (凌晨强信号唤醒)
- 月度复盘自动报表
- 实盘和 paper 收益对比页

---

## 文件结构检查清单

```
quant_system/
├── app.py                          [✏️ 加 2 行]
├── data/
│   └── save_intraday_data.py       [新]
├── dashboard/
│   ├── __init__.py                  [新]
│   └── orb_routes.py                [新]
├── signals/
│   └── orb_strategy.py              [新]
├── templates/
│   ├── index.html                  [✏️ 加 2 行]
│   └── orb_tab.html                [新]
├── output/
│   ├── klines_5m/                   [新, 12 个 JSON]
│   ├── charts_orb/                  [新, 12 张 PNG]
│   └── orb_signals.jsonl           [信号日志, 运行后生成]
├── generate_orb_charts.py           [新]
├── longport_history_backtest.py     [已有]
├── longport_stress_3_tests.py       [已有]
└── signal_live_longport.py          [已有]
```
