# quant-system — Claude 工作守则

## 🔒 铁律: 所有自动下单必须是 LongPort 模拟盘 (直到用户明确说切换)

**必须一直用模拟盘, 直到用户明确口头说"切真钱/切实盘"为止** —— 没有截止日期 (原 6/1 已作废, 2026-06-15 用户明确: "在我说切换之前要一直用模拟盘")。本仓库的 `signal_live_longport.py` + `live_executor.py` 配合 `LIVE_TRADING=true` 会真实下单, 必须确保对接的是 `lb_papertrading`.

### 改任何代码前必须 double-check 模拟盘

每次准备:
- 修改 `live_executor.py` / `signal_live_longport.py` / launchd plist
- 重启实盘进程
- 改变 `POSITION_USD` / 仓位算法
- 接新策略 / 改下单逻辑

**之前**, 必须执行三处独立验证:

```bash
source ~/.longport_creds.env && python3 -c "
import base64, json
parts = '$LONGPORT_ACCESS_TOKEN'.split('.')
p = json.loads(base64.urlsafe_b64decode(parts[1] + '=' * (-len(parts[1]) % 4)))
from longport.openapi import Config, TradeContext
t = TradeContext(Config.from_env())
print(f'JWT ac: {p[\"ac\"]}')
print(f'JWT ik: {p[\"ik\"]}')
for ch in t.stock_positions().channels: print(f'API channel: {ch.account_channel}')
"
```

**全部三处必须显示 `lb_papertrading`**. 缺一处或不一致 → 立即停手, 告知用户.

### 已知凭证文件
- `~/.longport_creds.env` — 凭证 (chmod 600, 不入仓库)
- 当前 access token (2026-04-30 至 2026-07-25 有效) 是 paper trading token

### 模拟盘标志特征
1. `JWT.ac == "lb_papertrading"` (account class)
2. `JWT.ik` 以 `lb_papertrading_` 开头 (institution key)
3. `TradeContext.stock_positions().channels[].account_channel == "lb_papertrading"`

如果上面任何一项不匹配, **不能** 启动 LIVE_TRADING.

## 项目结构 (实盘相关)

```
signal_live_longport.py  Multi-Strategy Live Dispatcher (主入口)
live_executor.py         订单执行 (LIT / MO + stop/tp + OCO + force_close)
backtest_engine.py       Fill-realistic 回测引擎
signals/strategies/
  intraday_pool.py       8 个策略池 (ORB5_Z / ORB15_VWAP / DC20 / ST_10_3 / VWAP_PB 等)
  indicators.py          VWAP / RSI / EMA / Donchian / Supertrend / ATR / ...
output/best_strategy_per_ticker.csv  每只票的最优策略分配
com.filmhousebot.orbsignal.plist     launchd 配置 (周一-五 21:00 SGT 自启)
```

## 每日运行节奏 (SGT 时区)

```
21:00 SGT (= 09:00 ET 盘前)  launchd 自动启动
21:30 SGT (= 09:30 ET 开盘)  开始派发信号
其间                         每 30s 拉 K 线 → 跑策略 → place_entry
03:50 SGT (= 15:50 ET)       force_close_all (撤单+市价平仓)
03:55 SGT (= 15:55 ET)       脚本 clean exit
```

## 重要环境变量

```
LIVE_TRADING=true   真下单 (但仍是模拟盘); false 则 dry-run
POSITION_USD=10000  每信号目标资金 (在 live_executor.py 里)
LONGPORT_*          API 凭证
TELEGRAM_BOT_TOKEN  推送 (可选)
TELEGRAM_CHAT_ID    推送 (可选)
```

## 不要做的事

- ❌ 不要在 ~/.longport_creds.env 里换成真钱账户的 token (6/1 前)
- ❌ 不要绕过 `_LIVE` / `lb_papertrading` 检查
- ❌ 不要无 double-check 就重启实盘进程
- ❌ 不要在没跑过 backtest_engine 的 fill-realistic 验证下加新策略
