# bots/enrich — Discord #期权-波段-enrich 跟单 bot

监听站长信号 → 解析 → LongPort **模拟盘**买期权 → 机械出场管理。

> 🔒 **铁律**: 全程 LongPort 模拟盘 (`lb_papertrading`)，直到用户明确说"切真钱"。
> bot 启动时三重校验 (JWT.ac / JWT.ik / API channel)，不过即退出。详见仓库根 `CLAUDE.md`。

---

## 当前状态 (2026-07-20)

| 项 | 值 |
|---|---|
| 运行模式 | **DRY_RUN** (`ENRICH_LIVE=false`) — 只解析播报，不下单 |
| 出场策略 | 纯机械，**无 LLM** |
| 回归测试 | 82 仿真场景 / 18 变异 / 83 单元断言 — 全绿 |
| 归档 job | 每日 05:10 SGT，**一天都不能断**(长桥期权 K 线只留约 1 月) |

**出场规则** (2026-07-19 定稿)：
进场跟站长限价不追高 + TTL 20 分不接刀 → +30% 卖 ⅓ → +60% 卖 ⅓ (武装) →
runner 武装后守 15 分 9ema 连破 2 根 → 止损 -60% 全程无保本 → 到期强平。
hedge 当 lotto 跟 ⅓ 仓。站长出场消息仅提醒，不触发动作。

---

## 目录

```
bots/enrich/
  discord_enrich_bot.py      主 bot (常驻)
  enrich_parser.py           信号解析 (真身；仓库根有转发垫片供回测脚本用)
  enrich_archive.py          每日归档: 期权5分K + 消息 + 订单流水 + 自动 commit
  run_sim.py                 ← 改完 bot 跑这个
  test_enrich_bot_safety.py  单元断言 (按不变式 I1–I5 组织)
  sim/
    fakes.py                     假券商/假行情/假时钟 (券商持独立真值)
    fake_discord.py              假 message/channel/client + 内存状态文件
    harness.py                   把假件接到 bot 真实逻辑上
    scenario_api.py              场景契约
    scenarios/normal.py          17 个正常链路场景
    scenarios/adversarial.py     27 个对抗场景
    scenarios/reduce_ambig.py    16 个 — 镜像减仓 + 歧义单消歧
    scenarios/discord_layer.py   17 个 — catch_up / on_message / main 启动
    scenarios/regression_guards.py 6 个 — 专守各道防线(每条都配了变异)
    mutation_check.py            变异测试 — 看守"测试本身"
    smoke.py                     端到端冒烟
  launchd/*.plist            launchd 配置
  attic/                     历史副本，不参与运行
```

**不在这个目录里的依赖**(留在仓库根，故意的)：

| 东西 | 位置 | 为什么 |
|---|---|---|
| 状态/日志 | `output/enrich_*` | 归档 job 和一堆回测脚本都按仓库根找 |
| 期权 K 线 | `data/enrich_bars/` | 同上 |
| 出场正则 | `backtest_andy.py` | bot 复用它的 `parse_entry`/`EXIT_*_RE`；`signal_history.py` 也在用 |
| 回测研究 | `bt_*.py` / `backtest_enrich.py` / `mirror_*` / `proxy_*` | 一次性研究脚本，搬走要改 20+ 处 import，风险大于收益 |
| LLM 分类器 | `llm_classifier.py` | bot 已不用；`build_interp_audit.py` / `backtest_agent_ab.py` 还在用 |

⚠️ **`OUT` 指向仓库根的 `output/`，不是本目录。** 见 `discord_enrich_bot.py` 里的注释——
搬目录时这里最容易静默把状态清零。

---

## 改完 bot 必须跑的三套

```bash
cd /Users/xin/Documents/Claude/Projects/money/quant_system
/usr/local/bin/python3 bots/enrich/run_sim.py                 # 82 端到端场景
/usr/local/bin/python3 bots/enrich/sim/mutation_check.py      # 18 个人为缺陷
/usr/local/bin/python3 bots/enrich/test_enrich_bot_safety.py  # 83 单元断言
```

必须用 `/usr/local/bin/python3`（系统 `python3` 没有 `longport` 模块）。

**为什么是三套而不是一套**：
- `run_sim` 用**假券商的独立真值**裁判，抓裸空 / 账实不符 / 孤儿单 / 弃仓 / 挂卖单超持仓
- `mutation_check` 故意把 bot 参数/守卫改坏，验证场景**不是空转** —— 上一版单元测试把
  `ensure_protection` stub 成 no-op，54 项全绿却漏掉了真正的裸空路径
- 单元断言按订单生命周期不变式 I1–I5 组织，含反向回归用例

⚠️ **变异必须指向真正由它守住的场景**，否则测的是别的防线。踩过两次：
`MECH_EMA_N` 指向未武装场景（armed 闸在 N 判断之前）、`AMBIG_MIN_MARGIN` 指向价格跑动场景
（实际被 `MAX_DEV` 拦住）。两次都报成"场景空转"，其实是映射错了。
判断方法：拆掉这道防线，那个场景**真的会红吗**？不会就说明另有防线在守。

跑单个场景：`run_sim.py -v 关键字`

---

## 订单生命周期不变式

状态机 `pending → open → closing → closed`：

- **I1 提交成功 ≠ 成交** — `sold`/`closed` 只由 `manage_positions ⓪` 依 `executed_quantity` 更新
- **I2 撤单只是"请求撤销"** — 未确认终态 = 单子还活着，禁止在它之上再发一张卖单
- **I3 宁可少卖，绝不多卖** — 少卖亏损上限是权利金，多卖 = 裸空期权 = 无限风险。
  终极保证是 `_broker_qty()`：每笔卖出以券商实际持仓封顶
- **I4 入场单在途时绝不标 closed** — 否则那张买单成交后成为无人管理的孤儿多头
- **I5 价格不可用(0/陈旧/停牌)时不得据以判止损**

---

## 运维

```bash
# 启停
launchctl unload ~/Library/LaunchAgents/com.xin.enrichbot.plist
cp bots/enrich/launchd/com.xin.enrichbot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.xin.enrichbot.plist

# 日志
tail -f output/enrich_bot.log

# 模拟盘三重校验(改任何下单相关代码前必做)
source ~/.longport_creds.env && python3 -c "..."   # 见仓库根 CLAUDE.md
```

⚠️ 改 `ENRICH_LIVE` 要**同时**改仓库版和 `~/Library/LaunchAgents/` 里那份，否则下次
`cp` 安装会静默切换运行模式。

---

## 已知未处理

- `_submit` 无真正幂等 — LongPort SDK 3.0.23 和 4.3.3 都**没有** `client_request_id`
  (只在 REST 层有)。现在靠应用层 `_live_sell_order` + `_broker_qty` 封顶，
  **降低重复概率，不是保证**
- `reconcile_with_broker()` **从未对真实 API 跑过**(只在 `LIVE=true` 时执行)，字段名
  是照 SDK 签名推断的
- `_recent_exits` 只按**文本**去重、不带标的 — 站长 10 分钟内对两只票发同样措辞的出场，
  第二条会被静默丢弃。目前被"无票名出场解析成 `ticker='*'`"绕开
- `account_equity_usd` 多币种聚合取第一个正币种，非 USD 按 7.8 折算
- Discord 通知从未生效 — 凭证/plist/shell 里都没有 `DISCORD_WEBHOOK_URL`
