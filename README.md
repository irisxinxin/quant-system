# quant-system

日内 + 波段量化系统。LongPort 模拟盘实盘下单 + Telegram 信号推送 + fill-realistic 回测引擎。

---

## ⚠️⚠️ 回测前必读：防止用旧数据 ⚠️⚠️

> **2026-06 踩过的坑**：`output/klines_5m/*.json` 是 **git 跟踪**的，回测脚本看到 JSON 已存在就"已有JSON, 跳过抓取"，于是一直吃几周前的旧数据 → 回测结论实质错误（19/31 只票推荐度变了）。`rm` 删文件还会被沙箱拦 / `git pull` 恢复。**务必按下面流程确认数据新鲜。**

### 1. 每次回测前先验数据新鲜度

```bash
python3 -c "import json; m=json.load(open('output/klines_5m/_manifest.json')); \
ds=sorted(set(t['last'][:10] for t in m['tickers'])); \
print('数据截止日:', ds[-3:], f'({len(m[\"tickers\"])}只)')"
```

末日期应接近**今天**。若停在几周前 = 数据过期，**必须强制重抓**（见下）。

### 2. 强制重抓最新数据（绕过所有缓存）

```bash
# 关键: --force 绕过 JSON存在跳过 + pkl 24h缓存; 删文件没用(沙箱/git会恢复)
python3 fetch_and_matrix_new6.py --force AAOI MU DELL MXL ...   # 列出要刷新的票
```

- `--force` 会让 `fetch_history(force=True)` 重新拉到**今天**，并覆盖 JSON。
- 抓完脚本会打印每只的 `first→last`，**逐只核对 last = 今天附近**。
- 不加 `--force` 时，已有 JSON 的票会被跳过（用于增量补新票）。

### 3. 不要信旧的 `output/strategy_matrix.csv`

那是某次旧运行（2026-05-01）的产物，**又旧又可能缺票**（当时漏了 LITE/SNDK/CRCL）。
要全票一致排名，**用 `--force` 重跑生成 `output/full_matrix_fresh.csv`**，别直接读老 matrix。

### 4. phantom 陷阱：别只看 PF 排序

限价单策略(ORB15_VWAP/VWAP_PB/DC20)常有高 phantom（信号没成交）。
`phantom≥25% + 成交<60` 的"高 PF"是少数走运成交撑的虚高，实盘复现不了。
**只信 `phantom<25% + 成交≥60` 的策略**（市价单 ORB5_Z/ST_10_3 通常 0% phantom）。

---

## 实盘交易（LongPort 模拟盘）

```bash
LIVE_TRADING=true python3 signal_live_longport.py   # 真下单(仍是模拟盘)
LIVE_TRADING=false ...                              # dry-run 不下单
```

- **票池**: `output/best_strategy_per_ticker.csv`（每票一个最优日内策略 + 档位）
- **优先级**: `output/intraday_priority.csv`（近期胜率/PF/每笔期望值/排名）→ Telegram 信号显示 `⭐#N/17`，多信号同时来时按 # 取舍
- **🔒 改实盘配置/重启进程前必做模拟盘三验**（见 `CLAUDE.md`）：`JWT ac` / `ik` / `API channel` 三处都必须是 `lb_papertrading`

## 回测基础设施

| 文件 | 作用 |
|---|---|
| `backtest_engine.py` | fill-realistic 回测（限价检 bar.Low、市价按 bar.Open、phantom 判定） |
| `signals/strategies/intraday_pool.py` | 8 个日内策略（ORB5_Z / ORB15_VWAP / DC20 / ST_10_3 / VWAP_PB ...） |
| `fetch_and_matrix_new6.py` | 抓 5m + 跑 8 策略矩阵找最优（支持 `--force`） |
| `run_strategy_matrix.py` | 全票×全策略矩阵 → `best_strategy_per_ticker.csv` |
| `output/klines_5m/*.json` | 5m K 线数据（git 跟踪，⚠见上方防呆提示） |

## 数据诚实铁律

- 回测必须按真实可操作路线算：`bar.High > 限价 ≠ 能 fill`，限价单要检 `bar.Low`，市价单按 `bar.Open`
- 不为凑数造假信号；样本不足/无干净策略就如实标 ❌，别硬给
- 6/1 前所有自动下单必须是 LongPort 模拟盘（详见 `CLAUDE.md`）
