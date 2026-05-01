"""
signal_live_longport.py — Multi-Strategy Live Dispatcher

每只票按 output/best_strategy_per_ticker.csv 跑各自最优策略:
  - ORB5_Z (CRWV/INTC/IREN/MSFL/NBIS)
  - ORB15_VWAP (OKLO/AMZN/TSLZ/EOSE)
  - DC20 Donchian (RKLB/HOOD/NVDS)
  - ST_10_3 Supertrend (PLTR/SOXL/AMD)
  - VWAP_PB (TSLL/MSFU/CIFR)

主循环 (每 30s):
  1. 对每只票, 拉最近 50 根 5m K 线
  2. 按 strategy_map[symbol] 调用对应策略
  3. 策略返回 TradePlan → live_executor.place_entry
  4. 心跳: check_fills_and_arm + reconcile_oco
  5. 15:50 ET: force_close_all

环境变量:
  LIVE_TRADING=true   才真下单
  LONGPORT_*          凭证
  TELEGRAM_*          可选, 推送通知
"""
import os
import sys
import time
import json
import subprocess
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from longport.openapi import Config, QuoteContext, Period, AdjustType
except ImportError:
    print("❌ longport 包未安装"); sys.exit(1)

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))
from signals.strategies.intraday_pool import STRATEGIES
import live_executor

ET = ZoneInfo("US/Eastern")
SGT = ZoneInfo("Asia/Singapore")

# 时段
MARKET_OPEN  = dtime(9, 30)
MARKET_CLOSE = dtime(15, 55)
FORCE_CLOSE_AT = dtime(15, 50)

# 日志
LOG_FILE = PROJECT_DIR / "signals_live_longport.jsonl"

# 主循环
POLL_INTERVAL = 30   # 秒

# ═══ 加载策略分配 ═══
def load_strategy_map() -> dict:
    """读 output/best_strategy_per_ticker.csv, 排除 verdict != OK 的票"""
    csv_path = PROJECT_DIR / "output" / "best_strategy_per_ticker.csv"
    if not csv_path.exists():
        print(f"❌ 找不到 {csv_path}"); sys.exit(1)
    df = pd.read_csv(csv_path)
    df_ok = df[df["verdict"] == "OK"]
    smap = dict(zip(df_ok["symbol"], df_ok["strategy"]))
    return smap

STRATEGY_MAP = load_strategy_map()
TICKERS = list(STRATEGY_MAP.keys())

# ═══ 工具 ═══
def now_str() -> str:
    return f"{datetime.now(ET).strftime('%H:%M:%S ET')} / {datetime.now(SGT).strftime('%H:%M:%S SGT')}"

def send_telegram(title: str, body: str, silent: bool = False):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    try:
        import urllib.request, urllib.parse
        msg = f"<b>{title}</b>\n\n<pre>{body}</pre>"
        data = urllib.parse.urlencode({
            "chat_id": chat_id, "text": msg, "parse_mode": "HTML",
            "disable_notification": "true" if silent else "false",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, method="POST")
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"   ⚠️ Telegram 失败: {e}")

def alert(title: str, body: str, strong: bool = True):
    print(f"\n{'═' * 70}\n{title}\n{body}\n{'═' * 70}")
    if strong:
        print("\a", end="", flush=True)
        try:
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{body[:200]}" with title "{title}" sound name "Glass"'],
                capture_output=True, timeout=3)
        except Exception:
            pass
    send_telegram(title, body, silent=not strong)

def log_event(payload: dict):
    payload["log_time"] = datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")

# ═══ K 线数据加载 (LongPort) ═══
def fetch_recent_bars(quote_ctx, symbol: str, count: int = 50) -> pd.DataFrame:
    """拉最近 N 根 5m K 线, 转成策略期望的 DataFrame 格式"""
    try:
        bars = list(quote_ctx.candlesticks(symbol, Period.Min_5, count, AdjustType.NoAdjust))
    except Exception as e:
        print(f"   ⚠️ {symbol} 拉 K 线失败: {e}")
        return pd.DataFrame()
    if not bars: return pd.DataFrame()
    rows = []
    for b in bars:
        ts_et = b.timestamp.astimezone(ET)
        rows.append({"timestamp": ts_et,
                     "Open": float(b.open), "High": float(b.high),
                     "Low": float(b.low), "Close": float(b.close),
                     "Volume": int(b.volume)})
    df = pd.DataFrame(rows).set_index("timestamp")
    return df

def split_today_full(df: pd.DataFrame) -> tuple:
    """
    切分 today_df + full_df, 关键: 只取**常规交易时段**(09:30-15:55 ET).
    LongPort 5m K 线包含盘前 (04:00 ET 起) 数据, 盘前流动性差 + 价格跳跃,
    若不过滤策略会用盘前 K 线当 OR, 在 ET 5 AM 误触发 (5/1 已踩坑).
    """
    today_et = datetime.now(ET).date()
    rth_start = dtime(9, 30)
    rth_end = dtime(15, 55)
    # full_df 也只保留 RTH (避免策略指标算到盘前 V 突变)
    full_rth = df[df.index.map(lambda x: rth_start <= x.time() < rth_end)]
    df_today = full_rth[full_rth.index.map(lambda x: x.date() == today_et)]
    return df_today, full_rth

# ═══ 已派发记录 (防止重复 dispatch) ═══
_DISPATCHED_TODAY: set = set()  # {symbol} - 当天已派发过策略

def reset_dispatch_state():
    global _DISPATCHED_TODAY
    _DISPATCHED_TODAY = set()

# ═══ 策略派发 ═══
def dispatch_strategies(quote_ctx):
    """每个轮询周期调用. 对每只票: 拉数据 → 跑策略 → 派发"""
    # 安全保险: 只在开盘后派发 (盘前数据流动性差, 易误触发)
    n_et = datetime.now(ET)
    if n_et.time() < MARKET_OPEN:
        return  # 盘前不派发
    today_et = n_et.date()
    for symbol in TICKERS:
        if symbol in _DISPATCHED_TODAY:
            continue   # 今天已派发过 (idempotent in-memory)
        strategy_name = STRATEGY_MAP[symbol]
        strategy_fn = STRATEGIES.get(strategy_name)
        if strategy_fn is None:
            print(f"   ⚠️ {symbol} 策略 {strategy_name} 不存在")
            continue

        full_df = fetch_recent_bars(quote_ctx, symbol, count=50)
        if full_df.empty:
            continue
        today_df, _full = split_today_full(full_df)
        if today_df.empty:
            continue
        # 美股开盘前 (today_df 长度 < 1) 跳过
        try:
            plans = strategy_fn(today_df, full_df)
        except Exception as e:
            print(f"   ⚠️ {symbol} 策略 {strategy_name} 报错: {e}")
            continue
        if not plans:
            continue

        # 派发 plan (long-only, 取第一个)
        plan = plans[0]
        if plan.side != "long":
            continue   # 我们只做多

        result = live_executor.place_entry(symbol, plan, is_replay=False)
        if result.get("ok"):
            _DISPATCHED_TODAY.add(symbol)
            entry_px = result["limit_px"]; stop_px = result["stop_px"]; tp_px = result["tp_px"]
            risk_pct = (entry_px - stop_px) / entry_px * 100
            reward_pct = (tp_px - entry_px) / entry_px * 100
            title = f"🚀 {symbol} {strategy_name} 信号 ({plan.order_type})"
            body = (f"时间: {now_str()}\n"
                    f"策略: {strategy_name}\n"
                    f"入场: {entry_px}  ({plan.order_type})\n"
                    f"止损: {stop_px}  ({-risk_pct:+.2f}%)\n"
                    f"止盈: {tp_px}  ({reward_pct:+.2f}%)\n"
                    f"股数: {result['qty']}\n"
                    f"备注: {plan.note}")
            alert(title, body, strong=True)
            log_event({"event": "entry_placed", "symbol": symbol, "strategy": strategy_name,
                       "order_type": plan.order_type, "limit_px": entry_px,
                       "stop_px": stop_px, "tp_px": tp_px, "qty": result["qty"],
                       "note": plan.note, "entry_id": result["entry_id"]})
        else:
            reason = result.get("reason", "")
            if "幂等" in reason or "回放" in reason:
                _DISPATCHED_TODAY.add(symbol)   # 已下过, 加入 dispatched 防止再算
            else:
                # dry-run 或其他, 也跳过免得每周期算
                _DISPATCHED_TODAY.add(symbol)
                print(f"   💤 {symbol} {strategy_name}: {reason}")


# ═══ 主循环 ═══
def main():
    missing = [k for k in ["LONGPORT_APP_KEY", "LONGPORT_APP_SECRET", "LONGPORT_ACCESS_TOKEN"]
               if not os.environ.get(k)]
    if missing:
        print(f"❌ 缺环境变量: {missing}"); sys.exit(1)

    print("=" * 70)
    print(f"🟢 ORB Multi-Strategy 实盘 (Plan-Driven)")
    print(f"   {now_str()}")
    print(f"   监测 {len(TICKERS)} 只 (按 best_strategy_per_ticker.csv):")
    for sym in TICKERS:
        print(f"     {sym:<10}  →  {STRATEGY_MAP[sym]}")
    print("=" * 70)

    print("\n🔌 连接 LongPort...")
    cfg = Config.from_env()
    quote_ctx = QuoteContext(cfg)
    print("   ✅ 连接成功")

    # 初始化下单模块
    live_executor.init()

    # 打印当前账户状态
    if live_executor._LIVE:
        try:
            from longport.openapi import TradeContext
            tc = TradeContext(cfg)
            ch_names = []
            for ch in tc.stock_positions().channels:
                ch_names.append(ch.account_channel)
            print(f"   📊 账户类型: {' / '.join(ch_names)}")
            if "lb_papertrading" in ch_names:
                print("   ✅ 确认是模拟盘")
            else:
                print("   ⚠️ 警告: 不是 lb_papertrading!")
        except Exception as e:
            print(f"   ⚠️ 查账户失败: {e}")

    log_event({"event": "session_start", "tickers": TICKERS, "strategies": STRATEGY_MAP})

    # 启动通知
    send_telegram(
        "🟢 多策略 ORB 实盘启动",
        f"时间: {now_str()}\n"
        f"监测 {len(TICKERS)} 只\n"
        f"模式: {'LIVE' if live_executor._LIVE else 'DRY-RUN'}\n"
        f"策略分布:\n  ORB5_Z: {sum(1 for v in STRATEGY_MAP.values() if v == 'ORB5_Z')}\n"
        f"  ORB15_VWAP: {sum(1 for v in STRATEGY_MAP.values() if v == 'ORB15_VWAP')}\n"
        f"  DC20: {sum(1 for v in STRATEGY_MAP.values() if v == 'DC20')}\n"
        f"  ST_10_3: {sum(1 for v in STRATEGY_MAP.values() if v == 'ST_10_3')}\n"
        f"  VWAP_PB: {sum(1 for v in STRATEGY_MAP.values() if v == 'VWAP_PB')}",
        silent=True)

    print(f"\n🔄 主循环启动 (每 {POLL_INTERVAL}s 派发一轮)")
    last_status_print = time.time()
    last_dispatch = 0.0
    force_close_done = False
    last_dispatch_day = None

    try:
        while True:
            n_et = datetime.now(ET)

            # 周末退出
            if n_et.weekday() >= 5:
                print(f"\n{now_str()} — 周末, 退出")
                break

            # 跨日重置 dispatched
            if last_dispatch_day != n_et.date():
                reset_dispatch_state()
                last_dispatch_day = n_et.date()

            # 收盘前 5min 强平
            if not force_close_done and n_et.time() >= FORCE_CLOSE_AT and n_et.time() < MARKET_CLOSE:
                try: live_executor.force_close_all()
                except Exception as e: print(f"\n⚠️ 强平失败: {e}")
                force_close_done = True

            if n_et.time() >= MARKET_CLOSE:
                print(f"\n{now_str()} — 已收盘, 退出")
                break

            # 主派发 + fill 检查
            if time.time() - last_dispatch >= POLL_INTERVAL:
                last_dispatch = time.time()
                try:
                    dispatch_strategies(quote_ctx)
                    live_executor.check_fills_and_arm_brackets()
                except Exception as e:
                    print(f"   ⚠️ 派发/fill 检查异常: {e}")

            # 5min 一次心跳 + OCO
            if time.time() - last_status_print > 300:
                last_status_print = time.time()
                try: live_executor.reconcile_oco()
                except Exception: pass
                dispatched_n = len(_DISPATCHED_TODAY)
                pending_n = len(live_executor._PENDING_ENTRIES)
                open_n = sum(1 for r in live_executor._OPEN_POSITIONS.values() if not r.get("closed"))
                print(f"\n[{now_str()}] 心跳: 已派发 {dispatched_n}/{len(TICKERS)}  待成交 {pending_n}  持仓 {open_n}")

            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n⛔ 用户中断 ({now_str()})")
        log_event({"event": "user_interrupt"})

    log_event({"event": "session_end"})
    send_telegram("📊 收盘", f"时间: {now_str()}\n请在 LongPort App 查 P&L", silent=True)


if __name__ == "__main__":
    main()
