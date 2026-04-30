"""
signal_live_longport.py — ORB 实时信号 (LongPort 实时数据源)

使用方法:
  1. 配置环境变量 LONGPORT_APP_KEY / LONGPORT_APP_SECRET / LONGPORT_ACCESS_TOKEN
  2. 美股开盘前启动:  python3 signal_live_longport.py
  3. 触发信号 → 响铃 + macOS 通知 + 终端打印下单参数
  4. 自己去 longbridge / moomoo paper 账户手动下单

支持:
  - 实时 5m K 线 WebSocket 推送 (无延迟)
  - 多股票并行监测
  - 自动会话切换 (跨日重置)
  - 强/弱信号分级 (强信号才会通过通知打扰你)
"""
import os
import sys
import time
import json
import signal as signal_module
import subprocess
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from longport.openapi import (
        Config, QuoteContext, Period, AdjustType, SubType, PushCandlestick
    )
except ImportError:
    print("❌ longport 包未安装")
    print("   运行:  pip3 install --break-system-packages longport")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

# 监测股票 — 从 signals.orb_strategy 同步配置
# 默认监测核心 9 只 (高胜率 ≥ 64%); 改环境变量 ORB_TICKERS 可自定义
sys.path.insert(0, str(Path(__file__).parent))
try:
    from signals.orb_strategy import TICKER_CONFIG
    # 监测列表: 真实回测 (限价入场, qty=10) 下 1y 美元 P&L 显著为正的 9 只
    # 已剔: AMD, MSFL, NVDS, HOOD, MSFU, EOSE, SOXS, CIFR (1y 负贡献)
    # 已剔: SOXL (-$59), NVDL (-$16)  — 1y 实际美元负贡献
    # 已剔: TSLZ (+$26)  — 价格低, 10 股美元 P&L 太小, 资金占用产出比差
    TOP9_PORTFOLIO = [
        "CRWV.US", "RKLB.US", "OKLO.US", "INTC.US", "IREN.US",  # 🟢 1y 大正
        "TSLL.US", "PLTR.US", "NBIS.US", "AMZN.US",              # 🟡 1y 正贡献
    ]
    _custom = os.environ.get("ORB_TICKERS", "").strip()
    _watch = _custom.split(",") if _custom else TOP9_PORTFOLIO
    TICKERS = {
        sym: {"entry_slip": TICKER_CONFIG[sym]["entry_slip"],
              "desc": TICKER_CONFIG[sym]["category"]}
        for sym in _watch if sym in TICKER_CONFIG
    }
except ImportError:
    # fallback: 写死核心 9 只
    TICKERS = {
        "AMZN.US": {"entry_slip": 0.0002, "desc": "超大盘"},
        "PLTR.US": {"entry_slip": 0.0003, "desc": "AI大盘"},
        "RKLB.US": {"entry_slip": 0.0008, "desc": "太空小盘"},
        "IREN.US": {"entry_slip": 0.0015, "desc": "BTC挖矿小盘"},
        "INTC.US": {"entry_slip": 0.0003, "desc": "半导体大盘"},
        "TSLL.US": {"entry_slip": 0.0005, "desc": "2x多 TSLA"},
        "TSLZ.US": {"entry_slip": 0.0008, "desc": "2x空 TSLA"},
        "SOXL.US": {"entry_slip": 0.0003, "desc": "3x多 半导体"},
        "SOXS.US": {"entry_slip": 0.0005, "desc": "3x空 半导体"},
    }

# 策略参数
OR_BARS = 3                         # 3 根 5m K 线 = 15 分钟 OR
TARGET_R = 2.0                       # 止盈 = 入场 + 2 × OR 范围
RVOL_THRESHOLD = 1.5                 # 突破时 RVOL ≥ 1.5 才有效
RVOL_LOOKBACK = 20                   # 滚动均量回看根数
MIN_OR_RANGE_PCT = 0.003             # OR 范围 < 0.3% 视为过窄

# 强信号判定 (满足才发出"叮咚响铃")
STRONG_RVOL = 2.5                    # 强信号 RVOL 门槛
STRONG_OR_RANGE_PCT = 0.020          # 强信号 OR 范围门槛 (2%)

# 时区
ET = ZoneInfo("US/Eastern")
SGT = ZoneInfo("Asia/Singapore")

# 时段
MARKET_OPEN  = dtime(9, 30)
USER_CUTOFF  = dtime(13, 0)
MARKET_CLOSE = dtime(15, 55)
FORCE_CLOSE_AT = dtime(15, 50)   # 提前 5 分钟强平所有持仓

# 日志
LOG_FILE = Path(__file__).parent / "signals_live_longport.jsonl"

# ═══════════════════════════════════════════════════════════════════════
# 状态机
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TickerState:
    symbol: str
    bars_today: list = field(default_factory=list)
    # 跨天滚动 V 缓冲, 最近 RVOL_LOOKBACK 根 (含当前) — lenient 算法 (= Pine ta.sma(volume, 20))
    recent_volumes: list = field(default_factory=list)
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    or_range: Optional[float] = None
    or_range_pct: Optional[float] = None
    or_formed: bool = False
    too_narrow: bool = False
    breakout_alerted: bool = False
    or_entry_placed: bool = False    # 标记: OR 形成后已挂限价 LIT 单
    last_day: Optional[str] = None

    def reset_for_new_day(self, new_day: str):
        # 注意: recent_volumes 跨天保留, 不清空 (lenient 算法跨天滚动)
        self.bars_today = []
        self.or_high = None
        self.or_low = None
        self.or_range = None
        self.or_range_pct = None
        self.or_formed = False
        self.too_narrow = False
        self.breakout_alerted = False
        self.last_day = new_day

    def on_bar(self, bar) -> Optional[dict]:
        """处理一根新的 5m K 线，返回事件 (or None)"""
        bar_dt_et = bar.timestamp.astimezone(ET)
        bar_day = bar_dt_et.date().isoformat()
        bar_time = bar_dt_et.time()

        # 跨日切换
        if self.last_day != bar_day:
            self.reset_for_new_day(bar_day)

        # 只处理常规交易时段
        if bar_time < MARKET_OPEN or bar_time >= MARKET_CLOSE:
            return None

        # 重复 K 线? (LongPort 推送可能更新当前未完成的 K 线)
        # 用 timestamp 判断是否是新 K 线
        new_bar = {
            "ts": bar_dt_et,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
        }
        if self.bars_today and self.bars_today[-1]["ts"] == bar_dt_et:
            self.bars_today[-1] = new_bar
            if self.recent_volumes:
                self.recent_volumes[-1] = new_bar["volume"]
        else:
            self.bars_today.append(new_bar)
            self.recent_volumes.append(new_bar["volume"])
            if len(self.recent_volumes) > RVOL_LOOKBACK:
                self.recent_volumes.pop(0)

        # OR 形成 (前 3 根)
        if not self.or_formed and len(self.bars_today) >= OR_BARS:
            first_3 = self.bars_today[:OR_BARS]
            self.or_high = max(b["high"] for b in first_3)
            self.or_low = min(b["low"] for b in first_3)
            self.or_range = self.or_high - self.or_low
            mid = (self.or_high + self.or_low) / 2
            self.or_range_pct = self.or_range / mid
            self.too_narrow = self.or_range_pct < MIN_OR_RANGE_PCT
            self.or_formed = True
            return {"event": "or_formed"}

        # 检测突破 (仅作显示/Telegram 通知用; 真正下单已在 OR 形成时挂 LIT 限价单)
        # RVOL 用 lenient (跨天滚动 SMA 20) — 与限价入场组合下回测最优
        if self.or_formed and not self.too_narrow and not self.breakout_alerted:
            current = self.bars_today[-1]
            if current["high"] > self.or_high and current["close"] > self.or_high:
                if len(self.recent_volumes) >= 5:
                    avg_vol = sum(self.recent_volumes) / len(self.recent_volumes)
                    rvol = current["volume"] / max(avg_vol, 1)
                    if rvol >= RVOL_THRESHOLD:
                        self.breakout_alerted = True
                        return {"event": "breakout", "rvol": rvol, "bar": current}
        return None


# ═══════════════════════════════════════════════════════════════════════
# 全局状态 + 工具
# ═══════════════════════════════════════════════════════════════════════

states = {symbol: TickerState(symbol) for symbol in TICKERS}

# 历史回测指标 (启动时一次性计算, 信号触发时附加到消息中)
HISTORICAL_STATS = {}

# 启动时回放历史 K 线: 标记为 True, 避免触发响铃/macOS 通知, 仅 Telegram 静默推送 [补发]
IS_REPLAY = False


def compute_historical_stats():
    """启动时算一次每只票的 18 月历史回测指标"""
    try:
        from signals.orb_strategy import (
            backtest_signals_history, summary_stats, TICKER_CONFIG as TC
        )
    except ImportError:
        return

    print("\n📊 计算 18 月历史回测指标...")
    for sym in TICKERS:
        try:
            trades = backtest_signals_history(sym)
            stats = summary_stats(trades) if not trades.empty else {}
            cfg = TC.get(sym, {})
            HISTORICAL_STATS[sym] = {
                "stats": stats,
                "score": cfg.get("score", 3),
                "category": cfg.get("category", ""),
                "suitable": cfg.get("suitable_intraday", True),
            }
            print(f"   {sym:<10} 胜率 {stats.get('positive_pct', 0):>4.1f}%  "
                  f"累计 {stats.get('cumulative_return_pct', 0):>+7.1f}%  "
                  f"PF {stats.get('profit_factor', 0):<5}  "
                  f"交易 {stats.get('trades', 0):>3}笔")
        except Exception as e:
            print(f"   {sym:<10} ⚠️ 计算失败: {e}")
            HISTORICAL_STATS[sym] = None


def format_historical_section(symbol):
    """生成信号附加的"历史表现"段落"""
    h = HISTORICAL_STATS.get(symbol)
    if not h or not h.get("stats"):
        return ""
    s = h["stats"]
    score_label = {1: "🎯 核心 (高胜率)", 2: "✅ 次要", 3: "⚠️ 不推荐"}.get(h["score"], "")
    return (
        f"\n📈 历史表现 (18 月回测):\n"
        f"   日内胜率: {s.get('positive_pct', 0):.1f}%\n"
        f"   累计收益: {s.get('cumulative_return_pct', 0):+.1f}%\n"
        f"   盈亏比 PF: {s.get('profit_factor', 0)}\n"
        f"   交易笔数: {s.get('trades', 0)}\n"
        f"   平均盈/亏: +{s.get('avg_win_pct', 0)}% / {s.get('avg_loss_pct', 0)}%\n"
        f"   评级: {score_label}"
    )


def now_str() -> str:
    return f"{datetime.now(ET).strftime('%H:%M:%S ET')} / {datetime.now(SGT).strftime('%H:%M:%S SGT')}"


def send_telegram(title: str, body: str, silent: bool = False):
    """发送 Telegram 通知 (手机推送, 极稳)"""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return  # 未配置 Telegram, 跳过

    try:
        import urllib.request
        import urllib.parse
        msg = f"<b>{title}</b>\n\n<pre>{body}</pre>"
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "HTML",
            "disable_notification": "true" if silent else "false",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=data,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"   ⚠️ Telegram 发送失败: {e}")


def alert(title: str, body: str, strong: bool = True):
    """终端打印 + 响铃 + macOS 通知 + Telegram 推送"""
    if IS_REPLAY:
        title = "[补发] " + title
    print(f"\n{'═' * 70}")
    print(title)
    print(body)
    print('═' * 70)

    # 历史回放期间不响铃 / 不弹 macOS 通知 (避免启动时一片叮咚), 仅静默 Telegram
    if strong and not IS_REPLAY:
        print("\a", end="", flush=True)
        try:
            sound = "Glass"
            subprocess.run(
                ["osascript", "-e",
                 f'display notification "{body[:200]}" with title "{title}" sound name "{sound}"'],
                capture_output=True, timeout=3,
            )
        except Exception:
            pass

    # Telegram: 强信号有声推送; 弱信号或回放统一静默
    send_telegram(title, body, silent=(not strong) or IS_REPLAY)


def log_event(payload: dict):
    payload["log_time"] = datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def on_candlestick(symbol: str, event: PushCandlestick):
    state = states.get(symbol)
    if state is None:
        return

    result = state.on_bar(event)
    if result is None:
        return

    config = TICKERS[symbol]

    if result["event"] == "or_formed":
        status = "✅ 可交易" if not state.too_narrow else "⚠️ OR 过窄 SKIP"
        msg = (f"📊 [{now_str()}] {symbol} OR 已形成\n"
               f"   High={state.or_high:.3f}  Low={state.or_low:.3f}  "
               f"范围={state.or_range_pct*100:.2f}%  {status}")
        print(msg)
        log_event({
            "event": "or_formed", "symbol": symbol,
            "or_high": state.or_high, "or_low": state.or_low,
            "or_range_pct": state.or_range_pct * 100,
            "too_narrow": state.too_narrow,
        })
        # OR 形成且不过窄 → 立刻挂 LIT 限价入场单 (代替原"突破时市价单")
        if not state.too_narrow and not state.or_entry_placed:
            try:
                import live_executor
                slip = config["entry_slip"]
                r = live_executor.place_or_entry(
                    symbol, state.or_high, state.or_low, state.or_range_pct,
                    entry_slip=slip, is_replay=IS_REPLAY)
                if r.get("ok"):
                    print(f"   📌 {symbol} LIT 入场单已挂 @ {r['target_px']}  id={r['entry_id']}")
                    state.or_entry_placed = True
                else:
                    print(f"   💤 {symbol} 未挂入场单: {r.get('reason')}")
            except Exception as e:
                print(f"   ⚠️ {symbol} 挂入场单异常: {e}")

    elif result["event"] == "breakout":
        rvol = result["rvol"]
        bar = result["bar"]

        # ── 信号过滤器: 反向 ETF 配对趋势 + OR 范围异常 ──
        try:
            from signals.orb_strategy import filter_signal
            passed, reason = filter_signal(symbol, state.or_range_pct)
            if not passed:
                msg = (f"⛔ 过滤器跳过 {symbol} 突破信号\n"
                       f"   时间: {bar['ts'].strftime('%H:%M:%S ET')}\n"
                       f"   原因: {reason}")
                print(f"\n{msg}")
                log_event({
                    "event": "breakout_filtered", "symbol": symbol,
                    "reason": reason, "or_range_pct": state.or_range_pct * 100,
                    "rvol": rvol, "bar_time": str(bar["ts"]),
                })
                # 静默 Telegram (让你知道哪些被过滤了, 但不响铃)
                send_telegram(f"⛔ {symbol} 信号被过滤", reason, silent=True)
                return  # 不继续处理这个突破
        except ImportError:
            pass  # 过滤器加载失败时, 让信号通过 (向下兼容)

        slip = config["entry_slip"]
        entry = state.or_high * (1 + slip)
        stop = state.or_low
        tp = state.or_high + TARGET_R * state.or_range
        risk_pct = (entry - stop) / entry * 100
        reward_pct = (tp - entry) / entry * 100

        # 强弱判定
        is_strong = rvol >= STRONG_RVOL and state.or_range_pct >= STRONG_OR_RANGE_PCT

        title = f"{'🚀🚀🚀' if is_strong else '🚀'} {symbol} 多头突破" + (" [强]" if is_strong else "")
        body = (
            f"突破时间: {bar['ts'].strftime('%H:%M:%S ET')}\n"
            f"突破价: {bar['high']:.3f}  RVOL: {rvol:.2f}\n"
            f"OR 范围: {state.or_range_pct*100:.2f}%\n"
            f"\n📥 下单参数:\n"
            f"   入场:  {entry:.3f}\n"
            f"   止损:  {stop:.3f}  ({risk_pct:+.2f}%)\n"
            f"   止盈:  {tp:.3f}  ({reward_pct:+.2f}%)\n"
            f"   R:R:   {TARGET_R}\n"
            + format_historical_section(symbol) +
            f"\n\n⚠️ 03:55 SGT 前手动平仓"
        )
        # 价格突破事件: 仅通知 (真正下单已在 OR 形成时挂 LIT 限价单, 价格触到自动成交)
        body += "\n\n💡 LIT 限价入场单已在 OR 形成时挂出, 价格触到自动成交"
        alert(title, body, strong=is_strong)
        log_event({
            "event": "breakout", "symbol": symbol, "rvol": rvol,
            "entry": entry, "stop": stop, "tp": tp,
            "or_range_pct": state.or_range_pct * 100,
            "is_strong": is_strong,
            "bar_time": str(bar["ts"]),
        })


# ═══════════════════════════════════════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════════════════════════════

def main():
    # 验证环境变量
    missing = [k for k in ["LONGPORT_APP_KEY", "LONGPORT_APP_SECRET", "LONGPORT_ACCESS_TOKEN"]
               if not os.environ.get(k)]
    if missing:
        print("❌ 缺少环境变量:")
        for k in missing:
            print(f"   {k}")
        print("\n请先设置:")
        print('   export LONGPORT_APP_KEY="..."')
        print('   export LONGPORT_APP_SECRET="..."')
        print('   export LONGPORT_ACCESS_TOKEN="..."')
        sys.exit(1)

    print("=" * 70)
    print(f"🟢 ORB 实时信号 (LongPort 数据源)")
    print(f"   {now_str()}")
    print(f"   监测 {len(TICKERS)} 只股票:")
    for sym, cfg in TICKERS.items():
        print(f"     {sym:<10}  {cfg['desc']}")
    print(f"   日志: {LOG_FILE}")
    print("=" * 70)

    # 建立连接
    print("\n🔌 连接 LongPort OpenAPI...")
    config = Config.from_env()
    ctx = QuoteContext(config)
    print("   ✅ 连接成功")

    # 拉历史 K 线作为上下文 (如果脚本启动时已经过开盘)
    # 初始化模拟盘下单模块 (必须在历史回放前调用, 否则今天的 OR 形成时 _LIVE 还是 False)
    try:
        import live_executor
        live_executor.init()
    except Exception as e:
        print(f"   ⚠️ live_executor 初始化失败: {e}")

    # 关键: 区分"昨天/前天 K 线 (回放, IS_REPLAY=True 不下单)"和"今天 K 线 (live missed, IS_REPLAY=False 要下单)"
    global IS_REPLAY
    print("\n📥 加载历史 K 线 (昨天前回放静默, 今天 K 线视为 live)...")
    symbols = list(TICKERS.keys())
    today_et = datetime.now(ET).date()
    for sym in symbols:
        try:
            historical = ctx.candlesticks(sym, Period.Min_5, 100, AdjustType.NoAdjust)
            count = 0
            for bar in historical:
                bar_date = bar.timestamp.astimezone(ET).date()
                IS_REPLAY = bar_date < today_et   # 昨天/前天 = 回放; 今天 = live
                on_candlestick(sym, bar)
                count += 1
            print(f"   {sym:<10} 加载 {count} 根历史 K 线")
        except Exception as e:
            print(f"   ⚠️ {sym} 历史加载失败: {e}")
    IS_REPLAY = False

    # 计算历史回测指标 (用于信号附加显示)
    compute_historical_stats()

    # 改用主动轮询 (替代 WebSocket subscribe), 每 30s 拉最新 5 根 5m K 线, on_bar 内部 dedup
    print(f"\n🔄 改用主动轮询 (每 30s 拉最新 5 根 5m K 线, 不依赖 WebSocket 推送)")

    log_event({"event": "session_start", "tickers": symbols})

    # 开机 Telegram 通知 (确认 bot 在线)
    send_telegram(
        "🟢 ORB 信号机启动",
        f"时间: {now_str()}\n"
        f"监测 {len(symbols)} 只:\n  {' / '.join(s.replace('.US','') for s in symbols)}\n\n"
        f"OR 形成: 9:30-9:45 ET (21:30-21:45 SGT)\n"
        f"今日有信号会推送到这里",
        silent=True,
    )

    # 主循环: 轮询拉 K 线 + 心跳 + OCO 调和 + 强平
    last_status_print = time.time()
    last_poll = 0.0
    POLL_INTERVAL = 30   # 每 30s 主动拉一次
    force_close_done = False
    try:
        while True:
            n_et = datetime.now(ET)

            # 周末或非交易日
            if n_et.weekday() >= 5:
                print(f"\n{now_str()} — 周末，退出")
                break

            # 收盘前 5 分钟: 撤所有未成交子单 + 市价强平 (一次性)
            if not force_close_done and n_et.time() >= FORCE_CLOSE_AT and n_et.time() < MARKET_CLOSE:
                try:
                    import live_executor
                    live_executor.force_close_all()
                except Exception as e:
                    print(f"\n⚠️ 强平失败: {e}")
                force_close_done = True

            # 收盘后退出
            if n_et.time() >= MARKET_CLOSE:
                print(f"\n{now_str()} — 已收盘，退出")
                break

            # 主动轮询: 每 30 秒拉一次最新 5 根 5m K 线, 喂给 on_candlestick (内部 dedup)
            if time.time() - last_poll >= POLL_INTERVAL:
                last_poll = time.time()
                for sym in symbols:
                    try:
                        latest = ctx.candlesticks(sym, Period.Min_5, 5, AdjustType.NoAdjust)
                        for bar in latest:
                            on_candlestick(sym, bar)
                    except Exception as e:
                        print(f"   ⚠️ {sym} 轮询失败: {e}")

            # 每 30s 轮询时也检查 LIT 入场单是否成交 + arm 子单 (重要!)
            if time.time() - last_poll < 5:  # 紧跟轮询调用
                try:
                    import live_executor
                    live_executor.check_fills_and_arm_brackets(ctx)
                except Exception as e:
                    print(f"   ⚠️ check_fills 异常: {e}")

            # 每 5 分钟打印一次状态 + OCO 调和
            if time.time() - last_status_print > 300:
                last_status_print = time.time()
                try:
                    import live_executor
                    live_executor.reconcile_oco()
                except Exception:
                    pass
                print(f"\n[{now_str()}] 心跳: ", end="")
                summary = []
                for sym, st in states.items():
                    if st.breakout_alerted:
                        summary.append(f"{sym}=突破")
                    elif st.too_narrow:
                        summary.append(f"{sym}=窄")
                    elif st.or_formed:
                        summary.append(f"{sym}=待")
                    else:
                        summary.append(f"{sym}={len(st.bars_today)}/3")
                print(" | ".join(summary))

            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n⛔ 用户中断 ({now_str()})")
        log_event({"event": "user_interrupt"})

    # 收盘汇总
    print(f"\n📊 当日汇总:")
    for sym, st in states.items():
        if st.breakout_alerted:
            print(f"   {sym:<10} ✅ 已触发突破")
        elif st.too_narrow:
            print(f"   {sym:<10} ⚪ OR 过窄跳过")
        elif st.or_formed:
            print(f"   {sym:<10} ➖ OR 形成但未突破")
        else:
            print(f"   {sym:<10} ❌ OR 未形成 (数据不足)")

    log_event({"event": "session_end"})

    # 收盘 Telegram 汇总
    triggered = [s for s, st in states.items() if st.breakout_alerted]
    narrow = [s for s, st in states.items() if st.too_narrow]
    no_break = [s for s, st in states.items() if st.or_formed and not st.too_narrow and not st.breakout_alerted]
    summary = f"⏰ 当日结束: {now_str()}\n\n"
    summary += f"🚀 触发: {len(triggered)} 只"
    if triggered: summary += f"\n  {', '.join(s.replace('.US','') for s in triggered)}"
    summary += f"\n➖ 未突破: {len(no_break)} 只"
    summary += f"\n⚪ OR 过窄: {len(narrow)} 只"
    summary += "\n\n请检查 paper 账户成交情况, 03:55 SGT 前手动平仓未触发的"
    send_telegram("📊 ORB 当日汇总", summary, silent=True)


if __name__ == "__main__":
    main()
