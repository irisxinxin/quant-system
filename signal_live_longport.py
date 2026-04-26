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
    from signals.orb_strategy import CORE_PORTFOLIO, TICKER_CONFIG
    _custom = os.environ.get("ORB_TICKERS", "").strip()
    _watch = _custom.split(",") if _custom else CORE_PORTFOLIO
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

# 日志
LOG_FILE = Path(__file__).parent / "signals_live_longport.jsonl"

# ═══════════════════════════════════════════════════════════════════════
# 状态机
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TickerState:
    symbol: str
    bars_today: list = field(default_factory=list)
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    or_range: Optional[float] = None
    or_range_pct: Optional[float] = None
    or_formed: bool = False
    too_narrow: bool = False
    breakout_alerted: bool = False
    last_day: Optional[str] = None

    def reset_for_new_day(self, new_day: str):
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
        if self.bars_today and self.bars_today[-1]["ts"] == bar_dt_et:
            self.bars_today[-1] = {
                "ts": bar_dt_et,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            }
        else:
            self.bars_today.append({
                "ts": bar_dt_et,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            })

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

        # 检测突破 (只检查最新一根)
        if self.or_formed and not self.too_narrow and not self.breakout_alerted:
            current = self.bars_today[-1]
            if current["high"] > self.or_high and current["close"] > self.or_high:
                # RVOL
                lookback_bars = self.bars_today[-RVOL_LOOKBACK-1:-1]
                if len(lookback_bars) >= 5:  # 至少 5 根才有意义
                    avg_vol = sum(b["volume"] for b in lookback_bars) / len(lookback_bars)
                    rvol = current["volume"] / max(avg_vol, 1)
                    if rvol >= RVOL_THRESHOLD:
                        self.breakout_alerted = True
                        return {"event": "breakout", "rvol": rvol, "bar": current}
        return None


# ═══════════════════════════════════════════════════════════════════════
# 全局状态 + 工具
# ═══════════════════════════════════════════════════════════════════════

states = {symbol: TickerState(symbol) for symbol in TICKERS}


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
    print(f"\n{'═' * 70}")
    print(title)
    print(body)
    print('═' * 70)

    if strong:
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

    # Telegram: 强信号默认有声推送, 弱信号静默 (避免半夜被吵)
    send_telegram(title, body, silent=not strong)


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

    elif result["event"] == "breakout":
        rvol = result["rvol"]
        bar = result["bar"]
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
            f"\n⚠️ 03:55 SGT 前手动平仓"
        )
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
    print("\n📥 加载今日历史 K 线...")
    symbols = list(TICKERS.keys())
    for sym in symbols:
        try:
            historical = ctx.candlesticks(sym, Period.Min_5, 100, AdjustType.NoAdjust)
            count = 0
            for bar in historical:
                states[sym].on_bar(bar)
                count += 1
            print(f"   {sym:<10} 加载 {count} 根历史 K 线")
        except Exception as e:
            print(f"   ⚠️ {sym} 历史加载失败: {e}")

    # 注册回调
    ctx.set_on_candlestick(on_candlestick)

    # 订阅实时推送
    print(f"\n📡 订阅实时 5m K 线推送...")
    ctx.subscribe_candlesticks(symbols, Period.Min_5)
    print("   ✅ 订阅成功，等待推送...")

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

    # 主循环：定期打印心跳 + 检查是否到收盘
    last_status_print = time.time()
    try:
        while True:
            n_et = datetime.now(ET)

            # 周末或非交易日
            if n_et.weekday() >= 5:
                print(f"\n{now_str()} — 周末，退出")
                break

            # 收盘后退出
            if n_et.time() >= MARKET_CLOSE:
                print(f"\n{now_str()} — 已收盘，退出")
                break

            # 每 5 分钟打印一次状态
            if time.time() - last_status_print > 300:
                last_status_print = time.time()
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

            time.sleep(10)

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
