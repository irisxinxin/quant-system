"""
signal_live.py — ORB 实时信号监测器（paper / 手动下单版）

使用说明:
  每个美股交易日 9:30 ET (21:30 SGT) 前启动:
    python3 signal_live.py

  脚本会自动:
    1. 等待开盘 + 前 15 分钟形成 OR
    2. 9:45 ET 后监测每个标的的突破
    3. 突破触发 → 响铃 + 桌面通知 + 打印下单参数
    4. 到 13:00 ET (01:00 SGT+1) 停止

  手动操作:
    - 看到信号后去 moomoo / longbridge 下单
    - 用括号单: 限价入场 + 止损 + 止盈 (R:R=2)
    - 15:55 ET (03:55 SGT+1) 前手动平掉仍持仓的单 (或用定时订单)

数据源: yfinance (15分钟延迟，paper阶段可用)
后续升级: 切到 moomoo 实时 API (futu-api)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import json
import subprocess
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf


# ==============================================================
# 配置
# ==============================================================
TICKERS = [
    # (ticker, entry_slip, stop_slip, desc)
    ("OKLO", 0.0015, 0.0030, "核能小盘"),
    ("IREN", 0.0015, 0.0030, "BTC挖矿"),
    ("CIFR", 0.0015, 0.0030, "BTC挖矿"),
    ("CRWV", 0.0010, 0.0020, "AI算力"),
]

OR_BARS = 3                       # 3 根 5m K线 = 15 分钟 OR
TARGET_R = 2.0                     # 止盈 = 入场 + 2 × OR范围
MIN_OR_RANGE_PCT = 0.003           # OR 范围至少 0.3%
RVOL_THRESHOLD = 1.5               # 突破K线量能 >= 20根均量 × 1.5
RVOL_LOOKBACK_BARS = 20

ET = ZoneInfo("US/Eastern")
SGT = ZoneInfo("Asia/Singapore")

MARKET_OPEN_ET = dtime(9, 30)
OR_FORMED_ET = dtime(9, 45)
USER_CUTOFF_ET = dtime(13, 0)      # 用户最晚盯盘时间 (01:00 SGT)
MARKET_CLOSE_ET = dtime(16, 0)

POLL_INTERVAL_SEC = 60              # 每 60 秒查一次

LOG_FILE = Path(__file__).parent / "signals_live.jsonl"
SESSION_FILE = Path(__file__).parent / "signal_session.json"


# ==============================================================
# 工具函数
# ==============================================================
def now_et():
    return datetime.now(ET)


def now_sgt():
    return datetime.now(SGT)


def is_within_watch_hours():
    """当前是否在用户盯盘窗口内 (9:30-13:00 ET, 周一到周五)"""
    n = now_et()
    if n.weekday() >= 5:
        return False
    cur = n.time()
    return MARKET_OPEN_ET <= cur <= USER_CUTOFF_ET


def fmt_time():
    n_et = now_et()
    n_sgt = now_sgt()
    return f"{n_et.strftime('%H:%M:%S ET')} / {n_sgt.strftime('%H:%M:%S SGT')}"


def beep_and_notify(title: str, msg: str):
    """终端响铃 + macOS 桌面通知"""
    print("\a", end="", flush=True)   # 终端响铃
    try:
        # macOS 原生通知
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{msg}" with title "{title}" sound name "Glass"'],
            capture_output=True, timeout=3,
        )
    except Exception:
        pass


def log_event(payload: dict):
    """追加到 jsonl"""
    payload["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def fetch_intraday(ticker: str) -> pd.DataFrame:
    """拉今日 5m K 线 (yfinance 15 分钟延迟)"""
    try:
        df = yf.download(ticker, period="5d", interval="5m",
                         auto_adjust=True, progress=False, prepost=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        print(f"   ⚠️ {ticker} 数据获取失败: {e}")
        return pd.DataFrame()


def today_bars(df: pd.DataFrame) -> pd.DataFrame:
    """过滤出今日 ET 交易时段的 bar"""
    if df.empty:
        return df
    today = now_et().date()
    # 转换 index 到 ET
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df_et = df.tz_convert(ET)
    return df_et[df_et.index.date == today]


# ==============================================================
# 策略状态
# ==============================================================
class TickerState:
    def __init__(self, ticker, entry_slip, stop_slip, desc):
        self.ticker = ticker
        self.entry_slip = entry_slip
        self.stop_slip = stop_slip
        self.desc = desc
        self.or_high = None
        self.or_low = None
        self.or_range = None
        self.or_mid = None
        self.or_range_pct = None
        self.or_printed = False
        self.too_narrow = False
        self.breakout_alerted = False
        self.last_status_print = None

    def try_compute_or(self, today_df: pd.DataFrame) -> bool:
        """当今日已有至少 3 根 9:30-9:45 的 K 线时计算 OR"""
        if self.or_high is not None:
            return True
        if len(today_df) < OR_BARS:
            return False
        first_3 = today_df.head(OR_BARS)
        self.or_high = float(first_3["High"].max())
        self.or_low = float(first_3["Low"].min())
        self.or_range = self.or_high - self.or_low
        self.or_mid = (self.or_high + self.or_low) / 2
        self.or_range_pct = self.or_range / self.or_mid
        if self.or_range_pct < MIN_OR_RANGE_PCT:
            self.too_narrow = True
        return True

    def check_breakout(self, today_df: pd.DataFrame) -> dict:
        """检测是否突破 + 量能确认"""
        if self.too_narrow or self.breakout_alerted or self.or_high is None:
            return None
        if len(today_df) < OR_BARS + 1:
            return None

        # 取 OR 之后的所有 bar，找第一个突破
        post_or = today_df.iloc[OR_BARS:]
        for idx, row in post_or.iterrows():
            if row["High"] > self.or_high and row["Close"] > self.or_high:
                # RVOL 检查
                lookback_end = today_df.index.get_loc(idx)
                lookback_start = max(0, lookback_end - RVOL_LOOKBACK_BARS)
                vol_ma = today_df.iloc[lookback_start:lookback_end]["Volume"].mean()
                rvol = row["Volume"] / max(vol_ma, 1)
                if rvol >= RVOL_THRESHOLD:
                    entry = self.or_high * (1 + self.entry_slip)
                    stop = self.or_low
                    tp = self.or_high + TARGET_R * self.or_range
                    risk_pct = (entry - stop) / entry * 100
                    reward_pct = (tp - entry) / entry * 100
                    return {
                        "ticker": self.ticker,
                        "breakout_bar": idx.strftime("%H:%M:%S %Z"),
                        "break_price": float(row["High"]),
                        "break_close": float(row["Close"]),
                        "rvol": rvol,
                        "entry": entry,
                        "stop": stop,
                        "tp": tp,
                        "or_range_pct": self.or_range_pct * 100,
                        "risk_pct": risk_pct,
                        "reward_pct": reward_pct,
                    }
        return None


# ==============================================================
# 主循环
# ==============================================================
def main():
    states = {t[0]: TickerState(*t) for t in TICKERS}

    print("\n" + "=" * 72)
    print(f"🟢 ORB 实时信号监测器 启动")
    print(f"   当前: {fmt_time()}")
    print(f"   标的: {[t[0] for t in TICKERS]}")
    print(f"   盯盘: 09:30-13:00 ET (21:30-01:00 SGT 夏令时)")
    print(f"   数据: yfinance (~15分钟延迟)")
    print(f"   日志: {LOG_FILE}")
    print("=" * 72)

    # 如果还没到开盘，等到开盘
    while not is_within_watch_hours():
        n = now_et()
        if n.weekday() >= 5:
            print(f"⚠️ 今天是周末 ({n.strftime('%A')}), 退出")
            return
        if n.time() < MARKET_OPEN_ET:
            seconds_until_open = (
                datetime.combine(n.date(), MARKET_OPEN_ET, tzinfo=ET) - n
            ).total_seconds()
            if seconds_until_open > 60:
                mins = seconds_until_open / 60
                print(f"⏰ 距离开盘还有 {mins:.1f} 分钟 ({fmt_time()})，等待中...")
                time.sleep(min(300, seconds_until_open - 30))
                continue
            else:
                time.sleep(5)
                continue
        else:
            print(f"⏰ 已过盯盘截止时间 (13:00 ET)，退出")
            return

    log_event({"event": "session_start", "tickers": [t[0] for t in TICKERS]})

    loop_count = 0
    while is_within_watch_hours():
        loop_count += 1
        print(f"\n── [{fmt_time()}] 第 {loop_count} 轮检查 ──")

        for ticker_cfg in TICKERS:
            ticker = ticker_cfg[0]
            state = states[ticker]

            df = fetch_intraday(ticker)
            if df.empty:
                continue

            today = today_bars(df)
            if len(today) == 0:
                print(f"   {ticker:<6}: 无今日数据")
                continue

            # 尝试计算 OR
            if state.or_high is None:
                if state.try_compute_or(today):
                    # 第一次打印 OR
                    status = "✅ 可交易" if not state.too_narrow else "⚠️ OR 过窄, SKIP"
                    print(f"   {ticker:<6}: OR 已形成  H={state.or_high:.3f} "
                          f"L={state.or_low:.3f}  范围={state.or_range_pct*100:.2f}%  {status}")
                    log_event({
                        "event": "or_formed",
                        "ticker": ticker,
                        "or_high": state.or_high,
                        "or_low": state.or_low,
                        "or_range_pct": state.or_range_pct * 100,
                        "too_narrow": state.too_narrow,
                    })
                else:
                    elapsed_bars = len(today)
                    print(f"   {ticker:<6}: OR 形成中 ({elapsed_bars}/{OR_BARS} 根)")
                    continue

            # 已成过窄，跳过
            if state.too_narrow:
                continue

            # 检查突破
            if not state.breakout_alerted:
                signal = state.check_breakout(today)
                if signal:
                    state.breakout_alerted = True

                    # 打印信号 + 响铃
                    print("\n" + "🚨" * 24)
                    print(f"🚨🚨 突破信号! {ticker} ({state.desc})")
                    print("🚨" * 24)
                    print(f"   突破时间: {signal['breakout_bar']}")
                    print(f"   突破价格: {signal['break_price']:.3f}")
                    print(f"   RVOL:     {signal['rvol']:.2f}")
                    print(f"   OR 范围: {signal['or_range_pct']:.2f}%")
                    print()
                    print(f"   📥 下单参数 (moomoo/longbridge 手动下括号单):")
                    print(f"      方向:      BUY (多)")
                    print(f"      入场价:    {signal['entry']:.3f}")
                    print(f"      止损价:    {signal['stop']:.3f}   ({signal['risk_pct']:+.2f}%)")
                    print(f"      止盈价:    {signal['tp']:.3f}   ({signal['reward_pct']:+.2f}%)")
                    print(f"      R:R:       {TARGET_R:.1f}")
                    print()
                    print(f"   ⚠️  15:55 ET (03:55 SGT) 前手动平仓（如未触发止损/止盈）")
                    print("=" * 72)

                    beep_and_notify(
                        f"🚨 {ticker} ORB 突破",
                        f"Entry {signal['entry']:.2f} Stop {signal['stop']:.2f} TP {signal['tp']:.2f}"
                    )

                    log_event({"event": "breakout", **signal})
                else:
                    last_price = today["Close"].iloc[-1]
                    dist_to_high = (state.or_high - last_price) / state.or_high * 100
                    print(f"   {ticker:<6}: 当前 {last_price:.3f}  距OR高 {dist_to_high:+.2f}%  等待突破")
            else:
                last_price = today["Close"].iloc[-1]
                print(f"   {ticker:<6}: 已下单 (突破后监测，见上)")

        # 总结当轮
        active = sum(1 for s in states.values() if s.or_high and not s.too_narrow and not s.breakout_alerted)
        alerted = sum(1 for s in states.values() if s.breakout_alerted)
        print(f"   📊 监测中: {active} 只 | 已触发: {alerted} 只 | 距截止: "
              f"{(datetime.combine(now_et().date(), USER_CUTOFF_ET, tzinfo=ET) - now_et()).seconds // 60} 分钟")

        time.sleep(POLL_INTERVAL_SEC)

    print(f"\n{fmt_time()} — 盯盘结束，汇总:")
    for ticker, state in states.items():
        status = "已触发" if state.breakout_alerted else (
            "OR 过窄跳过" if state.too_narrow else "未突破"
        )
        print(f"   {ticker}: {status}")

    log_event({"event": "session_end", "summary": {
        t: ("alerted" if s.breakout_alerted else "narrow" if s.too_narrow else "no_breakout")
        for t, s in states.items()
    }})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，退出")
        log_event({"event": "user_interrupt"})
