#!/usr/bin/env python3
"""
signal_live_hk.py — 港股 2x ETF 信号提示 (独立于美股系统; 仅 Telegram 提醒, 不自动下单)

监测: 7747.HK 三星2x(ORB5_Z) / 7709.HK 海力士2x(TTM_SQ)
港股时段 HKT 09:30-16:00(午休12:00-13:00)。信号触发 → Telegram 提醒(带近期胜率/PF), 你手动下单。

⚠ 为什么只提醒不下单: LongPort 模拟盘下单路径(live_executor)只接美股, 港股没接。
   本脚本纯只读行情 + Telegram, 不碰任何下单, 安全。

用法:
  python3 signal_live_hk.py           # 港股时段循环监测
  python3 signal_live_hk.py --once    # 跑一次(需在港股时段内)
  python3 signal_live_hk.py --test    # 测试: 无视时段, 用最近交易日数据跑一遍看会不会出信号
"""
import os, sys, time
import urllib.request, urllib.parse
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from longport.openapi import Config, QuoteContext, Period, AdjustType
from signals.strategies.intraday_pool import STRATEGIES

HKT = ZoneInfo("Asia/Hong_Kong")
HK_OPEN, HK_CLOSE, POLL_SEC = dtime(9, 30), dtime(16, 0), 30

# 港股池: symbol -> dict(名称, 策略, 近期胜率, 近期PF, 近期每笔%)  ← 来自 hk_2x_backtest.py 回测
HK_POOL = {
    "7747.HK": dict(name="三星2x",   strat="ORB5_Z", rec_win=42, rec_pf=1.67, rec_avg=0.66),
    "7709.HK": dict(name="海力士2x", strat="TTM_SQ", rec_win=53, rec_pf=1.62, rec_avg=0.34),
}

_DISPATCHED = set()   # (sym, date) 今日已提醒, 防重复


def send_telegram(title: str, body: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN"); chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print(f"\n[无TG凭证·本地打印]\n{title}\n{body}\n"); return
    data = urllib.parse.urlencode({"chat_id": chat, "text": f"{title}\n{body}"}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, method="POST"), timeout=10)
        print(f"   ✅ TG已发: {title}")
    except Exception as e:
        print(f"   ⚠️ TG失败: {e}")


def _hk_rth(t) -> bool:
    """港股常规交易时段: 上午 09:30-12:00, 下午 13:00-16:00 (HKT). 排除 09:00-09:20 集合竞价."""
    return (dtime(9, 30) <= t < dtime(12, 0)) or (dtime(13, 0) <= t < dtime(16, 0))

def fetch_5m(ctx, sym, count=200) -> pd.DataFrame:
    bars = ctx.candlesticks(sym, Period.Min_5, count, AdjustType.NoAdjust)
    if not bars:
        return pd.DataFrame()
    # 时间戳归一到 HKT (LongPort 可能返回 tz-aware), 用于 RTH 过滤
    idx = [b.timestamp.astimezone(HKT) for b in bars]
    df = pd.DataFrame({"Open": [float(b.open) for b in bars], "High": [float(b.high) for b in bars],
                       "Low": [float(b.low) for b in bars], "Close": [float(b.close) for b in bars],
                       "Volume": [float(b.volume) for b in bars]},
                      index=idx).sort_index()
    # 关键修 (6/15): 过滤掉盘前集合竞价 K 线 (美股 5/1 同类坑, 港股之前漏改).
    # 否则 ORB5_Z 的 iloc[0] / TTM_SQ 的 index[0]+30min 会锚在竞价 bar 上, OR/止损/方向全错.
    df = df[[_hk_rth(ts.time()) for ts in df.index]]
    if df.empty:
        return df
    df["date"] = df.index.date
    return df


def check_and_alert(ctx, force_day=None):
    """force_day: 测试用, 指定"今天"=最近交易日; 正常运行用真实今天。"""
    now = datetime.now(HKT)
    for sym, cfg in HK_POOL.items():
        df = fetch_5m(ctx, sym)
        if df.empty:
            continue
        day = force_day or now.date()
        today_df = df[df["date"] == day]
        if len(today_df) < 2:
            continue
        key = (sym, day)
        if key in _DISPATCHED:
            continue
        fn = STRATEGIES.get(cfg["strat"])
        try:
            plans = [p for p in fn(today_df, df) if p.side == "long"]
        except Exception as e:
            print(f"   ⚠️ {sym} {cfg['strat']} 报错: {e}"); continue
        if not plans:
            continue
        p = plans[0]
        risk = (p.limit_price - p.stop_price) / p.limit_price * 100
        rew = (p.tp_price - p.limit_price) / p.limit_price * 100
        title = f"🇭🇰🚀 {cfg['name']} ({sym}) {cfg['strat']} 信号"
        body = (f"时间(HKT): {now:%m-%d %H:%M}\n"
                f"📊 近期: 胜率{cfg['rec_win']}% · PF{cfg['rec_pf']} · 每笔+{cfg['rec_avg']}%\n"
                f"入场~{p.limit_price:.2f}  止损 {p.stop_price:.2f} ({-risk:.1f}%)  止盈 {p.tp_price:.2f} (+{rew:.1f}%)\n"
                f"⚠️ 港股·请手动下单(本系统不自动执行)\n"
                f"备注: {p.note}")
        send_telegram(title, body)
        _DISPATCHED.add(key)


def main():
    args = sys.argv[1:]
    ctx = QuoteContext(Config.from_env())
    pool_str = ", ".join(f"{k}={v['name']}/{v['strat']}" for k, v in HK_POOL.items())
    print(f"🇭🇰 港股2x信号提示 (仅提醒不下单) | 池: {pool_str}")

    if "--test" in args:
        # 用最近交易日数据跑一遍, 验证端到端
        sample = fetch_5m(ctx, "7747.HK")
        last_day = sample["date"].max() if not sample.empty else None
        print(f"🧪 测试模式: 用最近交易日 {last_day} 数据跑策略")
        check_and_alert(ctx, force_day=last_day)
        print("🧪 测试完成 (上面若无信号=该日策略未触发, 正常)")
        return

    if "--once" in args:
        check_and_alert(ctx); return

    print("循环监测中 (港股时段, Ctrl+C 退出)...")
    while True:
        now = datetime.now(HKT)
        if now.time() > HK_CLOSE:
            print("港股已收盘, 退出"); break
        if HK_OPEN <= now.time() <= HK_CLOSE:
            try:
                check_and_alert(ctx)
            except Exception as e:
                print(f"   loop err: {e}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
