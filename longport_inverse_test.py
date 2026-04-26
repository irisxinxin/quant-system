"""
longport_inverse_test.py — 反向 ETF 数据 + ORB 回测
拉 TSLZ / NVDS / SOXS 18 月 5m 历史, 跑 long-only ORB 看是否值得纳入组合
"""
import os
import sys
import time
import pickle
from pathlib import Path
from datetime import timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from longport.openapi import Config, QuoteContext, Period, AdjustType
from backtest_orb_validated import orb_backtest_v2, buy_and_hold_return

CACHE_DIR = Path(__file__).parent / "cache" / "longport_history"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

INVERSES = [
    ("TSLZ.US", 0.0008, 0.0015, "2x 反向 TSLA"),
    ("NVDS.US", 0.0008, 0.0015, "2x 反向 NVDA"),
    ("SOXS.US", 0.0005, 0.0010, "3x 反向 半导体"),
    ("BITI.US", 0.0010, 0.0020, "2x 反向 BTC"),
]
BATCH_SIZE = 1000
MAX_BATCHES = 30


def fetch_history(ctx, symbol):
    cache_file = CACHE_DIR / f"{symbol.replace('.', '_')}_5m.pkl"
    if cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < 24:
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    print(f"  {symbol:<10} 拉取中...", end="", flush=True)
    all_bars = []
    seen = set()

    try:
        batch = ctx.candlesticks(symbol, Period.Min_5, BATCH_SIZE, AdjustType.NoAdjust)
        if not batch:
            print(" ❌ 无数据")
            return pd.DataFrame()
        for c in batch:
            if c.timestamp not in seen:
                seen.add(c.timestamp); all_bars.append(c)
        oldest = batch[0].timestamp
        bd = 1
        while bd < MAX_BATCHES:
            try:
                older = ctx.history_candlesticks_by_offset(
                    symbol, Period.Min_5, AdjustType.NoAdjust,
                    forward=False, count=BATCH_SIZE, time=oldest)
                if not older: break
                new_n = 0
                for c in older:
                    if c.timestamp not in seen:
                        seen.add(c.timestamp); all_bars.append(c); new_n += 1
                if new_n == 0: break
                oldest = min(c.timestamp for c in older)
                bd += 1
                time.sleep(0.3)
            except Exception as e:
                print(f"\n   翻页中止: {e}"); break
    except Exception as e:
        print(f" ❌ {e}")
        return pd.DataFrame()

    if not all_bars:
        print(" ❌"); return pd.DataFrame()

    df = pd.DataFrame([{
        "timestamp": b.timestamp,
        "Open": float(b.open), "High": float(b.high), "Low": float(b.low),
        "Close": float(b.close), "Volume": int(b.volume),
    } for b in all_bars])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # HK timezone → ET, filter regular hours
    et = ZoneInfo("US/Eastern"); hk = ZoneInfo("Asia/Hong_Kong")
    if df.index.tz is None:
        df.index = df.index.tz_localize(hk)
    df_et = df.tz_convert(et)
    mask = ((df_et.index.time >= pd.Timestamp("09:30").time()) &
            (df_et.index.time < pd.Timestamp("16:00").time()) &
            (df_et.index.weekday < 5))
    df = df_et[mask]

    with open(cache_file, "wb") as f:
        pickle.dump(df, f)

    days = (df.index[-1] - df.index[0]).days
    print(f" ✅ {len(df)} 根, 跨 {days} 天")
    return df


def main():
    OUT = open("/tmp/inv_result.txt", "w")
    def p(s):
        print(s, flush=True)
        OUT.write(s + "\n"); OUT.flush()

    config = Config.from_env()
    ctx = QuoteContext(config)
    p("📥 拉取反向 ETF 18 月 5m 数据...")

    base_params = {
        "or_bars": 3, "target_r": 2.0, "use_volume_filter": True,
        "rvol_threshold": 1.5, "commission": 0.0, "normal_slip": 0.0005,
        "long_only": True,
    }

    p(f"\n{'='*100}")
    p(f"{'Ticker':<10} {'类型':<14} {'根数':>6} {'天数':>5} "
      f"{'策略收益':>10} {'B&H':>9} {'差值':>9} {'交易':>5} {'胜率':>7} {'PF':>6}")
    p("-"*100)

    results = []
    for sym, e_slip, s_slip, note in INVERSES:
        df = fetch_history(ctx, sym)
        if df.empty or len(df) < 500:
            continue

        df_bt = df.copy()
        if df_bt.index.tz is not None:
            df_bt.index = df_bt.index.tz_localize(None)

        try:
            r = orb_backtest_v2(df_bt, entry_slip=e_slip, stop_slip=s_slip, **base_params)
            bh = buy_and_hold_return(df_bt) * 100
            m = r["metrics"]
            ret = float(m["总收益率"].rstrip("%"))
            days = (df.index[-1] - df.index[0]).days
            p(f"{sym:<10} {note:<14} {len(df):>6} {days:>5} "
              f"{ret:>+9.1f}% {bh:>+8.1f}% {ret-bh:>+8.1f}% "
              f"{m['交易次数']:>5} {m['真实胜率']:>7} {m['盈亏比']:>6}")
            results.append((sym, ret, bh, m))
        except Exception as e:
            p(f"{sym:<10} ❌ {e}")

    print(f"\n{'='*100}")
    print("【组合分析】")
    print("="*100)
    print("""
关键判断:
  1. 反向 ETF 自身 ORB 收益 > B&H → 值得加入组合
  2. 反向 ETF 与对应原 ETF (TSLL/NVDL/SOXL) 同时监测,
     当原 ETF 涨破 OR_High → BUY 原 ETF
     当原 ETF 跌破 OR_Low → BUY 反向 ETF
     等价于双向交易, 不需要 short
""")
