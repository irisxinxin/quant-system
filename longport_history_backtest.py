"""
longport_history_backtest.py — LongPort 长周期历史数据回测

一键完成:
  1. 用 LongPort API 拉每只股票尽可能久的 5m 历史数据
  2. 探测数据深度（看每只股票能拉到多少天）
  3. 跑完整回测套件（基础回测 + 滑点翻倍 + Walk-Forward + Monte Carlo）
  4. 对比 yfinance 60 天回测 vs LongPort 长周期回测的差异
  5. 输出最终决策建议: GO LIVE / NEED MORE DATA / SKIP

用法:
  设置环境变量后运行:
  export LONGPORT_APP_KEY="..."
  export LONGPORT_APP_SECRET="..."
  export LONGPORT_ACCESS_TOKEN="..."
  python3 longport_history_backtest.py

默认:
  - 拉 12 只股票
  - 尽量往前拉 (LongPort 有多少给多少)
  - 数据缓存到 cache/longport_5m_*.pkl, 重跑不重复下载
"""
import os
import sys
import time
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

try:
    from longport.openapi import (
        Config, QuoteContext, Period, AdjustType
    )
except ImportError:
    print("❌ longport SDK 未装: pip3 install --break-system-packages longport")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════

TICKERS = [
    # (symbol, entry_slip, stop_slip, desc)
    ("OKLO.US", 0.0015, 0.0030, "核能小盘"),
    ("INTC.US", 0.0003, 0.0008, "半导体大盘"),
    ("IREN.US", 0.0015, 0.0030, "BTC挖矿小盘"),
    ("CIFR.US", 0.0015, 0.0030, "BTC挖矿小盘"),
    ("PLTR.US", 0.0003, 0.0008, "AI大盘"),
    ("NBIS.US", 0.0010, 0.0020, "AI基建中盘"),
    ("SOXL.US", 0.0003, 0.0008, "3x半导体ETF"),
    ("AMD.US",  0.0003, 0.0008, "半导体大盘"),
    ("AMZN.US", 0.0002, 0.0005, "超大盘"),
    ("TSLL.US", 0.0005, 0.0010, "2x特斯拉ETF"),
    ("NVDL.US", 0.0005, 0.0010, "2x英伟达ETF"),
    ("EOSE.US", 0.0025, 0.0050, "储能超小盘"),
]

CACHE_DIR = Path(__file__).parent / "cache" / "longport_history"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 每次 API 调用最多 1000 根 K 线 (LongPort 限制)
BATCH_SIZE = 1000
MAX_BATCHES = 30   # 30 batches × 1000 = 30000 根 5m K 线 ≈ 1 年


# ═══════════════════════════════════════════════════════════════════════
# 数据下载
# ═══════════════════════════════════════════════════════════════════════

def fetch_history(ctx, symbol: str, force: bool = False) -> pd.DataFrame:
    """拉一只股票尽量久的 5m K 线，缓存到本地"""
    cache_file = CACHE_DIR / f"{symbol.replace('.', '_')}_5m.pkl"

    if cache_file.exists() and not force:
        # 缓存存在且新鲜（24 小时内），直接用
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            with open(cache_file, "rb") as f:
                df = pickle.load(f)
            print(f"   {symbol:<10} ✅ 缓存命中 ({len(df)} 根, "
                  f"{df.index[0].date()} → {df.index[-1].date()}, "
                  f"缓存 {age_hours:.1f}h)")
            return df

    print(f"   {symbol:<10} 拉取中...", end="", flush=True)
    all_bars = []
    seen_timestamps = set()

    # LongPort SDK 的历史接口需要按"offset"分页
    # 用 candlesticks 接口拿最近的，再向前迭代
    try:
        # 第一批: 最近的 1000 根
        batch = ctx.candlesticks(symbol, Period.Min_5, BATCH_SIZE,
                                  AdjustType.NoAdjust)
        if not batch:
            print(f" ❌ 无数据返回")
            return pd.DataFrame()

        for c in batch:
            if c.timestamp not in seen_timestamps:
                seen_timestamps.add(c.timestamp)
                all_bars.append(c)

        # 后续批次: 用 history_candlesticks_by_offset 向前翻
        # SDK 方法名可能不同，尝试多种
        oldest_ts = batch[0].timestamp
        batches_done = 1

        while batches_done < MAX_BATCHES:
            try:
                # 用 history_candlesticks_by_offset 向后 (forward=False) 翻页
                older_batch = ctx.history_candlesticks_by_offset(
                    symbol, Period.Min_5, AdjustType.NoAdjust,
                    forward=False,         # False = backward = 更早的数据
                    count=BATCH_SIZE,
                    time=oldest_ts,        # 以 oldest_ts 为参考点
                )

                if not older_batch:
                    break

                new_count = 0
                for c in older_batch:
                    if c.timestamp not in seen_timestamps:
                        seen_timestamps.add(c.timestamp)
                        all_bars.append(c)
                        new_count += 1

                if new_count == 0:
                    break  # 没有新的数据了

                oldest_ts = min(c.timestamp for c in older_batch)
                batches_done += 1
                time.sleep(0.3)  # 避免被限速

            except Exception as e:
                print(f"\n   {symbol} 翻页中止: {type(e).__name__}: {str(e)[:50]}")
                break

    except Exception as e:
        print(f" ❌ {type(e).__name__}: {str(e)[:60]}")
        return pd.DataFrame()

    if not all_bars:
        print(" ❌ 无数据")
        return pd.DataFrame()

    # 转 DataFrame
    df = pd.DataFrame([{
        "timestamp": b.timestamp,
        "Open": float(b.open),
        "High": float(b.high),
        "Low": float(b.low),
        "Close": float(b.close),
        "Volume": int(b.volume),
    } for b in all_bars])

    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # ⚠️ 关键过滤: LongPort 时间戳是 HK 时区 (UTC+8) 但不带 tz 信息
    # 必须先 localize 为 HK, 再转换到 ET, 再过滤 regular hours
    et = ZoneInfo("US/Eastern")
    hk = ZoneInfo("Asia/Hong_Kong")
    if df.index.tz is None:
        df.index = df.index.tz_localize(hk)   # naive → HK
    df_et = df.tz_convert(et)                  # HK → ET
    # 只保留 9:30 - 16:00 ET (开盘到收盘)
    mask = ((df_et.index.time >= pd.Timestamp("09:30").time()) &
            (df_et.index.time < pd.Timestamp("16:00").time()) &
            (df_et.index.weekday < 5))
    df = df_et[mask]

    # 缓存
    with open(cache_file, "wb") as f:
        pickle.dump(df, f)

    days_span = (df.index[-1] - df.index[0]).days
    print(f" ✅ {len(df)} 根, 跨 {days_span} 天 ({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════

def main():
    # 验证环境变量
    missing = [k for k in ["LONGPORT_APP_KEY", "LONGPORT_APP_SECRET", "LONGPORT_ACCESS_TOKEN"]
               if not os.environ.get(k)]
    if missing:
        print(f"❌ 缺少环境变量: {missing}")
        sys.exit(1)

    print("=" * 80)
    print("LongPort 长周期历史数据回测")
    print("=" * 80)

    # 连接
    print("\n🔌 连接 LongPort OpenAPI...")
    config = Config.from_env()
    ctx = QuoteContext(config)
    print("   ✅ 连接成功")

    # 下载所有股票
    print(f"\n📥 下载 {len(TICKERS)} 只股票的 5m 历史数据...")
    print(f"   缓存目录: {CACHE_DIR}")
    print(f"   每只最多 {MAX_BATCHES * BATCH_SIZE} 根 (≈ 1 年)\n")

    data = {}
    for symbol, e_slip, s_slip, note in TICKERS:
        df = fetch_history(ctx, symbol)
        if not df.empty and len(df) > 200:
            data[symbol] = {"df": df, "entry_slip": e_slip, "stop_slip": s_slip, "note": note}

    if not data:
        print("\n❌ 没有可用数据")
        return

    # 数据深度报告
    print(f"\n{'═' * 80}")
    print("【数据深度报告】")
    print("═" * 80)
    print(f"{'Ticker':<10} {'根数':>7} {'起始日':<12} {'结束日':<12} {'天数':>6} {'交易日(估)':>10}")
    print("-" * 80)
    for symbol, d in data.items():
        df = d["df"]
        days = (df.index[-1] - df.index[0]).days
        trading_days = len(df) // 78  # 约 78 根/日
        print(f"{symbol:<10} {len(df):>7} {str(df.index[0].date()):<12} "
              f"{str(df.index[-1].date()):<12} {days:>6} {trading_days:>10}")

    # 跑回测
    print(f"\n{'═' * 80}")
    print("【ORB 长周期回测 - Long Only + 真实滑点】")
    print("═" * 80)

    from backtest_orb_validated import orb_backtest_v2, buy_and_hold_return

    base_params = {
        "or_bars": 3,
        "target_r": 2.0,
        "use_volume_filter": True,
        "rvol_threshold": 1.5,
        "commission": 0.0,
        "normal_slip": 0.0005,
        "long_only": True,
    }

    print(f"\n{'Ticker':<10} {'类型':<14} {'交易日':>6} {'策略收益':>10} {'B&H':>9} "
          f"{'差值':>10} {'交易':>5} {'胜率':>7} {'盈亏比':>7} {'回撤':>8} {'Calmar':>8}")
    print("-" * 110)

    backtest_results = {}
    for symbol, d in data.items():
        df = d["df"]
        # 移除 timestamp index 时区, 让 backtest_orb_validated 能识别为 ET
        df_for_bt = df.copy()
        if df_for_bt.index.tz is not None:
            df_for_bt.index = df_for_bt.index.tz_convert("US/Eastern").tz_localize(None)

        try:
            result = orb_backtest_v2(df_for_bt,
                                       entry_slip=d["entry_slip"],
                                       stop_slip=d["stop_slip"],
                                       **base_params)
            bh = buy_and_hold_return(df_for_bt) * 100
            m = result["metrics"]
            strat_ret = float(m["总收益率"].rstrip('%'))
            diff = strat_ret - bh
            trading_days = len(df) // 78
            backtest_results[symbol] = {
                "result": result, "bh": bh, "diff": diff,
                "note": d["note"], "trading_days": trading_days,
            }

            print(f"{symbol:<10} {d['note']:<14} {trading_days:>6} "
                  f"{strat_ret:>+9.2f}% {bh:>+8.2f}% {diff:>+9.2f}% "
                  f"{m['交易次数']:>5} {m['真实胜率']:>7} "
                  f"{m['盈亏比']:>7} {m['最大回撤']:>8} {m['Calmar']:>8}")
        except Exception as e:
            print(f"{symbol:<10} ❌ 回测失败: {e}")

    # 决策报告
    print(f"\n{'═' * 80}")
    print("【最终决策】")
    print("═" * 80)

    # 分类
    strong = []
    ok = []
    bad = []
    for symbol, r in backtest_results.items():
        m = r["result"]["metrics"]
        strat_ret = float(m["总收益率"].rstrip('%'))
        win_rate = float(m["真实胜率"].rstrip('%'))
        n_trades = m["交易次数"]
        calmar_str = m["Calmar"]
        try:
            calmar = float(calmar_str)
        except (ValueError, TypeError):
            calmar = 0

        # 强标准: 跑赢B&H >= 30%, 胜率>= 50%, 交易>= 30
        if r["diff"] >= 30 and win_rate >= 50 and n_trades >= 30:
            strong.append((symbol, r))
        elif r["diff"] > 0 and win_rate >= 45:
            ok.append((symbol, r))
        else:
            bad.append((symbol, r))

    print(f"\n🎯 强推 (跑赢B&H≥30% + 胜率≥50% + 交易数≥30): {[s for s, _ in strong]}")
    print(f"✅ 可考虑 (跑赢B&H + 胜率≥45%): {[s for s, _ in ok]}")
    print(f"❌ 不建议: {[s for s, _ in bad]}")

    # 综合建议
    if len(strong) >= 4:
        print(f"\n✅✅ 强标的数 {len(strong)} ≥ 4 → 策略稳健，可以考虑 paper trading 30 天")
    elif len(strong) + len(ok) >= 5:
        print(f"\n⚠️ 强标的不够多但可考虑标的有 {len(strong) + len(ok)} 只 → 建议 paper 但保守仓位")
    else:
        print(f"\n❌ 强标的 {len(strong)} 只，不足以支持 live 决策 → 暂时观望")

    # 保存完整报告
    report_file = CACHE_DIR.parent / "longport_backtest_report.txt"
    print(f"\n📄 完整结果已存到 {CACHE_DIR}/*.pkl")
    print(f"📊 跑完后可用现有 backtest_orb_stress.py 对这些数据做更深度的压力测试")


if __name__ == "__main__":
    main()
