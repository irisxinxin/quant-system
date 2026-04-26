"""
backtest_orb.py — Opening Range Breakout 开盘区间突破回测

策略规则:
  1. 每日开盘后前 N 根K线 = Opening Range (OR)
     - OR High = 窗口内最高点
     - OR Low  = 窗口内最低点
  2. OR 形成后，监测突破:
     - 价格突破 OR High + 可选量能过滤 → 做多
     - 价格跌破 OR Low + 可选量能过滤 → 做空（long_only=False时）
  3. 每日最多 1 笔交易（首次突破）
  4. 止损 = OR 另一端 (多单: OR Low, 空单: OR High)
  5. 止盈 = 入场 ± target_R × OR_Range
  6. 强制平仓 = 收盘前 N 根（避免隔夜）

测试对象: OKLO / IREN / HOOD / NVDL / TSLL
测试维度:
  - OR 长度: 15分钟 (3根5m) vs 30分钟 (6根5m)
  - 量能过滤: RVOL > 1.5 vs 关闭
  - 止盈目标: 2R vs 3R
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from backtest_soxl_5m import fetch_intraday


def orb_backtest(
    df: pd.DataFrame,
    or_bars: int = 3,               # 3 根5m = 15min 开盘区间
    target_r: float = 2.0,           # 止盈 = 入场 + 2 × OR_range
    use_volume_filter: bool = True,
    rvol_threshold: float = 1.5,     # 突破K线的成交量 / 20根K线均量
    rvol_lookback: int = 20,
    long_only: bool = True,
    eod_exit_bars: int = 4,          # 收盘前 4 根（20分钟）强制平仓
    commission: float = 0.0005,
    slippage: float = 0.0005,
    min_or_range_pct: float = 0.003, # OR 范围至少 0.3%（过滤过窄区间）
) -> dict:
    """Opening Range Breakout 回测（用 5m 数据）"""
    if df.empty:
        return {}

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    v = df["Volume"].values
    dates = df.index

    # 标记每日K线序号
    day_ids = pd.Series(dates).dt.date.values
    day_change = np.concatenate([[True], day_ids[1:] != day_ids[:-1]])
    bar_of_day = np.zeros(len(df), dtype=int)
    cnt = 0
    for i, change in enumerate(day_change):
        if change:
            cnt = 0
        bar_of_day[i] = cnt
        cnt += 1
    bars_per_day = pd.Series(bar_of_day).groupby(pd.Series(day_ids)).transform("max").values + 1
    bars_to_close = bars_per_day - bar_of_day - 1  # 到当日收盘还剩多少根

    # 滚动均量
    vol_ma = pd.Series(v).rolling(rvol_lookback, min_periods=1).mean().values

    n = len(df)
    daily_returns = np.zeros(n)
    round_trip = (commission + slippage) * 2

    # 按交易日分组处理
    unique_days = pd.Series(day_ids).unique()
    trades = []
    skipped_narrow = 0
    skipped_volume = 0
    no_breakout_days = 0

    for day in unique_days:
        day_idx = np.where(day_ids == day)[0]
        if len(day_idx) < or_bars + 5:
            continue  # 当日数据不足

        # 1. 计算 OR
        or_start = day_idx[0]
        or_end = day_idx[or_bars - 1]  # 最后一根OR K线索引（inclusive）
        or_high = h[or_start:or_end + 1].max()
        or_low  = l[or_start:or_end + 1].min()
        or_range = or_high - or_low
        or_mid = (or_high + or_low) / 2

        # 过滤过窄的 OR（通常是低波动日）
        if or_range / or_mid < min_or_range_pct:
            skipped_narrow += 1
            continue

        # 2. 监测突破（从 OR 结束下一根开始）
        position = 0   # 0 = 空仓, 1 = 多, -1 = 空
        entry_price = stop_price = tp_price = np.nan
        entry_idx = None
        breakout_found = False

        for j in range(or_bars, len(day_idx)):
            idx = day_idx[j]

            # 持仓检查（优先止损/止盈/日末平仓）
            if position != 0:
                # 日末强制平仓
                if bars_to_close[idx] <= eod_exit_bars:
                    exit_px = c[idx]
                    pnl_pct = position * (exit_px - entry_price) / entry_price - round_trip
                    daily_returns[idx] = pnl_pct
                    trades.append({
                        "day": day, "entry_date": dates[entry_idx], "exit_date": dates[idx],
                        "result": "日末平仓",
                        "entry": round(entry_price, 3),
                        "stop": round(stop_price, 3),
                        "tp": round(tp_price, 3),
                        "exit": round(exit_px, 3),
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "or_range_pct": round(or_range / or_mid * 100, 2),
                        "direction": "多" if position == 1 else "空",
                    })
                    position = 0
                    break

                # 止损
                stop_hit = (position == 1 and l[idx] <= stop_price) or \
                           (position == -1 and h[idx] >= stop_price)
                tp_hit = (position == 1 and h[idx] >= tp_price) or \
                         (position == -1 and l[idx] <= tp_price)

                if stop_hit:
                    exit_px = stop_price
                    pnl_pct = position * (exit_px - entry_price) / entry_price - round_trip
                    daily_returns[idx] = pnl_pct
                    trades.append({
                        "day": day, "entry_date": dates[entry_idx], "exit_date": dates[idx],
                        "result": "止损",
                        "entry": round(entry_price, 3),
                        "stop": round(stop_price, 3),
                        "tp": round(tp_price, 3),
                        "exit": round(exit_px, 3),
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "or_range_pct": round(or_range / or_mid * 100, 2),
                        "direction": "多" if position == 1 else "空",
                    })
                    position = 0
                    break
                elif tp_hit:
                    exit_px = tp_price
                    pnl_pct = position * (exit_px - entry_price) / entry_price - round_trip
                    daily_returns[idx] = pnl_pct
                    trades.append({
                        "day": day, "entry_date": dates[entry_idx], "exit_date": dates[idx],
                        "result": "止盈",
                        "entry": round(entry_price, 3),
                        "stop": round(stop_price, 3),
                        "tp": round(tp_price, 3),
                        "exit": round(exit_px, 3),
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "or_range_pct": round(or_range / or_mid * 100, 2),
                        "direction": "多" if position == 1 else "空",
                    })
                    position = 0
                    break
                continue

            # 空仓时检测突破
            if breakout_found:
                continue

            # 量能过滤
            if use_volume_filter:
                rvol = v[idx] / max(vol_ma[idx], 1)
                if rvol < rvol_threshold:
                    continue

            # 突破上沿做多
            if h[idx] > or_high and c[idx] > or_high:
                entry_price = or_high   # 突破点入场
                stop_price = or_low
                tp_price = entry_price + target_r * or_range
                position = 1
                entry_idx = idx
                breakout_found = True

            # 跌破下沿做空
            elif (not long_only) and l[idx] < or_low and c[idx] < or_low:
                entry_price = or_low
                stop_price = or_high
                tp_price = entry_price - target_r * or_range
                position = -1
                entry_idx = idx
                breakout_found = True

        # 当日结束仍持仓则按最后价平仓（保险）
        if position != 0 and entry_idx is not None:
            last_idx = day_idx[-1]
            exit_px = c[last_idx]
            pnl_pct = position * (exit_px - entry_price) / entry_price - round_trip
            daily_returns[last_idx] = pnl_pct
            trades.append({
                "day": day, "entry_date": dates[entry_idx], "exit_date": dates[last_idx],
                "result": "收盘平仓",
                "entry": round(entry_price, 3),
                "stop": round(stop_price, 3),
                "tp": round(tp_price, 3),
                "exit": round(exit_px, 3),
                "pnl_pct": round(pnl_pct * 100, 3),
                "or_range_pct": round(or_range / or_mid * 100, 2),
                "direction": "多" if position == 1 else "空",
            })

        if not breakout_found and position == 0:
            no_breakout_days += 1

    # 计算指标
    ret_series = pd.Series(daily_returns, index=dates, name="ORB")
    total_ret = (1 + ret_series).prod() - 1
    num_trades = len(trades)

    if num_trades > 0:
        td = pd.DataFrame(trades)
        wins = (td["result"] == "止盈").sum()
        losses = (td["result"] == "止损").sum()
        eod_exits = ((td["result"] == "日末平仓") | (td["result"] == "收盘平仓")).sum()
        win_rate = wins / num_trades
        profit_trades = td[td["pnl_pct"] > 0]
        loss_trades  = td[td["pnl_pct"] < 0]
        avg_win = profit_trades["pnl_pct"].mean() if len(profit_trades) > 0 else 0
        avg_loss = loss_trades["pnl_pct"].mean() if len(loss_trades) > 0 else 0
        profit_factor = (
            profit_trades["pnl_pct"].sum() / abs(loss_trades["pnl_pct"].sum())
            if len(loss_trades) > 0 else np.inf
        )
    else:
        td = pd.DataFrame()
        win_rate = avg_win = avg_loss = profit_factor = 0
        wins = losses = eod_exits = 0

    equity = (1 + ret_series).cumprod()
    max_dd = ((equity / equity.cummax()) - 1).min()
    days_in_period = (dates[-1] - dates[0]).days
    years = days_in_period / 365.25 if days_in_period > 0 else 1/365
    ann_ret = (1 + total_ret) ** (1/years) - 1 if total_ret > -1 else -1
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

    metrics = {
        "区间": f"{dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}",
        "总交易日": len(unique_days),
        "无突破天": no_breakout_days,
        "OR过窄": skipped_narrow,
        "总收益率": f"{total_ret*100:.2f}%",
        "年化收益": f"{ann_ret*100:.2f}%",
        "最大回撤": f"{max_dd*100:.2f}%",
        "Calmar": f"{calmar:.2f}",
        "交易次数": num_trades,
        "止盈次数": int(wins),
        "止损次数": int(losses),
        "日末平仓": int(eod_exits),
        "胜率": f"{win_rate*100:.1f}%",
        "平均盈利": f"{avg_win:.3f}%",
        "平均亏损": f"{avg_loss:.3f}%",
        "盈亏比": f"{profit_factor:.2f}" if profit_factor != np.inf else "∞",
    }

    return {"returns": ret_series, "trades": td, "metrics": metrics, "equity": equity}


# ============================================================
# 主入口: 矩阵化测试
# ============================================================
if __name__ == "__main__":
    tickers = ["OKLO", "IREN", "HOOD", "NVDL", "TSLL"]

    # 测试配置矩阵
    configs = [
        {"name": "A: 15分OR+量能+2R",   "or_bars": 3, "use_volume_filter": True,  "target_r": 2.0, "long_only": True},
        {"name": "B: 15分OR+无量能+2R", "or_bars": 3, "use_volume_filter": False, "target_r": 2.0, "long_only": True},
        {"name": "C: 30分OR+量能+2R",   "or_bars": 6, "use_volume_filter": True,  "target_r": 2.0, "long_only": True},
        {"name": "D: 15分OR+量能+3R",   "or_bars": 3, "use_volume_filter": True,  "target_r": 3.0, "long_only": True},
        {"name": "E: 15分OR+量能+2R双向", "or_bars": 3, "use_volume_filter": True, "target_r": 2.0, "long_only": False},
    ]

    print("=" * 100)
    print("Opening Range Breakout 回测 | 5m数据 | OKLO/IREN/HOOD/NVDL/TSLL")
    print("=" * 100)

    # 为每只股票下载数据
    data = {}
    for t in tickers:
        try:
            df = fetch_intraday(t, period="60d", interval="5m")
            if not df.empty:
                data[t] = df
                print(f"  {t}: {len(df)} 根5m")
        except Exception as e:
            print(f"  {t}: 下载失败 {e}")

    # 矩阵运行
    all_results = {}  # {config_name: {ticker: result}}
    for cfg in configs:
        name = cfg.pop("name")
        all_results[name] = {}
        for t, df in data.items():
            result = orb_backtest(df.copy(), **cfg)
            all_results[name][t] = result

    # 按配置打印
    for cfg_name, ticker_results in all_results.items():
        print(f"\n{'=' * 100}")
        print(f"{cfg_name}")
        print("=" * 100)
        print(f"{'Ticker':<8} {'总收益':>9} {'交易':>5} {'胜率':>7} {'盈亏比':>7} "
              f"{'均盈':>8} {'均亏':>8} {'Calmar':>7} {'回撤':>8} {'无突破日':>8}")
        print("-" * 100)
        for t, r in ticker_results.items():
            m = r["metrics"]
            print(f"{t:<8} {m['总收益率']:>9} {m['交易次数']:>5} {m['胜率']:>7} "
                  f"{m['盈亏比']:>7} {m['平均盈利']:>8} {m['平均亏损']:>8} "
                  f"{m['Calmar']:>7} {m['最大回撤']:>8} {m['无突破天']:>8}")

    # 跨配置总结表
    print(f"\n{'=' * 100}")
    print("跨配置横向对比：每个 Ticker 最佳配置")
    print("=" * 100)
    print(f"{'Ticker':<8} {'最佳配置':<30} {'总收益':>9} {'胜率':>7} {'交易':>5} {'Calmar':>7}")
    print("-" * 100)
    for t in data.keys():
        best_cfg = None
        best_ret = -999
        for cfg_name, ticker_results in all_results.items():
            if t not in ticker_results:
                continue
            m = ticker_results[t]["metrics"]
            ret = float(m["总收益率"].rstrip('%'))
            if ret > best_ret:
                best_ret = ret
                best_cfg = (cfg_name, m)
        if best_cfg:
            name, m = best_cfg
            print(f"{t:<8} {name:<30} {m['总收益率']:>9} {m['胜率']:>7} "
                  f"{m['交易次数']:>5} {m['Calmar']:>7}")

    # 过滤出达到判定标准的组合
    print(f"\n{'=' * 100}")
    print("达到判定标准 (交易≥30, 胜率≥40%, Calmar≥3) 的组合")
    print("=" * 100)
    passing = []
    for cfg_name, ticker_results in all_results.items():
        for t, r in ticker_results.items():
            m = r["metrics"]
            trades_n = m["交易次数"]
            win_rate = float(m["胜率"].rstrip('%'))
            calmar = float(m["Calmar"]) if m["Calmar"] != "inf" and m["Calmar"] != "-inf" else 0
            if trades_n >= 30 and win_rate >= 40 and calmar >= 3:
                passing.append((cfg_name, t, m))
                print(f"  ✅ {cfg_name} | {t} | 收益 {m['总收益率']} | "
                      f"胜率 {m['胜率']} | 交易 {trades_n} | Calmar {m['Calmar']}")
    if not passing:
        print("  ⚠️ 没有任何组合满足全部条件")

    # 打印最佳组合的样本交易
    best_overall = None
    best_score = -999
    for cfg_name, ticker_results in all_results.items():
        for t, r in ticker_results.items():
            ret = float(r["metrics"]["总收益率"].rstrip('%'))
            n_trades = r["metrics"]["交易次数"]
            if n_trades >= 10 and ret > best_score:
                best_score = ret
                best_overall = (cfg_name, t, r)

    if best_overall:
        cfg_name, t, r = best_overall
        print(f"\n{'=' * 100}")
        print(f"全局最佳组合（交易≥10）: {cfg_name} | {t} | 总收益 {r['metrics']['总收益率']}")
        print("样本交易（前15笔）:")
        print("=" * 100)
        if not r["trades"].empty:
            print(r["trades"].head(15).to_string())
