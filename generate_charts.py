"""
generate_charts.py — 为每只股票生成近1年K线+买卖点图
输出: output/charts/<TICKER>.png
用法:
  python3 generate_charts.py              # 全部73只
  python3 generate_charts.py NVDA STX     # 指定股票
  python3 generate_charts.py --force      # 强制重新生成
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mplfinance as mpf
import yfinance as yf
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "output", "charts")
os.makedirs(OUT, exist_ok=True)

# ── 深色主题 ──────────────────────────────────────────────────────────
STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        wick={"up": "#26a69a", "down": "#ef5350"},
        edge={"up": "#26a69a", "down": "#ef5350"},
        volume={"up": "#26a69a44", "down": "#ef535044"},
    ),
    facecolor="#0d1117", figcolor="#0d1117",
    gridcolor="#1e2430", gridstyle="--", gridaxis="both",
    rc={
        "axes.labelcolor": "#8b949e",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "axes.titlecolor": "#e6edf3",
        "font.size": 9,
        "font.family": "DejaVu Sans",
    },
)


def fetch_ohlcv(ticker: str, days: int = 420) -> pd.DataFrame | None:
    """用 yfinance 拿足够长的 OHLCV 数据"""
    end   = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"),
                         auto_adjust=True, progress=False)
        if df.empty or len(df) < 20:
            return None
        # 标准化列名
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.capitalize() for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        return df.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception as e:
        print(f"[yfinance error: {e}]", end="")
        return None


def fetch_trades(ticker: str):
    """从 backtest_review 取最优策略的交易列表"""
    sys.path.insert(0, BASE)
    from backtest_review import get_optimal_and_signals
    result = get_optimal_and_signals(ticker)
    if result.get("error"):
        return [], "", []

    strats = result.get("strategies", [])
    if not strats:
        return [], "", []

    # 优先用 rank1；但如果 rank1 近1年完全无交易，才换有近期成交的策略
    best = strats[0]
    rank1_trades = best.get("trades", [])
    rank1_recent = [t for t in rank1_trades if t.get("entry_date", "") >= "2025-04-20"]
    if not rank1_recent:
        # rank1 近1年无成交，换有最多近期交易的策略
        best_recent = 0
        for s in strats:
            n = sum(1 for t in s.get("trades", []) if t.get("entry_date", "") >= "2025-04-20")
            if n > best_recent:
                best_recent = n
                best = s

    trades = best.get("trades", [])
    label  = best.get("label", "")
    return trades, label, strats


def nearest_date(target: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """找离 target 最近的交易日（±5天内）"""
    if target in index:
        return target
    pos = index.searchsorted(target)
    candidates = []
    for p in [pos - 1, pos]:
        if 0 <= p < len(index):
            candidates.append(index[p])
    if not candidates:
        return None
    best = min(candidates, key=lambda d: abs((d - target).days))
    if abs((best - target).days) > 5:
        return None
    return best


def make_chart(ticker: str, force: bool = False) -> bool:
    out_path = os.path.join(OUT, f"{ticker}.png")
    if os.path.exists(out_path) and not force:
        print(f"  {ticker}: skip (exists)")
        return True

    print(f"  {ticker}: ", end="", flush=True)

    # 1. 取 OHLCV
    df = fetch_ohlcv(ticker, days=420)
    if df is None:
        print("NO DATA")
        return False

    # 2. 取交易记录
    trades, label, all_strats = fetch_trades(ticker)

    # 3. 均线
    df["EMA20"] = df["Close"].ewm(span=20,  adjust=False).mean()
    df["EMA60"] = df["Close"].ewm(span=60,  adjust=False).mean()
    df["MA200"] = df["Close"].rolling(200, min_periods=80).mean()

    add_plots = [
        mpf.make_addplot(df["EMA20"], color="#f0b429", width=0.9, alpha=0.85),
        mpf.make_addplot(df["EMA60"], color="#b388ff", width=0.9, alpha=0.85),
        mpf.make_addplot(df["MA200"], color="#4fc3f7", width=0.8, alpha=0.65),
    ]

    # 4. 整理买卖点
    cutoff = df.index[0]
    buy_series  = pd.Series(np.nan, index=df.index, dtype=float)
    sell_series = pd.Series(np.nan, index=df.index, dtype=float)
    buy_labels, sell_labels = [], []
    open_trade = None

    for t in trades:
        ed = pd.Timestamp(t["entry_date"])
        if ed < cutoff:
            continue
        idx = nearest_date(ed, df.index)
        if idx is not None:
            buy_series[idx] = df.loc[idx, "Low"] * 0.965
            buy_labels.append((idx, t["entry_px"]))

        if t.get("is_open"):
            open_trade = t
        elif t.get("exit_date"):
            xd = pd.Timestamp(t["exit_date"])
            xidx = nearest_date(xd, df.index)
            if xidx is not None:
                sell_series[xidx] = df.loc[xidx, "High"] * 1.035
                pnl = t.get("pnl_pct", 0)
                sell_labels.append((xidx, t.get("exit_px", t["entry_px"] * (1 + pnl / 100)), pnl))

    n_buy  = buy_series.notna().sum()
    n_sell = sell_series.notna().sum()

    if n_buy:
        add_plots.append(mpf.make_addplot(buy_series,  type="scatter",
                                          markersize=90, marker="^",
                                          color="#00e676", alpha=0.95, panel=0))
    if n_sell:
        add_plots.append(mpf.make_addplot(sell_series, type="scatter",
                                          markersize=90, marker="v",
                                          color="#ff1744", alpha=0.95, panel=0))

    # 5. 持仓横线
    extra_kwargs = {}
    if open_trade:
        extra_kwargs["hlines"] = dict(
            hlines=open_trade["entry_px"],
            linewidths=0.7, linestyle="--", colors="#f0b42988"
        )

    # 6. 绘图
    fig, axes = mpf.plot(
        df,
        type="candle",
        style=STYLE,
        volume=True,
        addplot=add_plots,
        figsize=(15, 7),
        panel_ratios=(4, 1),
        tight_layout=True,
        returnfig=True,
        warn_too_much_data=9999,
        **extra_kwargs,
    )
    ax = axes[0]

    # 7. 标题
    current_price = df["Close"].iloc[-1]
    date_str = df.index[-1].strftime("%Y-%m-%d")
    pnl_str = ""
    if open_trade:
        pnl = (current_price / open_trade["entry_px"] - 1) * 100
        days = (df.index[-1] - pd.Timestamp(open_trade["entry_date"])).days
        pnl_str = f"  ▶ 持仓{days}天 +{pnl:.1f}%"
    ax.set_title(
        f"{ticker}  ·  {label}{pnl_str}  ·  ${current_price:.2f} ({date_str})",
        fontsize=11, color="#e6edf3", pad=8, loc="left", fontweight="bold"
    )

    # 8. 图例
    legend_items = [
        mpatches.Patch(color="#f0b429", label="EMA20"),
        mpatches.Patch(color="#b388ff", label="EMA60"),
        mpatches.Patch(color="#4fc3f7", label="MA200"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#00e676",
                   markersize=9, label=f"Buy ({n_buy})"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#ff1744",
                   markersize=9, label=f"Sell ({n_sell})"),
    ]
    if open_trade:
        legend_items.append(
            mpatches.Patch(color="#f0b42966",
                           label=f"Entry ${open_trade['entry_px']:.1f}")
        )
    ax.legend(handles=legend_items, loc="upper left", fontsize=8,
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", framealpha=0.9)

    # 9. 价格标注（最近8笔）
    x_vals = list(df.index)
    for idx, price in buy_labels[-8:]:
        xi = x_vals.index(idx) if idx in x_vals else None
        if xi is None: continue
        y = buy_series[idx]
        ax.annotate(f"${price:.1f}", xy=(xi, y * 0.985), fontsize=6,
                    color="#00e676", ha="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#0d1117",
                              ec="#00e67666", alpha=0.85))

    for idx, price, pnl in sell_labels[-8:]:
        xi = x_vals.index(idx) if idx in x_vals else None
        if xi is None: continue
        y = sell_series[idx]
        color = "#ff1744" if pnl < 0 else "#ff6090"
        ax.annotate(f"${price:.1f}\n{pnl:+.1f}%", xy=(xi, y * 1.01),
                    fontsize=6, color=color, ha="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="#0d1117",
                              ec=f"{color}66", alpha=0.85))

    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close(fig)
    print(f"OK  (buy:{n_buy} sell:{n_sell})")
    return True


def main():
    args   = sys.argv[1:]
    force  = "--force" in args
    tickers_arg = [a for a in args if not a.startswith("--")]

    if tickers_arg:
        tickers = [t.upper() for t in tickers_arg]
    else:
        df_csv  = pd.read_csv(os.path.join(BASE, "output", "top3_strategies.csv"))
        tickers = sorted(df_csv["ticker"].unique().tolist())

    print(f"\n生成K线图：{len(tickers)} 只  →  {OUT}\n")
    ok, fail = 0, []
    for t in tickers:
        try:
            if make_chart(t, force=force):
                ok += 1
            else:
                fail.append(t)
        except Exception as e:
            print(f"ERR: {e}")
            fail.append(t)

    print(f"\n完成：{ok}/{len(tickers)} 张")
    if fail:
        print(f"失败：{fail}")


if __name__ == "__main__":
    main()
