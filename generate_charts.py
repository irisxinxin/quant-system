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
from scipy import stats as sp_stats
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


def compute_channels(df: pd.DataFrame, pivot_window: int = 8):
    """
    检测近期趋势通道与震荡区间。
    使用真实 pivot 点连线（非回归外推），确保通道贴近实际价格。
    返回 dict:
      {
        "type": "ascending" | "descending" | "consolidation" | "none",
        "support":    [(xi, price), (xi, price)],  # 支撑线两端点
        "resistance": [(xi, price), (xi, price)],  # 阻力线两端点
        "consol_band": (low, high) | None,
        "consol_x": (x0, x1) | None,
      }
    """
    n = len(df)
    if n < 40:
        return {"type": "none"}

    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    # ── 1. 找 pivot high / low（最近 90 根 K 线）────────────────────────
    window   = pivot_window
    lookback = min(n, 90)
    start_i  = n - lookback
    x_end    = n - 1

    avg_price = closes[-lookback:].mean()
    avg_range = (highs[-lookback:] - lows[-lookback:]).mean()  # 平均日内振幅

    pivot_hi, pivot_lo = [], []
    for i in range(start_i + window, n - window):
        if highs[i] == highs[i - window : i + window + 1].max():
            pivot_hi.append((i, highs[i]))
        if lows[i]  == lows[i  - window : i + window + 1].min():
            pivot_lo.append((i, lows[i]))

    result = {"type": "none", "support": [], "resistance": [],
              "consol_band": None, "consol_x": None}

    SLOPE_UP   =  0.04   # %/bar
    SLOPE_DOWN = -0.04
    MIN_WIDTH  = avg_price * 0.018   # 最小通道宽度：均价的 1.8%

    def proj(x, x0, y0, slope):
        """从 (x0,y0) 用斜率 slope 算出 x 处的 y 值"""
        return y0 + slope * (x - x0)

    # ── 2. 上升通道：找连续上升的 pivot low ───────────────────────────────
    # 筛选出单调递增的 pivot lows（每个比前一个高）
    asc_lows = []
    for pt in pivot_lo:
        if not asc_lows or pt[1] > asc_lows[-1][1]:
            asc_lows.append(pt)
    asc_lows = asc_lows[-3:]   # 取最近 3 个

    if len(asc_lows) >= 2:
        x0_lo, y0_lo = asc_lows[0]
        x1_lo, y1_lo = asc_lows[-1]
        slope = (y1_lo - y0_lo) / max(x1_lo - x0_lo, 1)
        slope_pct = slope / avg_price * 100

        if slope_pct >= SLOPE_UP:
            # 在 x0_lo ~ x_end 范围内找最高的 pivot high → 平行阻力线
            hi_in_range = [ph for ph in pivot_hi if ph[0] >= x0_lo]
            if hi_in_range:
                x_hi, y_hi = max(hi_in_range, key=lambda p: p[1])
            else:
                x_hi, y_hi = x1_lo, y1_lo + avg_range * 1.5

            # 平行阻力线：斜率相同，过 (x_hi, y_hi)
            y_res_x0 = proj(x0_lo, x_hi, y_hi, slope)
            y_res_end = proj(x_end, x_hi, y_hi, slope)
            y_sup_x0  = proj(x0_lo, x0_lo, y0_lo, slope)  # = y0_lo
            y_sup_end = proj(x_end, x0_lo, y0_lo, slope)

            width_end = y_res_end - y_sup_end
            if width_end >= MIN_WIDTH:
                result["type"]       = "ascending"
                result["support"]    = [(x0_lo, y_sup_x0),  (x_end, y_sup_end)]
                result["resistance"] = [(x0_lo, y_res_x0), (x_end, y_res_end)]

    # ── 3. 下降通道：找连续下降的 pivot high ──────────────────────────────
    if result["type"] == "none":
        desc_highs = []
        for pt in pivot_hi:
            if not desc_highs or pt[1] < desc_highs[-1][1]:
                desc_highs.append(pt)
        desc_highs = desc_highs[-3:]

        if len(desc_highs) >= 2:
            x0_hi, y0_hi = desc_highs[0]
            x1_hi, y1_hi = desc_highs[-1]
            slope = (y1_hi - y0_hi) / max(x1_hi - x0_hi, 1)
            slope_pct = slope / avg_price * 100

            if slope_pct <= SLOPE_DOWN:
                # 在范围内找最低 pivot low → 平行支撑线
                lo_in_range = [pl for pl in pivot_lo if pl[0] >= x0_hi]
                if lo_in_range:
                    x_lo, y_lo = min(lo_in_range, key=lambda p: p[1])
                else:
                    x_lo, y_lo = x1_hi, y1_hi - avg_range * 1.5

                y_sup_x0  = proj(x0_hi, x_lo, y_lo, slope)
                y_sup_end = proj(x_end, x_lo, y_lo, slope)
                y_res_x0  = proj(x0_hi, x0_hi, y0_hi, slope)  # = y0_hi
                y_res_end = proj(x_end, x0_hi, y0_hi, slope)

                width_end = abs(y_res_end - y_sup_end)
                if width_end >= MIN_WIDTH:
                    result["type"]       = "descending"
                    result["support"]    = [(x0_hi, y_sup_x0),  (x_end, y_sup_end)]
                    result["resistance"] = [(x0_hi, y_res_x0), (x_end, y_res_end)]

    # ── 4. 震荡区间：BB 宽度收窄 ──────────────────────────────────────────
    if result["type"] == "none":
        bb_std = pd.Series(closes).rolling(20).std().iloc[-1]
        bb_mean = closes[-20:].mean()
        bb_width_pct = bb_std / bb_mean * 100 if bb_mean else 99

        if bb_width_pct < 6.0:
            consol_start = max(0, n - 60)
            zone_lo = np.percentile(lows[consol_start:],   8)
            zone_hi = np.percentile(highs[consol_start:], 92)
            result["type"]        = "consolidation"
            result["consol_band"] = (zone_lo, zone_hi)
            result["consol_x"]    = (consol_start, x_end)

    return result


def draw_channels(ax, channels: dict, n_bars: int):
    """
    在 mplfinance axes 上绘制趋势通道或震荡区间。
    mplfinance 的 x 轴对应整数 bar index，ax 已由 returnfig=True 拿到。
    """
    ctype = channels.get("type", "none")
    if ctype == "none":
        return

    if ctype in ("ascending", "descending"):
        color_line = "#26a69a" if ctype == "ascending" else "#ef5350"

        sup = channels["support"]      # [(x0,y0),(x1,y1)]
        res = channels["resistance"]

        xs_s = [sup[0][0], sup[1][0]]
        ys_s = [sup[0][1], sup[1][1]]
        xs_r = [res[0][0], res[1][0]]
        ys_r = [res[0][1], res[1][1]]

        # 通道边界线：更粗、实线
        ax.plot(xs_s, ys_s, color=color_line, linewidth=1.8,
                linestyle="--", alpha=0.90, zorder=4)
        ax.plot(xs_r, ys_r, color=color_line, linewidth=1.8,
                linestyle="--", alpha=0.90, zorder=4)

        # 通道填充（明显可见）
        xs_fill = np.linspace(xs_s[0], xs_s[1], 50)
        slope_s  = (ys_s[1] - ys_s[0]) / max(xs_s[1] - xs_s[0], 1)
        slope_r  = (ys_r[1] - ys_r[0]) / max(xs_r[1] - xs_r[0], 1)
        ys_fill_s = ys_s[0] + slope_s * (xs_fill - xs_s[0])
        ys_fill_r = ys_r[0] + slope_r * (xs_fill - xs_r[0])
        ax.fill_between(xs_fill, ys_fill_s, ys_fill_r,
                        color=color_line, alpha=0.18, zorder=2)

        # 标注通道类型
        label = "↗ 上升通道" if ctype == "ascending" else "↘ 下降通道"
        mid_x = xs_r[1] - 2
        mid_y = (ys_r[1] + ys_s[1]) / 2
        ax.annotate(label, xy=(mid_x, mid_y), fontsize=8,
                    color=color_line, ha="right", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117",
                              ec=color_line, alpha=0.9, linewidth=1.2))

    elif ctype == "consolidation":
        band = channels["consol_band"]
        cx   = channels["consol_x"]
        if band and cx:
            lo, hi = band
            x0, x1 = cx
            ax.fill_between([x0, x1], lo, hi,
                            color="#ffd700", alpha=0.10, zorder=2)
            ax.plot([x0, x1], [lo, lo],
                    color="#ffd700", linewidth=1.5, linestyle="--", alpha=0.80)
            ax.plot([x0, x1], [hi, hi],
                    color="#ffd700", linewidth=1.5, linestyle="--", alpha=0.80)
            ax.annotate("⬌ 震荡区间",
                        xy=(x1 - 2, hi * 1.003), fontsize=8,
                        color="#ffd700", ha="right", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117",
                                  ec="#ffd700", alpha=0.9, linewidth=1.2))


def compute_entry_signal(df: pd.DataFrame, label: str) -> pd.Series:
    """
    根据策略标签解析进场条件，返回布尔 Series（每根K线是否满足进场信号）。
    不含门控（gate）条件，只看原始进场信号是否触发。
    """
    # 提取进场部分：label = "dc20+cmf | soxx | ma_x" → entry = "dc20+cmf"
    entry = label.split("|")[0].strip().lower() if label else ""

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── DC20 ───────────────────────────────────────────────────────────────
    dc20 = close > high.rolling(20).max().shift(1)

    # ── CMF (14) ───────────────────────────────────────────────────────────
    mf  = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    cmf = mf.rolling(14).sum() / volume.rolling(14).sum()

    # ── RSI (14) ───────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # ── Bollinger Band ─────────────────────────────────────────────────────
    bb_mean = close.rolling(20).mean()
    bb_std  = close.rolling(20).std()
    bb_lo   = bb_mean - 2 * bb_std
    bb_hi   = bb_mean + 2 * bb_std

    # ── EMA cross ──────────────────────────────────────────────────────────
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    ema_cross = (ema20 > ema60) & (ema20.shift(1) <= ema60.shift(1))

    # ── 匹配 ───────────────────────────────────────────────────────────────
    import re
    if "dc20" in entry and "cmf" in entry:
        return dc20 & (cmf > 0)
    elif "dc20" in entry and "ema" in entry:
        return dc20 | ema_cross
    elif "dc20" in entry:
        return dc20
    elif "bb_lo" in entry or "bb_hi" in entry:
        return (close < bb_lo) | (close > bb_hi)
    elif "rsi" in entry:
        m = re.search(r'rsi(\d+)', entry)
        thresh = int(m.group(1)) if m else 35
        return rsi < thresh
    elif "ema" in entry:
        return ema_cross
    elif "vol" in entry:
        vol_ma = volume.rolling(20).mean()
        return (volume > vol_ma * 1.5) & (close > close.shift(1))
    else:
        return pd.Series(False, index=df.index)


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

    # 4b. 信号确认标记：近90天内满足进场条件但未产生新买点（持仓中继续触发）
    signal_series = pd.Series(np.nan, index=df.index, dtype=float)
    n_signal = 0
    try:
        raw_signal = compute_entry_signal(df, label)
        cutoff_90  = df.index[-90] if len(df) >= 90 else df.index[0]
        recent_sig = raw_signal[df.index >= cutoff_90]
        for idx in recent_sig[recent_sig].index:
            if pd.isna(buy_series.get(idx, np.nan)):   # 没有已有买点才画
                signal_series[idx] = df.loc[idx, "Low"] * 0.975
                n_signal += 1
    except Exception:
        pass

    if n_buy:
        add_plots.append(mpf.make_addplot(buy_series,  type="scatter",
                                          markersize=90, marker="^",
                                          color="#00e676", alpha=0.95, panel=0))
    if n_sell:
        add_plots.append(mpf.make_addplot(sell_series, type="scatter",
                                          markersize=90, marker="v",
                                          color="#ff1744", alpha=0.95, panel=0))
    if n_signal:
        add_plots.append(mpf.make_addplot(signal_series, type="scatter",
                                          markersize=35, marker="^",
                                          color="#ffd700", alpha=0.70, panel=0))

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

    # 6b. 趋势通道 / 震荡区间
    try:
        channels = compute_channels(df)
        draw_channels(ax, channels, n_bars=len(df))
    except Exception as e:
        pass  # 通道绘制失败不影响主图

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
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#ffd700",
                   markersize=7, alpha=0.8, label=f"Signal ({n_signal})"),
    ]
    if open_trade:
        legend_items.append(
            mpatches.Patch(color="#f0b42966",
                           label=f"Entry ${open_trade['entry_px']:.1f}")
        )
    # 通道图例
    try:
        ctype = channels.get("type", "none")
        if ctype == "ascending":
            legend_items.append(
                plt.Line2D([0],[0], color="#26a69a", linewidth=1.2,
                           linestyle="--", label="↗ 上升通道"))
        elif ctype == "descending":
            legend_items.append(
                plt.Line2D([0],[0], color="#ef5350", linewidth=1.2,
                           linestyle="--", label="↘ 下降通道"))
        elif ctype == "consolidation":
            legend_items.append(
                mpatches.Patch(color="#ffd70033", label="⬌ 震荡区间"))
    except Exception:
        pass
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
    print(f"OK  (buy:{n_buy} sell:{n_sell} sig:{n_signal})")
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
