"""
optimize_stocks.py — 批量策略优化
对每只股票测试 60 种策略组合，找 Calmar 最优策略

用法：
  python3 optimize_stocks.py                     # 使用内置列表
  python3 optimize_stocks.py AAPL NVDA MSFT      # 自定义列表
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.WARNING)

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from data.downloader import get_prices, get_ohlcv
from strategies.router import classify_ticker
from backtest.engine import backtest
from config import CTA_LOOKBACKS, CTA_VOL_WIN

DEFAULT_TICKERS = [
    "SNOW", "SNDK", "CRCL", "PLTR", "HIMS", "VRT", "INTC", "GOOG",
    "MSFT", "OKLO", "VST", "ALB", "HOOD", "IREN", "RKLB", "SOXX",
    "BE", "AMD", "ORCL", "AVGO", "EOSE", "GLD",
]

# 数据不足时跳过
MIN_BARS = 400

# 近期权重：优化目标 = 60%×近1年Calmar + 25%×近3月收益率 + 15%×全期Calmar
W_1Y  = 0.60
W_3M  = 0.25
W_ALL = 0.15

# 热门板块 ETF（资金追踪用）
HOT_SECTOR_ETFS = {
    "半导体":   "SMH",
    "AI/科技":  "IGV",
    "核能":     "NLR",
    "太空/国防":"XAR",
    "加密":     "IBIT",
    "生物科技":  "IBB",
    "能源":     "XLE",
    "金融":     "XLF",
    "工业":     "XLI",
    "消费成长":  "XLY",
}


# ──────────────────────────────────────────────
# CTA 日线信号序列
# ──────────────────────────────────────────────

def _cta_series(px: pd.Series) -> pd.Series:
    r = px.pct_change().dropna()
    sigs = []
    for lk in CTA_LOOKBACKS:
        m = r.rolling(lk).mean() * 252
        v = r.rolling(CTA_VOL_WIN).std() * np.sqrt(252)
        sigs.append((m / v.replace(0, np.nan)).clip(-1, 1))
    return pd.concat(sigs, axis=1).mean(axis=1)


# ──────────────────────────────────────────────
# 分段绩效（1M / 3M / 1Y / 全期）
# ──────────────────────────────────────────────

def _period_stats(ret: pd.Series) -> dict:
    """计算一段收益率序列的 总收益% / CAGR% / MaxDD% / Calmar"""
    if ret.empty or len(ret) < 2:
        return {"ret": 0.0, "cagr": 0.0, "dd": 0.0, "calmar": 0.0}
    cum     = (1 + ret).cumprod()
    total   = (cum.iloc[-1] - 1) * 100
    dd      = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    n_years = len(ret) / 252
    cagr    = ((cum.iloc[-1] ** (1 / n_years)) - 1) * 100 if n_years > 0.05 else total
    calmar  = abs(cagr / dd) if dd < 0 else 0.0
    return {"ret": round(total, 1), "cagr": round(cagr, 1),
            "dd": round(dd, 1), "calmar": round(calmar, 2)}


def _multi_period(strat_ret: pd.Series) -> dict:
    """返回 1M/3M/1Y/全期 四段绩效"""
    return {
        "1M":  _period_stats(strat_ret.iloc[-21:]  if len(strat_ret) >= 21  else strat_ret),
        "3M":  _period_stats(strat_ret.iloc[-63:]  if len(strat_ret) >= 63  else strat_ret),
        "1Y":  _period_stats(strat_ret.iloc[-252:] if len(strat_ret) >= 252 else strat_ret),
        "All": _period_stats(strat_ret),
    }


def _recency_score(periods: dict) -> float:
    """近期加权综合评分（越高越好）"""
    c1y  = periods["1Y"]["calmar"]
    r3m  = periods["3M"]["ret"] / 30.0   # 标准化到 ~0~1 范围
    call = periods["All"]["calmar"]
    return W_1Y * c1y + W_3M * r3m + W_ALL * call


# ──────────────────────────────────────────────
# 热门板块扫描
# ──────────────────────────────────────────────

def scan_hot_sectors() -> list:
    """扫描热门板块 1M/3M 表现，按近1月涨幅排序"""
    rows = []
    for name, ticker in HOT_SECTOR_ETFS.items():
        try:
            px = get_prices(ticker)
            if px.empty or len(px) < 65:
                continue
            r1m = round(float(px.pct_change(21).iloc[-1]) * 100, 1)
            r3m = round(float(px.pct_change(63).iloc[-1]) * 100, 1)
            r1w = round(float(px.pct_change(5).iloc[-1])  * 100, 1)
            ohlcv = get_ohlcv(ticker)
            vol_r = round(float(ohlcv["Volume"].iloc[-1] /
                          ohlcv["Volume"].rolling(20).mean().iloc[-1]), 1)
            rows.append({"sector": name, "ticker": ticker,
                         "1W%": r1w, "1M%": r1m, "3M%": r3m, "vol_r": vol_r})
        except Exception:
            pass
    rows.sort(key=lambda x: x["1M%"], reverse=True)
    return rows


# ──────────────────────────────────────────────
# 状态机：把入场/CTA/出场信号合成仓位
# ──────────────────────────────────────────────

def _make_pos(entry: np.ndarray, cta_ok: np.ndarray, exit_cond: np.ndarray) -> np.ndarray:
    """
    Logic:
      - 入场: entry==1 且 cta_ok==1（当天触发，次日由 backtest engine shift执行）
      - 持仓: 只要 exit_cond==0 就继续持有
      - 离场: exit_cond==1 立即清仓
    """
    pos = np.zeros(len(entry), dtype=float)
    in_trade = False
    for i in range(len(entry)):
        if not in_trade:
            if entry[i] and cta_ok[i]:
                in_trade = True
                pos[i] = 1.0
        else:
            if exit_cond[i]:
                in_trade = False
            else:
                pos[i] = 1.0
    return pos


# ──────────────────────────────────────────────
# 单股优化
# ──────────────────────────────────────────────

def optimize_ticker(
    ticker: str,
    smh_cta: pd.Series,
    spy_cta: pd.Series,
) -> dict:
    """对单只股票跑 60 种组合，返回最优策略信息"""
    try:
        prices = get_prices(ticker)
        ohlcv  = get_ohlcv(ticker)

        if prices.empty or len(prices) < MIN_BARS:
            return {"ticker": ticker, "error": f"数据不足（{len(prices)}行，需≥{MIN_BARS}）"}

        # ── 分类 ──
        info = classify_ticker(ticker)
        asset_type = info["type"]
        ann_vol    = info.get("vol") or round(float(prices.pct_change().tail(252).std() * 252**0.5), 2)

        # ── 指标 ──
        e20  = prices.ewm(span=20, adjust=False).mean()
        e60  = prices.ewm(span=60, adjust=False).mean()
        ma50 = prices.rolling(50).mean()
        ma200= prices.rolling(200).mean()

        hi    = ohlcv["High"].reindex(prices.index)
        lo    = ohlcv["Low"].reindex(prices.index)
        vol   = ohlcv["Volume"].reindex(prices.index).fillna(0)
        dc20h = hi.rolling(20).max().shift(1)

        bb_m  = prices.rolling(20).mean()
        bb_s  = prices.rolling(20).std()
        bb_lo = bb_m - 2 * bb_s
        bb_hi = bb_m + 2 * bb_s

        d    = prices.diff()
        gain = d.clip(lower=0).rolling(10).mean()
        loss = (-d.clip(upper=0)).rolling(10).mean()
        rsi  = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50)

        # ── 主力资金流向指标 ──────────────────────────────────
        # OBV (On-Balance Volume)：价涨累加量，价跌累减量
        obv_dir = np.sign(prices.diff().fillna(0))
        obv     = (vol * obv_dir).cumsum()
        obv_ma20 = obv.rolling(20).mean()

        # CMF (Chaikin Money Flow 20)：主力买卖压力
        hl = (hi - lo).replace(0, np.nan)
        mf_mult = ((prices - lo) - (hi - prices)) / hl   # -1~+1，正=资金流入
        cmf = (mf_mult * vol).rolling(20).sum() / vol.rolling(20).sum().replace(0, np.nan)

        # MFI (Money Flow Index 14)：量价版RSI
        tp      = (hi + lo + prices) / 3
        raw_mf  = tp * vol
        pos_mf  = raw_mf.where(tp > tp.shift(1), 0.0)
        neg_mf  = raw_mf.where(tp < tp.shift(1), 0.0)
        mf_rat  = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
        mfi     = (100 - 100 / (1 + mf_rat)).fillna(50)

        # 放量突破（量能>1.5x 均量 且 价格站上EMA20）
        vol_ratio    = vol / vol.rolling(20).mean().replace(0, np.nan)
        vol_surge_up = (vol_ratio > 1.5) & (prices > e20)

        # ── CTA 日线序列（对齐到本股票日期）──
        combo_cta_s = (
            smh_cta.reindex(prices.index).ffill().fillna(0) +
            spy_cta.reindex(prices.index).ffill().fillna(0)
        ) / 2

        # ── 入场条件 ──
        entries = {
            # 原有趋势信号
            "ema2060":    (e20 > e60).fillna(False).astype(float),
            "dc20":       (prices > dc20h).fillna(False).astype(float),
            "dc20|ema":   ((prices > dc20h) | (e20 > e60)).fillna(False).astype(float),
            "ma5200":     (ma50 > ma200).fillna(False).astype(float),
            "bb_lo":      (prices < bb_lo).fillna(False).astype(float),
            # 主力资金流向信号（新增）
            "obv_up":     (obv > obv_ma20).fillna(False).astype(float),          # OBV站上均线=主力净买入
            "cmf_pos":    (cmf > 0.05).fillna(False).astype(float),              # CMF>5%=明显资金流入
            "mfi_os":     (mfi < 35).fillna(False).astype(float),                # MFI超卖=主力低吸
            "vol_surge":  vol_surge_up.fillna(False).astype(float),              # 放量站上EMA20=主力拉升
            # 组合：价格突破 + 主力资金双重确认
            "dc20+obv":   ((prices > dc20h) & (obv > obv_ma20)).fillna(False).astype(float),
            "dc20+cmf":   ((prices > dc20h) & (cmf > 0.0)).fillna(False).astype(float),
            "vol+ema":    (vol_surge_up & (e20 > e60)).fillna(False).astype(float),
        }

        # ── CTA 过滤 ──
        cta_gates = {
            "none":  pd.Series(1.0, index=prices.index),
            "combo": (combo_cta_s > 0).astype(float),
            "spy":   (spy_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
            "smh":   (smh_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
        }

        # ── 出场条件 ──
        exits = {
            "ema_x":    (e20 < e60).fillna(False).astype(float),
            "ma_x":     (ma50 < ma200).fillna(False).astype(float),
            "rsi80":    (rsi > 80).astype(float),
            # 资金出逃信号出场
            "obv_down": (obv < obv_ma20).fillna(False).astype(float),  # OBV跌破均线=主力出逃
            "cmf_neg":  (cmf < -0.05).fillna(False).astype(float),     # CMF<-5%=明显资金流出
        }

        results = []
        for en, e_sig in entries.items():
            for cn, c_sig in cta_gates.items():
                for xn, x_sig in exits.items():
                    # bb_lo 是均值回归：出场 = 触及上轨 OR 附加条件
                    if en == "bb_lo":
                        bb_exit_base = (prices > bb_hi).fillna(False)
                        add_exit = {
                            "ema_x":    (e20 < e60).fillna(False),
                            "ma_x":     (ma50 < ma200).fillna(False),
                            "rsi80":    (rsi > 80),
                            "obv_down": (obv < obv_ma20).fillna(False),
                            "cmf_neg":  (cmf < -0.05).fillna(False),
                        }.get(xn, pd.Series(False, index=prices.index))
                        eff_exit = (bb_exit_base | add_exit).astype(float)
                    else:
                        eff_exit = x_sig

                    pos = _make_pos(
                        e_sig.fillna(0).values,
                        c_sig.fillna(0).values,
                        eff_exit.fillna(0).values,
                    )
                    sig_s = pd.Series(pos, index=prices.index)

                    in_mkt = sig_s.mean()
                    if in_mkt < 0.03 or in_mkt > 0.97:   # 过滤极端情况
                        continue

                    try:
                        res    = backtest(prices, sig_s)
                        m      = res["metrics"]
                        strat_ret = res["returns"]

                        # 全期指标
                        calmar = float(m.get("Calmar比率", 0))
                        cagr   = float(m.get("年化收益(CAGR)", "0%").replace("%", "")) / 100
                        dd     = float(m.get("最大回撤", "0%").replace("%", "")) / 100
                        sharpe = float(m.get("Sharpe比率", 0))
                        n_tr   = len(res["trades"])

                        # 分段绩效（近1年/3月/1月）
                        periods = _multi_period(strat_ret)
                        score   = _recency_score(periods)

                        results.append({
                            "entry":     en,
                            "cta":       cn,
                            "exit":      xn,
                            "score":     score,      # 近期加权综合评分（排序用）
                            "calmar":    calmar,
                            "cagr":      cagr,
                            "dd":        dd,
                            "sharpe":    sharpe,
                            "n_trades":  n_tr,
                            "in_market": in_mkt,
                            "periods":   periods,
                        })
                    except Exception:
                        pass

        if not results:
            return {"ticker": ticker, "type": asset_type, "error": "无有效策略"}

        # 按近期加权综合评分排序
        results.sort(key=lambda x: x["score"], reverse=True)
        best = results[0]

        return {
            "ticker":     ticker,
            "type":       asset_type,
            "ann_vol":    round(ann_vol * 100, 1) if ann_vol else "?",
            "entry":      best["entry"],
            "cta":        best["cta"],
            "exit":       best["exit"],
            "score":      round(best["score"], 2),
            "calmar":     round(best["calmar"], 2),
            "cagr":       round(best["cagr"] * 100, 1),
            "dd":         round(best["dd"] * 100, 1),
            "sharpe":     round(best["sharpe"], 2),
            "n_trades":   best["n_trades"],
            "in_market":  round(best["in_market"] * 100, 1),
            "periods":    best["periods"],
            "top3":       results[:3],
            "error":      None,
        }

    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main(tickers: list | None = None) -> list:
    if tickers is None:
        tickers = DEFAULT_TICKERS

    print(f"\n{'='*72}")
    print(f"  策略批量优化  共 {len(tickers)} 只  (60种组合/只)")
    print(f"{'='*72}")

    # 预取共用数据
    print("📥 获取参考数据 SMH / SPY ...")
    smh_px   = get_prices("SMH")
    spy_px   = get_prices("SPY")
    smh_cta  = _cta_series(smh_px)
    spy_cta  = _cta_series(spy_px)
    print(f"   SMH CTA当前: {smh_cta.iloc[-1]:.2f}  SPY CTA当前: {spy_cta.iloc[-1]:.2f}")

    print(f"\n⚡ 并行优化中...\n")

    # ── 热门板块扫描 ──
    print("\n🔥 热门板块扫描...")
    hot = scan_hot_sectors()
    print(f"\n  {'板块':8s}  {'ETF':5s}  {'1周':>6s}  {'1月':>6s}  {'3月':>7s}  {'量能':>5s}")
    print(f"  {'-'*46}")
    for s in hot:
        hot_mark = " 🔥" if s["1M%"] > 5 else (" 📈" if s["1M%"] > 0 else " 📉")
        print(f"  {s['sector']:8s}  {s['ticker']:5s}  {s['1W%']:>+5.1f}%  {s['1M%']:>+5.1f}%  {s['3M%']:>+6.1f}%  {s['vol_r']:>4.1f}x{hot_mark}")

    print(f"\n⚡ 并行优化中（近期加权评分）...\n")

    raw_results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(optimize_ticker, t, smh_cta, spy_cta): t
            for t in tickers
        }
        for future in as_completed(futures):
            t   = futures[future]
            res = future.result()
            raw_results[t] = res
            if res.get("error"):
                print(f"  ❌ {t:8s}  {res['error']}")
            else:
                p = res["periods"]
                print(
                    f"  ✅ {t:8s}  {res['type']}类  "
                    f"{res['entry']}+{res['cta']}+{res['exit']:10s}  "
                    f"评分={res['score']:4.2f}  "
                    f"1M={p['1M']['ret']:+5.1f}%  3M={p['3M']['ret']:+6.1f}%  "
                    f"1Y_Calmar={p['1Y']['calmar']:4.2f}"
                )

    valid  = [raw_results[t] for t in tickers if not raw_results[t].get("error")]
    errors = [raw_results[t] for t in tickers if raw_results[t].get("error")]
    valid.sort(key=lambda x: x["score"], reverse=True)

    # ── 汇总表（含分段表现）──
    print(f"\n{'='*100}")
    print(f"  最优策略汇总（近期加权评分降序）  排序权重: 1Y_Calmar×60% + 3M收益×25% + 全期×15%")
    print(f"{'='*100}")
    print(f"  {'股票':6s}  {'类':2s}  {'最优策略':28s}  {'评分':>5s}  {'全期':>7s}  │  {'1月':>6s}  {'3月':>7s}  {'1年':>7s}  {'1年DD':>7s}")
    print(f"  {'-'*98}")

    for r in valid:
        p    = r["periods"]
        strat = f"{r['entry']}+{r['cta']}+{r['exit']}"
        # 近期颜色标记（用文字代替颜色）
        m1  = f"{p['1M']['ret']:>+5.1f}%"
        m3  = f"{p['3M']['ret']:>+6.1f}%"
        m1y = f"{p['1Y']['cagr']:>+6.1f}%"
        dd1y= f"{p['1Y']['dd']:>+6.1f}%"
        mark = "🔥" if p["3M"]["ret"] > 15 else ("📈" if p["3M"]["ret"] > 0 else "📉")
        print(
            f"  {r['ticker']:6s}  {r['type']:2s}  {strat:28s}  {r['score']:>5.2f}  "
            f"{r['cagr']:>+6.1f}%  │  {m1}  {m3}  {m1y}  {dd1y}  {mark}"
        )

    if errors:
        print(f"\n  ❌ 跳过（数据不足）: {', '.join(r['ticker'] for r in errors)}")

    # ── 近期最热 Top 5 ──
    by_3m = sorted(valid, key=lambda x: x["periods"]["3M"]["ret"], reverse=True)
    print(f"\n{'='*72}")
    print(f"  近3月最强势 Top 5（策略实际赚的，不是股票涨幅）")
    print(f"{'='*72}")
    print(f"  {'股票':6s}  {'策略':28s}  {'1月':>6s}  {'3月':>7s}  {'1年CAGR':>9s}  {'1年DD':>7s}")
    print(f"  {'-'*70}")
    for r in by_3m[:5]:
        p = r["periods"]
        print(
            f"  {r['ticker']:6s}  {r['entry']}+{r['cta']}+{r['exit']:22s}  "
            f"{p['1M']['ret']:>+5.1f}%  {p['3M']['ret']:>+6.1f}%  "
            f"{p['1Y']['cagr']:>+8.1f}%  {p['1Y']['dd']:>+6.1f}%"
        )

    # ── Top 3 细节 ──
    print(f"\n{'='*72}")
    print(f"  各股前 3 策略明细（近期加权评分排序）")
    print(f"{'='*72}")
    for r in valid:
        p = r["periods"]
        print(f"\n  {r['ticker']} ({r['type']}类, 波动{r['ann_vol']}%)  "
              f"1M:{p['1M']['ret']:+.1f}%  3M:{p['3M']['ret']:+.1f}%  "
              f"1Y_Calmar:{p['1Y']['calmar']:.2f}")
        for i, s in enumerate(r.get("top3", [])[:3], 1):
            sp = s["periods"]
            print(
                f"    #{i} {s['entry']}+{s['cta']}+{s['exit']:15s}  "
                f"评分={s['score']:.2f}  "
                f"1Y_CAGR={sp['1Y']['cagr']:+.1f}%  1Y_DD={sp['1Y']['dd']:+.1f}%  "
                f"3M={sp['3M']['ret']:+.1f}%  1M={sp['1M']['ret']:+.1f}%"
            )

    # ── 保存 ──
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    if valid:
        rows = []
        for r in valid:
            rows.append({
                "ticker":      r["ticker"],
                "type":        r["type"],
                "ann_vol_pct": r["ann_vol"],
                "entry":       r["entry"],
                "cta":         r["cta"],
                "exit":        r["exit"],
                "calmar":      r["calmar"],
                "cagr_pct":    r["cagr"],
                "max_dd_pct":  r["dd"],
                "sharpe":      r["sharpe"],
                "n_trades":    r["n_trades"],
                "in_market_pct": r["in_market"],
            })
        pd.DataFrame(rows).to_csv(out_dir / "strategy_optimization.csv", index=False)
        print(f"\n  💾 已保存 output/strategy_optimization.csv")

    print()
    return valid


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    main(tickers)
