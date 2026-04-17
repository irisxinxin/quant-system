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

SECTOR_GROUPS = {
    "🔵 大盘/核心":        ["QQQ", "SPY", "GOOG", "META", "TSLA", "AMZN"],
    "⚡ 半导体/AI算力":    ["NVDA", "ASML", "TSM", "AMD", "ARM", "AVGO", "AEHR", "TXN", "MRVL", "KLAC"],
    "💾 存储":             ["MU", "WDC", "STX", "SNDK"],
    "🏗 AI电力/数据中心":  ["BE", "VRT", "ETN", "GEV", "PWR"],
    "🌐 光子/高速连接":    ["LITE", "COHR", "FN", "AAOI", "LWLG", "VIAV", "CLS", "CIEN", "GLW", "TSEM"],
    "🚚 物流/运输":        ["ODFL", "XPO", "JBHT", "PCAR", "CMI"],
    "🏭 工业/航天制造":    ["CAT", "DE", "HWM", "ITT", "EME", "AME"],
    "💰 金融":             ["MS", "CBOE", "TRV"],
    "🪙 加密/Fintech":    ["COIN", "MSTR"],
    "🔋 电池/稀土":        ["MP", "ALB", "EOSE"],
    "🚀 太空/机器人":      ["LUNR", "PL", "TER", "RKLB"],
}

# 所有 ticker（去重保序）
_seen: set = set()
DEFAULT_TICKERS: list = []
for _tickers in SECTOR_GROUPS.values():
    for _t in _tickers:
        if _t not in _seen:
            _seen.add(_t)
            DEFAULT_TICKERS.append(_t)


def get_sector(ticker: str) -> str:
    """返回 ticker 所属板块名，未知返回 '其他'"""
    for sec, tks in SECTOR_GROUPS.items():
        if ticker.upper() in tks:
            return sec
    return "其他"


# 数据不足时跳过（250≈1年交易日，足以计算1Y/3M/1M分段绩效）
MIN_BARS = 250

# 近期权重：优化目标 = 50%×近1年Calmar + 25%×近3月收益率 + 15%×2026年初至今Calmar + 10%×全期Calmar
# 2026年AI量化资金主导市场，YTD权重单独拎出加强
W_1Y  = 0.50
W_3M  = 0.25
W_YTD = 0.15   # 2026 YTD：重点奖励在今年上涨行情中表现好的策略
W_ALL = 0.10

YTD_START = "2026-01-01"   # 2026年初至今

# 热门板块 ETF（资金追踪用）
HOT_SECTOR_ETFS = {
    "半导体":   "SMH",
    "软件/SaaS": "IGV",
    "军工/航天": "XAR",
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
    """返回 1M/3M/YTD/1Y/全期 五段绩效"""
    ytd = strat_ret[strat_ret.index >= YTD_START] if not strat_ret.empty else strat_ret.iloc[0:0]
    return {
        "1M":  _period_stats(strat_ret.iloc[-21:]  if len(strat_ret) >= 21  else strat_ret),
        "3M":  _period_stats(strat_ret.iloc[-63:]  if len(strat_ret) >= 63  else strat_ret),
        "YTD": _period_stats(ytd if len(ytd) >= 10 else strat_ret.iloc[-63:]),  # 2026至今，不足时降级3M
        "1Y":  _period_stats(strat_ret.iloc[-252:] if len(strat_ret) >= 252 else strat_ret),
        "All": _period_stats(strat_ret),
    }


CALMAR_CAP = 4.0   # 防止极少笔交易的过拟合策略"垄断"排名

def _recency_score(periods: dict, avg_hold: float = 1.0, win_rate: float = 0.5) -> float:
    """近期加权综合评分（越高越好）

    权重组成：
      50% × 近1年Calmar（风险收益）
      25% × 近3月总收益（最近行情实战）
      15% × 2026 YTD Calmar（AI量化资金主导年份）
      10% × 全期Calmar（长期稳健性）
      +持仓时长加成（趋势型策略奖励）
      +胜率加成（高胜率策略额外奖励，低胜率惩罚）
    """
    c1y  = min(periods["1Y"]["calmar"], CALMAR_CAP)
    r3m  = periods["3M"]["ret"] / 30.0   # 标准化到 ~0~1 范围
    cytd = min(periods["YTD"]["calmar"], CALMAR_CAP)  # 2026 YTD
    call = min(periods["All"]["calmar"], CALMAR_CAP)
    # 持仓时长加成：平均持仓越长（趋势型）得分越高，上限0.5分
    hold_bonus = min(avg_hold / 20.0, 0.5)
    # 胜率加成：以50%为中轴，每高1%加0.02，每低1%减0.02，上限±0.3
    # 55%胜率 → +0.1；45%胜率 → -0.1；60%以上 → +0.2封顶
    wr_bonus = max(-0.3, min(0.3, (win_rate - 0.50) * 2.0))
    return W_1Y * c1y + W_3M * r3m + W_YTD * cytd + W_ALL * call + 0.15 * hold_bonus + 0.10 * wr_bonus


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
    qqq_cta: pd.Series | None = None,
    extra_ctas: dict | None = None,
) -> dict:
    """对单只股票跑所有组合，返回最优策略信息"""
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

        # ── 更多指标（用于新信号）──
        hi20_max   = hi.rolling(20).max()          # 近20日最高价（trailing stop 基准）
        ema20_band = (prices >= e20 * 0.97) & (prices <= e20 * 1.04) & (e20 > e60)  # 回踩EMA20
        ema_accel  = (e20 - e20.shift(5)) > 0      # EMA20 向上加速（斜率>0）

        # ── 超卖深度入场（用于低点买入）──
        rsi_deep_os = rsi < 35    # RSI深度超卖
        rsi_ext_os  = rsi < 28    # RSI极端超卖（更高质量信号，更少触发）

        # ── 入场条件 ──
        entries = {
            # 趋势突破信号
            "ema2060":    (e20 > e60).fillna(False).astype(float),
            "dc20":       (prices > dc20h).fillna(False).astype(float),
            "dc20|ema":   ((prices > dc20h) | (e20 > e60)).fillna(False).astype(float),
            "ma5200":     (ma50 > ma200).fillna(False).astype(float),
            "bb_lo":      (prices < bb_lo).fillna(False).astype(float),
            # 主力资金信号
            "obv_up":     (obv > obv_ma20).fillna(False).astype(float),
            "cmf_pos":    (cmf > 0.05).fillna(False).astype(float),
            "mfi_os":     (mfi < 35).fillna(False).astype(float),
            "vol_surge":  vol_surge_up.fillna(False).astype(float),
            # 趋势内回调买入（EMA20 支撑 + 上升趋势）
            "ema20_dip":  ema20_band.fillna(False).astype(float),
            "ema20_dip+obv": (ema20_band & (obv > obv_ma20)).fillna(False).astype(float),
            # 组合确认
            "dc20+obv":   ((prices > dc20h) & (obv > obv_ma20)).fillna(False).astype(float),
            "dc20+cmf":   ((prices > dc20h) & (cmf > 0.0)).fillna(False).astype(float),
            "vol+ema":    (vol_surge_up & (e20 > e60)).fillna(False).astype(float),
            # 低点买入：超卖反弹（适合大跌后低吸，不依赖趋势方向）
            "rsi35":      rsi_deep_os.fillna(False).astype(float),        # RSI<35 深度超卖
            "rsi28":      rsi_ext_os.fillna(False).astype(float),         # RSI<28 极端超卖
        }

        # ── CTA 过滤 ──
        qqq_s = qqq_cta.reindex(prices.index).ffill().fillna(0) if qqq_cta is not None \
                else pd.Series(0.0, index=prices.index)
        cta_gates = {
            "none":  pd.Series(1.0, index=prices.index),
            "combo": (combo_cta_s > 0).astype(float),
            "soft":  (combo_cta_s > -0.25).astype(float),   # 宽松：SMH+SPY 轻微负向也允许入场
            "spy":   (spy_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
            "smh":   (smh_cta.reindex(prices.index).ffill().fillna(0) > 0).astype(float),
            "qqq":   (qqq_s > 0).astype(float),             # 纳指趋势，更贴近科技股
        }

        # 动态注入行业 CTA（soxx/igv/xly/xar/ibit 等）
        if extra_ctas:
            for _cn, _cs in extra_ctas.items():
                if _cn not in cta_gates:
                    cta_gates[_cn] = (_cs.reindex(prices.index).ffill().fillna(0) > 0).astype(float)

        # ── 出场条件 ──
        rsi_was_hot  = rsi.rolling(5).max() > 70          # 近5天RSI曾超70
        exits = {
            "ema_x":     (e20 < e60).fillna(False).astype(float),
            "ma_x":      (ma50 < ma200).fillna(False).astype(float),
            "rsi80":     (rsi > 80).astype(float),
            # 资金出逃信号
            "obv_down":  (obv < obv_ma20).fillna(False).astype(float),
            "cmf_neg":   (cmf < -0.05).fillna(False).astype(float),
            # Trailing stop：从近20日高点回撤触发（适合趋势股持仓更久）
            "trail_8":   (prices < hi20_max * 0.92).fillna(False).astype(float),   # 回撤8%止损
            "trail_12":  (prices < hi20_max * 0.88).fillna(False).astype(float),   # 回撤12%止损
            # 动量衰减：RSI曾高位(>70)后真正回落到60以下才出场，避免追高后被小波动震出
            "rsi_fade":  ((rsi < 60) & rsi_was_hot).fillna(False).astype(float),
            # 超卖反弹止盈：RSI恢复到70以上，动量修复完成，适合低点买入策略的出场
            "rsi70":     (rsi > 70).astype(float),
        }

        results = []
        for en, e_sig in entries.items():
            for cn, c_sig in cta_gates.items():
                for xn, x_sig in exits.items():
                    # 超卖入场（bb_lo / rsi35 / rsi28）：均值回归策略
                    # 基础出场 = 价格恢复到 EMA20 以上（均值修复完成）OR 附加条件
                    if en in ("bb_lo", "rsi35", "rsi28"):
                        # bb_lo 用布林上轨作基础；rsi35/rsi28 用 EMA20 回升作基础
                        if en == "bb_lo":
                            base_exit = (prices > bb_hi).fillna(False)
                        else:
                            base_exit = (prices > e20).fillna(False)   # 价格恢复到 EMA20 以上
                        add_exit = {
                            "ema_x":    (e20 < e60).fillna(False),
                            "ma_x":     (ma50 < ma200).fillna(False),
                            "rsi80":    (rsi > 80),
                            "rsi70":    (rsi > 70),
                            "obv_down": (obv < obv_ma20).fillna(False),
                            "cmf_neg":  (cmf < -0.05).fillna(False),
                            "trail_8":  (prices < hi20_max * 0.92).fillna(False),
                            "trail_12": (prices < hi20_max * 0.88).fillna(False),
                        }.get(xn, pd.Series(False, index=prices.index))
                        eff_exit = (base_exit | add_exit).astype(float)
                    else:
                        eff_exit = x_sig

                    pos = _make_pos(
                        e_sig.fillna(0).values,
                        c_sig.fillna(0).values,
                        eff_exit.fillna(0).values,
                    )
                    sig_s = pd.Series(pos, index=prices.index)

                    in_mkt = sig_s.mean()
                    if in_mkt < 0.03 or in_mkt > 0.98:   # 过滤极端情况
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

                        # 过滤过拟合策略：全期交易次数太少 or 全期 CAGR 太低
                        if n_tr < 5:
                            continue
                        if cagr < 0.03:   # 全期 CAGR < 3%
                            continue

                        # 平均持仓天数（趋势型策略加成）
                        avg_hold = float(res["trades"]["days_held"].mean()) if n_tr > 0 else 1.0

                        # 交易胜率（盈利笔数/总笔数）
                        trades_df = res["trades"]
                        win_rate = float((trades_df["pnl_pct"] > 0).mean()) if n_tr > 0 else 0.5

                        # 分段绩效（近1年/3月/1月/YTD）
                        periods = _multi_period(strat_ret)
                        score   = _recency_score(periods, avg_hold, win_rate)

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
                            "win_rate":  win_rate,
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

        # ── 构建 top3：三个槽保证三种风格 ──
        #   #1: 综合最优（任何风格）
        #   #2: 最佳趋势持仓型出场（rsi_fade/ema_x/ma_x）— 吃大波段
        #   #3: 最佳低买反弹型入场（rsi35/rsi28/mfi_os/bb_lo）— 低点买入
        TREND_EXITS = {"rsi_fade", "ema_x", "ma_x"}
        DIP_ENTRIES = {"rsi35", "rsi28", "mfi_os", "bb_lo"}

        seen = {(best["entry"], best["cta"], best["exit"])}

        # #2: 最佳趋势出场
        slot2 = next((r for r in results
                      if r["exit"] in TREND_EXITS
                      and (r["entry"], r["cta"], r["exit"]) not in seen), None)
        if slot2 is None:   # 无趋势出场时取下一个高分策略
            slot2 = next((r for r in results
                          if (r["entry"], r["cta"], r["exit"]) not in seen), None)
        if slot2:
            seen.add((slot2["entry"], slot2["cta"], slot2["exit"]))

        # #3: 最佳低买策略
        slot3 = next((r for r in results
                      if r["entry"] in DIP_ENTRIES
                      and (r["entry"], r["cta"], r["exit"]) not in seen), None)
        if slot3 is None:   # 无低买策略时取下一个高分策略
            slot3 = next((r for r in results
                          if (r["entry"], r["cta"], r["exit"]) not in seen), None)

        top3 = [r for r in [best, slot2, slot3] if r is not None]

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
            "win_rate":   round(best["win_rate"] * 100, 1),
            "in_market":  round(best["in_market"] * 100, 1),
            "periods":    best["periods"],
            "top3":       top3,
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

    n_entries = 16   # 当前入场数
    n_ctas    = 6    # 当前CTA数
    n_exits   = 9    # 当前出场数
    print(f"\n{'='*72}")
    print(f"  策略批量优化  共 {len(tickers)} 只  ({n_entries}×{n_ctas}×{n_exits}={n_entries*n_ctas*n_exits}种组合/只)")
    print(f"{'='*72}")

    # 预取共用数据
    print("📥 获取参考数据 SMH / SPY / QQQ ...")
    smh_px   = get_prices("SMH")
    spy_px   = get_prices("SPY")
    qqq_px   = get_prices("QQQ")
    smh_cta  = _cta_series(smh_px)
    spy_cta  = _cta_series(spy_px)
    qqq_cta  = _cta_series(qqq_px)
    # 行业 CTA：soxx=费城半导体, igv=软件, xly=消费, xar=军工, ibit=加密
    _sector_etfs = {"soxx": "SOXX", "igv": "IGV", "xly": "XLY", "xar": "XAR", "ibit": "IBIT"}
    extra_ctas = {}
    for _name, _sym in _sector_etfs.items():
        try:
            extra_ctas[_name] = _cta_series(get_prices(_sym))
        except Exception:
            pass
    print(f"   SMH CTA: {smh_cta.iloc[-1]:.2f}  SPY CTA: {spy_cta.iloc[-1]:.2f}  QQQ CTA: {qqq_cta.iloc[-1]:.2f}")

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
            executor.submit(optimize_ticker, t, smh_cta, spy_cta, qqq_cta, extra_ctas): t
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
                    f"1Y_Calmar={p['1Y']['calmar']:4.2f}  "
                    f"胜率={res.get('win_rate', 0):4.1f}%"
                )

    valid  = [raw_results[t] for t in tickers if not raw_results[t].get("error")]
    errors = [raw_results[t] for t in tickers if raw_results[t].get("error")]

    # 按近期评分降序（板块内排序用）
    valid.sort(key=lambda x: x["score"], reverse=True)
    # 附加板块信息
    for r in valid:
        r["sector"] = get_sector(r["ticker"])

    # ── 按板块分组汇总 ──
    print(f"\n{'='*108}")
    print(f"  最优策略汇总（按板块分组）  排序权重: 1Y_Calmar×50% + 3M收益×25% + YTD(2026)×15% + 全期×10% + 胜率加成")
    print(f"{'='*108}")

    # 确定板块顺序（按 SECTOR_GROUPS 顺序，再加"其他"）
    ordered_sectors = list(SECTOR_GROUPS.keys()) + ["其他"]
    sector_map: dict[str, list] = {s: [] for s in ordered_sectors}
    for r in valid:
        sector_map.setdefault(r["sector"], []).append(r)

    for sec in ordered_sectors:
        rows_sec = sector_map.get(sec, [])
        if not rows_sec:
            continue
        print(f"\n  {sec}")
        print(f"  {'─'*104}")
        print(f"  {'股票':7s}  {'类':2s}  {'最优策略':30s}  {'评分':>5s}  {'全期CAGR':>8s}  │  {'1月':>6s}  {'3月':>7s}  {'YTD':>7s}  {'1年':>7s}  {'1年DD':>7s}  {'胜率':>6s}")
        print(f"  {'─'*120}")
        for r in rows_sec:
            p     = r["periods"]
            strat = f"{r['entry']}+{r['cta']}+{r['exit']}"
            mark  = "🔥" if p["3M"]["ret"] > 15 else ("📈" if p["3M"]["ret"] > 0 else "📉")
            print(
                f"  {r['ticker']:7s}  {r['type']:2s}  {strat:30s}  {r['score']:>5.2f}  "
                f"{r['cagr']:>+7.1f}%  │  "
                f"{p['1M']['ret']:>+5.1f}%  {p['3M']['ret']:>+6.1f}%  "
                f"{p['YTD']['ret']:>+6.1f}%  "
                f"{p['1Y']['cagr']:>+6.1f}%  {p['1Y']['dd']:>+6.1f}%  "
                f"{r.get('win_rate', 0):>5.1f}%  {mark}"
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
            wr = s.get("win_rate", 0)
            print(
                f"    #{i} {s['entry']}+{s['cta']}+{s['exit']:15s}  "
                f"评分={s['score']:.2f}  胜率={wr*100:.0f}%  "
                f"1Y_CAGR={sp['1Y']['cagr']:+.1f}%  1Y_DD={sp['1Y']['dd']:+.1f}%  "
                f"YTD={sp['YTD']['ret']:+.1f}%  3M={sp['3M']['ret']:+.1f}%  1M={sp['1M']['ret']:+.1f}%"
            )

    # ── 保存 ──
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    if valid:
        rows = []
        for r in valid:
            rows.append({
                "ticker":      r["ticker"],
                "sector":      r.get("sector", "其他"),
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
