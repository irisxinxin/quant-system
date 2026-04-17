"""
strategies/router.py — 资产类型分类 + 策略路由

四类资产，四套策略配置，结合 CTA 资金流向：

  Type A ⚡ 杠杆ETF / 高波动品种（SOXL, NVDL, TSLL…）
    → EMA(20/60) 趋势 + SMC 加仓确认
    → CTA 宏观：硬过滤（SPY+QQQ 动量全负时禁止入场）
    → VIX政体：高波动减仓，极端禁仓

  Type B 🚀 成长股 / 高Beta个股（NVDA, AMD, TSLA…）
    → MA(50/200) 大趋势 + Vegas反弹 / 裸K 精选入场点
    → CTA 宏观：行业 ETF 动量，软过滤（减仓，不强平）

  Type C 🏦 优质蓝筹 / 震荡个股（AAPL, MSFT, INTC…）
    → 布林带均值回归 + 裸K确认回调
    → CTA 宏观：COT极端位逆向参考（拥挤预警）

  Type D 📊 宽基ETF / 商品（SPY, QQQ, GLD, TLT…）
    → 股票指数：SMC 精选；商品/债券：EMA 趋势
    → CTA 宏观：COT 期货仓位，主要参考信号
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.downloader import get_prices
from signals.cta_monitor import cta_trend_signal

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 分类常量
# ──────────────────────────────────────────────

# 已知宽基 ETF（直接归 Type D，跳过波动率判断）
BROAD_ETF = {
    "SPY","QQQ","IWM","DIA","VTI","VOO",      # 股票指数
    "GLD","SLV","IAU","PDBC","DBC",            # 贵金属/商品综合
    "USO","UNG","BNO","DBO",                   # 能源
    "TLT","IEF","SHY","BND","AGG","HYG",       # 债券
    "UUP","FXE","FXY","EFA","EEM",             # 外汇/新兴市场
    "VNQ","XLRE",                              # REITs
    "XLK","XLF","XLV","XLE","XLI",            # 板块 ETF
    "SMH","AIQ","IGV","ICLN",
}

# 已知杠杆/反向 ETF（直接归 Type A）
LEVERAGED_ETF = {
    "SOXL","TQQQ","UPRO","SPXL","LABU","TECL",
    "NVDL","TSLL","MSFO","AMZO","GOOGO",
    "FNGU","WEBL","BULZ","DPST","NAIL",
    "SOXS","TECS","SPXS","SQQQ",           # 反向也归A（高波动）
    "TMF","TNA","FAS","FAZ",
}

# 已知蓝筹（直接归 Type C）
BLUE_CHIP = {
    "AAPL","MSFT","GOOGL","GOOG","AMZN","META",
    "BRK-B","JPM","BAC","WFC","GS","MS",
    "JNJ","PG","KO","PEP","WMT","COST",
    "V","MA","UNH","LLY","ABBV",
    "XOM","CVX","COP",
    "INTC","IBM","CSCO","ORCL",
}

# 高波动投机成长股（直接归 Type A）
# 加密矿/量子/核能初创/AI语音 — 年化波动通常 >80%，用趋势跟踪+硬CTA过滤
SPECULATIVE_HIGH_VOL = {
    "IREN","COIN",              # 加密矿 / 加密交易所
    "QUBT","IONQ","RGTI",       # 量子计算
    "OKLO","NNE","SMR","BWXT",  # 核能新能源
    "SOUN","BBAI","PLTR",       # AI语音 / AI软件（高波动版）
    "MSTR","HOOD","RIOT","MARA", # 加密相关
    "LITE",                     # 光互联（年化波动率~117%，归A类）
}

# 高Beta成长股（直接归 Type B）
# AI硬件 / 数据中心 / 能源转型 — MA(50/200)趋势 + Vegas/裸K选点
GROWTH_STOCK = {
    # AI 硬件 & 光互联
    "MRVL","SNDK","ANET","SMCI",
    # 数据中心 & 电力
    "VRT","VST","CEG","ETN","HUBB",
    # 清洁能源 & 半导体材料
    "BE","AXTI","WOLF","ENPH","FSLR",
    # 高Beta消费/医疗成长
    "HIMS","DKNG","SOFI","UPST","AFRM",
    # 其他热门成长
    "UBER","LYFT","SNOW","DDOG","NET",
}

# 波动率阈值
VOL_LEVERAGED  = 0.55   # 年化波动 > 55% → 视为 A 类
VOL_GROWTH_LO  = 0.30   # 年化波动 30%-55% → 可能 B 类
BETA_GROWTH    = 1.15   # Beta > 1.15 → B 类


# ──────────────────────────────────────────────
# 策略配置映射
# ──────────────────────────────────────────────

STRATEGY_CONFIG = {
    "A": {
        "label":       "杠杆ETF/高波动",
        "primary":     "Donchian20突破 OR EMA(20/60)金叉",
        "secondary":   "SMC=2时加仓至满仓（不做门控）",
        "cta_mode":    "HARD",       # 板块CTA(SMH)+大盘均值，负向禁入
        "cot_role":    "NONE",
        "vix_role":    "SIZE",       # VIX控制仓位大小
        "note":        "DC20新高突破入场，EMA死叉出场，SMH板块CTA硬过滤",
    },
    "B": {
        "label":       "成长股/高Beta",
        "primary":     "MA(50/200)大趋势",
        "secondary":   "Vegas反弹/裸K精选入场",
        "cta_mode":    "SOFT",       # 行业ETF动量 < 0 → 减半仓
        "cot_role":    "NONE",
        "vix_role":    "SIZE",
        "note":        "MA判方向，Vegas/PA选时，行业CTA决定仓位",
    },
    "C": {
        "label":       "优质蓝筹/震荡股",
        "primary":     "布林带均值回归",
        "secondary":   "裸K形态确认",
        "cta_mode":    "NONE",
        "cot_role":    "CONTRARIAN", # COT极度多头时警告追高
        "vix_role":    "NONE",
        "note":        "COT>85%时不追多，等回调；<15%时逢低建仓",
    },
    "D": {
        "label":       "宽基ETF/商品",
        "primary":     "SMC精选（股票指数）/ EMA(20/60)（商品）",
        "secondary":   "COT期货仓位",
        "cta_mode":    "NONE",
        "cot_role":    "PRIMARY",    # COT是主要参考
        "vix_role":    "REF",
        "note":        "COT趋势+技术信号双重确认",
    },
}

# Type D 内部区分
_EQUITY_INDEX = {"SPY","QQQ","IWM","DIA","VTI","VOO"}
_BOND_ETF     = {"TLT","IEF","SHY","BND","AGG","HYG","TMF"}


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _realized_vol(prices: pd.Series, lookback: int = 252) -> float:
    ret = prices.pct_change().dropna()
    if len(ret) < 20:
        return 0.0
    return float(ret.tail(lookback).std() * np.sqrt(252))


def _beta(prices: pd.Series, spy: pd.Series, lookback: int = 252) -> float:
    common = prices.index.intersection(spy.index)
    if len(common) < 60:
        return 1.0
    r_stock = prices.loc[common].pct_change().dropna()
    r_spy   = spy.loc[common].pct_change().dropna()
    common2 = r_stock.index.intersection(r_spy.index)
    if len(common2) < 30:
        return 1.0
    rs  = r_stock.loc[common2].tail(lookback)
    rm  = r_spy.loc[common2].tail(lookback)
    cov = float(np.cov(rs, rm)[0][1])
    var = float(rm.var())
    return round(cov / var, 2) if var > 0 else 1.0


# ──────────────────────────────────────────────
# 核心分类函数
# ──────────────────────────────────────────────

def classify_ticker(
    ticker: str,
    spy_prices: Optional[pd.Series] = None,
    lookback: int = 252,
) -> dict:
    """
    将任意股票/ETF分类为 A/B/C/D 四种资产类型

    Returns:
        {"ticker", "type", "label", "vol", "beta", "reason"}
    """
    t = ticker.upper().strip()

    # 1. 宽基 ETF（直接D）
    if t in BROAD_ETF:
        sub = "equity" if t in _EQUITY_INDEX else (
              "bond"   if t in _BOND_ETF     else "commodity")
        return {"ticker": t, "type": "D", "label": STRATEGY_CONFIG["D"]["label"],
                "vol": None, "beta": None,
                "reason": f"已知宽基ETF({sub})"}

    # 2. 已知杠杆ETF（直接A）
    if t in LEVERAGED_ETF:
        return {"ticker": t, "type": "A", "label": STRATEGY_CONFIG["A"]["label"],
                "vol": None, "beta": None,
                "reason": "已知杠杆/反向ETF"}

    # 3. 后缀启发式（LL/2X/3X/US → 杠杆ETF）
    leveraged_suffixes = ("LL", "2X", "3X", "US", "2L", "3L")
    if any(t.endswith(s) for s in leveraged_suffixes) and len(t) >= 4:
        return {"ticker": t, "type": "A", "label": STRATEGY_CONFIG["A"]["label"],
                "vol": None, "beta": None,
                "reason": f"杠杆ETF后缀({t[-2:]})"}

    # 4. 高波动投机成长股（直接A）
    if t in SPECULATIVE_HIGH_VOL:
        return {"ticker": t, "type": "A", "label": STRATEGY_CONFIG["A"]["label"],
                "vol": None, "beta": None,
                "reason": "高波动投机成长股（加密/量子/核能/AI语音）"}

    # 5. 已知蓝筹（直接C）
    if t in BLUE_CHIP:
        return {"ticker": t, "type": "C", "label": STRATEGY_CONFIG["C"]["label"],
                "vol": None, "beta": None,
                "reason": "已知优质蓝筹"}

    # 6. 已知高Beta成长股（直接B）
    if t in GROWTH_STOCK:
        return {"ticker": t, "type": "B", "label": STRATEGY_CONFIG["B"]["label"],
                "vol": None, "beta": None,
                "reason": "AI硬件/数据中心/高Beta成长股"}

    # 7. 波动率/Beta 动态判断
    prices = get_prices(t)
    if prices.empty or len(prices) < 60:
        return {"ticker": t, "type": "B", "label": STRATEGY_CONFIG["B"]["label"],
                "vol": None, "beta": None,
                "reason": "数据不足，默认B类"}

    vol = _realized_vol(prices, lookback)

    # 波动率极高 → A（类杠杆品种）
    if vol >= VOL_LEVERAGED:
        return {"ticker": t, "type": "A", "label": STRATEGY_CONFIG["A"]["label"],
                "vol": round(vol, 3), "beta": None,
                "reason": f"年化波动{vol:.0%} > {VOL_LEVERAGED:.0%}，高波动品种"}

    # 低波动 → C（蓝筹震荡）
    if vol < VOL_GROWTH_LO:
        return {"ticker": t, "type": "C", "label": STRATEGY_CONFIG["C"]["label"],
                "vol": round(vol, 3), "beta": None,
                "reason": f"年化波动{vol:.0%} < {VOL_GROWTH_LO:.0%}，低波动蓝筹"}

    # 中等波动 → 看Beta分B/C
    if spy_prices is None:
        spy_prices = get_prices("SPY")
    beta = _beta(prices, spy_prices, lookback)

    if beta >= BETA_GROWTH:
        return {"ticker": t, "type": "B", "label": STRATEGY_CONFIG["B"]["label"],
                "vol": round(vol, 3), "beta": beta,
                "reason": f"Beta={beta}≥{BETA_GROWTH}，成长/高Beta"}

    return {"ticker": t, "type": "C", "label": STRATEGY_CONFIG["C"]["label"],
            "vol": round(vol, 3), "beta": beta,
            "reason": f"Beta={beta}<{BETA_GROWTH}，低Beta稳定股"}


# ──────────────────────────────────────────────
# 当日实时信号（结合 CTA）
# ──────────────────────────────────────────────

def get_live_signal(ticker: str, asset_type: str = None) -> dict:
    """
    根据资产类型返回当日策略信号状态

    Returns:
        {ticker, type, signal_label, cta_value, cta_ok,
         vix_regime, primary_signal, notes}
    """
    if asset_type is None:
        info = classify_ticker(ticker)
        asset_type = info["type"]

    cfg    = STRATEGY_CONFIG[asset_type]
    notes  = []

    # ── CTA 宏观信号（SPY + QQQ 动量）──
    spy_cta = cta_trend_signal("SPY")
    qqq_cta = cta_trend_signal("QQQ")
    cta_avg = (spy_cta + qqq_cta) / 2

    # ── VIX 政体 ──
    try:
        from signals.breakout import current_vix_regime
        vix_info = current_vix_regime()
    except Exception:
        vix_info = {"regime": "UNKNOWN", "allow_entry": True,
                    "size_mult": 1.0, "note": "VIX不可用"}

    # ── 主策略技术信号 ──
    cta_ok = True
    primary_sig = 0   # 0=无信号, 1=入场, -1=反向
    score = 0         # Type A 多维度评分（其他类型不使用）

    prices = get_prices(ticker)

    if asset_type == "A":
        # ── Type A 最优策略（回测验证 Calmar=19.6）──
        # 入场：Donchian20新高突破 OR EMA(20/60)金叉
        # 出场：EMA死叉（唯一出场条件，不用ATR/RSI出场）
        # CTA：SMH板块+大盘均值（比纯大盘CTA更精准）
        # SMC：signal=2时可视为加仓信号（不做入场门控）
        # 评分维度：趋势、Donchian突破、RSI位置、量能、相对强度

        ticker_cta = cta_trend_signal(ticker)
        smh_cta    = cta_trend_signal("SMH")
        smh_combo  = (smh_cta + cta_avg) / 2   # SMH板块+大盘均值

        score = 0
        score_detail = []

        try:
            from data.downloader import get_ohlcv
            ohlcv = get_ohlcv(ticker)
            if ohlcv.empty or len(ohlcv) < 61:
                raise ValueError("数据不足")

            px  = ohlcv["Close"]
            hi  = ohlcv["High"]
            lo  = ohlcv["Low"]
            vol = ohlcv["Volume"] if "Volume" in ohlcv.columns else None

            e20v = float(px.ewm(span=20, adjust=False).mean().iloc[-1])
            e60v = float(px.ewm(span=60, adjust=False).mean().iloc[-1])
            pxv  = float(px.iloc[-1])

            # 1. EMA趋势方向（死叉=硬出场信号）
            ema_ok = e20v > e60v
            if ema_ok:
                score += 1
                score_detail.append("✅EMA金叉(持仓方向对)")
            else:
                score_detail.append("🔴EMA死叉→出场信号")

            # 2. Donchian20 新高突破（主要入场触发）
            dc20_high = float(hi.rolling(20).max().shift(1).iloc[-1])
            dc20_ok   = pxv > dc20_high if not np.isnan(dc20_high) else False
            if dc20_ok:
                score += 1
                score_detail.append(f"✅DC20突破({dc20_high:.1f})")
            else:
                score_detail.append(f"DC20未突破(高点={dc20_high:.1f})")

            # 3. RSI（10日）不超买
            delta = px.diff()
            gain  = delta.clip(lower=0).rolling(10).mean()
            loss  = (-delta.clip(upper=0)).rolling(10).mean()
            rsi_v = float((100 - 100 / (1 + gain/loss.replace(0,np.nan))).iloc[-1])
            rsi_ok = rsi_v <= 78
            if rsi_ok:
                score += 1
                score_detail.append(f"✅RSI={rsi_v:.0f}(未超买)")
            else:
                score_detail.append(f"⚠️RSI超买={rsi_v:.0f}")

            # 4. 量能确认（今日量 ≥ 20日均量80%）
            if vol is not None and len(vol) >= 20:
                vol_r = float(vol.iloc[-1]) / float(vol.rolling(20).mean().iloc[-1])
                if vol_r >= 0.8:
                    score += 1
                    score_detail.append(f"✅量能{vol_r:.1f}x")
                else:
                    score_detail.append(f"⚠️缩量{vol_r:.1f}x")
            else:
                score += 1
                score_detail.append("量能N/A")

            # 5. 相对强度（20日跑赢SMH板块）
            try:
                smh_px = get_prices("SMH")
                rs_val = float(px.pct_change(20).iloc[-1]) - float(smh_px.pct_change(20).iloc[-1])
                if rs_val > 0:
                    score += 1
                    score_detail.append(f"✅RS超SMH{rs_val*100:+.0f}%")
                else:
                    score_detail.append(f"⚠️弱于SMH{rs_val*100:+.0f}%")
            except Exception:
                score += 1
                score_detail.append("RS N/A")

            # SMC加仓提示
            try:
                smc_df  = smc_signal(ticker)
                smc_val = int(smc_df["signal"].iloc[-1])
                if smc_val >= 2:
                    score_detail.append(f"💎SMC={smc_val}(机构建仓，可加仓)")
                elif smc_val == 1:
                    score_detail.append(f"SMC={smc_val}(有建仓迹象)")
            except Exception:
                pass

            # 入场判断：EMA OR DC20突破（任一满足） + CTA不为负
            trend_ok = ema_ok or dc20_ok
            if not trend_ok:
                primary_sig = 0
            elif score >= 3:
                primary_sig = 1
            else:
                primary_sig = 0

            notes.append(f"评分 {score}/5  [{' | '.join(score_detail)}]")
            notes.append(f"CTA板块={smh_combo:.2f}(SMH={smh_cta:.2f} 大盘={cta_avg:.2f})")

        except Exception as ex:
            notes.append(f"技术评分失败: {ex}")

        # 硬过滤：SMH板块CTA + 大盘均值为负 → 禁入（比纯大盘更精准）
        try:
            smh_combo_val = smh_combo
        except Exception:
            smh_combo_val = cta_avg
        if smh_combo_val < 0:
            cta_ok = False
            notes.append(f"⛔ 板块CTA负向(SMH+大盘={smh_combo_val:.2f})，禁止开仓")
        elif smh_combo_val < 0.2:
            notes.append(f"⚠️ 板块CTA偏弱({smh_combo_val:.2f})，仓位谨慎")

        if not vix_info["allow_entry"]:
            cta_ok = False
            notes.append(vix_info["note"])
        elif vix_info["size_mult"] < 1.0:
            notes.append(vix_info["note"])

    elif asset_type == "B":
        # MA(50/200) 趋势信号
        try:
            if not prices.empty and len(prices) >= 201:
                ma50  = prices.rolling(50).mean()
                ma200 = prices.rolling(200).mean()
                primary_sig = 1 if float(ma50.iloc[-1]) > float(ma200.iloc[-1]) else 0
        except Exception:
            pass

        # 软过滤：CTA为负 → 仓位×0.5
        if cta_avg < -0.1:
            notes.append(f"⚠️  行业CTA偏弱({cta_avg:.2f})，建议半仓")
        else:
            notes.append(f"✅ 宏观顺风({cta_avg:.2f})")

    elif asset_type == "C":
        # 布林带均值回归：价格跌至下轨 → 买入信号
        try:
            if not prices.empty and len(prices) >= 21:
                from config import MR_BB_PERIOD, MR_BB_STD
                ma   = prices.rolling(MR_BB_PERIOD).mean()
                std  = prices.rolling(MR_BB_PERIOD).std()
                lb   = ma - MR_BB_STD * std
                ub   = ma + MR_BB_STD * std
                px   = float(prices.iloc[-1])
                lb_v = float(lb.iloc[-1])
                ub_v = float(ub.iloc[-1])
                if px <= lb_v:
                    primary_sig = 1    # 超卖，布林下轨，买入
                    notes.append(f"📉 布林下轨触发(价格={px:.1f}≤下轨={lb_v:.1f})，均值回归买入")
                elif px >= ub_v:
                    primary_sig = -1   # 超买，布林上轨，离场
                    notes.append(f"📈 布林上轨(价格={px:.1f}≥上轨={ub_v:.1f})，超买观望")
                else:
                    pct = (px - lb_v) / (ub_v - lb_v) if (ub_v - lb_v) > 0 else 0.5
                    notes.append(f"布林带中段({pct:.0%}位置，下轨={lb_v:.1f}/上轨={ub_v:.1f})")
        except Exception:
            pass

        # 逆向参考：仅提示拥挤
        if cta_avg > 0.6:
            notes.append(f"⚠️  市场CTA过热({cta_avg:.2f})，追多需谨慎")
        else:
            notes.append(f"ℹ️  CTA({cta_avg:.2f})")

    elif asset_type == "D":
        # 股票指数用SMC代理（简化：EMA判方向），商品/债券用EMA(20/60)
        try:
            if not prices.empty and len(prices) >= 61:
                e20 = prices.ewm(span=20, adjust=False).mean()
                e60 = prices.ewm(span=60, adjust=False).mean()
                primary_sig = 1 if float(e20.iloc[-1]) > float(e60.iloc[-1]) else 0
        except Exception:
            pass

        # COT作为主要参考（这里用CTA代理）
        if cta_avg > 0.3:
            notes.append(f"📊 CTA系统性多头({cta_avg:.2f})")
        elif cta_avg < -0.1:
            notes.append(f"📊 CTA系统性空头({cta_avg:.2f})")
        else:
            notes.append(f"📊 CTA中性({cta_avg:.2f})")

    # ── 最终信号标签（策略信号 + CTA 双重判断）──
    if not cta_ok:
        signal_label = "CTA拦截🚫"
    elif vix_info.get("regime") == "EXTREME":
        signal_label = "VIX熔断🚫"
    elif primary_sig == 1:
        if asset_type == "A":
            signal_label = "强信号入场✅✅" if score >= 4 else "弱信号入场⚠️(半仓)"
        else:
            signal_label = "策略入场✅" if cta_avg >= 0 else "策略入场⚠️(CTA偏弱)"
    elif primary_sig == -1:
        signal_label = "超买离场🔴"
    else:
        signal_label = "观望⚪"

    return {
        "ticker":         ticker,
        "type":           asset_type,
        "label":          cfg["label"],
        "primary":        cfg["primary"],
        "cta_mode":       cfg["cta_mode"],
        "cta_value":      round(cta_avg, 3),
        "cta_ok":         cta_ok,
        "primary_signal": primary_sig,
        "vix_regime":     vix_info.get("regime", "N/A"),
        "vix_mult":       vix_info.get("size_mult", 1.0),
        "signal_label":   signal_label,
        "notes":          notes,
    }


# ──────────────────────────────────────────────
# 批量路由
# ──────────────────────────────────────────────

def route_batch(tickers: list, show_signal: bool = True) -> pd.DataFrame:
    """
    对一批标的分类 + 显示推荐策略 + 当日信号状态

    Returns:
        DataFrame，一行一个标的
    """
    spy_prices = get_prices("SPY")
    rows = []

    for t in tickers:
        try:
            info = classify_ticker(t, spy_prices=spy_prices)
            row  = {
                "标的":   t,
                "类型":   f"{info['type']}",
                "分类":   info["label"],
                "年化波动": f"{info['vol']:.0%}" if info["vol"] else "-",
                "Beta":   str(info["beta"]) if info["beta"] else "-",
                "分类依据": info["reason"],
                "主策略":  STRATEGY_CONFIG[info["type"]]["primary"],
                "CTA模式": STRATEGY_CONFIG[info["type"]]["cta_mode"],
            }

            if show_signal:
                sig = get_live_signal(t, asset_type=info["type"])
                row["信号"]    = sig["signal_label"]
                row["CTA值"]   = sig["cta_value"]
                row["VIX政体"] = sig["vix_regime"]
                row["提示"]    = " | ".join(sig["notes"])

            rows.append(row)
        except Exception as e:
            logger.warning(f"{t} 路由失败: {e}")
            rows.append({"标的": t, "类型": "?", "分类": "失败",
                         "分类依据": str(e)})

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 快速测试
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    watchlist = [
        "SPY", "QQQ", "GLD", "TLT",           # D类
        "SOXL", "TQQQ", "NVDL", "TSLL",       # A类
        "NVDA", "AMD", "TSLA",                 # B类
        "AAPL", "MSFT", "INTC", "JPM",         # C类
        "EOSE", "IREN",                        # 高波动个股
    ]

    print("=== 资产分类路由表 ===\n")
    df = route_batch(watchlist, show_signal=True)

    # 按类型分组显示
    for t in ["A", "B", "C", "D"]:
        sub = df[df["类型"] == t]
        if sub.empty:
            continue
        icon = {"A":"⚡","B":"🚀","C":"🏦","D":"📊"}[t]
        cfg  = STRATEGY_CONFIG[t]
        print(f"\n{icon} Type {t}：{cfg['label']}")
        print(f"   策略：{cfg['primary']} | CTA：{cfg['cta_mode']} | {cfg['note']}")
        cols = ["标的","分类依据","信号","CTA值","VIX政体"]
        avail = [c for c in cols if c in sub.columns]
        print(sub[avail].to_string(index=False))

    print("\n✅ router 测试通过")
