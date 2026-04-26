"""
market_env_check.py — ORB 策略的市场环境识别器

每天开盘前（或开盘时）运行，自动判断：
  - 今天适合跑 ORB 吗？
  - 哪些标的建议开启，哪些建议观望？
  - 预期胜率和建议仓位？

依赖的市场指标（多因子打分）:
  1. VIX 水平与趋势   → 波动率环境
  2. SPY/QQQ 隔夜 Gap → 系统性事件冲击
  3. 板块强度          → 行业轮动
  4. 个股 RVOL        → 流动性与关注度
  5. 经济日历事件     → FOMC/CPI/NFP 等
  6. 月/季末          → 机构 rebalance 扭曲
  7. 假期邻近         → 低流动性日
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================
# 已知的 2026 关键经济事件日历（需要每年更新）
# ==============================================================
FOMC_DATES_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 16),
]
# CPI 通常为月中 (每月 10-13 日左右发布上月数据)
CPI_DATES_2026 = [
    date(2026, 1, 14), date(2026, 2, 11), date(2026, 3, 11),
    date(2026, 4, 14), date(2026, 5, 13), date(2026, 6, 11),
    date(2026, 7, 15), date(2026, 8, 12), date(2026, 9, 11),
    date(2026, 10, 14), date(2026, 11, 12), date(2026, 12, 10),
]
# NFP 每月第一个周五
def nfp_dates_for_year(year):
    dates = []
    for m in range(1, 13):
        d = date(year, m, 1)
        while d.weekday() != 4:  # 4 = Friday
            d += timedelta(days=1)
        dates.append(d)
    return dates
NFP_DATES_2026 = nfp_dates_for_year(2026)


# ==============================================================
# 因子检测函数
# ==============================================================
def get_vix_info() -> dict:
    """VIX 当前水平 + 5日变化"""
    vix = yf.download("^VIX", period="10d", progress=False, auto_adjust=True)
    if vix.empty:
        return {"level": None, "regime": "UNKNOWN", "change_5d": 0}
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    level = float(vix["Close"].iloc[-1])
    change_5d = float(vix["Close"].iloc[-1] / vix["Close"].iloc[-6] - 1) * 100 if len(vix) >= 6 else 0

    if level < 13:
        regime = "极低"
        orb_fit = "⚠️ 波动不足"
    elif level < 18:
        regime = "低"
        orb_fit = "⚠️ 偏不利"
    elif level < 25:
        regime = "正常"
        orb_fit = "✅ 适合"
    elif level < 32:
        regime = "高"
        orb_fit = "✅ 非常适合"
    else:
        regime = "极高"
        orb_fit = "⚠️ 过度恐慌，gap 风险大"
    return {"level": level, "regime": regime, "change_5d": change_5d, "orb_fit": orb_fit}


def get_spy_gap() -> dict:
    """SPY 隔夜 Gap 幅度"""
    spy = yf.download("SPY", period="5d", interval="1d", progress=False, auto_adjust=True)
    if spy.empty or len(spy) < 2:
        return {"gap_pct": 0, "status": "UNKNOWN"}
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    last_close = float(spy["Close"].iloc[-2])
    today_open = float(spy["Open"].iloc[-1])
    gap_pct = (today_open / last_close - 1) * 100

    abs_gap = abs(gap_pct)
    if abs_gap < 0.3:
        status = "正常"
        orb_fit = "✅ 安全"
    elif abs_gap < 0.8:
        status = "中等"
        orb_fit = "✅ 可做"
    elif abs_gap < 1.5:
        status = "较大"
        orb_fit = "⚠️ OR 可能偏极端"
    else:
        status = "极端"
        orb_fit = "❌ 建议观望"
    return {"gap_pct": gap_pct, "status": status, "orb_fit": orb_fit}


def get_sector_strength() -> dict:
    """关键板块 5 日强度"""
    sectors = {
        "AI/科技": "XLK",
        "半导体": "SMH",
        "加密": "IBIT",   # 比特币 ETF 代理
        "小盘": "IWM",
        "能源": "XLE",
    }
    result = {}
    for name, ticker in sectors.items():
        try:
            df = yf.download(ticker, period="10d", progress=False, auto_adjust=True)
            if df.empty or len(df) < 6:
                result[name] = {"5d_return": 0, "status": "UNKNOWN"}
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            ret_5d = float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100
            if ret_5d > 3:
                status = "🚀 强势"
            elif ret_5d > 1:
                status = "✅ 偏强"
            elif ret_5d > -1:
                status = "➡️ 震荡"
            elif ret_5d > -3:
                status = "🔻 偏弱"
            else:
                status = "💀 破位"
            result[name] = {"5d_return": ret_5d, "status": status}
        except Exception as e:
            result[name] = {"5d_return": 0, "status": f"Error: {e}"}
    return result


def get_ticker_rvol(ticker: str) -> dict:
    """个股的 RVOL（昨日成交量 / 20日均值）"""
    try:
        df = yf.download(ticker, period="30d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 21:
            return {"rvol": None, "avg_vol": None}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        last_vol = float(df["Volume"].iloc[-1])
        avg_vol = float(df["Volume"].iloc[-21:-1].mean())
        rvol = last_vol / avg_vol if avg_vol > 0 else 0
        # 同时返回近 5 日相对大盘的 alpha
        spy = yf.download("SPY", period="10d", progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        ticker_ret_5d = float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
        spy_ret_5d = float(spy["Close"].iloc[-1] / spy["Close"].iloc[-6] - 1) * 100 if len(spy) >= 6 else 0
        alpha_5d = ticker_ret_5d - spy_ret_5d
        return {
            "rvol": rvol,
            "ticker_ret_5d": ticker_ret_5d,
            "alpha_5d": alpha_5d,
            "avg_vol_m": avg_vol / 1e6,
        }
    except Exception as e:
        return {"rvol": None, "error": str(e)}


def check_calendar_events(check_date: date) -> list:
    """检查当日是否有重大经济事件"""
    events = []
    if check_date in FOMC_DATES_2026:
        events.append(("FOMC 利率决议", "❌ 强烈建议观望"))
    if check_date in CPI_DATES_2026:
        events.append(("CPI 数据发布", "⚠️ intraday 可能大幅反转"))
    if check_date in NFP_DATES_2026:
        events.append(("NFP 非农就业", "⚠️ 开盘 1 小时波动大"))

    # 月末/季末
    next_day = check_date + timedelta(days=1)
    if check_date.day >= 28 and next_day.month != check_date.month:
        events.append(("月末最后交易日", "⚠️ 机构 rebalance 扭曲"))
    if check_date.month in [3, 6, 9, 12] and check_date.day >= 28 and next_day.month != check_date.month:
        events.append(("季末", "⚠️ 算法订单大量涌出"))

    # 周末邻近
    if check_date.weekday() == 4:  # 周五
        events.append(("周五", "ℹ️ 下午 15:00 后流动性下降"))
    if check_date.weekday() == 0:  # 周一
        events.append(("周一", "ℹ️ 隔夜信息 2 天积累，gap 概率高"))

    return events


def score_ticker(ticker: str, rvol_info: dict, market_env: dict) -> dict:
    """
    综合打分：0-100
    - VIX 环境分（25分）
    - SPY Gap 分（20分）
    - 个股 RVOL 分（25分）
    - 板块强度分（15分）
    - 日历事件分（15分）
    """
    score = 0
    reasons = []

    # VIX
    vix = market_env["vix"]["level"]
    if vix and 15 <= vix <= 28:
        score += 25
        reasons.append("VIX 适宜")
    elif vix and 13 <= vix < 15:
        score += 12
        reasons.append("VIX 偏低")
    elif vix and 28 < vix <= 35:
        score += 18
        reasons.append("VIX 偏高但可控")
    else:
        reasons.append("VIX 不利")

    # SPY Gap
    gap = abs(market_env["spy_gap"]["gap_pct"])
    if gap < 0.5:
        score += 20
        reasons.append("Gap 小")
    elif gap < 1.0:
        score += 15
    elif gap < 1.5:
        score += 8
    else:
        reasons.append("Gap 过大，观望")

    # RVOL
    rvol = rvol_info.get("rvol")
    if rvol and rvol >= 1.5:
        score += 25
        reasons.append(f"RVOL={rvol:.2f}高")
    elif rvol and rvol >= 1.0:
        score += 18
    elif rvol and rvol >= 0.7:
        score += 10
    else:
        reasons.append("成交量不足")

    # 5 日 alpha
    alpha = rvol_info.get("alpha_5d", 0) or 0
    if abs(alpha) > 5:
        score += 15
        reasons.append(f"5日alpha {alpha:+.1f}%（有 momentum）")
    elif abs(alpha) > 2:
        score += 10

    # 日历事件扣分
    if market_env["events"]:
        has_critical = any("❌" in e[1] for e in market_env["events"])
        if has_critical:
            score -= 30
            reasons.append("⚠️ 重大日历事件")
        else:
            score += 15 - 5 * sum(1 for e in market_env["events"] if "⚠️" in e[1])

    score = max(0, min(100, score))
    if score >= 70:
        verdict = "🎯 强烈推荐"
    elif score >= 55:
        verdict = "✅ 推荐"
    elif score >= 40:
        verdict = "⚠️ 谨慎"
    else:
        verdict = "❌ 观望"

    return {"score": score, "verdict": verdict, "reasons": reasons}


# ==============================================================
# 主入口
# ==============================================================
def run_env_check(tickers: list = None, check_date: date = None):
    if check_date is None:
        check_date = date.today()
    if tickers is None:
        tickers = ["OKLO", "IREN", "HOOD", "TSLL", "PLTR", "NBIS", "CIFR", "CRWV"]

    print("=" * 80)
    weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][check_date.weekday()]
    print(f"🌐 ORB 策略市场环境识别 | {check_date} ({weekday_cn})")
    print("=" * 80)

    # 1. 市场宏观
    print("\n📊 【市场宏观指标】")
    vix_info = get_vix_info()
    if vix_info["level"]:
        print(f"  VIX 当前水平   : {vix_info['level']:.2f}  ({vix_info['regime']}) {vix_info['orb_fit']}")
        print(f"  VIX 5日变化    : {vix_info['change_5d']:+.1f}%")
    else:
        print(f"  VIX             : 数据无法获取")

    spy_info = get_spy_gap()
    print(f"  SPY 隔夜 Gap    : {spy_info['gap_pct']:+.2f}%  ({spy_info['status']}) {spy_info['orb_fit']}")

    # 2. 板块强度
    print("\n🎯 【板块5日强度】")
    sectors = get_sector_strength()
    for name, info in sectors.items():
        print(f"  {name:<8} {info['5d_return']:+6.2f}%  {info['status']}")

    # 3. 日历事件
    print(f"\n📅 【日历事件检查 - {check_date}】")
    events = check_calendar_events(check_date)
    if events:
        for name, status in events:
            print(f"  {name}: {status}")
    else:
        print(f"  ✅ 无重大日历事件")

    market_env = {
        "vix": vix_info,
        "spy_gap": spy_info,
        "sectors": sectors,
        "events": events,
    }

    # 4. 个股打分
    print(f"\n🎯 【个股 ORB 适配度评分】")
    print(f"  {'Ticker':<7} {'RVOL':>6} {'5日Alpha':>10} {'均量(M)':>10} {'分数':>6}  {'建议':<15} {'理由'}")
    print("  " + "-" * 100)
    ticker_scores = []
    for t in tickers:
        rvol_info = get_ticker_rvol(t)
        score_info = score_ticker(t, rvol_info, market_env)
        ticker_scores.append((t, rvol_info, score_info))

        rvol_str = f"{rvol_info.get('rvol', 0):.2f}" if rvol_info.get("rvol") else "N/A"
        alpha_str = f"{rvol_info.get('alpha_5d', 0):+.1f}%" if rvol_info.get("alpha_5d") is not None else "N/A"
        vol_str = f"{rvol_info.get('avg_vol_m', 0):.1f}" if rvol_info.get("avg_vol_m") else "N/A"
        reason_str = ", ".join(score_info["reasons"][:3])

        print(f"  {t:<7} {rvol_str:>6} {alpha_str:>10} {vol_str:>10} "
              f"{score_info['score']:>5}  {score_info['verdict']:<15} {reason_str}")

    # 5. 综合建议
    print(f"\n💡 【综合建议】")

    # 一票否决型事件
    has_critical_event = any("❌" in e[1] for e in events)
    if has_critical_event:
        print(f"\n  ❌ 今日有重大日历事件（FOMC/CPI），**建议全天观望**")
        return

    # 一票否决型宏观
    if vix_info["level"] and vix_info["level"] > 35:
        print(f"\n  ❌ VIX > 35，极度恐慌环境，gap 风险大，**建议观望**")
        return
    if abs(spy_info["gap_pct"]) > 2:
        print(f"\n  ❌ SPY 开盘 Gap > 2%，系统性冲击，**建议观望**")
        return

    # 按分数排序
    ticker_scores.sort(key=lambda x: x[2]["score"], reverse=True)

    top = [t for t, _, s in ticker_scores if s["score"] >= 70]
    ok = [t for t, _, s in ticker_scores if 55 <= s["score"] < 70]
    skip = [t for t, _, s in ticker_scores if s["score"] < 40]

    if top:
        print(f"\n  🎯 今日优先标的 (分数≥70): {', '.join(top)}")
    if ok:
        print(f"  ✅ 可开启 (分数55-69): {', '.join(ok)}")
    if skip:
        print(f"  ❌ 观望 (分数<40): {', '.join(skip)}")

    # 仓位建议
    eligible = top + ok
    if len(eligible) == 0:
        print(f"\n  ⚠️ 无合格标的，今日休息")
    else:
        per_stock = 100 / len(eligible) / 3  # 保守：每只最多占总资金 1/(3*N)
        print(f"\n  📊 仓位建议:")
        print(f"     合格标的 {len(eligible)} 只，单只建议仓位不超过总资金 {per_stock:.0f}%")
        print(f"     总暴露不超过 33%（其余 67% 保持现金应对意外）")
        print(f"     每只单笔最大亏损控制在总资金 1% 以内（2R 模型下单笔风险 ≈ 仓位 × 2%）")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_env_check()
