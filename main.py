"""
main.py — 量化系统主流程（v2）

运行模式：
  monitor  — 每日盘前：CTA仪表盘 + 板块热力图 + 资产路由信号
  router   — 仅显示资产分类路由表（快速查看各标的推荐策略）
  backtest — 对指定标的跑所有策略回测
  select   — 月末因子选股
  all      — 全量运行

用法示例：
  python main.py                              # 默认 monitor 模式
  python main.py --mode router               # 仅路由表
  python main.py --mode backtest --ticker SOXL
  python main.py --mode all --tickers NVDL TSLL SOXL NVDA AAPL SPY
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 默认观察列表 ──────────────────────────────────────────────────────────────
DEFAULT_WATCHLIST = [
    # D 类宽基
    "SPY", "QQQ", "GLD",
    # A 类杠杆/高波动
    "SOXL", "NVDL", "TSLL",
    # B 类成长
    "NVDA", "TSLA",
    # C 类蓝筹
    "AAPL", "MSFT", "INTC",
]


# ══════════════════════════════════════════════
# 1. 每日 CTA + 板块监控
# ══════════════════════════════════════════════

def run_daily_monitor():
    """每日盘前运行：VIX政体 + CTA仪表盘 + 板块热力图"""
    print("\n" + "="*65)
    print(f"  📊 量化系统日报  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*65)

    # ── VIX 政体（第一道防线）──
    print("\n【VIX 波动率政体】")
    try:
        from signals.breakout import current_vix_regime
        v = current_vix_regime()
        print(f"  {v['note']}")
    except Exception as e:
        logger.error(f"VIX状态失败: {e}")

    # ── CTA 仪表盘 ──
    print("\n【CTA 大盘信号（多周期动量）】")
    try:
        from signals.cta_monitor import run_cta_dashboard
        df_cta = run_cta_dashboard()
        print(df_cta[["资产","ETF","信号","方向","强度","仓位变化"]].to_string(index=False))
    except Exception as e:
        logger.error(f"CTA监控失败: {e}")

    # ── 板块热力图 ──
    print("\n【板块热力图】")
    try:
        from signals.sector_flows import sector_heatmap
        df_heat = sector_heatmap()
        print(df_heat[["板块","ETF","综合评分","趋势信号",
                        "RSI","3月超额","整体判断"]].to_string(index=False))
    except Exception as e:
        logger.error(f"板块监控失败: {e}")

    # ── COT（周五更新）──
    if datetime.now().weekday() == 4:
        print("\n【COT 期货仓位（周五更新）】")
        try:
            from data.cot_loader import get_cot_signals
            df_cot = get_cot_signals()
            if not df_cot.empty:
                print(df_cot[["净多头","百分位","Z-score","方向","警告"]].to_string())
        except Exception as e:
            logger.error(f"COT加载失败: {e}")


# ══════════════════════════════════════════════
# 2. 资产路由 + 策略推荐仪表盘（核心新功能）
# ══════════════════════════════════════════════

def run_router_dashboard(tickers: list = None):
    """
    对观察列表中每个标的：
      1. 自动分类（A/B/C/D）
      2. 显示推荐策略和 CTA 使用方式
      3. 输出当日信号状态
    """
    if tickers is None:
        tickers = DEFAULT_WATCHLIST

    print("\n" + "─"*65)
    print("【资产分类 × 策略路由 × 今日信号】")
    print("─"*65)

    try:
        from strategies.router import route_batch, STRATEGY_CONFIG

        df = route_batch(tickers, show_signal=True)

        TYPE_ICON = {"A": "⚡", "B": "🚀", "C": "🏦", "D": "📊"}
        TYPE_ORDER = ["A", "B", "C", "D"]

        for t_code in TYPE_ORDER:
            sub = df[df["类型"] == t_code]
            if sub.empty:
                continue
            cfg  = STRATEGY_CONFIG[t_code]
            icon = TYPE_ICON.get(t_code, "")
            print(f"\n{icon} Type {t_code}  {cfg['label']}")
            print(f"   策略：{cfg['primary'][:50]}")
            print(f"   CTA ：{cfg['cta_mode']} | {cfg['note']}")

            show_cols = ["标的", "分类依据", "信号", "CTA值", "VIX政体", "提示"]
            avail = [c for c in show_cols if c in sub.columns]
            print(sub[avail].to_string(index=False))

    except Exception as e:
        logger.error(f"路由仪表盘失败: {e}", exc_info=True)


# ══════════════════════════════════════════════
# 3. 策略回测对比
# ══════════════════════════════════════════════

def run_backtest_comparison(ticker: str = "SPY", start: str = "2021-01-01"):
    """运行所有策略回测，输出对比表（含新 Donchian 突破策略）"""
    print(f"\n【策略回测对比 — {ticker}（{start}起）】")
    try:
        from backtest.engine import run_all_strategies
        from signals.breakout import breakout_signal_series
        from data.downloader import get_prices
        from backtest.engine import backtest
        from backtest.metrics import compare_strategies

        # 先跑所有内置策略
        df = run_all_strategies(ticker, start=start)

        # 追加 Donchian 突破
        prices = get_prices(ticker, start=start)
        if not prices.empty:
            for period in [20, 55]:
                try:
                    sig = breakout_signal_series(ticker, period=period,
                                                 vix_filter=True,
                                                 long_only=True, start=start)
                    if not sig.empty:
                        sig = sig.clip(lower=0).reindex(prices.index).fillna(0)
                        res = backtest(prices, sig, name=f"Donchian{period}")
                        if df is not None and not df.empty:
                            # 将新行追加进结果 DataFrame
                            m = res["metrics"]
                            new_row = {c: m.get(c, None) for c in df.columns}
                            df.loc[f"Donchian突破({period}日)"] = new_row
                except Exception as e:
                    logger.warning(f"Donchian{period}失败: {e}")

        if df is not None and not df.empty:
            display_cols = ["年化收益(CAGR)", "Sharpe比率",
                            "最大回撤", "Calmar比率", "日胜率"]
            avail = [c for c in display_cols if c in df.columns]
            print(df[avail].to_string())
            print(f"\n🏆 最优Sharpe: {df['Sharpe比率'].idxmax()}")
            print(f"🛡️  最优Calmar: {df['Calmar比率'].idxmax()}")
    except Exception as e:
        logger.error(f"回测失败: {e}", exc_info=True)


# ══════════════════════════════════════════════
# 4. 因子选股（月末）
# ══════════════════════════════════════════════

def run_stock_selection():
    """月度选股（月末运行）"""
    from calendar import monthrange
    today = datetime.now()
    is_month_end = today.day >= monthrange(today.year, today.month)[1] - 2
    if not is_month_end:
        logger.info("非月末，跳过选股")
        return

    print("\n【月度因子选股 Top 20】")
    try:
        from signals.factors import monthly_stock_selection
        top = monthly_stock_selection(top_n=20)
        if not top.empty:
            cols = [c for c in ["momentum_12_1","low_vol",
                                "trend_quality","composite_score"]
                    if c in top.columns]
            print(top[cols].round(3).to_string())
    except Exception as e:
        logger.error(f"选股失败: {e}")


# ══════════════════════════════════════════════
# 5. Dual Momentum
# ══════════════════════════════════════════════

def run_dual_momentum():
    """查看 Dual Momentum 当前持仓建议"""
    print("\n【Dual Momentum 当前持仓建议】")
    try:
        from signals.trend_following import dual_momentum_signal
        dm = dual_momentum_signal()
        if not dm.empty:
            current = dm.iloc[-1]
            print(f"  当前持仓: {current['holding']}")
            print(f"  原因:     {current['reason']}")
    except Exception as e:
        logger.error(f"Dual Momentum失败: {e}")


# ══════════════════════════════════════════════
# 6. 风险检查
# ══════════════════════════════════════════════

def check_risk_status(portfolio_value: float = None):
    """检查当前风险状态"""
    print("\n【风险状态】")
    if portfolio_value:
        from portfolio.risk_manager import DrawdownGuard
        guard = DrawdownGuard()
        m = guard.check(portfolio_value)
        if m < 1.0:
            print(f"  ⚠️ 回撤熔断激活，仓位乘数: {m}")
        else:
            print(f"  ✅ 风控正常，仓位乘数: 1.0")
    else:
        # 显示当前 VIX 作为风险参考
        try:
            from signals.breakout import current_vix_regime
            v = current_vix_regime()
            print(f"  {v['note']}")
            print(f"  仓位乘数建议: ×{v['size_mult']}")
        except Exception:
            print("  VIX 数据不可用")


# ══════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="量化交易系统 v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py                             # 每日监控（默认）
  python main.py --mode router               # 查看资产分类路由
  python main.py --mode backtest --ticker SOXL --start 2021-01-01
  python main.py --mode all --tickers NVDL TSLL SOXL NVDA AAPL SPY
        """
    )
    parser.add_argument(
        "--mode",
        choices=["monitor", "router", "backtest", "select", "all"],
        default="monitor",
        help="运行模式"
    )
    parser.add_argument("--ticker",  default="SPY",        help="回测标的（backtest模式）")
    parser.add_argument("--start",   default="2021-01-01", help="回测起始日期")
    parser.add_argument(
        "--tickers", nargs="+",
        default=DEFAULT_WATCHLIST,
        help="观察列表（router/all模式）"
    )
    args = parser.parse_args()

    if args.mode == "monitor":
        run_daily_monitor()
        run_dual_momentum()
        run_router_dashboard(args.tickers)
        check_risk_status()

    elif args.mode == "router":
        run_router_dashboard(args.tickers)

    elif args.mode == "backtest":
        # 先显示该标的的分类
        try:
            from strategies.router import classify_ticker, STRATEGY_CONFIG
            info = classify_ticker(args.ticker)
            cfg  = STRATEGY_CONFIG[info["type"]]
            icon = {"A":"⚡","B":"🚀","C":"🏦","D":"📊"}.get(info["type"],"")
            print(f"\n{icon} {args.ticker} → Type {info['type']} ({info['label']})")
            print(f"   推荐策略: {cfg['primary']}")
            print(f"   CTA模式:  {cfg['cta_mode']}")
            print(f"   分类依据: {info['reason']}")
        except Exception:
            pass
        run_backtest_comparison(args.ticker, start=args.start)

    elif args.mode == "select":
        run_stock_selection()

    elif args.mode == "all":
        run_daily_monitor()
        run_dual_momentum()
        run_router_dashboard(args.tickers)
        run_backtest_comparison(args.ticker, start=args.start)
        run_stock_selection()
        check_risk_status()

    print("\n✅ 运行完毕")


if __name__ == "__main__":
    main()
