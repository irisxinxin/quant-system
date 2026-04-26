"""
dashboard/orb_routes.py — 短线 ORB 策略 dashboard 的 API 路由

供 app.py 集成. 用法:
  from dashboard.orb_routes import register_orb_routes
  register_orb_routes(app)

提供端点:
  GET /api/orb/manifest        # 标的列表 + 元信息
  GET /api/orb/symbol/{sym}    # 单只股票详情 (策略性能 + 最近交易)
  GET /api/orb/today           # 当日实时信号状态 (所有标的)
  GET /api/orb/signals_log     # 历史信号日志 (paper/live)
  GET /charts_orb/{sym}.png    # K 线图 (静态)
"""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.orb_strategy import (
    load_5m_data, backtest_signals_history, summary_stats,
    list_available_symbols, generate_signal, compute_or_for_today,
    TICKER_CONFIG, CORE_PORTFOLIO, ORB_PARAMS,
)


SIGNALS_LOG = Path(__file__).parent.parent / "output" / "orb_signals.jsonl"


def register_orb_routes(app: FastAPI):
    """把所有 ORB 相关路由注册到 FastAPI app"""

    @app.get("/api/orb/manifest")
    def orb_manifest():
        """标的列表 + 每只的回测概要"""
        items = []
        for sym in list_available_symbols():
            try:
                df = load_5m_data(sym)
                trades = backtest_signals_history(sym)
                stats = summary_stats(trades)
                cfg = TICKER_CONFIG.get(sym, {})
                items.append({
                    "symbol": sym,
                    "category": cfg.get("category", ""),
                    "score": cfg.get("score", 3),
                    "is_core": sym in CORE_PORTFOLIO,
                    "suitable_intraday": cfg.get("suitable_intraday", False),
                    "pair": cfg.get("pair", None),
                    "bars": len(df),
                    "first_date": df.index[0].strftime("%Y-%m-%d") if len(df) else None,
                    "last_date": df.index[-1].strftime("%Y-%m-%d") if len(df) else None,
                    "trades": stats.get("trades", 0),
                    "positive_pct": stats.get("positive_pct", 0),
                    "win_rate_strict": stats.get("win_rate_strict", 0),
                    "cumulative_return_pct": stats.get("cumulative_return_pct", 0),
                    "profit_factor": stats.get("profit_factor", 0),
                    "avg_win_pct": stats.get("avg_win_pct", 0),
                    "avg_loss_pct": stats.get("avg_loss_pct", 0),
                })
            except Exception as e:
                items.append({"symbol": sym, "error": str(e)})

        return {
            "params": ORB_PARAMS,
            "core_portfolio": CORE_PORTFOLIO,
            "symbols": items,
            "generated_at": datetime.now().isoformat(),
        }

    @app.get("/api/orb/symbol/{sym}")
    def orb_symbol(sym: str):
        """单只股票详情"""
        if not sym.endswith(".US"):
            sym = sym + ".US"
        df = load_5m_data(sym)
        if df.empty:
            raise HTTPException(404, f"无数据: {sym}")

        trades = backtest_signals_history(sym)
        stats = summary_stats(trades)
        cfg = TICKER_CONFIG.get(sym, {})

        # 最近 30 笔交易
        recent_trades = []
        if not trades.empty:
            for _, t in trades.tail(30).iterrows():
                recent_trades.append({
                    "day": t["day"],
                    "entry_time": t["entry_time"].strftime("%H:%M"),
                    "entry": t["entry_price"],
                    "stop": t["stop"],
                    "tp": t["tp"],
                    "exit": t["exit_price"],
                    "result": t["result"],
                    "pnl_pct": t["pnl_pct"],
                    "or_range_pct": t["or_range_pct"],
                })

        # 月度收益
        monthly = []
        if not trades.empty:
            trades_copy = trades.copy()
            trades_copy["entry_time"] = pd.to_datetime(trades_copy["entry_time"])
            trades_copy["month"] = trades_copy["entry_time"].dt.to_period("M").astype(str)
            for month, g in trades_copy.groupby("month"):
                month_pnl = ((1 + g["pnl_pct"] / 100).prod() - 1) * 100
                monthly.append({
                    "month": month,
                    "trades": len(g),
                    "pnl_pct": round(month_pnl, 2),
                })

        return {
            "symbol": sym,
            "category": cfg.get("category", ""),
            "score": cfg.get("score", 3),
            "is_core": sym in CORE_PORTFOLIO,
            "stats": stats,
            "recent_trades": recent_trades,
            "monthly": monthly,
            "config": {
                "entry_slip": cfg.get("entry_slip", 0),
                "stop_slip": cfg.get("stop_slip", 0),
            },
        }

    @app.get("/api/orb/today")
    def orb_today():
        """所有核心标的的当日实时状态"""
        et = ZoneInfo("US/Eastern")
        today = datetime.now(et).date()
        items = []
        for sym in CORE_PORTFOLIO:
            try:
                df = load_5m_data(sym)
                or_state = compute_or_for_today(df, today)
                signal = generate_signal(sym, today)
                items.append({
                    "symbol": sym,
                    "or_state": or_state or {"status": "NO_DATA"},
                    "signal": signal,
                })
            except Exception as e:
                items.append({"symbol": sym, "error": str(e)})

        return {
            "today": str(today),
            "now_et": datetime.now(et).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "items": items,
        }

    @app.get("/api/orb/signals_log")
    def orb_signals_log():
        """历史信号日志 (来自 signal_live_longport.py 写入)"""
        if not SIGNALS_LOG.exists():
            return {"signals": []}
        signals = []
        with open(SIGNALS_LOG) as f:
            for line in f:
                try:
                    signals.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # 只返回最近 200 条
        return {"signals": signals[-200:]}

    @app.get("/api/orb/chart_data/{sym}")
    def orb_chart_data(sym: str, days: int = 60):
        """
        给 Lightweight Charts 用的数据端点
        返回: { candles: [...], volumes: [...], markers: [...], or_lines: [...] }
        """
        if not sym.endswith(".US"):
            sym = sym + ".US"

        df = load_5m_data(sym)
        if df.empty:
            raise HTTPException(404, f"无数据: {sym}")

        # 取最近 N 个交易日
        unique_days = sorted(set(df.index.date))
        if len(unique_days) > days:
            cutoff = unique_days[-days]
            df = df[df.index.date >= cutoff]

        # OHLCV
        candles = []
        volumes = []
        for ts, row in df.iterrows():
            ts_unix = int(ts.timestamp())
            candles.append({
                "time": ts_unix,
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
            })
            color = "#26a69a" if row["Close"] >= row["Open"] else "#ef5350"
            volumes.append({
                "time": ts_unix,
                "value": int(row["Volume"]),
                "color": color,
            })

        # 历史信号 markers
        trades = backtest_signals_history(sym)
        if not trades.empty:
            trades["entry_time"] = pd.to_datetime(trades["entry_time"])
            trades["exit_time"] = pd.to_datetime(trades["exit_time"])
            window_trades = trades[trades["entry_time"] >= df.index[0]]
        else:
            window_trades = pd.DataFrame()

        markers = []
        for _, t in window_trades.iterrows():
            # 入场点
            markers.append({
                "time": int(t["entry_time"].timestamp()),
                "position": "belowBar",
                "color": "#10b981",
                "shape": "arrowUp",
                "text": f"BUY ${t['entry_price']:.2f}",
            })
            # 出场点 (按结果区分)
            exit_time = int(t["exit_time"].timestamp())
            if t["result"] == "TP":
                markers.append({
                    "time": exit_time, "position": "aboveBar",
                    "color": "#10b981", "shape": "arrowDown",
                    "text": f"TP +{t['pnl_pct']:.1f}%",
                })
            elif t["result"] == "STOP":
                markers.append({
                    "time": exit_time, "position": "aboveBar",
                    "color": "#ef4444", "shape": "circle",
                    "text": f"STOP {t['pnl_pct']:.1f}%",
                })
            else:  # EOD
                markers.append({
                    "time": exit_time, "position": "aboveBar",
                    "color": "#9ca3af", "shape": "square",
                    "text": f"EOD {t['pnl_pct']:.1f}%",
                })

        # 每日 OR 上下沿 (作为 price line 用线段表示)
        or_lines = []
        p_or_bars = 3
        for day, day_df in df.groupby(df.index.date):
            if len(day_df) < p_or_bars:
                continue
            first_n = day_df.head(p_or_bars)
            or_h = float(first_n["High"].max())
            or_l = float(first_n["Low"].min())
            day_start = int(day_df.index[0].timestamp())
            day_end = int(day_df.index[-1].timestamp())
            or_lines.append({
                "day_start": day_start,
                "day_end": day_end,
                "or_end": int(day_df.index[p_or_bars - 1].timestamp()),
                "or_high": round(or_h, 4),
                "or_low": round(or_l, 4),
            })

        # 汇总
        stats = summary_stats(window_trades) if not window_trades.empty else {}
        cfg = TICKER_CONFIG.get(sym, {})

        return {
            "symbol": sym,
            "category": cfg.get("category", ""),
            "is_core": sym in CORE_PORTFOLIO,
            "candles": candles,
            "volumes": volumes,
            "markers": markers,
            "or_lines": or_lines,
            "stats": stats,
            "trades_count": len(window_trades),
        }


# 兜底 import
try:
    import pandas as pd
except ImportError:
    pass
