"""
live_executor.py — LongPort 模拟盘自动下单 (限价 LIT 入场版)

新流程:
  1. OR 形成 (9:45 ET) → 立刻挂 LIT BUY 单 @ OR_high*(1+slip)
  2. 价格触到限价 → 自动成交在 OR_high 附近 (无 5m close 后市价单的 1.3% 滑点)
  3. 成交后心跳轮询发现成交 → 检查 RVOL (lenient, 跨天 SMA 20)
       - RVOL ≥ 1.5: 挂止损 + 止盈
       - RVOL < 1.5: 立刻市价平仓 (避免假突破)
  4. 心跳调和 OCO (止损/止盈任一成交则撤另一)
  5. 15:50 ET 强平: 撤所有未成交单 + 市价平所有持仓

环境变量:
  LIVE_TRADING=true   才会真下单, 否则只 dry-run 打印
  LONGPORT_*          凭证 (与 QuoteContext 共用)

入口函数:
  init()                       启动时初始化
  place_or_entry(...)          OR 形成时调用 → 挂 LIT 限价单
  check_fills_and_arm_brackets(quote_ctx)  心跳调用 → 检查成交 + arm 子单
  reconcile_oco()              心跳调用 → OCO 仿真
  force_close_all()            收盘前 15:50 调用
"""
import os
import json
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

# ═══ 全局状态 ═══
_TRADE_CTX = None
_LIVE: bool = False
_ORDER_FILE: Path = Path(__file__).parent / "orders_today.jsonl"

# 待成交 LIT 入场单 (OR 形成后已挂, 等触发)
# {symbol: {"date", "entry_id", "or_high", "or_low", "or_range", "qty", "filled": False, "armed": False}}
_PENDING_ENTRIES: dict = {}

# 已成交 + 已 arm 子单的持仓 (止损/止盈在挂)
# {symbol: {"date", "entry_id", "buy_fill_price", "stop_id", "tp_id", "or_high", "or_low", "or_range", "qty", "closed": False}}
_OPEN_POSITIONS: dict = {}

RVOL_LOOKBACK = 20
RVOL_THRESHOLD = 1.5
TARGET_R = 2.0
POSITION_USD = 10000         # 每个信号目标资金 ~$10000 USD, 按入场价动态算股数 (不融资)


def _today_str() -> str:
    return datetime.now(ET).date().isoformat()


def _round_tick(price) -> Decimal:
    """美股 tick = 0.01, round 到分"""
    return Decimal(f"{round(float(price) + 1e-9, 2):.2f}")


def _load_today_state():
    """脚本重启时, 重新载入今天已下过的单"""
    if not _ORDER_FILE.exists():
        return
    today = _today_str()
    try:
        with open(_ORDER_FILE) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("date") != today or not rec.get("symbol"):
                    continue
                sym = rec["symbol"]
                kind = rec.get("kind", "open_position")
                if kind == "pending_entry":
                    _PENDING_ENTRIES[sym] = rec
                else:
                    _OPEN_POSITIONS[sym] = rec
    except Exception as e:
        print(f"   ⚠️ 载入 orders_today.jsonl 失败: {e}")


def _persist(sym: str, kind: str, data: dict):
    rec = dict(data); rec["symbol"] = sym; rec["kind"] = kind
    rec["log_time"] = datetime.now().isoformat()
    with open(_ORDER_FILE, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def init():
    global _TRADE_CTX, _LIVE
    _LIVE = os.environ.get("LIVE_TRADING", "").lower() in ("1", "true", "yes")
    if _LIVE:
        try:
            from longport.openapi import Config, TradeContext
            cfg = Config.from_env()
            _TRADE_CTX = TradeContext(cfg)
            print("   ⚡ LIVE_TRADING=true → 模拟盘自动下单已启用 (限价 LIT 入场)")
        except Exception as e:
            print(f"   ❌ TradeContext 初始化失败: {e}")
            _LIVE = False
    else:
        print("   🛡️ LIVE_TRADING 未启用 (信号触发只 dry-run, 不真下单)")
    _load_today_state()
    if _PENDING_ENTRIES or _OPEN_POSITIONS:
        print(f"   📂 载入 今日待成交 {len(_PENDING_ENTRIES)} 个 / 已成交 {len(_OPEN_POSITIONS)} 个")


def place_or_entry(symbol: str, or_high: float, or_low: float, or_range_pct: float,
                   entry_slip: float = 0.001, is_replay: bool = False) -> dict:
    """
    OR 形成时调用 → 挂 LIT BUY 限价单 @ OR_high * (1+slip).
    is_replay=True 时跳过 (历史回放不下单).
    """
    today = _today_str()
    if symbol in _OPEN_POSITIONS and _OPEN_POSITIONS[symbol].get("date") == today:
        return {"ok": False, "reason": "今日已成交 (幂等跳过)"}
    if symbol in _PENDING_ENTRIES and _PENDING_ENTRIES[symbol].get("date") == today:
        return {"ok": False, "reason": "今日已挂单 (幂等跳过)"}
    if is_replay:
        return {"ok": False, "reason": "历史回放 (dry-run)"}
    if not _LIVE or _TRADE_CTX is None:
        return {"ok": False, "reason": "LIVE_TRADING=false (dry-run)"}

    target_px = _round_tick(or_high * (1 + entry_slip))
    # 按目标资金动态算股数, 至少 1 股
    shares = max(1, int(POSITION_USD / float(target_px)))
    qty = Decimal(str(shares))
    or_range = or_high - or_low

    try:
        from longport.openapi import OrderType, OrderSide, TimeInForceType, TriggerStatus
    except ImportError as e:
        return {"ok": False, "reason": f"import error: {e}"}

    # LIT BUY: 价格触到 trigger_price → 在 submitted_price 挂限价
    try:
        resp = _TRADE_CTX.submit_order(
            symbol=symbol,
            order_type=OrderType.LIT,
            side=OrderSide.Buy,
            submitted_quantity=qty,
            submitted_price=target_px,
            trigger_price=target_px,
            time_in_force=TimeInForceType.Day,
            remark="ORB-LIT-Entry",
        )
        rec = {
            "date": today, "entry_id": resp.order_id,
            "or_high": or_high, "or_low": or_low, "or_range": or_range,
            "or_range_pct": or_range_pct, "target_px": str(target_px),
            "qty": str(qty), "filled": False, "armed": False,
        }
        _PENDING_ENTRIES[symbol] = rec
        _persist(symbol, "pending_entry", rec)
        return {"ok": True, "entry_id": resp.order_id, "target_px": str(target_px)}
    except Exception as e:
        return {"ok": False, "reason": f"挂 LIT 入场单失败: {e}"}


def _compute_lenient_rvol(quote_ctx, symbol: str) -> float:
    """跨天 SMA 20 计算最近一根 K 线的 RVOL (lenient)"""
    try:
        from longport.openapi import Period, AdjustType
        bars = list(quote_ctx.candlesticks(symbol, Period.Min_5, RVOL_LOOKBACK + 1, AdjustType.NoAdjust))
        if len(bars) < 6: return 0.0
        avg_v = sum(int(b.volume) for b in bars) / len(bars)
        return float(bars[-1].volume) / max(avg_v, 1)
    except Exception as e:
        print(f"   ⚠️ {symbol} RVOL 计算失败: {e}")
        return 0.0


def check_fills_and_arm_brackets(quote_ctx):
    """心跳调用: 检查 LIT 入场单成交 → 检查 RVOL → arm 止损/止盈 (或市价平仓)"""
    if not _LIVE or _TRADE_CTX is None:
        return
    try:
        from longport.openapi import OrderType, OrderSide, TimeInForceType, OrderStatus
    except ImportError:
        return

    for sym, rec in list(_PENDING_ENTRIES.items()):
        if rec.get("filled") or rec.get("armed"):
            continue
        entry_id = rec.get("entry_id")
        if not entry_id:
            continue
        # 查订单状态
        try:
            d = _TRADE_CTX.order_detail(entry_id)
        except Exception as e:
            print(f"   ⚠️ {sym} 查 LIT 单状态失败: {e}")
            continue
        if d.status not in (OrderStatus.Filled, OrderStatus.PartialFilled):
            continue   # 还没成交, 等下一轮

        fill_price = float(d.executed_price) if d.executed_price else float(rec["target_px"])
        rec["filled"] = True
        rec["buy_fill_price"] = fill_price
        print(f"   ✅ {sym} LIT 单成交 @ ${fill_price:.2f}")

        # 检查 RVOL (lenient, 跨天 SMA 20)
        rvol = _compute_lenient_rvol(quote_ctx, sym)
        print(f"      RVOL = {rvol:.2f}")

        qty = Decimal(rec.get("qty", "1"))
        or_high = rec["or_high"]; or_low = rec["or_low"]; or_range = rec["or_range"]

        if rvol < RVOL_THRESHOLD:
            # RVOL 不够 → 立刻市价平仓 (避免假突破吃亏)
            print(f"   ⚠️ {sym} RVOL {rvol:.2f} < {RVOL_THRESHOLD}, 立刻市价平仓")
            try:
                close_resp = _TRADE_CTX.submit_order(
                    symbol=sym, order_type=OrderType.MO, side=OrderSide.Sell,
                    submitted_quantity=qty, time_in_force=TimeInForceType.Day,
                    remark="ORB-RVOL-Reject")
                rec["closed"] = True; rec["close_reason"] = "rvol_reject"
                rec["close_id"] = close_resp.order_id
            except Exception as e:
                print(f"      ❌ 市价平仓失败: {e}")
            _PENDING_ENTRIES.pop(sym, None)
            _persist(sym, "open_position", rec)
            continue

        # RVOL 够 → 挂止损 + 止盈
        stop_px = _round_tick(or_low)
        tp_px = _round_tick(or_high + TARGET_R * or_range)
        try:
            stop_resp = _TRADE_CTX.submit_order(
                symbol=sym, order_type=OrderType.MIT, side=OrderSide.Sell,
                submitted_quantity=qty, trigger_price=stop_px,
                time_in_force=TimeInForceType.Day, remark="ORB-Stop")
            rec["stop_id"] = stop_resp.order_id
            rec["stop_px"] = str(stop_px)
        except Exception as e:
            rec["stop_error"] = str(e); print(f"      ❌ 挂止损失败: {e}")

        try:
            tp_resp = _TRADE_CTX.submit_order(
                symbol=sym, order_type=OrderType.LO, side=OrderSide.Sell,
                submitted_quantity=qty, submitted_price=tp_px,
                time_in_force=TimeInForceType.Day, remark="ORB-TP")
            rec["tp_id"] = tp_resp.order_id
            rec["tp_px"] = str(tp_px)
        except Exception as e:
            rec["tp_error"] = str(e); print(f"      ❌ 挂止盈失败: {e}")

        rec["armed"] = True
        rec["closed"] = False
        _OPEN_POSITIONS[sym] = rec
        _PENDING_ENTRIES.pop(sym, None)
        _persist(sym, "open_position", rec)
        print(f"   🎯 {sym} 子单已挂: stop @ {stop_px}  tp @ {tp_px}")


def reconcile_oco():
    """OCO 仿真: 检查每个未平仓票的 stop / tp 状态, 任一成交则撤另一"""
    if not _LIVE or _TRADE_CTX is None:
        return
    try:
        from longport.openapi import OrderStatus
    except ImportError:
        return

    for sym, rec in list(_OPEN_POSITIONS.items()):
        if rec.get("closed"):
            continue
        stop_id = rec.get("stop_id")
        tp_id = rec.get("tp_id")
        stop_filled = tp_filled = False
        try:
            if stop_id:
                d = _TRADE_CTX.order_detail(stop_id)
                if d.status in (OrderStatus.Filled, OrderStatus.PartialFilled):
                    stop_filled = True
            if tp_id:
                d = _TRADE_CTX.order_detail(tp_id)
                if d.status in (OrderStatus.Filled, OrderStatus.PartialFilled):
                    tp_filled = True
        except Exception as e:
            print(f"   ⚠️ {sym} 查子单失败: {e}")
            continue

        if stop_filled and tp_id:
            try:
                _TRADE_CTX.cancel_order(tp_id)
                print(f"   🔁 {sym} 止损成交, 已撤止盈单")
            except Exception: pass
            rec["closed"] = True; rec["close_reason"] = "stop_filled"
            _persist(sym, "open_position", rec)
        elif tp_filled and stop_id:
            try:
                _TRADE_CTX.cancel_order(stop_id)
                print(f"   🔁 {sym} 止盈成交, 已撤止损单")
            except Exception: pass
            rec["closed"] = True; rec["close_reason"] = "tp_filled"
            _persist(sym, "open_position", rec)


def force_close_all():
    """收盘前 15:50 ET: 撤所有未成交单 + 市价平所有持仓"""
    if not _LIVE or _TRADE_CTX is None:
        return
    try:
        from longport.openapi import OrderType, OrderSide, TimeInForceType
    except ImportError:
        return

    print("\n🛑 收盘前强平: 撤所有未成交单 + 市价平仓")

    # 1) 撤所有未成交 LIT 入场单
    for sym, rec in list(_PENDING_ENTRIES.items()):
        eid = rec.get("entry_id")
        if eid and not rec.get("filled"):
            try:
                _TRADE_CTX.cancel_order(eid)
                print(f"   ↩️ 撤 LIT 入场单 {sym}")
            except Exception: pass

    # 2) 撤所有未成交子单
    for sym, rec in _OPEN_POSITIONS.items():
        if rec.get("closed"): continue
        for key in ("stop_id", "tp_id"):
            oid = rec.get(key)
            if oid:
                try:
                    _TRADE_CTX.cancel_order(oid)
                    print(f"   ↩️ 撤 {sym} {key}")
                except Exception: pass

    # 3) 市价平所有真持仓
    try:
        positions = _TRADE_CTX.stock_positions()
    except Exception as e:
        print(f"   ⚠️ 查持仓失败: {e}")
        return

    for ch in positions.channels:
        for p in ch.positions:
            qty = Decimal(str(p.quantity))
            if qty <= 0: continue
            try:
                _TRADE_CTX.submit_order(
                    symbol=p.symbol, order_type=OrderType.MO, side=OrderSide.Sell,
                    submitted_quantity=qty, time_in_force=TimeInForceType.Day,
                    remark="ORB-ForceClose")
                print(f"   ✅ 强平 {p.symbol} {qty} 股")
                rec = _OPEN_POSITIONS.get(p.symbol, {})
                rec["closed"] = True; rec["close_reason"] = "force_close"
                _OPEN_POSITIONS[p.symbol] = rec
                _persist(p.symbol, "open_position", rec)
            except Exception as e:
                print(f"   ❌ 强平 {p.symbol} 失败: {e}")
