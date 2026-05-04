"""
live_executor.py — LongPort 模拟盘自动下单 (Plan-Driven, 多策略版)

设计:
  外部传入 TradePlan (来自 signals.strategies.intraday_pool 的策略函数)
  本模块负责:
    - 按 plan.order_type ('LMT' / 'MKT') 下单
    - fill 后自动挂止损 (MIT) + 止盈 (LO)
    - OCO 调和 (止损/止盈 任一成交则撤另一)
    - 收盘前 15:50 ET 强平所有

入口:
  init()                       启动时调用
  place_entry(symbol, plan, position_usd, is_replay)  策略产生 plan 时调用
  check_fills_and_arm_brackets(quote_ctx=None)  心跳调用
  reconcile_oco()              心跳调用
  force_close_all()             15:50 ET 调用

环境变量:
  LIVE_TRADING=true   真下单 (模拟盘); 否则 dry-run
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

# 待成交订单 (LMT 还没 fill, 或 MKT 还在等订单状态确认)
# {symbol: {date, entry_id, strategy, order_type, limit_price, stop_price, tp_price, qty, filled, armed}}
_PENDING_ENTRIES: dict = {}

# 已 arm 子单的持仓
_OPEN_POSITIONS: dict = {}

POSITION_USD = 10000   # 每个信号目标资金, 按入场价动态算股数 (不融资)


def _today_str() -> str:
    return datetime.now(ET).date().isoformat()


def _round_tick(price) -> Decimal:
    """美股 tick = 0.01"""
    return Decimal(f"{round(float(price) + 1e-9, 2):.2f}")


def _load_today_state():
    """脚本重启时, 重新载入今天的订单状态"""
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
            print("   ⚡ LIVE_TRADING=true → 模拟盘自动下单已启用 (Plan-Driven 多策略)")
        except Exception as e:
            print(f"   ❌ TradeContext 初始化失败: {e}")
            _LIVE = False
    else:
        print("   🛡️ LIVE_TRADING 未启用 (策略只 dry-run)")
    _load_today_state()
    if _PENDING_ENTRIES or _OPEN_POSITIONS:
        print(f"   📂 载入: 待成交 {len(_PENDING_ENTRIES)} 个 / 已成交 {len(_OPEN_POSITIONS)} 个")


def place_entry(symbol: str, plan, position_usd: int = None, is_replay: bool = False) -> dict:
    """
    Plan-Driven 入场. plan = TradePlan 对象 (有 order_type/limit_price/stop_price/tp_price/note 等)
    is_replay=True 时跳过 (历史回放不下单).
    """
    today = _today_str()
    if symbol in _OPEN_POSITIONS and _OPEN_POSITIONS[symbol].get("date") == today:
        return {"ok": False, "reason": "今日已成交 (幂等)"}
    if symbol in _PENDING_ENTRIES and _PENDING_ENTRIES[symbol].get("date") == today:
        return {"ok": False, "reason": "今日已挂单 (幂等)"}
    if is_replay:
        return {"ok": False, "reason": "历史回放 (dry-run)"}
    if not _LIVE or _TRADE_CTX is None:
        return {"ok": False, "reason": "LIVE_TRADING=false (dry-run)"}

    if position_usd is None:
        position_usd = POSITION_USD

    # 解析 plan (策略名取自 note 第一段)
    strategy = (plan.note or "?").split()[0] if plan.note else "?"
    target_px = _round_tick(plan.limit_price)
    stop_px = _round_tick(plan.stop_price)
    tp_px = _round_tick(plan.tp_price)
    if float(target_px) <= 0:
        return {"ok": False, "reason": f"非法价格 {target_px}"}
    qty = Decimal(str(max(1, int(position_usd / float(target_px)))))

    try:
        from longport.openapi import OrderType, OrderSide, TimeInForceType
    except ImportError as e:
        return {"ok": False, "reason": f"import error: {e}"}

    rec = {
        "date": today, "strategy": strategy,
        "order_type": plan.order_type, "side": plan.side,
        "limit_price": float(target_px), "stop_price": float(stop_px), "tp_price": float(tp_px),
        "qty": str(qty), "filled": False, "armed": False,
        "note": plan.note,
    }

    try:
        if plan.order_type == "LMT":
            # 限价 LIT BUY (价格触到 trigger 后挂限价)
            resp = _TRADE_CTX.submit_order(
                symbol=symbol, order_type=OrderType.LIT, side=OrderSide.Buy,
                submitted_quantity=qty, submitted_price=target_px, trigger_price=target_px,
                time_in_force=TimeInForceType.Day,
                remark=f"{strategy}-LMT")
        elif plan.order_type == "MKT":
            # 市价单 (立即成交)
            resp = _TRADE_CTX.submit_order(
                symbol=symbol, order_type=OrderType.MO, side=OrderSide.Buy,
                submitted_quantity=qty, time_in_force=TimeInForceType.Day,
                remark=f"{strategy}-MKT")
        else:
            return {"ok": False, "reason": f"未知 order_type: {plan.order_type}"}

        rec["entry_id"] = resp.order_id
        _PENDING_ENTRIES[symbol] = rec
        _persist(symbol, "pending_entry", rec)
        return {"ok": True, "entry_id": resp.order_id, "qty": int(qty),
                "limit_px": float(target_px), "stop_px": float(stop_px), "tp_px": float(tp_px),
                "strategy": strategy, "order_type": plan.order_type}
    except Exception as e:
        return {"ok": False, "reason": f"submit_order 失败: {e}"}


def check_fills_and_arm_brackets(quote_ctx=None):
    """
    心跳调用. 检查所有 pending 订单是否已成交; 成交后挂 stop+tp.
    quote_ctx 参数保留但不再使用 (策略各自有 RVOL/趋势过滤, 不再后置检查).
    """
    if not _LIVE or _TRADE_CTX is None:
        return
    try:
        from longport.openapi import OrderType, OrderSide, TimeInForceType, OrderStatus
    except ImportError:
        return

    for sym, rec in list(_PENDING_ENTRIES.items()):
        if rec.get("armed"):
            continue
        entry_id = rec.get("entry_id")
        if not entry_id:
            continue
        try:
            d = _TRADE_CTX.order_detail(entry_id)
        except Exception as e:
            print(f"   ⚠️ {sym} 查单失败: {e}")
            continue
        if d.status not in (OrderStatus.Filled, OrderStatus.PartialFilled):
            continue   # 还没成交

        fill_price = float(d.executed_price) if d.executed_price else float(rec["limit_price"])
        # 只在第一次检测到 fill 时打印, 避免重试时刷屏
        was_filled_before = rec.get("filled", False)
        rec["filled"] = True
        rec["buy_fill_price"] = fill_price
        if not was_filled_before:
            print(f"   ✅ {sym} {rec['order_type']} 单成交 @ ${fill_price:.2f}  策略={rec['strategy']}")

        qty = Decimal(rec.get("qty", "1"))

        # ⚠️ 关键修复: LongPort papertrading 时序 bug — fill 后持仓需要时间 reflect
        # 直接挂 SELL stop/tp 会被 LongPort 误判为做空 (603301).
        # 等下次心跳时 query stock_positions 确认持仓 ≥ qty 后再挂.
        try:
            positions_now = _TRADE_CTX.stock_positions(symbols=[sym])
            actual_qty = 0
            for ch in positions_now.channels:
                for p in ch.positions:
                    if p.symbol == sym:
                        actual_qty = max(actual_qty, int(p.quantity))
            if actual_qty < int(qty):
                if not was_filled_before:
                    print(f"   ⏳ {sym} 持仓还未到账 ({actual_qty}/{int(qty)}), 等下次心跳再挂子单")
                continue   # 不 mark armed, 下轮重试
        except Exception as e:
            print(f"   ⚠️ {sym} 查持仓失败, 跳过本轮: {e}")
            continue
        stop_px = _round_tick(rec["stop_price"])
        tp_px = _round_tick(rec["tp_price"])

        # 安全检查: stop 必须 < fill, tp 必须 > fill (long-only)
        if float(stop_px) >= fill_price:
            print(f"   ⚠️ {sym} stop {stop_px} >= fill {fill_price}, 调整 stop = fill * 0.99")
            stop_px = _round_tick(fill_price * 0.99)
        if float(tp_px) <= fill_price:
            print(f"   ⚠️ {sym} tp {tp_px} <= fill {fill_price}, 调整 tp = fill * 1.02")
            tp_px = _round_tick(fill_price * 1.02)

        # 挂止损 (MIT)
        try:
            stop_resp = _TRADE_CTX.submit_order(
                symbol=sym, order_type=OrderType.MIT, side=OrderSide.Sell,
                submitted_quantity=qty, trigger_price=stop_px,
                time_in_force=TimeInForceType.Day,
                remark=f"{rec['strategy']}-Stop")
            rec["stop_id"] = stop_resp.order_id
            rec["stop_px"] = str(stop_px)
        except Exception as e:
            rec["stop_error"] = str(e)
            print(f"      ❌ 挂止损失败: {e}")

        # 挂止盈 (LO)
        try:
            tp_resp = _TRADE_CTX.submit_order(
                symbol=sym, order_type=OrderType.LO, side=OrderSide.Sell,
                submitted_quantity=qty, submitted_price=tp_px,
                time_in_force=TimeInForceType.Day,
                remark=f"{rec['strategy']}-TP")
            rec["tp_id"] = tp_resp.order_id
            rec["tp_px"] = str(tp_px)
        except Exception as e:
            rec["tp_error"] = str(e)
            print(f"      ❌ 挂止盈失败: {e}")

        rec["armed"] = True
        rec["closed"] = False
        _OPEN_POSITIONS[sym] = rec
        _PENDING_ENTRIES.pop(sym, None)
        _persist(sym, "open_position", rec)
        print(f"   🎯 {sym} 子单已挂: stop @ {stop_px}  tp @ {tp_px}")


def reconcile_oco():
    """OCO 仿真: 检查每个未平仓票的 stop/tp 状态, 任一成交则撤另一"""
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
                print(f"   🔁 {sym} 止损成交, 已撤止盈")
            except Exception:
                pass
            rec["closed"] = True; rec["close_reason"] = "stop_filled"
            _persist(sym, "open_position", rec)
        elif tp_filled and stop_id:
            try:
                _TRADE_CTX.cancel_order(stop_id)
                print(f"   🔁 {sym} 止盈成交, 已撤止损")
            except Exception:
                pass
            rec["closed"] = True; rec["close_reason"] = "tp_filled"
            _persist(sym, "open_position", rec)


def cleanup_orphan_positions():
    """
    9:30 ET 开盘后第一时间调用. 平掉**隔夜遗留**持仓 (_OPEN_POSITIONS 里没有 / 或都 closed 的, 但 LongPort 实际还有).
    场景: 上一交易日 force_close 部分 Rejected → 周末挂单过夜.
    LongPort papertrading 盘前禁止 SELL (603301), 必须等开盘后立刻 MO 平.
    """
    if not _LIVE or _TRADE_CTX is None:
        return 0
    try:
        from longport.openapi import OrderType, OrderSide, TimeInForceType
    except ImportError:
        return 0

    try:
        positions = _TRADE_CTX.stock_positions()
    except Exception as e:
        print(f"   ⚠️ cleanup_orphan: 查持仓失败 {e}")
        return 0

    closed_count = 0
    for ch in positions.channels:
        for p in ch.positions:
            qty = int(p.quantity)
            if qty <= 0:
                continue
            sym = p.symbol
            # 检查是不是 today active 的 (在 _OPEN_POSITIONS 且未 closed)
            today = _today_str()
            rec = _OPEN_POSITIONS.get(sym)
            if rec and rec.get("date") == today and not rec.get("closed"):
                continue   # 是今天活的, 跳过
            # 是 orphan, 平掉
            try:
                from decimal import Decimal
                resp = _TRADE_CTX.submit_order(
                    symbol=sym, order_type=OrderType.MO, side=OrderSide.Sell,
                    submitted_quantity=Decimal(str(qty)), time_in_force=TimeInForceType.Day,
                    remark="OrphanFromPrevDay")
                print(f"   🧹 平隔夜孤儿 {sym} {qty} 股, id={resp.order_id}")
                closed_count += 1
            except Exception as e:
                print(f"   ❌ 平孤儿 {sym} {qty} 失败: {e}")
    if closed_count > 0:
        print(f"   ✅ cleanup_orphan: 平了 {closed_count} 只隔夜遗留持仓")
    return closed_count


def force_close_all():
    """收盘前 15:50 ET: 撤所有未成交单 + 市价平所有持仓"""
    if not _LIVE or _TRADE_CTX is None:
        return
    try:
        from longport.openapi import OrderType, OrderSide, TimeInForceType
    except ImportError:
        return

    print("\n🛑 收盘前强平: 撤所有未成交单 + 市价平仓")

    # 撤所有 pending 入场单
    for sym, rec in list(_PENDING_ENTRIES.items()):
        eid = rec.get("entry_id")
        if eid and not rec.get("filled"):
            try:
                _TRADE_CTX.cancel_order(eid)
                print(f"   ↩️ 撤入场单 {sym}")
            except Exception:
                pass

    # 撤所有 open position 的 stop/tp 子单
    for sym, rec in _OPEN_POSITIONS.items():
        if rec.get("closed"):
            continue
        for key in ("stop_id", "tp_id"):
            oid = rec.get(key)
            if oid:
                try:
                    _TRADE_CTX.cancel_order(oid)
                    print(f"   ↩️ 撤 {sym} {key}")
                except Exception:
                    pass

    # 市价平所有真持仓
    try:
        positions = _TRADE_CTX.stock_positions()
    except Exception as e:
        print(f"   ⚠️ 查持仓失败: {e}")
        return

    for ch in positions.channels:
        for p in ch.positions:
            qty = Decimal(str(p.quantity))
            if qty <= 0:
                continue
            try:
                _TRADE_CTX.submit_order(
                    symbol=p.symbol, order_type=OrderType.MO, side=OrderSide.Sell,
                    submitted_quantity=qty, time_in_force=TimeInForceType.Day,
                    remark="ForceClose")
                print(f"   ✅ 强平 {p.symbol} {qty} 股")
                rec = _OPEN_POSITIONS.get(p.symbol, {})
                rec["closed"] = True; rec["close_reason"] = "force_close"
                _OPEN_POSITIONS[p.symbol] = rec
                _persist(p.symbol, "open_position", rec)
            except Exception as e:
                print(f"   ❌ 强平 {p.symbol} 失败: {e}")
