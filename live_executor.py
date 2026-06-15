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
import time as _time
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

# 卖单冷却: {symbol: 上次市价卖出 submit 的时间戳}. 防止合成平仓重试 / force_close 重叠
# 在持仓回写延迟窗口内对同一票重复下卖单 (超卖/反向开空).
_sell_cooldown: dict = {}
SELL_COOLDOWN_SEC = 60

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
    心跳调用. 检查所有 pending 入场单是否已成交; 成交 + 持仓到账后 → arm 合成 bracket.

    ⚠️ 关键修复 (6/15): 不再挂 resting 止损/止盈子单.
    LongPort 不支持原生 OCO: 同一持仓挂止盈(LO)+止损(MIT)两个全仓卖单, 先挂的锁仓,
    后挂的报 "Insufficient holdings" 被拒 → 历史上 52 个止损单全 Rejected/Canceled,
    仓位实际从无止损保护. 现在改成: fill 后只记录 stop_px/tp_px, 由 check_synthetic_exits
    每个心跳轮询现价, 触及就市价平 (合成 bracket). 一并消除 603301 误判做空.
    """
    if not _LIVE or _TRADE_CTX is None:
        return
    try:
        from longport.openapi import OrderStatus
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
        was_filled_before = rec.get("filled", False)
        rec["filled"] = True
        rec["buy_fill_price"] = fill_price
        if not was_filled_before:
            print(f"   ✅ {sym} {rec['order_type']} 单成交 @ ${fill_price:.2f}  策略={rec['strategy']}")

        qty = int(Decimal(rec.get("qty", "1")))

        # 确认持仓到账 (防 race), 用实际持仓数量
        try:
            positions_now = _TRADE_CTX.stock_positions(symbols=[sym])
            actual_qty = 0
            for ch in positions_now.channels:
                for p in ch.positions:
                    if p.symbol == sym:
                        actual_qty = max(actual_qty, int(p.quantity))
            if actual_qty < qty:
                if not was_filled_before:
                    print(f"   ⏳ {sym} 持仓还未到账 ({actual_qty}/{qty}), 等下次心跳")
                continue   # 不 arm, 下轮重试
        except Exception as e:
            print(f"   ⚠️ {sym} 查持仓失败, 跳过本轮: {e}")
            continue

        stop_px = float(_round_tick(rec["stop_price"]))
        tp_px = float(_round_tick(rec["tp_price"]))
        # 安全检查: stop < fill, tp > fill (long-only)
        if stop_px >= fill_price:
            stop_px = float(_round_tick(fill_price * 0.99))
        if tp_px <= fill_price:
            tp_px = float(_round_tick(fill_price * 1.02))

        # 合成 bracket: 只记录价位, 不挂子单
        rec["stop_px"] = stop_px
        rec["tp_px"] = tp_px
        rec["held_qty"] = actual_qty
        rec["armed"] = True
        rec["closed"] = False
        _OPEN_POSITIONS[sym] = rec
        _PENDING_ENTRIES.pop(sym, None)
        _persist(sym, "open_position", rec)
        print(f"   🎯 {sym} 合成 bracket 已 arm: stop≤{stop_px}  tp≥{tp_px}  (轮询触发, 不挂 resting 子单)")


def _market_sell(sym: str, qty: int, reason: str, force: bool = False) -> bool:
    """市价平仓 (合成 bracket / force_close 共用). 卖前以实际持仓为准, 返回成功/失败.
    冷却: 同一票 SELL_COOLDOWN_SEC 内不重复下卖单 (防回写延迟窗口内超卖). force=True 跳过冷却."""
    if not force:
        last = _sell_cooldown.get(sym, 0)
        if _time.time() - last < SELL_COOLDOWN_SEC:
            return False   # 冷却中, 上一张卖单可能还没回写, 不重复下
    try:
        from longport.openapi import OrderType, OrderSide, TimeInForceType
    except ImportError:
        return False
    try:
        resp = _TRADE_CTX.submit_order(
            symbol=sym, order_type=OrderType.MO, side=OrderSide.Sell,
            submitted_quantity=Decimal(str(qty)), time_in_force=TimeInForceType.Day,
            remark=reason[:30])
        _sell_cooldown[sym] = _time.time()
        print(f"   💰 {sym} 市价平 {qty} 股 ({reason}) id={resp.order_id}")
        return True
    except Exception as e:
        print(f"   ❌ {sym} 市价平失败 ({reason}): {e}")
        return False


def check_synthetic_exits(quote_ctx):
    """
    合成 bracket 核心: 心跳轮询所有未平仓位的现价, 触及 stop/tp 就市价平.
    替代 LongPort 不支持的 OCO resting 子单.
    """
    if not _LIVE or _TRADE_CTX is None or quote_ctx is None:
        return
    open_syms = [s for s, r in _OPEN_POSITIONS.items() if not r.get("closed")]
    if not open_syms:
        return
    # 批量拉现价 (一次 quote 调用, 省 API)
    # ⚠️ 逐条容错: 某只票停牌/盘前/当日无成交时 last_done 可能 None, float(None) 会炸.
    # 必须只跳过坏值, 不能让一只票拖垮整批 (否则所有持仓本轮漏检止损 = 原大bug复发).
    quotes = {}
    try:
        for q in quote_ctx.quote(open_syms):
            if q.last_done is None:
                continue
            try:
                quotes[q.symbol] = float(q.last_done)
            except (ValueError, TypeError):
                pass
    except Exception as e:
        print(f"   ⚠️ check_synthetic_exits 拉现价失败: {e}")
        return

    for sym in open_syms:
        rec = _OPEN_POSITIONS.get(sym)
        if not rec or rec.get("closed"):
            continue
        px = quotes.get(sym)
        if px is None:
            continue
        # 强制 float (旧 jsonl 记录里 stop_px/tp_px 可能是字符串)
        try:
            stop_px = float(rec["stop_px"]) if rec.get("stop_px") is not None else None
            tp_px = float(rec["tp_px"]) if rec.get("tp_px") is not None else None
        except (ValueError, TypeError):
            continue
        hit = None
        if stop_px is not None and px <= stop_px:
            hit = "stop"
        elif tp_px is not None and px >= tp_px:
            hit = "tp"
        if not hit:
            continue
        # 以实际持仓数量市价平 (防止超卖)
        try:
            positions_now = _TRADE_CTX.stock_positions(symbols=[sym])
            actual_qty = 0
            for ch in positions_now.channels:
                for p in ch.positions:
                    if p.symbol == sym:
                        actual_qty = max(actual_qty, int(p.quantity))
        except Exception as e:
            print(f"   ⚠️ {sym} 平仓前查持仓失败: {e}")
            continue
        if actual_qty <= 0:
            rec["closed"] = True; rec["close_reason"] = f"{hit}_already_flat"
            _persist(sym, "open_position", rec)
            continue
        if _market_sell(sym, actual_qty, f"{rec.get('strategy','?')}-{hit.upper()}@{px}"):
            rec["closed"] = True
            rec["close_reason"] = hit
            rec["close_px"] = px
            _persist(sym, "open_position", rec)


def reconcile_oco():
    """[DEPRECATED] 合成 bracket 取代了 OCO resting 子单, 此函数保留为 no-op 兼容旧调用."""
    return


_orphan_pending_retry: dict = {}   # {symbol: {"qty": int, "attempts": int}}

def cleanup_orphan_positions():
    """
    9:30 ET 开盘后第一时间调用. 平掉**隔夜遗留**持仓 (_OPEN_POSITIONS 里没有 / 或都 closed 的, 但 LongPort 实际还有).
    场景: 上一交易日 force_close 部分 Rejected → 周末挂单过夜.
    LongPort papertrading 盘前禁止 SELL (603301), 必须等开盘后立刻 MO 平.
    Reject 的会在下次心跳重试 (max 5 次).
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
                _orphan_pending_retry.pop(sym, None)
            except Exception as e:
                # 加入重试队列
                attempts = _orphan_pending_retry.get(sym, {}).get("attempts", 0) + 1
                _orphan_pending_retry[sym] = {"qty": qty, "attempts": attempts}
                print(f"   ❌ 平孤儿 {sym} {qty} 失败 (尝试 {attempts}/5): {e}")
    if closed_count > 0:
        print(f"   ✅ cleanup_orphan: 平了 {closed_count} 只隔夜遗留持仓")
    return closed_count


def retry_orphan_cleanup():
    """心跳调用: 重试 cleanup_orphan 失败的孤儿. max 5 次"""
    if not _LIVE or _TRADE_CTX is None:
        return
    try:
        from longport.openapi import OrderType, OrderSide, TimeInForceType
        from decimal import Decimal
    except ImportError:
        return
    for sym, info in list(_orphan_pending_retry.items()):
        if info["attempts"] >= 5:
            print(f"   ⚠️ 孤儿 {sym} 已尝试 {info['attempts']} 次仍失败, 放弃")
            _orphan_pending_retry.pop(sym, None)
            continue
        try:
            resp = _TRADE_CTX.submit_order(
                symbol=sym, order_type=OrderType.MO, side=OrderSide.Sell,
                submitted_quantity=Decimal(str(info["qty"])), time_in_force=TimeInForceType.Day,
                remark="OrphanFromPrevDay-Retry")
            print(f"   🧹 重试平孤儿 {sym} {info['qty']} 股 ✅ (第 {info['attempts']+1} 次)")
            _orphan_pending_retry.pop(sym, None)
        except Exception as e:
            info["attempts"] += 1
            print(f"   ❌ 重试 {sym} 仍失败 ({info['attempts']}/5): {str(e)[:80]}")


def force_close_all():
    """
    收盘前 15:50 ET: 撤所有未成交单 + 市价平所有持仓.
    关键修: 撤单后 sleep 3s 让 LongPort 处理, 再 submit MO sell, 避免 603301 时序冲突.
    Reject 的入隔夜孤儿队列, 下次开盘 cleanup_orphan 兜底.
    """
    import time as _time
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

    # 合成 bracket 后没有 resting 止损/止盈子单要撤了 (只可能有未成交 LIT 入场单, 上面已撤).
    # 撤单后等 LongPort 处理, 再 submit MO sell, 避免 603301 时序冲突.
    print("   ⏳ 等 3s 让撤单 settle...")
    _time.sleep(3)

    # 市价平所有真持仓 (用持仓 query, 不依赖 _OPEN_POSITIONS, 防遗漏)
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
                # Reject 的入孤儿队列, 等下交易日 cleanup_orphan 兜底
                _orphan_pending_retry[p.symbol] = {"qty": int(qty), "attempts": 0}
                print(f"      → 已加入孤儿队列, 下交易日开盘自动平")
