#!/usr/bin/env python3
"""sim/scenarios/strategy_v2.py — 对抗 2026-07-21 新策略(单档30卖½ + 保本)。

目的是【找 bug】, 不是刷覆盖: 每个场景断言"正确/安全行为", bot 错了就红。
攻击面 = 新策略独有的边界与竞态:
  · 保本的触发/不重复卖/whipsaw
  · 首档卖½的取整边界(绝不超卖)
  · 首档部分成交时保本尚未生效、剩仓仍受保护
写法约定见 scenario_api.py。通用不变式(裸空/账实/孤儿/弃仓)由 runner 自动跑。
(2026-07-21 用户拍板: 模拟单不做新旧隔离, 保本全局生效 → 原隔离场景已删。)
"""
import discord_enrich_bot as B
from sim.scenario_api import expect, expect_eq, expect_le, expect_ge
from types import SimpleNamespace

OSI = "HOOD260724C120000.US"
BUY = "$HOOD 7/24 $120 calls $1.00"
FLAT = (1.00, 1.00, 1.00)


def _isolate():
    B._rl_until = 0.0
    B._recent_exits.clear()
    B._closing.clear()
    B._optfail.clear()


def _open_new(s, equity=1200.0, path=(FLAT,), oi=500_000):
    """走真实链路开一个新策略仓(保本由全局 MECH_BE 控制)。"""
    _isolate()
    s.broker.equity = equity
    s.quotes.set_path(OSI, list(path))
    s.quotes.open_interest[OSI] = oi
    s.send(BUY)
    s.tick()
    return s.pos(OSI)


def sc_v2_sell_half_never_oversell(s):
    """首档卖½在各 filled 值下: 挂出的卖单张数绝不超过持仓(超了=裸空)。"""
    _isolate()
    fails = []
    for filled in (1, 2, 3, 4, 5, 7, 10):
        q1, _ = B._tp_params(filled, 1.0)
        fails += expect_le(q1, filled, f"filled={filled}: 首档卖{q1}张不得超过持仓")
        fails += expect_ge(q1, 1, f"filled={filled}: 首档至少卖1张")
    # 端到端: 建仓后 ensure_protection 挂的止盈腿不超持仓
    p = _open_new(s, equity=2000.0)
    if p:
        remain = p.get("filled", 0) - p.get("sold", 0)
        live_sell = sum(o.submitted_quantity - o.executed_quantity
                        for o in s.broker.live_orders(OSI) if o.side == "Sell")
        fails += expect_le(live_sell, remain + p.get("sold", 0),
                           f"挂出卖单{live_sell} 不得超持仓{p.get('filled')}")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 首档部分成交 —— 保本尚未生效(reduced只在全部成交置位), 剩仓仍受保护
# ══════════════════════════════════════════════════════════════════════════

def sc_v2_partial_first_tp_still_protected(s):
    """首档止盈单只部分成交时: reduced 不置位(保本未启用), 但剩仓必须仍有止损保护, 不裸奔。"""
    _isolate()
    fails = []
    # 6张仓, 首档卖3张, 但只成交1张后该单终态(流动性枯竭作废)
    s.broker.position[OSI] = 6
    s.quotes.set_path(OSI, [1.00] * 10)
    s.positions[OSI] = dict(ticker="HOOD", filled=6, sold=0, avg=1.00, qty=6,
                            right="C", expiry="2026-07-24", strike=120.0,
                            entry_order_id=None, stop_mult=0.5,
                            tp_order_id="TP", tp_qty=3, tp_order_id_counted=0,
                            reduced=False, tp1_done=False, armed=False,
                            status="open", opened="2026-07-21")
    # 让 tp 单只成交1张然后作废
    s.broker.liquidity[OSI] = 1
    s.broker.tick(step_price=False)
    s.clock.set_et(2026, 7, 21, 16, 30)            # Day 单作废 → 部分成交终态
    s.broker.liquidity.pop(OSI, None)
    s.tick(n=1)
    p = s.pos(OSI)
    # 部分成交后 reduced 不该置位(否则保本对未全部止盈的仓位提前生效, 且关闭剩余止盈通道)
    fails += expect(not p.get("reduced"), f"首档仅部分成交, reduced 不该置位: reduced={p.get('reduced')}")
    # 剩仓必须仍有止损保护(未 reduced → 走 -50%, 非保本)
    if (p.get("filled", 0) - p.get("sold", 0)) > 0:
        sp = B._stop_price(p)
        fails += expect(sp > 0, "剩仓必须有有效止损价(不裸奔)")
        fails += expect(abs(sp - p["avg"] * 0.5) < 1e-6,
                        f"未全部止盈 → 止损应仍是-50%不保本: {sp:.3f}")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 保本触发 —— 平净且不重复卖, 不裸空
# ══════════════════════════════════════════════════════════════════════════

def sc_v2_breakeven_triggers_flat_no_double(s):
    """首档止盈后价格跌回入场价 → 保本止损触发平净, 不重复卖, 券商不裸空。"""
    _isolate()
    fails = []
    # 6张, 摸+30%首档卖3张 → 保本; 再跌回入场价
    s.quotes.set_path(OSI, [1.00, 1.00, 1.35, 1.35, 1.00, 0.98, 0.98, 0.98, 0.98])
    p = _open_new(s, equity=1200.0, path=(FLAT,))
    s.quotes.set_path(OSI, [1.00, 1.35, 1.35, 1.00, 0.98, 0.98, 0.98])
    s.tick(n=7)
    p = s.pos(OSI) or {}
    fails += expect_ge(len(s.evs("tp_fill")), 1, "首档止盈应成交")
    fails += expect_ge(len(s.evs("stop_trigger")), 1, "跌回入场价应触发保本止损")
    fails += expect_ge(s.broker_pos(OSI), 0, "平仓不得裸空")
    # 保本止损价必须是入场价(不是-50%): 验证触发的确实是"保本"而非普通止损
    fails += expect(s.pos(OSI) is None or s.pos(OSI).get("status") in ("closing", "closed")
                    or (s.pos(OSI).get("filled", 0) - s.pos(OSI).get("sold", 0)) == 0,
                    "保本止损后仓位应进入平仓/已平")
    # 不重复卖由通用不变式(超卖/裸空)兜底
    return fails



