#!/usr/bin/env python3
"""sim/scenarios/strategy_f_overnight.py — 对抗 2026-07-22 F不对称过夜策略【线上真实配置】。

线上跑 MECH_BE=0(runner不保本, 让它扛回撤接大runner如GOOGL) +
F_EOD_CLOSE_UNREDUCED=1(未落袋满仓 15:40-16:00 收盘前平不裸扛过夜, 已落袋runner留过夜)。

⚠ feedback_paper_serves_live: sim 默认 MECH_BE=1(旧保本), 但线上 F 走 MECH_BE=0 ——
  那是【模拟盘全绿≠线上没bug】的坑。这里显式切到线上配置, 覆盖线上真实会走的路径。
攻击面:
  · runner 不保本后是否真维持-50%(回撤到入场价不被误砍 = 接得住GOOGL)
  · 未落袋满仓 15:40-16:00 是否收盘平(不裸扛过夜避IBM式-94% gap)且平净不裸空
  · 已落袋runner 15:50 是否留过夜(不被误平)
  · F窗口边界: <15:40不平、≥16:00不平(市场关闭无法市价卖)
"""
import discord_enrich_bot as B
from sim.scenario_api import expect, expect_eq, expect_le, expect_ge
from contextlib import contextmanager

OSI = "HOOD260724C120000.US"          # expiry 2026-07-24(周五); 默认时钟07-20周一=非到期日
BUY = "$HOOD 7/24 $120 calls $1.00"
FLAT = (1.00, 1.00, 1.00)


@contextmanager
def _f_config():
    """临时切到线上 F 配置(MECH_BE=0, F开), 退出恢复原值, 不泄漏给别的场景。"""
    be0, f0 = B.MECH_BE, B.F_EOD_CLOSE_UNREDUCED
    B.MECH_BE, B.F_EOD_CLOSE_UNREDUCED = False, True
    try:
        yield
    finally:
        B.MECH_BE, B.F_EOD_CLOSE_UNREDUCED = be0, f0


def _isolate():
    B._rl_until = 0.0
    B._recent_exits.clear()
    B._closing.clear()
    B._optfail.clear()


def _open_new(s, equity=1200.0, path=(FLAT,), oi=500_000):
    _isolate()
    s.broker.equity = equity
    s.quotes.set_path(OSI, list(path))
    s.quotes.open_interest[OSI] = oi
    s.send(BUY)
    s.tick()
    return s.pos(OSI)


# ══════════════════════════════════════════════════════════════════════════
# runner 不保本(MECH_BE=0): 首档止盈后止损仍-50%, 回撤到入场价不被砍
# ══════════════════════════════════════════════════════════════════════════

def sc_f_runner_no_breakeven_rides_minus50(s):
    """F(MECH_BE=0): 首档+30%卖½后 runner 止损仍是-50%(不移保本)。回撤到入场价1.00 > -50%(0.50)
    → runner 不该被止损平(这正是接住GOOGL: 砍在保本就吃不到次日爆发)。"""
    with _f_config():
        _isolate()
        fails = []
        # 摸+30%(1.30)首档卖½ → runner; 再跌回入场价1.00(GOOGL式回撤)
        s.quotes.set_path(OSI, [1.00, 1.35, 1.35, 1.00, 1.00, 1.00, 1.00])
        p = _open_new(s, equity=1200.0, path=(FLAT,))
        s.clock.set_et(2026, 7, 20, 12, 0)                 # 盘中, 远离F收盘窗
        s.quotes.set_path(OSI, [1.35, 1.35, 1.00, 1.00, 1.00])
        s.tick(n=5)
        p = s.pos(OSI)
        fails += expect_ge(len(s.evs("tp_fill")), 1, "首档止盈应成交(reduced置位)")
        if p and (p.get("filled", 0) - p.get("sold", 0)) > 0:
            sp = B._stop_price(p)
            fails += expect(abs(sp - p["avg"] * 0.5) < 1e-6,
                            f"F: reduced后MECH_BE=0 → runner止损应=-50%({p['avg']*0.5:.3f}), 实际{sp:.3f}(移了保本=错)")
            fails += expect(p.get("status") == "open",
                            "F: runner回撤到入场价(>-50%)不该被止损平, 应活着接后续爆发")
        else:
            fails += ["runner 不该在回撤到入场价时消失(MECH_BE=0应维持-50%不砍)"]
    return fails


# ══════════════════════════════════════════════════════════════════════════
# F 收盘平未落袋满仓 (15:40-16:00), runner 留过夜
# ══════════════════════════════════════════════════════════════════════════

def sc_f_eod_closes_unreduced(s):
    """F: 未落袋满仓(从没摸+30%) 非到期日 15:50 ET → 收盘前市价平(不裸扛过夜), 平净不裸空。"""
    with _f_config():
        _isolate()
        fails = []
        p = _open_new(s, equity=1200.0, path=(FLAT,))       # 价格平, 从没摸+30% → 未落袋满仓
        fails += expect(p is not None and not p.get("reduced"), "前置: 应有未落袋满仓(reduced=False)")
        s.clock.set_et(2026, 7, 20, 15, 50)                 # 非到期日(expiry07-24) 15:40-16:00窗内
        s.tick(n=3)                                         # F触发close_position + 市价卖成交settle
        p = s.pos(OSI)
        closed = p is None or p.get("status") in ("closing", "closed") \
            or (p.get("filled", 0) - p.get("sold", 0)) == 0
        fails += expect(closed, f"F: 未落袋满仓应在15:40-16:00收盘平, 实际status={p and p.get('status')}")
        fails += expect_eq(s.broker_pos(OSI), 0, "F收盘平后必须平净, 不留裸多/裸空")
    return fails


def sc_f_runner_holds_overnight(s):
    """F: 已落袋+30%的runner(reduced=True) 15:50 ET → 【不】被收盘平, 留过夜接大runner。"""
    with _f_config():
        _isolate()
        fails = []
        s.quotes.set_path(OSI, [1.00, 1.35, 1.35, 1.20, 1.20, 1.20, 1.20])
        p = _open_new(s, equity=1200.0, path=(FLAT,))
        s.clock.set_et(2026, 7, 20, 12, 0)
        s.quotes.set_path(OSI, [1.35, 1.35, 1.20, 1.20])
        s.tick(n=4)                                         # 摸+30%落袋 → runner(未武装, 价1.20)
        p = s.pos(OSI)
        fails += expect(p is not None and p.get("reduced"), "前置: 应有已落袋runner(reduced=True)")
        if p:
            s.clock.set_et(2026, 7, 20, 15, 50)             # 收盘窗内
            s.tick()
            p = s.pos(OSI)
            fails += expect(p is not None and p.get("status") == "open"
                            and (p.get("filled", 0) - p.get("sold", 0)) > 0,
                            "F: 已落袋runner不该被收盘平, 应留过夜(接GOOGL类回撤后爆发)")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# F 窗口边界: <15:40 不平、≥16:00 不平(市场关闭)
# ══════════════════════════════════════════════════════════════════════════

def sc_f_no_close_before_1540(s):
    """F: 15:40前不该平未落袋满仓(让它盘中继续跑, 还可能摸+30%落袋转runner)。"""
    with _f_config():
        _isolate()
        fails = []
        p = _open_new(s, equity=1200.0, path=(FLAT,))
        s.clock.set_et(2026, 7, 20, 15, 30)                 # 15:40前
        s.tick()
        p = s.pos(OSI)
        fails += expect(p is not None and p.get("status") == "open",
                        "F: 15:40前不该收盘平未落袋满仓")
    return fails


def sc_f_no_close_after_1600_market_closed(s):
    """F窗口封在15:40-16:00: 16:00后市场关闭, 市价卖成交不了 → F不该fire(等次日窗口)。
    这也是通用场景 sc_partial_exit_reprotect_by_remain(用16:05触发Day单撤销)不被F误伤的保证。"""
    with _f_config():
        _isolate()
        fails = []
        p = _open_new(s, equity=1200.0, path=(FLAT,))
        fails += expect(p is not None and not p.get("reduced"), "前置: 未落袋满仓")
        s.clock.set_et(2026, 7, 20, 16, 5)                  # 16:00后, 非到期日
        s.tick()
        p = s.pos(OSI)
        fails += expect(p is not None and p.get("status") == "open",
                        "F: 16:00后市场关闭不该fire收盘平(否则挂个成交不了的市价单), 应等次日15:40窗")
    return fails
