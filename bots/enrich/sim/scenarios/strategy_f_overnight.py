#!/usr/bin/env python3
"""sim/scenarios/strategy_f_overnight.py — 对抗 2026-07-22 F不对称过夜【阶梯】策略(线上真实配置)。

线上: MECH_BE=0(runner不保本扛回撤接大runner) + F过夜阶梯(收盘15:40-16:00按剩仓当前收益g):
  g<F_EOD_CLOSE_BELOW(30%)   → 全平不过夜(含未落袋满仓+回撤runner; 胜率引擎)
  30%≤g<F_EOD_TRIM_ABOVE(50%) → 原样留过夜
  g≥50%                       → 砍半剩仓降风险再留过夜(本日只砍一次)
⚠ feedback_paper_serves_live: sim默认MECH_BE=1(旧保本), 这里显式切MECH_BE=0覆盖线上真实路径。
攻击面: 三档边界、砍半不重复不裸空、窗口边界(<15:40/≥16:00不动)、runner回撤(XOM式)被<30%全平。
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


def _runner(s, price_after_tp1):
    """建仓 → 摸+30%首档卖½ → runner 停在 price_after_tp1。返回持仓。"""
    _isolate()
    s.quotes.set_path(OSI, [1.00, 1.35, 1.35, price_after_tp1])
    p = _open_new(s, equity=1200.0, path=(FLAT,))
    s.clock.set_et(2026, 7, 20, 12, 0)                    # 盘中, 远离F收盘窗
    s.quotes.set_path(OSI, [1.35, price_after_tp1, price_after_tp1])
    s.tick(n=3)
    return s.pos(OSI)


# ══════════════════════════════════════════════════════════════════════════
# runner 不保本(MECH_BE=0): 首档止盈后止损仍-50%, 回撤到入场价不被砍
# ══════════════════════════════════════════════════════════════════════════

def sc_f_runner_no_breakeven_rides_minus50(s):
    """F(MECH_BE=0): 首档+30%卖½后 runner 止损仍-50%(不移保本)。盘中回撤到入场价1.00 > -50%(0.50)
    → runner 不该被止损平(接GOOGL: 砍在保本就吃不到次日爆发)。"""
    with _f_config():
        fails = []
        p = _runner(s, 1.00)                              # 首档后回撤到入场价1.00(盘中12点, 非收盘窗)
        fails += expect_ge(len(s.evs("tp_fill")), 1, "首档止盈应成交(reduced置位)")
        if p and (p.get("filled", 0) - p.get("sold", 0)) > 0:
            sp = B._stop_price(p)
            fails += expect(abs(sp - p["avg"] * 0.5) < 1e-6,
                            f"F: reduced后MECH_BE=0 → runner止损应=-50%({p['avg']*0.5:.3f}), 实际{sp:.3f}(移了保本=错)")
            fails += expect(p.get("status") == "open",
                            "F: runner盘中回撤到入场价(>-50%)不该被止损平, 应活着接后续爆发")
        else:
            fails += ["runner 不该在盘中回撤到入场价时消失(MECH_BE=0应维持-50%不砍)"]
    return fails


# ══════════════════════════════════════════════════════════════════════════
# F 过夜阶梯三档: <30全平 / 30-50留 / ≥50砍半留
# ══════════════════════════════════════════════════════════════════════════

def sc_f_eod_closes_unreduced(s):
    """F阶梯: 未落袋满仓(从没摸+30%, g必<30%) 15:50 ET → 全平不裸扛过夜, 平净不裸空。"""
    with _f_config():
        _isolate()
        fails = []
        p = _open_new(s, equity=1200.0, path=(FLAT,))       # 价格平g≈0, 从没摸+30% → 未落袋满仓
        fails += expect(p is not None and not p.get("reduced"), "前置: 应有未落袋满仓(reduced=False)")
        s.clock.set_et(2026, 7, 20, 15, 50)                 # 非到期日 15:40-16:00窗内
        s.tick(n=3)                                         # 触发全平 + 市价卖成交settle
        p = s.pos(OSI)
        closed = p is None or p.get("status") in ("closing", "closed") \
            or (p.get("filled", 0) - p.get("sold", 0)) == 0
        fails += expect(closed, f"F阶梯: 未落袋满仓(g<30)应收盘全平, 实际status={p and p.get('status')}")
        fails += expect_eq(s.broker_pos(OSI), 0, "全平后必须平净, 不留裸多/裸空")
    return fails


def sc_f_eod_closes_weak_runner(s):
    """F阶梯: 已落袋runner但收盘回撤到 g<30%(如+10%) → 全平不过夜。
    这就是XOM式"当天回撤次日爆发"被误杀的路径 —— 尾部保护的必付代价(收盘时它和将崩单无法区分)。"""
    with _f_config():
        fails = []
        p = _runner(s, 1.10)                                # 首档落袋 → runner回撤到1.10(+10% < 30%)
        fails += expect(p is not None and p.get("reduced"), "前置: 已落袋runner(reduced=True)")
        if p:
            s.quotes.set_path(OSI, [1.10, 1.10])
            s.clock.set_et(2026, 7, 20, 15, 50)
            s.tick(n=3)
            p = s.pos(OSI)
            closed = p is None or p.get("status") in ("closing", "closed") \
                or (p.get("filled", 0) - p.get("sold", 0)) == 0
            fails += expect(closed, "F阶梯: 收盘g<30%的回撤runner应全平(不裸扛过夜)")
            fails += expect_eq(s.broker_pos(OSI), 0, "全平后不裸空")
    return fails


def sc_f_runner_holds_overnight(s):
    """F阶梯: 已落袋runner收盘时 g∈[30%,50%)(如+40%) → 原样留过夜(接GOOGL类)。"""
    with _f_config():
        fails = []
        p = _runner(s, 1.40)                                # runner停在1.40(+40%, 未武装)
        fails += expect(p is not None and p.get("reduced"), "前置: 已落袋runner(reduced=True)")
        if p:
            s.quotes.set_path(OSI, [1.40, 1.40])            # 收盘价 g=+40% ∈[30,50) → 原样留
            s.clock.set_et(2026, 7, 20, 15, 50)
            s.tick()
            p = s.pos(OSI)
            fails += expect(p is not None and p.get("status") == "open"
                            and (p.get("filled", 0) - p.get("sold", 0)) > 0,
                            "F阶梯: g∈[30,50)的runner不该被收盘平/砍, 应原样留过夜")
    return fails


def sc_f_eod_trims_strong_runner(s):
    """F阶梯: 收盘时 g≥50%(如+80%)且剩≥2张 → 砍半剩仓降风险, 剩下留过夜; 本日只砍一次, 不裸空。"""
    with _f_config():
        _isolate()
        fails = []
        s.broker.position[OSI] = 4                          # 券商侧runner 4张(已落袋 filled8/sold4)
        s.quotes.set_path(OSI, [1.80] * 8)                  # 现价1.80 = +80% (g≥50)
        s.positions[OSI] = dict(ticker="HOOD", filled=8, sold=4, avg=1.00, qty=8,
                                right="C", expiry="2026-07-24", strike=120.0,
                                entry_order_id=None, stop_mult=0.5,
                                reduced=True, tp1_done=True, armed=False,
                                status="open", opened="2026-07-20")
        s.clock.set_et(2026, 7, 20, 15, 50)                 # EOD窗内
        s.tick(n=3)                                         # 触发砍半(卖4//2=2) + settle
        p = s.pos(OSI)
        fails += expect(p is not None, "砍半后仓位应还在(非全平)")
        if p:
            rem = p.get("filled", 0) - p.get("sold", 0)
            fails += expect(p.get("status") == "open", "砍半后应回open留过夜")
            fails += expect(0 < rem < 4, f"应砍掉部分留部分过夜(剩{rem}张, 期望0<rem<4)")
            fails += expect_eq(p.get("eod_trim_date"), "2026-07-20", "应标记本日已砍(防窗口内重复砍)")
        fails += expect_le(s.broker_pos(OSI), 4, "券商持仓≤原剩仓, 不裸空")
        # 窗口内再tick: eod_trim_date 守卫 → 不该再砍
        before = (s.pos(OSI) or {}).get("sold", 0)
        s.tick()
        after = (s.pos(OSI) or {}).get("sold", 0)
        fails += expect_eq(after, before, "本日只砍一次: 窗口内再tick不该再砍")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# F 窗口边界: <15:40 不动、≥16:00 不动(市场关闭)
# ══════════════════════════════════════════════════════════════════════════

def sc_f_no_close_before_1540(s):
    """F: 15:40前不该动未落袋满仓(让它盘中继续跑, 还可能摸+30%落袋转runner)。"""
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
    """F窗口封在15:40-16:00: 16:00后市场关闭市价卖不了 → F不该动(等次日窗口)。
    也是通用场景 sc_partial_exit_reprotect_by_remain(用16:05触发Day单撤销)不被F误伤的保证。"""
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


# ══════════════════════════════════════════════════════════════════════════
# F 过夜阶梯边界 edge case (用户2026-07-22指出: 新策略的边界要补齐)
# ══════════════════════════════════════════════════════════════════════════

def sc_f_eod_1lot_runner_holds_not_trim(s):
    """F阶梯: g≥50%但剩仓只1张(不可分半, rem<2) → 原样留过夜, 不砍成0/不误全平(保留过夜上限)。"""
    with _f_config():
        _isolate()
        fails = []
        s.broker.position[OSI] = 1
        s.quotes.set_path(OSI, [1.80] * 4)                  # g=+80%≥50
        s.positions[OSI] = dict(ticker="HOOD", filled=2, sold=1, avg=1.00, qty=2,
                                right="C", expiry="2026-07-24", strike=120.0,
                                entry_order_id=None, stop_mult=0.5,
                                reduced=True, tp1_done=True, armed=False,
                                status="open", opened="2026-07-20")
        s.clock.set_et(2026, 7, 20, 15, 50)
        s.tick()
        p = s.pos(OSI)
        fails += expect(p is not None and p.get("status") == "open"
                        and (p.get("filled", 0) - p.get("sold", 0)) == 1,
                        "F阶梯: 1张runner g≥50 应原样留过夜(rem<2不砍), 不该砍成0/全平")
        fails += expect(not p.get("eod_trim_date"), "1张不触发砍半, 不该打trim标记")
    return fails


def sc_f_expiry_day_full_close_beats_ladder_trim(s):
    """F阶梯: 到期日15:40强平【优先于】过夜阶梯 —— 即使g≥50%强runner也必须全平, 不能砍半留(0DTE不可过夜/防行权)。"""
    with _f_config():
        _isolate()
        fails = []
        eosi = "HOOD260720C120000.US"                       # 到期=默认时钟日07-20
        s.broker.position[eosi] = 4
        s.quotes.set_path(eosi, [1.80] * 6)                 # g=+80%≥50 的强runner
        s.positions[eosi] = dict(ticker="HOOD", filled=4, sold=0, avg=1.00, qty=4,
                                 right="C", expiry="2026-07-20", strike=120.0,
                                 entry_order_id=None, stop_mult=0.5,
                                 reduced=True, tp1_done=True, armed=False,
                                 status="open", opened="2026-07-20")
        s.clock.set_et(2026, 7, 20, 15, 50)                 # 到期日收盘窗
        s.tick(n=3)
        p = s.pos(eosi)
        closed = p is None or p.get("status") in ("closing", "closed") \
            or (p.get("filled", 0) - p.get("sold", 0)) == 0
        fails += expect(closed, "到期日强runner必须全平(到期强平优先于阶梯砍半)")
        fails += expect(p is None or not p.get("eod_trim_date"),
                        "到期日走的应是全平不是阶梯砍半(不该打trim标记)")
        fails += expect_eq(s.broker_pos(eosi), 0, "到期日平净不裸空")
    return fails


def sc_f_eod_trim_guard_resets_next_day(s):
    """F阶梯: '本日只砍一次'守卫按【日期】—— 次日g仍≥50%应能【再砍一次】(不是砍一次就永久锁死)。"""
    with _f_config():
        _isolate()
        fails = []
        s.broker.position[OSI] = 4                          # 券商侧=本地remain(filled8-sold4), 一致
        s.quotes.set_path(OSI, [1.80] * 30)                 # 全程g=+80%
        s.positions[OSI] = dict(ticker="HOOD", filled=8, sold=4, avg=1.00, qty=8,
                                right="C", expiry="2026-07-24", strike=120.0,
                                entry_order_id=None, stop_mult=0.5,
                                reduced=True, tp1_done=True, armed=False,
                                status="open", opened="2026-07-20")
        s.clock.set_et(2026, 7, 20, 15, 50); s.tick(n=3)    # 第1夜: 砍半(卖2)
        sold_d1 = (s.pos(OSI) or {}).get("sold", 0)
        fails += expect(sold_d1 > 4, f"第1夜应砍半(sold从4升到{sold_d1})")
        s.clock.set_et(2026, 7, 21, 15, 50); s.tick(n=3)    # 第2夜(非到期日): 守卫按日期重置, 应再砍
        sold_d2 = (s.pos(OSI) or {}).get("sold", 0)
        fails += expect(sold_d2 > sold_d1,
                        f"第2夜g仍≥50应再砍一次(sold从{sold_d1}升到{sold_d2}); 守卫按日期重置不该永久锁死")
        fails += expect_le(s.broker_pos(OSI), 8, "多夜砍半累计不得超卖")
    return fails


def sc_f_eod_noquote_unreduced_conservative_close(s):
    """F阶梯: 收盘窗报价失效(OPRA权限/429/停牌)+未落袋满仓 → 保守全平(无价也不裸扛过夜)。"""
    with _f_config():
        _isolate()
        fails = []
        p = _open_new(s, equity=1200.0, path=(FLAT,))       # 未落袋满仓, 建仓时有价
        fails += expect(p is not None and not p.get("reduced"), "前置: 未落袋满仓")
        s.quotes.fail.add(OSI)                              # option_quote 抛错 → _option_last返None
        s.clock.set_et(2026, 7, 20, 15, 50)
        s.tick(n=3)
        p = s.pos(OSI)
        closed = p is None or p.get("status") in ("closing", "closed") \
            or (p.get("filled", 0) - p.get("sold", 0)) == 0
        fails += expect(closed, "F阶梯: 报价失效+未落袋满仓应保守全平(无价也不裸扛过夜)")
    return fails


def sc_f_eod_unreduced_g_above30_still_closed(s):
    """F加固(回放reviewer#1): 未落袋满仓(reduced=False)即使收盘 g≥30%(罕见: tp1因故没成交) → 仍必须
    全平不裸扛过夜。防"未落袋⟹g<30"的隐式不变式被tp1失败路径打破而满仓过夜。"""
    with _f_config():
        _isolate()
        fails = []
        s.broker.position[OSI] = 4
        s.quotes.set_path(OSI, [1.40] * 6)                  # g=+40% ∈[30,50): 无reduced守卫会被"原样留"
        s.positions[OSI] = dict(ticker="HOOD", filled=4, sold=0, avg=1.00, qty=4,
                                right="C", expiry="2026-07-24", strike=120.0,
                                entry_order_id=None, stop_mult=0.5,
                                reduced=False, tp1_done=False, armed=False,   # 关键: 未落袋满仓
                                status="open", opened="2026-07-20")
        s.clock.set_et(2026, 7, 20, 15, 50)
        s.tick(n=3)
        p = s.pos(OSI)
        closed = p is None or p.get("status") in ("closing", "closed") \
            or (p.get("filled", 0) - p.get("sold", 0)) == 0
        fails += expect(closed, "F加固: 未落袋满仓即使g≥30也必须收盘平(不裸扛过夜), 不该落到'原样留'")
        fails += expect_eq(s.broker_pos(OSI), 0, "平净不裸空")
    return fails
