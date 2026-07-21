#!/usr/bin/env python3
"""sim/scenarios/normal.py — 【正常链路】端到端仿真场景。

覆盖 bot 在"一切正常"时应该做对的事: 机械出场三档 / 9ema拖尾武装门 / 止损 / 到期强平
/ 入场TTL / 迟到闸 / 仓位档位(常规·lotto·0DTE) / 去重 / 敞口闸 / OI帽 / 权利金上限。

对抗性/故障注入场景在 sim/scenarios/adversarial.py, 这里只写"顺风路径"。

约定(见 sim/scenario_api.py): 每个 sc_* 返回失败原因列表, 空列表=通过。
通用不变式(裸空/账实不符/孤儿单/弃仓/挂卖单超持仓)由 run_sim.py 统一跑, 本文件不重复。
"""
from datetime import timedelta

import discord_enrich_bot as B
from sim.harness import ET
from sim.scenario_api import (expect, expect_eq, expect_ge, expect_le,
                              expect_in, expect_not_in, osi)

# ── 常量: 与场景里用的信号文本保持一致 ──
HOOD_120 = osi("HOOD", "260724", "C", 120)      # $HOOD 7/24 $120 calls
LIMIT = 0.83                                     # 信号权利金 = 入场限价
EQUITY = 100_000.0                               # run_sim 给每个场景的默认净值

# 建仓限价 0.83 → 一档 +30% = $1.08, 二档 +60% = $1.33, 止损 -60% = $0.33
TP1_PX = round(LIMIT * 1.30, 2)
TP2_PX = round(LIMIT * 1.60, 2)

# 15分9ema 用的标的K线: 平台整理后连破2根 / 一路上行不破
CANDLES_BREAK2 = [100.0] * 18 + [95.0, 90.0]
CANDLES_RISING = [100.0 + i for i in range(20)]


def _remain(s, o):
    """bot 账本上的剩余张数。"""
    p = s.pos(o) or {}
    return p.get("filled", 0) - p.get("sold", 0)


def _live_sells(s, o):
    return [x for x in s.broker.live_orders(o) if x.side == "Sell"]


def _buy_orders(s):
    return [x for x in s.broker.orders.values() if x.side == "Buy"]


def _reasons(s, ev):
    return [e.get("reason") for e in s.evs(ev)]


# ══════════════════════════════════════════════════════════════════════════
# 1-5: 机械出场三档 + 9ema 拖尾
# ══════════════════════════════════════════════════════════════════════════

def sc_full_mechanical_exit(s):
    """完整机械出场(2026-07-21单档定案): 建仓→+30%卖½→首档后保本→剩仓摸+60%武装9ema→runner在手。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.95, 1.10, 1.20, 1.35, 1.40, 1.25, 1.20, 1.20])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.quotes.candles["HOOD"] = CANDLES_RISING      # 标的一路上行 → 9ema不破, runner该留着
    s.send("$HOOD 7/24 $120 calls $.83")
    s.run(ticks=10)

    p = s.pos(HOOD_120) or {}
    f = []
    f += expect_eq(p.get("status"), "open", "跑完后仓位应仍为 open(runner在手)")
    f += expect_ge(len(s.evs("tp_fill")), 1, "首档止盈(+30%)应成交")
    f += expect_eq(len(s.evs("tp2_fill")), 0, "单档策略: 不该再有二档止盈成交")
    f += expect_eq(p.get("armed"), True, "剩仓摸到+60%后 runner 应武装9ema (armed=True)")
    # 首档卖½, 剩½ runner
    filled, sold = p.get("filled", 0), p.get("sold", 0)
    f += expect_ge(filled, 2, "常规单张数应≥2才谈得上对半分")
    f += expect(abs(sold - round(filled * 0.5)) <= 1,
                f"首档应卖≈½仓: filled={filled} sold={sold}")
    f += expect_ge(_remain(s, HOOD_120), 1, "runner 应至少保留1张")
    f += expect_eq(s.broker_pos(HOOD_120), _remain(s, HOOD_120), "runner 张数应与券商侧一致")
    # 新策略核心: 首档止盈后止损移到入场价(保本)
    f += expect(B._stop_price(p) >= p.get("avg", 0) * 0.999,
                f"首档止盈后止损应移到保本(入场价): stop={B._stop_price(p):.3f} avg={p.get('avg')}")
    f += expect_not_in("ema_exit", [e.get("ev") for e in s.events],
                       "标的未破9ema, 不该有9ema出场")
    return f


def sc_tp1_only_no_arm(s):
    """只涨到+35%: 首档成交、未摸+60%不武装(armed保持False), runner靠保本止损守着。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.95, 1.10, 1.12, 1.12, 1.05, 1.00, 1.00, 1.00])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.quotes.candles["HOOD"] = CANDLES_RISING
    s.send("$HOOD 7/24 $120 calls $.83")
    s.run(ticks=10)

    p = s.pos(HOOD_120) or {}
    f = []
    f += expect_ge(len(s.evs("tp_fill")), 1, "首档止盈(+30%)应成交")
    f += expect_eq(len(s.evs("runner_armed")), 0, "最高仅+35%, 未摸+60%不该武装")
    f += expect(not p.get("armed"), f"未摸到+60%, 不该武装9ema: armed={p.get('armed')}")
    f += expect_eq(p.get("status"), "open", "仓位应仍为 open")
    f += expect_ge(_remain(s, HOOD_120), 1, "首档后应还有剩仓(runner)")
    # 未武装的 runner 靠保本止损守(首档已止盈 → 保本生效)
    f += expect(B._stop_price(p) >= p.get("avg", 0) * 0.999,
                "首档后剩仓应有保本止损兜底(即使未武装9ema)")
    return f


def sc_stop_loss_60pct(s):
    """直接跌穿-60%: 轮询止损触发, 全部平掉, 券商侧净持仓归零。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.70, 0.50, 0.30, 0.28, 0.28, 0.28, 0.28])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.quotes.candles["HOOD"] = CANDLES_RISING
    s.send("$HOOD 7/24 $120 calls $.83")
    s.run(ticks=9)

    p = s.pos(HOOD_120) or {}
    f = []
    f += expect_ge(len(s.evs("stop_trigger")), 1, "跌破成本×0.4应触发轮询止损")
    f += expect_eq(p.get("status"), "closed", "止损后仓位应标记 closed")
    f += expect_eq(s.broker_pos(HOOD_120), 0, "止损后券商侧净持仓应归零")
    f += expect_eq(len(s.broker.live_orders(HOOD_120)), 0, "止损后不该残留任何活单")
    f += expect_eq(len(s.evs("tp_fill")), 0, "一路下跌, 不该有止盈成交")
    return f


def sc_armed_ema_exit(s):
    """武装后标的15分9ema连破2根 → runner全平(9ema读的是标的不是期权)。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 1.10, 1.35, 1.40, 1.30, 1.30, 1.30, 1.30, 1.30])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.quotes.candles["HOOD"] = CANDLES_BREAK2      # 标的连破2根9ema
    s.send("$HOOD 7/24 $120 calls $.83")
    s.run(ticks=12)

    p = s.pos(HOOD_120) or {}
    f = []
    f += expect_ge(len(s.evs("runner_armed")), 1, "先要摸到+60%把runner武装起来(单档: 靠价格摸+60%, 不靠二档成交)")
    f += expect_ge(len(s.evs("ema_exit")), 1, "武装后标的连破2根9ema应触发出场")
    ev = (s.evs("ema_exit") or [{}])[0]
    f += expect_eq(ev.get("ticker"), "HOOD", "9ema出场应记录【标的】ticker")
    f += expect_ge(ev.get("break_count", 0), 2, "破位根数应≥MECH_EMA_N=2")
    f += expect_eq(p.get("status"), "closed", "9ema出场后仓位应 closed")
    f += expect_eq(s.broker_pos(HOOD_120), 0, "9ema出场后券商侧应清仓")
    return f


def sc_unarmed_ema_no_exit(s):
    """未武装时标的9ema破位不应出场(防早盘回踩误洗肥尾 — 定稿关键)。"""
    # 期权始终在 +20% 以内, 从未摸到 +60% → 不武装
    s.quotes.set_path(HOOD_120, [0.83, 0.85, 0.95, 1.00, 1.00, 0.95, 0.95, 0.95, 0.95, 0.95])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.quotes.candles["HOOD"] = CANDLES_BREAK2      # 标的已经连破2根, 但runner没武装
    s.send("$HOOD 7/24 $120 calls $.83")
    s.run(ticks=10)

    p = s.pos(HOOD_120) or {}
    f = []
    f += expect(not p.get("armed"), f"从未摸到+60%, armed应为假: {p.get('armed')}")
    f += expect_eq(len(s.evs("ema_exit")), 0, "未武装时9ema破位【不应】出场")
    f += expect_eq(len(s.evs("exit_submit")), 0, "未武装时不该发出任何平仓卖单")
    f += expect_eq(p.get("status"), "open", "仓位应原封不动继续持有")
    f += expect_ge(_remain(s, HOOD_120), 1, "仓位应还在手上")
    return f


# ══════════════════════════════════════════════════════════════════════════
# 6-8: 到期强平 / 入场TTL / 迟到闸
# ══════════════════════════════════════════════════════════════════════════

def sc_expiry_force_close(s):
    """到期日 15:40 ET 之后 → 剩仓强平(防归零/行权)。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.quotes.candles["HOOD"] = CANDLES_RISING
    s.send("$HOOD 7/24 $120 calls $.83")
    s.run(ticks=3)                                  # 先在 7/20 建好仓
    held_before = _remain(s, HOOD_120)

    s.clock.set_et(2026, 7, 24, 15, 45)             # 到期日盘尾, 过了15:40强平线
    s.run(ticks=4)

    p = s.pos(HOOD_120) or {}
    f = []
    f += expect_ge(held_before, 1, "跳到到期日之前应先真的持有仓位")
    f += expect_in("到期强平", _reasons(s, "exit_submit"), "应以【到期强平】为由发出平仓单")
    f += expect_eq(p.get("status"), "closed", "到期强平后仓位应 closed")
    f += expect_eq(s.broker_pos(HOOD_120), 0, "到期强平后券商侧应清仓")
    f += expect_eq(len(s.broker.live_orders(HOOD_120)), 0, "到期强平后不该残留活单(含GTC止盈腿)")
    return f


def sc_entry_ttl_cancel(s):
    """入场限价单挂满TTL(20分)未成交 → 撤单, 仓位closed, 券商侧无活单。"""
    # 价格全程高于限价 0.83 → 买单永远碰不到, 只能被TTL撤掉(不接刀)
    s.quotes.set_path(HOOD_120, [1.50] * 6)
    s.quotes.open_interest[HOOD_120] = 500_000
    s.send("$HOOD 7/24 $120 calls $.83")
    p0 = s.pos(HOOD_120) or {}
    submitted = p0.get("entry_order_id")
    s.run(ticks=22)                                  # 22分钟 > ENTRY_TTL_SEC(1200s)

    p = s.pos(HOOD_120) or {}
    f = []
    f += expect(submitted is not None, "应先真的提交了入场限价单")
    f += expect_ge(len(s.evs("entry_ttl_cancel")), 1, "挂满20分钟未成交应触发TTL撤单")
    f += expect_eq(p.get("filled", 0), 0, "全程不触价, 不该有任何成交")
    f += expect_eq(p.get("status"), "closed", "TTL撤单后仓位应标 closed")
    f += expect(p.get("entry_order_id") is None, "TTL撤单后应清掉 entry_order_id")
    f += expect_eq(len(s.broker.live_orders(HOOD_120)), 0, "TTL撤单后券商侧不该还有活单")
    f += expect_eq(s.broker_pos(HOOD_120), 0, "从未成交, 券商侧持仓应为0")
    return f


def sc_stale_signal_no_entry(s):
    """迟到信号(消息时间距今>180s) → 严禁补单(用户铁律)。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.83, 0.83])
    s.quotes.open_interest[HOOD_120] = 500_000
    # 直接喂一个 10 分钟前的消息时间戳 (s.send 只会用"当下", 测不到迟到闸)
    late = s.clock.now(ET) - timedelta(minutes=10)
    B._handle("$HOOD 7/24 $120 calls $.83", s.clock.now(ET).date(), 991001,
              s.seen, s.positions, late)
    s.run(ticks=3)

    f = []
    f += expect_ge(len(s.evs("stale_buy_skipped")), 1, "迟到10分钟的买入信号应被迟到闸拦下")
    f += expect_eq(len(s.evs("entry_submit")), 0, "迟到信号【绝不允许】补单")
    f += expect_not_in(HOOD_120, s.positions, "迟到信号不该建任何仓位记录")
    f += expect_eq(len(_buy_orders(s)), 0, "券商侧不该收到任何买单")
    return f


# ══════════════════════════════════════════════════════════════════════════
# 9-11: 仓位档位 (常规 ½ / lotto·hedge ⅓ / 0DTE ⅒ / 缺到期)
# ══════════════════════════════════════════════════════════════════════════

def sc_zero_dte_tenth_size(s):
    """0DTE信号走⅒仓位档(ZERO_DTE_FRAC=0.10), 张数明显小于常规单。"""
    o0 = osi("HOOD", "260720", "C", 120)             # 0DTE = 当天(2026-07-20)
    s.quotes.set_path(o0, [0.83, 0.83, 0.83, 0.83])
    s.quotes.open_interest[o0] = 500_000
    s.send("$HOOD 0DTE $120 calls $.83")

    p = s.pos(o0) or {}
    qty = p.get("qty", 0)
    want_0dte = int((EQUITY * 0.10) // (LIMIT * 100))     # ⅒档 → 120张
    want_reg = int((EQUITY * 0.50) // (LIMIT * 100))      # 常规½档 → 602张
    f = []
    f += expect(p, f"0DTE信号应建仓 {o0}")
    f += expect_eq(qty, want_0dte, "0DTE应按净值×ZERO_DTE_FRAC(0.10)定张")
    f += expect_le(qty, want_reg // 3, f"0DTE张数应明显小于常规单(常规≈{want_reg}张)")
    f += expect_eq(p.get("expiry"), "2026-07-20", "0DTE到期应=信号当天")
    return f


def sc_lotto_hedge_third_size(s):
    """lotto / hedge 信号走⅓档(LOTTO_FRAC=0.3333)。"""
    o_lotto = osi("HOOD", "260724", "C", 120)
    o_hedge = osi("HOOD", "260724", "C", 125)
    for o in (o_lotto, o_hedge):
        s.quotes.set_path(o, [0.83, 0.83, 0.83, 0.83])
        s.quotes.open_interest[o] = 500_000
    s.send("Lotto $HOOD 7/24 $120 calls $.83")
    s.send("$HOOD 7/24 $125 calls $.83 hedge")

    want_lotto = int((EQUITY * 0.3333) // (LIMIT * 100))   # ⅓档 → 401张
    want_reg = int((EQUITY * 0.50) // (LIMIT * 100))
    pl, ph = s.pos(o_lotto) or {}, s.pos(o_hedge) or {}
    f = []
    f += expect(pl, f"lotto信号应建仓 {o_lotto}")
    f += expect(ph, f"hedge信号应建仓 {o_hedge} (2026-07-19起hedge按lotto小仓跟, 不再跳过)")
    f += expect_eq(pl.get("qty"), want_lotto, "lotto应按净值×LOTTO_FRAC(⅓)定张")
    f += expect_eq(ph.get("qty"), want_lotto, "hedge应按lotto档(⅓)定张")
    f += expect_le(pl.get("qty", 10**9), want_reg, "lotto档张数应小于常规½档")
    f += expect_ge(len(s.evs("hedge_as_lotto")), 1, "hedge单应留下 hedge_as_lotto 事件")
    return f


def sc_noexpiry_lotto_follow(s):
    """BUY_NOEXPIRY(缺到期日) → 推断到最近周五并按lotto档直接跟。"""
    # 2026-07-20 是周一 → 推断到期 = 本周五 7/24
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.83, 0.83])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.send("$HOOD $120 calls $.83")                  # 无 weekly / 无 M/D / 无 0DTE

    p = s.pos(HOOD_120) or {}
    want_lotto = int((EQUITY * 0.3333) // (LIMIT * 100))
    f = []
    f += expect_ge(len(s.evs("noexpiry_accept")), 1, "缺到期信号应走 BUY_NOEXPIRY 接受分支")
    f += expect(p, f"BUY_NOEXPIRY 应直接建仓 {HOOD_120} (ARM 0DTE漏单教训)")
    f += expect_eq(p.get("expiry"), "2026-07-24", "缺到期应推断为最近周五 7/24")
    f += expect_eq(p.get("qty"), want_lotto, "BUY_NOEXPIRY 应按lotto档(⅓)定张, 不能按常规½档")
    f += expect_ge(len(s.evs("entry_submit")), 1, "应真的提交了入场单")
    return f


# ══════════════════════════════════════════════════════════════════════════
# 12-13: 去重
# ══════════════════════════════════════════════════════════════════════════

def sc_dup_same_day_skip(s):
    """同一合约当天重复信号 → 跳过, 不重复建仓。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.83, 0.83])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.send("$HOOD 7/24 $120 calls $.83", msg_id=700001)
    s.send("$HOOD 7/24 $120 calls $.83", msg_id=700002)   # 站长重发同一条

    p = s.pos(HOOD_120) or {}
    f = []
    f += expect_eq(len(s.evs("entry_submit")), 1, "同合约当天只该建仓一次")
    f += expect_eq(len(_buy_orders(s)), 1, "券商侧只该收到一张买单")
    f += expect_eq(p.get("qty"), int((EQUITY * 0.50) // (LIMIT * 100)),
                   "仓位记录应保持第一次建仓的张数")
    return f


def sc_dup_cross_day_open_skip(s):
    """同一合约跨日复述、但前一笔仍未平仓 → 跳过建仓, 且旧仓位记录不被冲掉。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.90, 0.90, 0.90, 0.90])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.quotes.candles["HOOD"] = CANDLES_RISING
    s.send("$HOOD 7/24 $120 calls $.83", msg_id=710001)
    s.run(ticks=3)                                   # 成交, 挂上保护腿
    before = dict(s.pos(HOOD_120) or {})

    # 次日站长又复述同一合约 (去重key含日期 → 消息级/交易级去重都拦不住, 只能靠持仓守卫)
    s.send("$HOOD 7/24 $120 calls $.83", msg_id=710002, at_et=(2026, 7, 21, 10, 0))
    after = s.pos(HOOD_120) or {}

    f = []
    f += expect_ge(before.get("filled", 0), 1, "第一笔应已真实成交")
    f += expect_ge(len(s.evs("dup_open_skip")), 1, "跨日复述未平仓合约应命中重复建仓守卫")
    f += expect_eq(len(s.evs("entry_submit")), 1, "不该为同一未平仓合约再提交入场单")
    f += expect_eq(len(_buy_orders(s)), 1, "券商侧买单仍应只有一张")
    # 旧账本不能被 positions[osi]=dict(...) 整条覆盖
    f += expect_eq(after.get("filled"), before.get("filled"), "旧仓位 filled 不该被覆盖清零")
    f += expect_eq(after.get("avg"), before.get("avg"), "旧仓位 avg 不该被覆盖清零(止损前置条件)")
    f += expect_eq(after.get("opened"), before.get("opened"), "旧仓位建仓日不该被改写")
    f += expect_eq(after.get("tp_order_id"), before.get("tp_order_id"),
                   "旧的GTC止盈腿ID不该丢失(否则变孤儿单)")
    return f


# ══════════════════════════════════════════════════════════════════════════
# 14-16: 风控闸门
# ══════════════════════════════════════════════════════════════════════════

def sc_gross_exposure_cap(s):
    """总敞口闸: 连发多条常规信号, 累计在险权利金不得超过 净值×MAX_GROSS_FRAC。"""
    strikes = [120, 125, 130, 135]
    syms = [osi("HOOD", "260724", "C", k) for k in strikes]
    for o in syms:
        s.quotes.set_path(o, [0.83, 0.83, 0.83, 0.83])
        s.quotes.open_interest[o] = 500_000
    for i, k in enumerate(strikes):
        s.send(f"$HOOD 7/24 ${k} calls $.83", msg_id=720000 + i)

    # 在险权利金 = Σ 计划张数 × 限价 × 100 (全部还挂在途, sold=0)
    gross = sum((p.get("qty", 0) - p.get("sold", 0)) * p.get("limit", 0) * 100
                for p in s.positions.values() if p.get("status") in B.ACTIVE_STATUSES)
    cap = EQUITY * B.MAX_GROSS_FRAC
    opened = [o for o in syms if o in s.positions]

    f = []
    f += expect_le(gross, cap, f"累计在险权利金不得超过 净值×{B.MAX_GROSS_FRAC:.0%}")
    f += expect_ge(len(s.evs("gross_cap_skip")), 1, "额度用尽后应有信号被敞口闸拒掉")
    f += expect_le(len(opened), 3, "净值×100% / (每笔≈50%) 最多容得下2-3笔常规单")
    f += expect_ge(len(opened), 2, "敞口闸不该把前面正常的信号也一并拒掉")
    f += expect_not_in(syms[-1], s.positions, "额度耗尽后最后一条信号不该建仓")
    return f


def sc_oi_cap_limits_qty(s):
    """OI很小时张数被OI帽(10%)压下来, 防模拟盘假成交失真。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.83, 0.83])
    s.quotes.open_interest[HOOD_120] = 300           # OI只有300 → 帽 = 30张
    s.send("$HOOD 7/24 $120 calls $.83")

    p = s.pos(HOOD_120) or {}
    want_budget_qty = int((EQUITY * 0.50) // (LIMIT * 100))     # 不设帽时会下 602 张
    want_cap = max(1, int(300 * B.OI_CAP_PCT))                  # = 30
    f = []
    f += expect(p, "OI小但仍应建仓(只压张数, 不拒单)")
    f += expect_eq(p.get("qty"), want_cap, f"张数应被OI帽压到 OI×{B.OI_CAP_PCT:.0%}")
    f += expect_le(p.get("qty", 10**9), want_budget_qty,
                   f"OI帽后张数必须小于纯预算张数({want_budget_qty})")
    return f


def sc_premium_over_max_rejected(s):
    """权利金 > MAX_PREMIUM($5) 上限 → 拒绝下单。"""
    o = osi("HOOD", "260724", "C", 120)
    s.quotes.set_path(o, [5.50, 5.50, 5.50])
    s.quotes.open_interest[o] = 500_000
    s.send("$HOOD 7/24 $120 calls $5.50")            # 5.50 > MAX_PREMIUM=5.0
    s.run(ticks=3)

    f = []
    f += expect_eq(len(s.evs("entry_submit")), 0, "超上限权利金不该提交入场单")
    f += expect_not_in(o, s.positions, "超上限权利金不该建仓位记录")
    f += expect_eq(len(_buy_orders(s)), 0, "券商侧不该收到任何买单")
    f += expect(any("权利金" in l and "上限" in l for l in s.logs),
                "应留下【权利金超上限拒绝】的日志痕迹")
    return f


# ══════════════════════════════════════════════════════════════════════════
# 17: 保护腿连续性
# ══════════════════════════════════════════════════════════════════════════

def sc_protection_after_tp1(s):
    """首档止盈成交后剩余仓位不能"卖完就裸奔" —— 单档策略下保护 = 保本止损轮询。"""
    s.quotes.set_path(HOOD_120, [0.83, 0.83, 0.95, 1.10, 1.12, 1.10, 1.05, 1.05, 1.05])
    s.quotes.open_interest[HOOD_120] = 500_000
    s.quotes.candles["HOOD"] = CANDLES_RISING
    s.send("$HOOD 7/24 $120 calls $.83")
    s.run(ticks=9)

    p = s.pos(HOOD_120) or {}
    remain = _remain(s, HOOD_120)
    sells = _live_sells(s, HOOD_120)
    sell_qty = sum(o.submitted_quantity - o.executed_quantity for o in sells)
    f = []
    f += expect_ge(len(s.evs("tp_fill")), 1, "首档止盈应已成交")
    f += expect_ge(remain, 1, "首档止盈后应还有剩仓(runner)")
    f += expect_le(sell_qty, remain, "挂出的卖单张数不得超过剩仓(超了=裸空)")
    # 单档策略: 首档后不再挂二档卖单, 保护改由【保本止损轮询】承担
    f += expect(p.get("tp2_order_id") is None, "单档策略: 不该有二档止盈腿")
    f += expect(B._stop_price(p) >= p.get("avg", 0) * 0.999,
                f"首档后止损应移到保本(入场价), 剩仓不裸奔: stop={B._stop_price(p):.3f} avg={p.get('avg')}")
    f += expect_ge(p.get("stop_mult", 0), 0.01, "止损通道应仍然武装(stop_mult>0)")
    f += expect_eq(len(s.evs("stop_trigger")), 0, "价格未跌回入场价, 不该误触保本止损")
    return f
