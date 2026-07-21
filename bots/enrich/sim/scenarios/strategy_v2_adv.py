#!/usr/bin/env python3
"""sim/scenarios/strategy_v2_adv.py — 对抗 2026-07-21 单档保本策略, 【找 bug】。

每个场景断言"正确/安全行为", bot 错了就红。攻击面(主 agent 未覆盖的):
  · 保本时序: 首档止盈前绝不能保本; 首档后价格跌回入场价必须触发
  · 保本 whipsaw: 同轮止盈+保本止损, 不重复卖不裸空
  · 武装口径: bot 用 last_done 判 +60% 武装(回测用 bar.high), high 穿但收盘没穿会漏武装
  · 武装先于首档: last 直接摸 +60% 但 reduced 尚未置位时, 9ema 出场会不会把全仓当 runner 平
  · 隔离: 老仓位(无 be)全生命周期都不保本/不被就地加 be
  · 卖½ 取整 + 部分入场 + 崩溃重启下 be 持久
通用不变式(裸空/账实/孤儿/弃仓)由 runner 自动跑。
"""
import discord_enrich_bot as B
from sim.scenario_api import (expect, expect_eq, expect_le, expect_ge,
                              expect_not_in, osi)

TK = "HOOD"
OSI = osi("HOOD", "260724", "C", 120)
BUY = "$HOOD 7/24 $120 calls $1.00"
FLAT = (1.00, 1.00, 1.00)


def _open_new(s, equity=1200.0, oi=500_000):
    """走真实链路开一个新策略仓(带 be=True)。equity=1200 → 6张。"""
    B._rl_until = 0.0
    s.broker.equity = equity
    s.quotes.set_path(OSI, [FLAT])
    s.quotes.open_interest[OSI] = oi
    s.send(BUY)
    s.tick()
    return s.pos(OSI)


def _downtrend_candles(n=20):
    """强下跌 15m 收盘序列 —— 保证最近若干根都在 9ema 下方(break_count 大)。"""
    return [float(x) for x in range(n, 0, -1)]


def _uptrend_candles(n=20):
    """强上涨 —— 最近的 close 都在 9ema 上方(break_count=0, 不触 9ema 出场)。"""
    return [float(x) for x in range(1, n + 1)]


# ══════════════════════════════════════════════════════════════════════════
# 1. 保本时序 —— 首档止盈【前】绝不保本
# ══════════════════════════════════════════════════════════════════════════

def sc_adv_no_breakeven_before_tp1(s):
    """首档止盈前 be=True 但 reduced=False → 止损必须仍是 -50%, 不是保本(入场价)。

    若 _stop_price 误在 reduced 前就保本, 价格跌到入场价下方一点(-2%)就会清仓 ——
    等于从没止盈过就在 -2% 割肉。断言: -2% 不触发, -50% 才触发。
    """
    fails = []
    p = _open_new(s)
    fails += expect(bool(p) and p.get("be") is True, "新仓应带 be=True")
    fails += expect(not p.get("reduced"), "开仓时尚未首档止盈")
    if p:
        fails += expect(abs(B._stop_price(p) - p["avg"] * 0.5) < 1e-6,
                        f"首档前止损必须=-50%(avg×0.5), 实际 {B._stop_price(p):.3f}")
    # 跌到 -2%(0.98): 远在 -50% 止损线之上, 不能触发任何出场
    s.quotes.set_path(OSI, [(0.98, 0.98, 0.98)] * 4)
    s.tick(n=4)
    fails += expect_eq(s.evs("stop_trigger"), [],
                       "首档止盈前跌到 -2% 绝不能触发保本止损(未止盈就 -2% 割肉=保本逻辑越界)")
    fails += expect_eq(s.broker_pos(OSI), 6, "-2% 时应仍持有全部 6 张")
    return fails


def sc_adv_crash_before_tp1_stops_at_minus50(s):
    """首档止盈前直接崩到 -55%: 必须按 -50% 触发止损并平净(不是保本, 不裸奔)。"""
    fails = []
    p = _open_new(s)
    s.quotes.set_path(OSI, [(0.45, 0.45, 0.45)] * 3)   # -55%
    s.tick(n=3)
    fails += expect_ge(len(s.evs("stop_trigger")), 1, "-55% 必须触发 -50% 止损")
    # 触发的是 -50% 而非保本: stop_px 应约 0.50 而非 1.00
    st = s.evs("stop_trigger")
    if st:
        fails += expect(abs(st[0].get("stop_px", 0) - 0.50) < 0.02,
                        f"首档前止损价应≈0.50(-50%)而非保本1.00: 实际 {st[0].get('stop_px')}")
        fails += expect(st[0].get("be") is False,
                        f"首档前不应标记为保本止损: be={st[0].get('be')}")
    s.run(ticks=3)
    fails += expect_eq(s.broker_pos(OSI), 0, "止损后应平净")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 2. 保本触发全生命周期 —— 首档后跌回入场价, 平净不超卖
# ══════════════════════════════════════════════════════════════════════════

def sc_adv_breakeven_full_lifecycle(s):
    """首档 +30% 卖½ → 保本激活 → 跌回入场价 → 保本止损平净。全程 sold≤filled, 券商不裸空。"""
    fails = []
    p = _open_new(s)
    fails += expect_eq(p["filled"], 6, "应成交 6 张")
    # 摸 +30% 触首档(GTC限价 @1.30 卖3张)
    s.quotes.set_path(OSI, [(1.35, 1.40, 1.30)])
    s.tick()
    p = s.pos(OSI)
    fails += expect_ge(len(s.evs("tp_fill")), 1, "首档止盈应成交")
    fails += expect(p.get("reduced") is True, "首档全成交后 reduced 应置位")
    fails += expect(abs(B._stop_price(p) - p["avg"]) < 1e-6,
                    f"首档后止损应=入场价(保本): {B._stop_price(p):.3f} avg={p['avg']}")
    fails += expect_eq(p.get("sold"), 3, "首档应只卖 3 张(卖½)")
    # 跌回入场价 → 保本止损
    s.quotes.set_path(OSI, [(1.00, 1.00, 1.00)] * 2)
    s.tick(n=2)
    p = s.pos(OSI)
    fails += expect_ge(len(s.evs("stop_trigger")), 1, "跌回入场价应触发保本止损")
    bt = [e for e in s.evs("stop_trigger") if e.get("be")]
    fails += expect_ge(len(bt), 1, "触发的应是【保本】止损(be=True)")
    s.run(ticks=3)
    fails += expect_eq(s.broker_pos(OSI), 0, "保本平净后券商侧应为 0")
    fails += expect_eq(p.get("sold"), p.get("filled"), "sold 应收敛到 filled(不多不少)")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 3. 武装口径 —— bot 用 last_done, 回测用 bar.high
# ══════════════════════════════════════════════════════════════════════════

def sc_adv_arming_high_only_no_arm(s):
    """一根 bar high 穿 +60% 但 last_done 收在 +60% 下方 —— bot 不武装(保守)。

    这是主 agent 让我判断的口径差异。这里【断言 bot 的实际行为】以坐实差异存在:
    武装靠 last_done, 所以 high 穿而 last 未穿时 armed 应保持 False。
    (若断言反了变红, 说明 bot 其实按 high 武装; 绿则坐实"漏武装"差异客观存在。)
    """
    fails = []
    p = _open_new(s)
    # 先首档止盈(否则 reduced 前的语义另说), 用 last=high=1.35 一次穿 +30%
    s.quotes.set_path(OSI, [(1.35, 1.35, 1.35)])
    s.tick()
    p = s.pos(OSI)
    fails += expect(p.get("reduced") is True, "首档应已止盈")
    # 现在造一根 high=1.70(>+60%) 但 last_done=1.45(<+60%) 的 bar
    s.quotes.set_path(OSI, [(1.45, 1.70, 1.40)] * 2)
    s.tick(n=2)
    p = s.pos(OSI)
    fails += expect(not p.get("armed"),
                    f"high 穿 +60% 但 last_done 没穿 → bot(按 last_done)不武装: armed={p.get('armed')}")
    fails += expect_eq(s.evs("runner_armed"), [], "不应记 runner_armed 事件")
    # 反证: last_done 真穿 +60% 时必须武装
    s.quotes.set_path(OSI, [(1.65, 1.65, 1.65)] * 2)
    s.tick(n=2)
    p = s.pos(OSI)
    fails += expect(p.get("armed") is True, "last_done 真穿 +60% 后必须武装")
    return fails


def sc_adv_armed_then_reverse_caught_by_breakeven(s):
    """runner 武装(last 摸 +60%)后价格反转跌回入场价, 9ema 不破 —— 保本止损必须兜住, 不裸奔。"""
    fails = []
    p = _open_new(s)
    s.quotes.candles[TK] = _uptrend_candles()      # 9ema 不破(上升趋势)
    # 首档 + 武装
    s.quotes.set_path(OSI, [(1.35, 1.35, 1.35), (1.65, 1.65, 1.65), (1.65, 1.65, 1.65)])
    s.tick(n=3)
    p = s.pos(OSI)
    fails += expect(p.get("reduced") is True, "首档应止盈")
    fails += expect(p.get("armed") is True, "应已武装(last 摸 +60%)")
    fails += expect_eq(s.evs("ema_exit"), [], "9ema 不破 → 不应有 9ema 出场")
    # 反转跌回入场价: 保本止损必须触发(runner 不能跌破入场价还没人管)
    s.quotes.set_path(OSI, [(1.00, 1.00, 1.00)] * 2)
    s.tick(n=2)
    p = s.pos(OSI)
    fails += expect_ge(len(s.evs("stop_trigger")), 1, "武装后跌回入场价, 保本止损必须兜住")
    s.run(ticks=3)
    fails += expect_eq(s.broker_pos(OSI), 0, "最终应平净, 不留裸奔 runner")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 4. 武装先于首档 —— last 直接摸 +60% 而 reduced 尚未置位
#    (若可达: 9ema 出场会把全仓当 runner 平, 跳过 +30% 首档卖½)
# ══════════════════════════════════════════════════════════════════════════

def sc_adv_arm_before_tp1_no_oversell(s):
    """价格一步摸 +60% 触发首档(1.30限价)成交 + 武装同轮 —— 无论先后, 绝不超卖/裸空。

    构造: 一根 bar last=high=1.65 直接穿 +30% 与 +60%。首档 GTC 限价 @1.30 会成交(high≥1.30),
    ④ 同轮 last=1.65 触发武装。断言两条通道合计卖出量绝不超过持仓。
    """
    fails = []
    p = _open_new(s)
    s.quotes.candles[TK] = _uptrend_candles()
    s.quotes.set_path(OSI, [(1.65, 1.65, 1.65)] * 2)
    s.tick(n=2)
    p = s.pos(OSI)
    # 首档卖½=3, 武装, 剩 3 runner。无论顺序, sold≤filled 且在挂卖单≤持仓
    live_sell = sum(o.submitted_quantity - o.executed_quantity
                    for o in s.broker.live_orders(OSI) if o.side == "Sell")
    fails += expect_le(live_sell, s.broker_pos(OSI), "在挂卖单不得超过券商持仓")
    fails += expect_le(p.get("sold", 0), p.get("filled", 0), "sold 不得超过 filled")
    fails += expect_le(p.get("sold", 0), 3, "首档最多卖½=3 张(未 9ema 出场时不应更多)")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 5. 隔离 —— 老仓位(无 be)全生命周期
# ══════════════════════════════════════════════════════════════════════════

def sc_adv_old_position_lifecycle_never_breakeven(s):
    """老仓位(无 be 键, stop_mult=0.4)跑首档止盈→跌回入场价: 绝不保本, 绝不被加 be,
    且价格在入场价上方(-60%止损线之上)时不能被平掉。"""
    fails = []
    B._rl_until = 0.0
    # 老仓位: 已建仓 open, 6张, 无 be 键, stop_mult=0.4, 有 GTC 一档止盈腿
    s.broker.position[OSI] = 6
    s.quotes.set_path(OSI, [(1.35, 1.40, 1.30)])   # 摸 +30% 触老仓一档
    s.positions[OSI] = dict(ticker=TK, filled=6, sold=0, avg=1.00, qty=6,
                            right="C", expiry="2026-07-24", strike=120.0,
                            entry_order_id=None, stop_mult=0.4,
                            reduced=False, tp1_done=False, armed=False,
                            tp_order_id=None, tp2_order_id=None,
                            status="open", opened="2026-07-20")
    s.tick()      # ①b 补挂止盈腿 + 若穿价则成交
    p = s.pos(OSI)
    fails += expect_not_in("be", p, "老仓位跑一轮不得被加 be 键")
    # 跌回入场价 1.00: 老仓 -60% 止损线=0.40, 1.00 远在上方 → 不能触发止损
    s.quotes.set_path(OSI, [(1.00, 1.00, 1.00)] * 3)
    s.tick(n=3)
    p = s.pos(OSI)
    fails += expect_not_in("be", p, "老仓位跌回入场价仍不得被加 be 键")
    fails += expect(abs(B._stop_price(p) - p["avg"] * 0.4) < 1e-6,
                    f"老仓位止损应始终 -60%(avg×0.4): {B._stop_price(p):.3f}")
    fails += expect_eq(s.evs("stop_trigger"), [],
                       "老仓位在入场价(远高于-60%线)绝不能触发止损 —— 若触发说明被新保本逻辑污染")
    fails += expect_eq(s.broker_pos(OSI), 6 - p.get("sold", 0), "券商持仓应与账本一致")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 6. 卖½ 取整 —— 1 张仓
# ══════════════════════════════════════════════════════════════════════════

def sc_adv_one_lot_tp1_closes_clean(s):
    """1 张仓摸 +30%: round(1×0.5)=0→max(1)=1 → 卖光整仓, 干净平仓无孤儿单。"""
    fails = []
    # equity=200 → 200×0.5//100 = 1 张
    p = _open_new(s, equity=200.0)
    fails += expect_eq(p["filled"], 1, f"应只成交 1 张, 实际 {p['filled']}")
    q1, _ = B._tp_params(1, p["avg"])
    fails += expect_eq(q1, 1, "1张仓首档应卖 1 张(round(0.5)=0 被 max(1) 兜住)")
    s.quotes.set_path(OSI, [(1.35, 1.40, 1.30)])
    s.tick()
    s.run(ticks=2)
    p = s.pos(OSI)
    fails += expect_eq(s.broker_pos(OSI), 0, "1张仓首档卖光后券商侧应为 0")
    fails += expect_eq(p.get("sold"), 1, "应卖出 1 张")
    fails += expect(p.get("status") == "closed", f"应干净平仓: status={p.get('status')}")
    fails += expect(not s.broker.live_orders(OSI), "平仓后不得留活单(孤儿)")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 7. 部分入场 + 保本
# ══════════════════════════════════════════════════════════════════════════

def sc_adv_partial_entry_then_breakeven(s):
    """入场只部分成交 → 已成交部分首档止盈 → 保本 → 跌回入场价平净。全程按已成交量算, 不超卖。"""
    fails = []
    B._rl_until = 0.0
    s.broker.equity = 6000.0                       # 计划 30 张
    s.broker.liquidity[OSI] = 4                    # 每 tick 最多成交 4 张
    s.quotes.set_path(OSI, [FLAT, (1.10, 1.10, 1.05)])  # 首格成交部分, 次格高于限价不再成交
    s.quotes.open_interest[OSI] = 500_000
    s.send(BUY)
    s.tick()
    s.broker.liquidity[OSI] = 0                    # 冻结让撤单确认
    s.tick()
    p = s.pos(OSI)
    filled = p["filled"]
    fails += expect(0 < filled < 30, f"应为部分成交: 实际 {filled}/30")
    fails += expect(p.get("be") is True, "部分入场的新仓仍应带 be=True")
    fails += expect_eq(filled, s.broker_pos(OSI), "filled 必须=券商持仓")
    del s.broker.liquidity[OSI]
    # 首档止盈
    s.quotes.set_path(OSI, [(1.35, 1.40, 1.30)])
    s.tick()
    p = s.pos(OSI)
    fails += expect(p.get("reduced") is True, "已成交部分应能首档止盈")
    fails += expect_le(p.get("sold", 0), filled, "首档卖出不得超过已成交量")
    fails += expect(abs(B._stop_price(p) - p["avg"]) < 1e-6, "首档后应保本")
    # 跌回入场价 → 保本
    s.quotes.set_path(OSI, [(1.00, 1.00, 1.00)] * 2)
    s.tick(n=2)
    s.run(ticks=2)
    fails += expect_eq(s.broker_pos(OSI), 0, "最终平净")
    fails += expect_le(p.get("sold", 0), filled, "总卖出不得超过已成交量(否则裸空)")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 8. 崩溃重启 —— be+reduced 落盘后保本仍生效
# ══════════════════════════════════════════════════════════════════════════

def sc_adv_breakeven_survives_restart(s):
    """新仓首档止盈后崩溃, 用落盘快照恢复 → 保本必须仍生效(be+reduced 都在, 跌回入场价触发)。"""
    fails = []
    p = _open_new(s)
    s.quotes.set_path(OSI, [(1.35, 1.40, 1.30)])
    s.tick()
    # 取最后一次落盘快照当作"重启后从磁盘读回的状态"
    snaps = [snap for name, snap in s.saved if OSI in snap]
    fails += expect(bool(snaps), "应至少落盘一次")
    if snaps:
        restored = {k: dict(v) for k, v in snaps[-1].items()}
        rp = restored[OSI]
        fails += expect(rp.get("be") is True, "重启恢复的仓位必须仍带 be=True")
        fails += expect(rp.get("reduced") is True, "重启恢复的仓位必须仍带 reduced=True")
        fails += expect(abs(B._stop_price(rp) - rp["avg"]) < 1e-6,
                        f"重启后保本止损必须仍=入场价: {B._stop_price(rp):.3f}")
        # 用恢复的状态继续驱动: 券商侧仍有 runner 3 张, 跌回入场价必须触发保本平净
        s.positions.clear()
        s.positions[OSI] = rp
        s.quotes.set_path(OSI, [(1.00, 1.00, 1.00)] * 2)
        s.tick(n=2)
        s.run(ticks=3)
        fails += expect_ge(len(s.evs("stop_trigger")), 1, "重启后跌回入场价必须触发保本止损")
        fails += expect_eq(s.broker_pos(OSI), 0, "重启后保本仍应把 runner 平净")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# 9. 【BUG】保护腿挂单丢响应(券商已收单/客户端超时) → 孤儿保护腿泄漏
#    根因: ensure_protection 的 _submit 遇 submit_but_lose_response 返回 ok=False,
#          bot 不记 tp_order_id, 但券商侧订单已存在 → 无人追踪的孤儿卖单。
#    下面三条断言【安全/正确行为】, bot 违反即红 —— 坐实缺陷。
# ══════════════════════════════════════════════════════════════════════════

def sc_adv_lostresp_no_orphan_protective_leg(s):
    """挂 tp1 保护腿时券商已收单但客户端丢响应 → 券商侧【不得】留下 bot 不追踪的孤儿卖单。

    bot 明确声称防御"券商已收单但客户端认为失败"(见 _live_sell_order docstring), 但该防御
    只覆盖【出场】提交, 不覆盖【保护腿】提交。后果: ensure_protection 挂 tp1 丢响应后, tp_order_id
    保持 None, 而同轮 ①b 再挂一张 → 券商侧出现【两张】tp1 限价卖单(合计=满仓), bot 只认 1 张。
    正确: 券商侧在挂卖单量应=bot 追踪的保护腿量(不多)。
    """
    fails = []
    B._rl_until = 0.0
    s.broker.equity = 1200.0
    s.quotes.open_interest[OSI] = 500_000
    s.quotes.candles[TK] = _uptrend_candles()
    s.quotes.set_path(OSI, [FLAT])
    s.send(BUY)
    s.broker.submit_but_lose_response = 1     # 入场成交后挂 tp1 时: 券商收单, 客户端丢响应
    s.tick()
    p = s.pos(OSI)
    live_sell = sum(o.submitted_quantity - o.executed_quantity
                    for o in s.broker.live_orders(OSI) if o.side == "Sell")
    tracked = int(p.get("tp_qty", 0) if p.get("tp_order_id") else 0)
    fails += expect_eq(live_sell, tracked,
                       f"券商侧在挂卖单({live_sell}张)必须=bot追踪的保护腿({tracked}张) —— "
                       "多出的是丢响应泄漏的孤儿 tp 腿, bot 不认识它")
    fails += expect_le(live_sell, p.get("filled", 0),
                       f"券商侧在挂卖单({live_sell})不得超过持仓({p.get('filled')})")
    # 后果坐实: 摸 +30% 时两张 tp 腿都成交 → 整仓被平, 单档卖½的 runner 被摧毁
    s.quotes.set_path(OSI, [(1.35, 1.40, 1.30)] * 2)
    s.tick(n=2)
    fails += expect(s.broker_pos(OSI) > 0,
                    f"+30% 后应保留½=runner, 实际券商持仓={s.broker_pos(OSI)}张 —— "
                    "双 tp 腿把整仓在 +30% 全平, runner 归零")
    return fails


def sc_adv_lostresp_orphan_causes_naked_short(s):
    """全真实链路: 保护腿丢响应留孤儿 → 崩盘出场(市价平、标closed)→ 孤儿残留 → 价格回弹成交 → 裸空。

    链路(只注入 bot 声称要防御的网络故障):
      ① 挂 tp1 丢响应 → 孤儿 tp1 限价卖单(½仓 @+30%)残留券商侧, bot 不追踪
      ② -50% 崩盘出场, 同一次网络故障令 today_orders 查询失败 → _live_sell_order 认领防线失灵
         → _start_exit 补发满仓市价卖 → 在崩盘价成交 → 券商归零, 仓位标 closed
      ③ 孤儿限价单仍活; 价格回弹到 +30% → 孤儿成交 → 券商净持仓转负 = 裸空期权(风险无上限)
         且仓位已 closed, manage_positions 永不再管它 = 弃仓的裸空。
    断言: 任何时刻券商净持仓 >= 0(不裸空)。
    """
    fails = []
    B._rl_until = 0.0
    s.broker.equity = 1200.0
    s.quotes.open_interest[OSI] = 500_000
    s.quotes.candles[TK] = _uptrend_candles()
    s.quotes.set_path(OSI, [FLAT])
    s.send(BUY)
    s.broker.submit_but_lose_response = 1
    s.tick()
    # -50% 崩盘 + today_orders 失败(同一次网络抖动)
    s.quotes.set_path(OSI, [(0.45, 0.45, 0.45)])
    s.broker.fail_today_orders = True
    s.tick()
    s.broker.fail_today_orders = False
    s.run(ticks=2)
    p = s.pos(OSI)
    # 价格回弹 +30%: 残留孤儿限价单成交
    s.quotes.set_path(OSI, [(1.35, 1.40, 1.30)])
    s.run(ticks=3)
    fails += expect_ge(s.broker_pos(OSI), 0,
                       f"券商净持仓必须>=0(裸空=未授权期权空头, 风险无上限): 实际={s.broker_pos(OSI)}")
    live = s.broker.live_orders(OSI)
    fails += expect(not (p and p.get("status") == "closed" and live),
                    f"仓位标 closed 后券商侧不得留活单(孤儿): status={p.get('status') if p else None} live={live}")
    return fails


def sc_adv_exit_never_claims_protective_leg(s):
    """出场时 _start_exit 的"认领在途卖单"绝不能把【保护腿】(tp 限价单)当成出场单认领。

    _live_sell_order 找到的孤儿 tp1 是 +30% 限价、半仓; 认领它当全平市价出场 → 位置进入假 closing,
    而 ④ 轮询止损要求 status==open → 止损被禁用; 该限价单在崩盘里永不成交 → 止损盲区裸奔至逃生舱。
    断言: 崩盘期间必须有真止损保护(不能因假 closing 而止损失效)。
    """
    fails = []
    B._rl_until = 0.0
    s.broker.position[OSI] = 6
    # 券商侧孤儿 tp1 限价卖单(3@1.30), bot 不追踪 —— 等价于挂 tp1 丢响应后的券商状态
    r = s.broker.submit_order(symbol=OSI, order_type="LO", side="Sell",
                              submitted_quantity=3, time_in_force="GoodTilCanceled",
                              submitted_price=1.30, remark="tp1")
    orphan = r.order_id
    s.positions[OSI] = dict(ticker=TK, filled=6, sold=0, avg=1.00, qty=6, right="C",
                            expiry="2026-07-24", strike=120.0, entry_order_id=None,
                            stop_mult=0.5, be=True, reduced=False, tp1_done=False,
                            armed=False, tp_order_id=None, tp2_order_id=None,
                            status="open", opened="2026-07-21")
    s.quotes.set_path(OSI, [(0.45, 0.45, 0.45)])   # -55% → 应触发 -50% 止损
    s.tick()
    p = s.pos(OSI)
    ex = s.broker.orders.get(p.get("exit_order_id")) if p.get("exit_order_id") else None
    if ex is not None:
        fails += expect(ex.order_id != orphan,
                        "出场单不得是被认领的孤儿 tp 限价腿(应是新提交的市价出场单)")
        fails += expect_eq(ex.order_type, "MO",
                           f"出场单必须是市价单(MO), 实际认领了 {ex.order_type}@{ex.price}(限价)")
    # 继续崩到 -80%: 无论如何必须有真止损, 不能因假 closing 而 6 张裸奔
    s.quotes.set_path(OSI, [(0.20, 0.20, 0.20)])
    s.run(ticks=4)
    fails += expect_ge(len(s.evs("stop_trigger")), 1,
                       "崩到 -80% 必须触发止损 —— 若 0 次说明认领孤儿致假 closing, ④ 止损被禁用(止损盲区)")
    fails += expect_eq(s.broker_pos(OSI), 0,
                       f"崩盘后应已止损平净, 实际券商仍持 {s.broker_pos(OSI)} 张裸奔")
    return fails
