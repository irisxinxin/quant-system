#!/usr/bin/env python3
"""sim/scenarios/adversarial.py — 对抗/故障链路端到端场景。

目标是【证伪】bot 自称的三条不变式:
  I1 提交成功≠成交, sold/closed 只能由 manage_positions ⓪ 依据 executed_quantity 更新
  I2 撤单只是"请求撤销", 未查到终态即视为挂单仍活着, 禁止在其上再发卖单
  I3 宁可少卖(亏损上限=权利金), 绝不多卖(裸空期权=无限风险)

约定(见 sim/scenario_api.py): 每个场景返回【失败原因列表】, 空列表=通过。
通用不变式(裸空/账实不符/孤儿单/弃仓/挂卖单超持仓)由 run_sim.py 自动跑, 这里只写场景特有预期。

关于假件时序(读 fakes.py 得出, 写场景必须心里有数):
  · s.tick() 一轮 = clock+60s → broker.tick() → bot manage_positions()
  · manage_positions 每个仓位开头 time.sleep(0.4) → 又推一次 broker.tick()
  · cancel_and_reconcile 内部 _await_terminal 最多再 sleep 5 次 → 又推 5 次 broker.tick()
  · FakeBroker.tick() 结尾无条件 quotes.step() —— 所以【行情下标在每次 broker.tick 都前进】,
    上面这几处 sleep 都会推进价格路径。撤单竞态就是靠这个精确构造出来的。
"""
from sim.scenario_api import (expect, expect_eq, expect_ge, expect_le, expect_in,
                              expect_not_in, osi)

TK = "HOOD"
OSI = osi("HOOD", "260724", "C", 120)      # HOOD 2026-07-24 $120 CALL
BUY = "$HOOD 7/24 $120 calls $1.00"        # 权利金$1.00 → 张数 = 净值×0.5 // 100

FLAT = (1.00, 1.00, 1.00)                  # 成本价横盘: 入场单可成交, 止盈/止损都不触
IDLE = (1.10, 1.10, 1.05)                  # 高于限价: 入场单不成交; 低于一档: 止盈不成交
CRASH = (0.30, 0.30, 0.30)                 # -70%: 触 -60% 止损
TP1HIT = (1.30, 1.30, 1.30)                # 恰好触一档止盈 $1.30
TP1SPIKE = (1.35, 1.40, 1.30)              # 高点穿一档


# ── 通用夹具 ──

def _isolate():
    """清掉 Sim.__enter__ 没有复位的 bot 模块级全局 —— 否则场景之间会互相污染。

    · _rl_until: 429退避截止时刻。假时钟每个场景都从同一时刻起跑, 所以上一个场景注入的
      限流退避会让【后续所有场景】的 _order_state/_option_last 直接返回 None(仓位永远卡 pending)。
    · _recent_exits: 出场文本10分钟去重表, 同样按假时钟计时, 跨场景同文本会被静默吞掉。
    """
    import discord_enrich_bot as B
    B._rl_until = 0.0
    B._recent_exits.clear()


def _signal(s, equity=1200.0, path=(FLAT,), oi=50000):
    """发一条标准买入信号(未推进行情)。equity 决定张数: 净值×0.5 // $100。"""
    _isolate()
    s.broker.equity = equity
    s.quotes.set_path(OSI, list(path))
    s.quotes.open_interest[OSI] = oi
    s.send(BUY)
    return s


def _open(s, equity=1200.0, path=(FLAT,), oi=50000):
    """建仓并跑满一轮: 入场单全成交 → status=open → 挂好 ⅓@+30% / ⅓@+60% 两条保护腿。"""
    _signal(s, equity=equity, path=path, oi=oi)
    s.tick()
    return s.pos(OSI)


def _sell_orders(s, sym=OSI):
    return [o for o in s.broker.orders.values() if o.side == "Sell" and o.symbol == sym]


def _buy_orders(s, sym=OSI):
    return [o for o in s.broker.orders.values() if o.side == "Buy" and o.symbol == sym]


def _sold_at_broker(s, sym=OSI):
    return sum(f[4] for f in s.broker.fills if f[2] == sym and f[3] == "Sell")


def _bought_at_broker(s, sym=OSI):
    return sum(f[4] for f in s.broker.fills if f[2] == sym and f[3] == "Buy")


def _live_sell_qty(s, sym=OSI):
    return sum(o.submitted_quantity - o.executed_quantity for o in s.broker.live_orders(sym)
               if o.side == "Sell")


def _has_protection(p):
    return bool(p.get("tp_order_id") or p.get("tp2_order_id") or p.get("stop_order_id"))


def _alerts_like(s, *words):
    return [a for a in s.alerts if any(w in a for w in words)]


# ══════════════════════════════════════════════════════════════════════════
# A 竞态类
# ══════════════════════════════════════════════════════════════════════════

def sc_cancel_race_leg_fills_during_poll(s):
    """撤单请求已发出、正在轮询确认终态时保护腿恰好成交 —— 成交量必须原样计入 sold 且只计一次。"""
    fails = []
    p = _open(s, equity=6000.0)          # 30张 → 一档10张@1.30 二档10张@1.60
    fails += expect_eq(p["filled"], 30, "入场应全成交30张")
    tp_oid = p.get("tp_order_id")
    fails += expect(bool(tp_oid), "开仓后应有一档止盈腿")

    # 关键构造: 前两格行情摸不到 $1.30(broker.tick + manage 的 sleep 各吃一格),
    # 第三格才穿价 —— 那一格正好落在 cancel_and_reconcile 的轮询 sleep 里。
    s.quotes.set_path(OSI, [IDLE, IDLE, TP1SPIKE])
    s.clock.set_et(2026, 7, 24, 15, 45)   # 到期日强平 → 触发 close_position 撤所有腿
    s.tick()

    tp = s.broker.orders.get(tp_oid)
    fails += expect_eq(tp.status, "Filled", "一档腿应在撤单轮询期内成交(竞态构造失败则场景无效)")
    fails += expect_eq(p.get("sold", 0), tp.executed_quantity,
                       "撤单竞态成交必须计入sold(既不能漏记=后续超卖, 也不能重复记=虚增已卖)")
    fails += expect_eq(p.get("tp_order_id_counted"), tp.executed_quantity, "该腿的已计数应等于实际成交")

    s.run(ticks=3)                        # 剩仓市价卖出收尾
    fails += expect_eq(s.broker_pos(OSI), 0, "收尾后券商侧应无持仓")
    fails += expect_eq(p.get("sold"), 30, "总卖出应恰好=总买入30张(不多不少)")
    return fails


def sc_entry_ttl_cancel_race_backfill(s):
    """入场单TTL撤单时订单在轮询期内成交 —— filled/avg 必须回填, 且已成交部分必须立刻有保护。"""
    fails = []
    _signal(s, equity=1200.0, path=(IDLE,))    # 6张 @$1.00, 行情一直高于限价 → 不成交
    s.run(ticks=19)                            # 19分钟: 尚未到 20 分钟 TTL
    p = s.pos(OSI)
    fails += expect_eq(p["status"], "pending", "19分钟时入场单应仍在途")
    fails += expect_eq(p["filled"], 0, "19分钟时不应有成交")

    # 第20分钟: 前两格仍不触价(①看到 exq=0 → 走 ④d TTL 撤单), 第三格穿价 → 撤单轮询期内成交
    s.quotes.set_path(OSI, [IDLE, IDLE, (1.00, 1.02, 0.98)])
    s.tick()

    fails += expect_eq(p["filled"], s.broker_pos(OSI), "TTL撤单竞态成交必须回填 filled(否则真实多头失联)")
    fails += expect_ge(p["filled"], 1, "本场景应确实成交(竞态构造失败则场景无效)")
    fails += expect(p.get("avg", 0) > 0,
                    f"竞态成交后 avg 必须回填: 实际 avg={p.get('avg')!r} —— avg=0 会让 "
                    "ensure_protection 早退、④轮询止损的 avg>0 前置失败, 该仓位止盈腿与"
                    "-60%止损【全部失效】, 且 entry_order_id 已清空后再无路径补回 avg")
    fails += expect(_has_protection(p),
                    "TTL竞态成交后必须立刻配上保护腿(实际一条都没有 → 全额权利金裸奔)")

    # 反证资金后果: 之后崩到 -85% 也不会止损
    s.quotes.set_path(OSI, [(0.15, 0.15, 0.15)])
    s.run(ticks=4)
    fails += expect(bool(s.evs("stop_trigger")) or p.get("status") == "closed",
                    "价格跌到 -85% 仍未触发任何止损/平仓 —— 该仓位只能等到期强平, "
                    "最坏情况归零(本笔全额权利金)")
    return fails


def sc_protective_leg_partial_then_cancelled(s):
    """保护腿部分成交后被撤(PartialWithdrawal) —— 已成交量必须记入 sold, 剩仓卖单不得超过持仓。"""
    fails = []
    p = _open(s, equity=6000.0)                # 30张, 一档10张@1.30
    s.broker.liquidity[OSI] = 1                # 每 broker.tick 最多成交1张
    s.quotes.set_path(OSI, [TP1HIT])
    s.tick()                                   # 一档腿部分成交(未完成)
    tp_oid = p.get("tp_order_id")
    partial = s.broker.orders[tp_oid].executed_quantity
    fails += expect(0 < partial < 10, f"一档腿应处于部分成交(实际{partial}/10)")

    s.broker.liquidity[OSI] = 0                # 冻结成交, 让撤单能干净落地
    s.clock.set_et(2026, 7, 24, 15, 45)
    s.tick()                                   # 到期强平 → 撤腿 → PartialWithdrawal

    fails += expect_eq(s.broker.orders[tp_oid].status, "PartialWithdrawal", "该腿应以部分撤销终结")
    fails += expect_eq(p.get("sold"), partial, "部分成交量必须记入 sold(漏记则剩仓被高估→超卖)")
    fails += expect_eq(p.get("exit_qty"), 30 - partial, "市价平仓张数必须按 filled-sold 算")
    fails += expect_le(_live_sell_qty(s), s.broker_pos(OSI), "在挂卖单不得超过券商侧持仓")

    del s.broker.liquidity[OSI]
    s.run(ticks=3)
    fails += expect_eq(s.broker_pos(OSI), 0, "收尾应平净")
    fails += expect_eq(_sold_at_broker(s), 30, "总卖出应恰好30张")
    return fails


def sc_exit_order_stuck_pending_cancel(s):
    """卖单永远卡在 PendingCancel —— 必须告警且绝不重复卖; 卡死解除后仓位应能回到正常管理。"""
    fails = []
    p = _open(s, equity=1200.0)
    s.broker.liquidity[OSI] = 0                # 卖单永不成交
    s.quotes.set_path(OSI, [CRASH])
    s.tick()                                   # -70% → 止损 → 市价卖在途
    exit_oid = p.get("exit_order_id")
    fails += expect(bool(exit_oid), "止损应已提交市价卖单")
    fails += expect_eq(p["status"], "closing", "卖单在途期间应为 closing")
    fails += expect_eq(p.get("sold", 0), 0, "I1: 提交成功不得记 sold")

    s.broker.stuck_cancel.add(exit_oid)        # 逃生舱撤单也撤不动
    s.run(ticks=8)                             # 超过 EXIT_STUCK_SEC=300s

    fails += expect_eq(len(_sell_orders(s)), 3, "卡死期间不得再发新卖单(应只有 tp1/tp2/exit 三张)")
    fails += expect(bool(_alerts_like(s, "卡死", "人工")),
                    "卖单卡死且仓位失去全部自动保护时必须告警")
    fails += expect_eq(p["status"], "closing", "撤不掉的卖单仍在途, 不得擅自改状态")
    fails += expect_eq(p.get("sold", 0), 0, "卡死期间不得凭猜测记账")

    # 解除卡死 → 逃生舱应把仓位捞回 open 并重配保护
    s.broker.stuck_cancel.discard(exit_oid)
    s.quotes.set_path(OSI, [FLAT])
    s.run(ticks=3)
    fails += expect_eq(p["status"], "open", "卡死解除后仓位必须回到正常管理, 不得永久卡在 closing")
    fails += expect(_has_protection(p), "回到 open 后必须重新配保护腿")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# B 故障注入类
# ══════════════════════════════════════════════════════════════════════════

def sc_fail_detail_during_stoploss(s):
    """查单持续 429 期间恰好该止损 —— 不得凭猜测下卖单, 也不得弃仓, 且必须告警止损通道失效。"""
    fails = []
    p = _open(s, equity=1200.0)
    before = len(_sell_orders(s))
    s.broker.fail_detail = 999                 # order_detail 全灭
    s.quotes.set_path(OSI, [CRASH])
    s.run(ticks=8)

    fails += expect_eq(len(_sell_orders(s)), before, "查不到订单状态时绝不能盲发卖单(I2)")
    fails += expect_not_in(p["status"], ("closed",), "查单失败不得弃仓")
    fails += expect_eq(p.get("sold", 0), 0, "没有 executed_quantity 佐证就不得记 sold(I1)")
    fails += expect_eq(s.evs("exit_submit"), [], "不得提交任何新卖单")
    fails += expect(bool(_alerts_like(s, "止损通道失效", "止损")),
                    "止损通道整段失效必须告警(否则用户以为 -60% 止损还在)")
    fails += expect_eq(s.broker_pos(OSI), 6, "券商侧持仓应原样保留")
    return fails


def sc_fail_cancel_blocks_new_sell(s):
    """撤单接口持续失败 —— 未撤净的挂单之上绝不允许再发卖单(I2), 且出场意图必须留待重试。"""
    fails = []
    p = _open(s, equity=1200.0)
    before = len(_sell_orders(s))
    s.broker.fail_cancel = 999
    s.quotes.set_path(OSI, [CRASH])
    s.run(ticks=5)

    fails += expect_eq(len(_sell_orders(s)), before, "撤单没撤净就不得再发卖单(旧腿+新单=超卖裸空)")
    fails += expect_eq(p.get("pending_action"), "full_exit", "推迟的平仓必须记为待办, 否则出场意图丢失")
    fails += expect_eq(p["status"], "open", "未发出卖单时不得进入 closing")
    fails += expect_le(_live_sell_qty(s), s.broker_pos(OSI), "在挂卖单不得超过持仓")

    s.broker.fail_cancel = 0                   # 故障恢复
    s.run(ticks=4)
    fails += expect_eq(s.broker_pos(OSI), 0, "撤单恢复后待办平仓必须真的执行")
    fails += expect_eq(p["status"], "closed", "平净后应标 closed")
    return fails


def sc_submit_lost_response_no_double_sell(s):
    """券商已收单但客户端超时(submit_but_lose_response) —— 下一轮绝不允许重复卖出同一批仓位。"""
    fails = []
    p = _open(s, equity=1200.0)                # 6张
    s.broker.submit_but_lose_response = 1
    s.quotes.set_path(OSI, [CRASH])
    s.tick()                                   # 止损 → 卖单进了券商, 客户端认为失败

    fails += expect(bool(s.evs("exit_submit_failed")), "本场景应确实触发一次'提交失败'(构造校验)")
    s.run(ticks=4)                             # 券商侧那张市价单会立刻成交

    sold = _sold_at_broker(s)
    fails += expect_le(sold, 6, f"总卖出必须≤总持仓6张: 实际卖了{sold}张 —— "
                                "重复提交导致同一批仓位被卖两次")
    fails += expect_ge(s.broker_pos(OSI), 0,
                       f"I3 裸空: 券商侧净持仓 {s.broker_pos(OSI)} —— 卖超的是【期权空头】, "
                       "风险无上限, 且账户凭空多出未授权空仓")
    fails += expect_le(len([o for o in _sell_orders(s) if o.order_type == "MO"]), 1,
                       "同一次出场意图只应存在一张市价卖单")
    fails += expect_le(_live_sell_qty(s), max(0, s.broker_pos(OSI)),
                       "任何时刻在挂卖单都不得超过持仓")
    return fails


def sc_rejected_sell_returns_to_open(s):
    """卖单被券商拒绝 —— 仓位必须退回 open 继续被管理并重配保护腿, 绝不能标 closed。"""
    fails = []
    p = _open(s, equity=1200.0)
    s.broker.reject_next_submit = 1
    s.quotes.set_path(OSI, [CRASH])
    s.tick()                                   # 止损 → 提交 → 立刻 Rejected
    fails += expect_eq(p["status"], "closing", "提交成功(拿到委托号)后应进 closing 等成交确认")
    fails += expect_eq(p.get("sold", 0), 0, "I1: 拿到委托号不等于成交")

    s.quotes.set_path(OSI, [FLAT])             # 价格回来, 避免止损立刻重触发干扰断言
    s.tick()                                   # ⓪ 对账: Rejected 是终态, 成交0张

    fails += expect_eq(p["status"], "open", "被拒的卖单必须让仓位退回 open, 不得标 closed(弃仓)")
    fails += expect_eq(p.get("sold", 0), 0, "被拒订单不得计入 sold")
    fails += expect_eq(s.broker_pos(OSI), 6, "券商侧仍持有6张")
    fails += expect(_has_protection(p), "退回 open 后必须重新配上保护腿")
    return fails


def sc_persist_failure_fail_stop(s):
    """关键状态落盘失败(券商已收单/已挂腿) —— 必须进 fail_stop 并停止一切自动交易 + 告警。"""
    fails = []
    _signal(s, equity=1200.0, path=(FLAT,))
    s.save_ok = False                          # 入场成交、准备挂保护腿的那一刻落盘挂了
    s.tick()
    p = s.pos(OSI)

    fails += expect_eq(p.get("fail_stop"), True, "落盘失败必须把仓位置 fail_stop")
    fails += expect(bool(s.evs("persist_failed")), "必须记 persist_failed 事件")
    fails += expect(bool(_alerts_like(s, "落盘失败", "fail-stop")), "必须告警等人工处理")

    n_orders = len(s.broker.orders)
    s.quotes.set_path(OSI, [CRASH])            # 之后就算崩盘也不许自动动手
    s.run(ticks=5)
    fails += expect_eq(len(s.broker.orders), n_orders, "fail_stop 后不得再提交任何订单")
    fails += expect_eq(p.get("sold", 0), 0, "fail_stop 后不得改动账本")
    return fails


def sc_fail_balance_degenerate_sizing(s):
    """账户净值拿不到 —— 定张退化路径不得下出异常大单(净值缺失≠无限预算)。"""
    fails = []
    s.broker.fail_balance = True
    _signal(s, equity=1200.0, path=(FLAT,))
    p = s.pos(OSI)
    fails += expect(p is not None, "净值拿不到时仍应能按保守张数下单或干脆不下单")
    if p:
        orders = _buy_orders(s)
        fails += expect_eq(len(orders), 1, "应只提交一张入场单")
        fails += expect_le(orders[0].submitted_quantity, 2,
                           "净值未知时必须退化到固定小张数, 不得按 0/未知预算放大")
    s.run(ticks=3)
    fails += expect_le(_bought_at_broker(s), 2, "实际买入张数必须受控")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# C 部分成交类
# ══════════════════════════════════════════════════════════════════════════

def sc_partial_entry_protected_and_rest_cancelled(s):
    """流动性受限致入场只成交一部分 —— 已成交部分必须立刻有保护, 剩余买腿必须被撤(不摊)。"""
    fails = []
    s.broker.liquidity[OSI] = 1
    _signal(s, equity=6000.0, path=(FLAT,))    # 计划30张
    s.tick()
    p = s.pos(OSI)
    s.broker.liquidity[OSI] = 0                # 冻结: 让撤单能确认终态
    s.tick()

    ent = [o for o in _buy_orders(s)][0]
    fails += expect(0 < p["filled"] < 30, f"应为部分成交(实际{p['filled']}/30)")
    fails += expect_eq(p["filled"], s.broker_pos(OSI), "本地 filled 必须等于券商持仓")
    fails += expect_in(ent.status, ("PartialWithdrawal", "Canceled", "Filled"),
                       "剩余买腿必须被撤(否则整日挂着专接崩盘穿价)")
    fails += expect_eq(p.get("entry_order_id"), None, "撤净后应清空 entry_order_id")
    fails += expect_eq(p["status"], "open", "已成交部分必须转 open 受管理")
    fails += expect(_has_protection(p), "已成交部分必须立刻配上保护腿")
    fails += expect_le(_live_sell_qty(s), p["filled"], "保护腿张数不得超过已成交张数")
    return fails


def sc_partial_exit_reprotect_by_remain(s):
    """卖单只部分成交 —— sold 只加实际成交量, 剩仓退回 open 且保护腿必须按 remain 而非 filled 定张。"""
    fails = []
    p = _open(s, equity=6000.0)                # 30张
    s.broker.liquidity[OSI] = 0
    s.quotes.set_path(OSI, [CRASH])
    s.tick()                                   # 止损 → 市价卖 30 张在途, 一张都成交不了
    exit_oid = p.get("exit_order_id")
    fails += expect(bool(exit_oid), "止损应已提交市价卖")

    s.quotes.set_path(OSI, [FLAT])             # 价格回到成本(避免收尾时止损再触发)
    s.broker.liquidity[OSI] = 2
    s.run(ticks=2)                             # 卖单吃到一部分
    s.broker.liquidity[OSI] = 0
    s.clock.set_et(2026, 7, 20, 16, 5)         # 收盘: Day 单以 PartialWithdrawal 终结
    s.tick()

    ex = s.broker.orders[exit_oid]
    done = ex.executed_quantity
    remain = 30 - done
    fails += expect(0 < done < 30, f"卖单应为部分成交(实际{done}/30)")
    fails += expect_eq(p.get("sold"), done, "sold 只能加实际成交量")
    fails += expect_eq(p["status"], "open", "剩仓必须退回 open")
    fails += expect_eq(s.broker_pos(OSI), remain, "券商侧应剩 remain 张")
    fails += expect(_has_protection(p), "剩仓必须重新配保护腿")
    legs = p.get("tp_qty", 0) + p.get("tp2_qty", 0)
    fails += expect_le(legs, remain,
                       f"保护腿总张数({legs})必须按 remain({remain})定张, 按 filled(30)定张会挂出"
                       "多于持仓的卖单 = 裸空期权")
    fails += expect_le(_live_sell_qty(s), s.broker_pos(OSI), "在挂卖单不得超过持仓")
    return fails


def sc_tp1_partial_cancel_keeps_channel_open(s):
    """一档止盈腿部分成交后被撤 —— tp1_done 不得置位, 否则该止盈通道对剩余张数永久关闭。"""
    fails = []
    p = _open(s, equity=6000.0)                # 30张, 一档10张@1.30
    tp_oid = p["tp_order_id"]
    s.broker.liquidity[OSI] = 1
    s.quotes.set_path(OSI, [TP1HIT])
    s.tick()
    done = s.broker.orders[tp_oid].executed_quantity
    fails += expect(0 < done < 10, f"一档腿应部分成交(实际{done}/10)")

    s.broker.liquidity[OSI] = 0                # 冻结后由外部撤单(模拟券商侧/人工撤)
    s.broker.cancel_order(tp_oid)
    s.tick()

    fails += expect_in(s.broker.orders[tp_oid].status, ("PartialWithdrawal",), "该腿应部分撤销")
    fails += expect_eq(p.get("sold"), done, "已成交部分必须记账(防超卖)")
    fails += expect(p.get("tp1_done") is not True,
                    "一档只成交一部分就被撤, 不得置 tp1_done —— 置位后 ensure_protection 的 "
                    "not tp1_done 守卫会永久关闭一档通道, 剩余张数再无 +30% 出口")
    fails += expect(bool(p.get("tp_order_id")) and p["tp_order_id"] != tp_oid,
                    "撤掉后必须为剩仓补挂新的一档止盈腿")
    fails += expect_le(_live_sell_qty(s), s.broker_pos(OSI), "补挂后在挂卖单不得超过持仓")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# D 行情异常类
# ══════════════════════════════════════════════════════════════════════════

def _no_false_stop(s, p, what):
    fails = []
    fails += expect_eq(s.evs("stop_trigger"), [], f"{what}: 不得据此判止损")
    fails += expect_eq(s.evs("exit_submit"), [], f"{what}: 不得下卖单")
    fails += expect_eq(p["status"], "open", f"{what}: 仓位应保持 open")
    fails += expect_eq(p.get("sold", 0), 0, f"{what}: 不得记账")
    fails += expect_eq(s.broker_pos(OSI), 6, f"{what}: 券商侧持仓不变")
    return fails


def sc_zero_quote_no_false_stop(s):
    """期权 last_done=0(无成交) —— 0 ≤ 任何止损价, 绝不能据此打出假止损。"""
    p = _open(s, equity=1200.0)
    s.quotes.set_path(OSI, [(0.0, 0.0, 0.0)])
    s.run(ticks=6)
    return _no_false_stop(s, p, "报价为0")


def sc_stale_quote_no_false_stop(s):
    """报价陈旧(远超 QUOTE_MAX_AGE_SEC) —— 拿几小时前的旧价判止损等于看历史, 必须跳过。"""
    p = _open(s, equity=1200.0)
    s.quotes.quote_age_sec[OSI] = 4 * 3600
    s.quotes.set_path(OSI, [CRASH])
    s.run(ticks=6)
    return _no_false_stop(s, p, "报价陈旧4小时")


def sc_halted_no_false_stop(s):
    """trade_status=Halted(停牌/熔断) —— 停牌期间的价格不可据以交易。"""
    p = _open(s, equity=1200.0)
    s.quotes.trade_status[OSI] = "Halted"
    s.quotes.set_path(OSI, [CRASH])
    s.run(ticks=6)
    return _no_false_stop(s, p, "停牌")


def sc_quote_channel_down_alerts(s):
    """期权报价接口持续抛错 —— 模拟盘唯一止损通道失效, 必须告警且不得误动作。"""
    fails = []
    p = _open(s, equity=1200.0)
    s.quotes.fail.add(OSI)
    s.run(ticks=8)
    fails += expect(bool(s.evs("stop_channel_down")), "报价通道死掉必须记 stop_channel_down 事件")
    fails += expect(bool(_alerts_like(s, "止损通道失效")), "必须告警(否则用户看不出止损已不存在)")
    fails += _no_false_stop(s, p, "报价接口全灭")
    return fails


def sc_dead_quote_still_force_closed_at_expiry(s):
    """报价整日失效(止损通道全灭) —— 到期强平这道最后防线必须照常生效, 不能把仓位留到归零。"""
    fails = []
    p = _open(s, equity=1200.0)
    s.quotes.fail.add(OSI)
    s.run(ticks=5)
    fails += expect_eq(p["status"], "open", "报价失效期间仓位应仍在管理中")
    s.clock.set_et(2026, 7, 24, 15, 45)
    s.run(ticks=4)
    fails += expect_eq(p["status"], "closed", "到期强平必须执行, 不受报价通道影响")
    fails += expect_eq(s.broker_pos(OSI), 0, "到期日收盘前必须平净, 防归零/被行权")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# E 组合攻击
# ══════════════════════════════════════════════════════════════════════════

# (原 _mirror_mode 夹具随 mirror 出场模式一并删除 —— 2026-07-20。
#  下面两条场景改用 close_position 直接触发: 现实里的触发源是 9ema 拖尾 / 到期强平,
#  它们同样是"这一轮的出场意图必须留到下一轮"的一次性动作。)


def sc_pending_action_kept_when_closing(s):
    """卖单在途期间收到站长全平信号 —— 一次性意图不得丢弃(记 pending_action), 也不得重复卖。"""
    import discord_enrich_bot as B
    fails = []
    p = _open(s, equity=1200.0)
    s.broker.liquidity[OSI] = 0
    s.quotes.set_path(OSI, [CRASH])
    s.tick()                                   # 止损 → 市价卖在途(卡住不成交)
    fails += expect_eq(p["status"], "closing", "应处于卖单在途状态")
    n_sell = len(_sell_orders(s))

    B.close_position(s.positions, OSI, "9ema拖尾出场")   # 卖单在途时又来一个全平意图
    fails += expect_eq(p.get("pending_action"), "full_exit",
                       "closing 期间收到的全平意图必须记为待办(一次性动作, 丢了就永久丢了)")
    fails += expect(bool(s.evs("exit_deferred_closing")), "必须记 exit_deferred_closing 事件")
    fails += expect_eq(len(_sell_orders(s)), n_sell, "不得在已有卖单之上重复卖")

    del s.broker.liquidity[OSI]
    s.run(ticks=3)
    fails += expect_eq(s.broker_pos(OSI), 0, "卖单成交后应平净")
    fails += expect_eq(p["status"], "closed", "平净后应标 closed")
    fails += expect_eq(p.get("pending_action"), None, "待办执行完必须清掉")
    return fails


def sc_exit_intent_lost_on_submit_failure(s):
    """站长全平信号遇上卖单提交失败 —— 一次性出场意图必须留待重试, 不能静默丢弃。"""
    import discord_enrich_bot as B
    fails = []
    p = _open(s, equity=1200.0)
    s.broker.fail_submit = 1                    # 下一次 submit 网络超时(券商没收到)
    B.close_position(s.positions, OSI, "9ema拖尾出场")
    fails += expect(bool(s.evs("exit_submit_failed")), "本场景应确实触发一次提交失败(构造校验)")
    fails += expect_eq(p["status"], "open", "提交失败时仓位应保持 open")
    fails += expect_eq(p.get("pending_action"), "full_exit",
                       "卖单提交失败后必须把一次性出场意图记为待办 —— 不记的话这次出场判定"
                       "从此永久丢失, 仓位只剩 -60% 止损/到期强平兜底")

    s.quotes.set_path(OSI, [FLAT])
    s.run(ticks=4)                              # 提交故障已恢复, 应自动重试
    fails += expect_eq(s.broker_pos(OSI), 0,
                       "故障恢复后必须重试并真的平掉 —— 实际仓位仍在, 出场意图被吞")
    return fails


def sc_buy_signal_during_closing_no_overwrite(s):
    """卖单在途期间又收到同合约买入信号 —— 不得覆盖 positions[osi] 把 exit_order_id 冲掉。"""
    fails = []
    p = _open(s, equity=1200.0)
    s.broker.liquidity[OSI] = 0
    s.quotes.set_path(OSI, [CRASH])
    s.tick()
    exit_oid, filled = p.get("exit_order_id"), p["filled"]
    fails += expect(bool(exit_oid), "应有卖单在途")

    n_buy = len(_buy_orders(s))
    s.clock.set_et(2026, 7, 21, 10, 0)             # 次日复述同一合约(绕过当日消息级去重)
    s.send(BUY)

    p2 = s.pos(OSI)
    fails += expect(p2 is p, "positions[osi] 不得被整条覆盖")
    fails += expect_eq(p2.get("exit_order_id"), exit_oid, "在途卖单ID不得被冲掉(冲掉=卖单失联)")
    fails += expect_eq(p2["filled"], filled, "filled 不得被清零")
    fails += expect_eq(p2["status"], "closing", "状态不得被打回 pending")
    fails += expect_eq(len(_buy_orders(s)), n_buy, "已有未平仓位时不得重复建仓")
    fails += expect(bool(s.evs("dup_open_skip")), "应记 dup_open_skip")
    return fails


def sc_entry_inflight_zero_net_not_closed(s):
    """入场单仍在途 + 保护腿成交致净仓归零 —— 绝不能标 closed, 否则后续成交变孤儿仓。"""
    fails = []
    s.broker.liquidity[OSI] = 1
    s.broker.fail_cancel = 999                     # 撤不掉 → 入场单持续在途
    _signal(s, equity=6000.0, path=(FLAT, IDLE))   # 只有第一格能成交
    s.tick()
    p = s.pos(OSI)
    fails += expect_eq(p["filled"], 1, "应只成交1张")
    fails += expect(bool(p.get("entry_order_id")), "撤单失败 → 入场单必须仍被视为在途")
    fails += expect_eq(p["status"], "open", "已成交部分应转 open")

    s.quotes.set_path(OSI, [TP1SPIKE])
    s.tick()                                       # 一档止盈成交 → 净仓归零

    fails += expect_eq(p.get("sold"), 1, "止盈成交应记账")
    fails += expect_not_in(p["status"], ("closed",),
                           "入场单仍在途时净仓归零【不得】标 closed —— 标了之后入场单再成交, "
                           "那批多头永远无人管理(无止损/无强平/日志无痕)")
    fails += expect_eq(s.broker_pos(OSI), 0, "此刻券商侧应为0")

    s.broker.fail_cancel = 0
    s.quotes.set_path(OSI, [FLAT])
    s.run(ticks=3)                                 # 入场单又成交一些 → 必须被接管
    fails += expect_eq(p["filled"] - p.get("sold", 0), s.broker_pos(OSI), "后续成交必须纳入账本")
    if s.broker_pos(OSI) > 0:
        fails += expect(_has_protection(p) or p["status"] == "closing",
                        "后续成交的张数必须重新获得保护")
    return fails


def sc_expiry_plus_stuck_exit_plus_dead_quote(s):
    """到期日 + 卖单卡死 + 报价失效 三重叠加 —— 至少必须告警, 且绝不能留下重复卖单。"""
    fails = []
    p = _open(s, equity=1200.0)
    s.broker.liquidity[OSI] = 0
    s.clock.set_et(2026, 7, 24, 15, 42)
    s.tick()                                       # 到期强平 → 卖单在途但成交不了
    exit_oid = p.get("exit_order_id")
    fails += expect(bool(exit_oid), "到期强平应提交市价卖")
    s.broker.stuck_cancel.add(exit_oid)
    s.quotes.fail.add(OSI)                         # 报价也没了
    s.run(ticks=8)

    fails += expect_eq(len([o for o in _sell_orders(s) if o.order_type == "MO"]), 1,
                       "三重故障下也只能有一张市价卖单")
    fails += expect(bool(_alerts_like(s, "卡死", "人工")),
                    "到期后仍有无保护持仓且卖单撤不掉 —— 必须告警求人工介入")
    fails += expect_eq(p.get("sold", 0), 0, "没有成交佐证不得记账")
    fails += expect_eq(p["filled"] - p.get("sold", 0), s.broker_pos(OSI), "账实必须一致")
    return fails


def sc_double_stop_trigger_no_double_sell(s):
    """止损条件连续多轮成立 —— closing 期间必须完全静默, 不得每轮各发一张市价卖。"""
    fails = []
    p = _open(s, equity=1200.0)
    s.broker.liquidity[OSI] = 0                    # 卖单不成交 → 止损条件持续成立
    s.quotes.set_path(OSI, [CRASH])
    s.run(ticks=4)                                 # 4 轮都满足止损条件

    mo = [o for o in _sell_orders(s) if o.order_type == "MO"]
    fails += expect_eq(len(mo), 1, f"只应有一张市价卖单, 实际{len(mo)}张 = 重复卖出")
    fails += expect_eq(p.get("sold", 0), 0, "一张都没成交, sold 必须为0(I1)")
    fails += expect_le(_live_sell_qty(s), s.broker_pos(OSI), "在挂卖单不得超过持仓")
    return fails


def sc_tp_and_stop_same_bar(s):
    """同一根K线里既穿一档止盈又击穿止损(巨幅震荡bar) —— 两条通道各卖一次, 合计不得超过持仓。"""
    fails = []
    p = _open(s, equity=1200.0)                    # 6张, 一档2张@1.30
    s.quotes.set_path(OSI, [(0.30, 1.35, 0.25)])   # 最高$1.35(触一档) 最新$0.30(触止损)
    s.tick()

    fails += expect_eq(p.get("sold"), 2, "一档止盈应恰好成交2张")
    fails += expect_eq(p.get("exit_qty"), 4, "止损应只卖剩下的4张, 不得按 filled 全额再卖一遍")
    s.run(ticks=3)
    fails += expect_eq(_sold_at_broker(s), 6, "总卖出必须恰好6张(止盈+止损不得重叠卖)")
    fails += expect_eq(s.broker_pos(OSI), 0, "应平净")
    fails += expect_eq(p.get("sold"), p.get("filled"), "sold 应收敛到 filled")
    return fails


def sc_stuck_entry_leg_blocks_expiry_flatten(s):
    """入场买腿查不到终态(撤单请求发了但状态不确定) —— 不得因此放弃对【已成交部分】的到期强平。"""
    fails = []
    s.broker.liquidity[OSI] = 1
    _signal(s, equity=6000.0, path=(FLAT, IDLE))   # 计划30张, 只有第一格能成交
    ent_oid = _buy_orders(s)[0].order_id
    s.broker.stuck_cancel.add(ent_oid)             # 这张买腿永远查不到终态
    s.tick()
    p = s.pos(OSI)
    fails += expect_ge(p["filled"], 1, "应有部分成交")
    fails += expect(bool(p.get("entry_order_id")), "撤单未确认 → 买腿必须仍视为在途(I2)")

    s.clock.set_et(2026, 7, 24, 15, 45)            # 到期日 15:40 之后
    s.run(ticks=6)

    fails += expect_eq(s.broker_pos(OSI), 0,
                       "到期日必须把【已成交的多头】强平掉 —— 实际被一张撤不掉的【买】腿"
                       "无限期阻塞: ④d 入场TTL 在撤单未确认时 continue, 而 ④到期强平 就写在它"
                       "下面, 于是每一轮都在 ④d 早退, 强平永远执行不到。持仓被留到期权到期"
                       "(归零/被行权), 而卖掉已成交的多头根本不需要先撤买腿(卖≤已成交量不可能裸空)")
    # 注意: 买腿仍在途时【正确行为是保持 open】而不是 closed ——
    # manage_positions 首行会 continue 掉 closed 仓位, 标 closed 会让那张买单后续成交
    # 变成无人管理的孤儿多头。所以这里断言"不得 closed", 并在下面验证卡死解除后能干净收口。
    fails += expect(p["status"] != "closed",
                    f"买腿仍在途时不得标 closed(会让后续成交变孤儿仓): 实际={p['status']}")
    # 卡死解除 → 买腿走完终态 → 仓位应能干净收口, 且券商侧不留活单
    s.broker.stuck_cancel.clear()
    s.run(ticks=3)
    p = s.pos(OSI)
    fails += expect_eq(p["status"], "closed", "卡死解除后应干净收口")
    fails += expect(not s.broker.live_orders(OSI),
                    f"收口后券商侧不应留活单: {s.broker.live_orders(OSI)}")
    fails += expect(bool(_alerts_like(s, "推迟", "人工", "TTL")),
                    "而且整个过程只有一行 log + journal, 没有任何 _alert —— "
                    "到期日平不掉仓这种事必须响亮告警")
    return fails


def sc_gross_cap_blocks_second_position(s):
    """敞口闸: 第一笔已压满净值时第二只票不得再开 —— 防购买力不足随机拒单/全仓压在期权上。"""
    fails = []
    _open(s, equity=1200.0)                        # 6张×$100 = $600 = 净值×0.5
    n_buy = len(s.broker.orders)
    s.quotes.set_path("TSLA260724C120000.US", [FLAT])
    s.quotes.open_interest["TSLA260724C120000.US"] = 50000
    s.send("$TSLA 7/24 $120 calls $1.00")
    s.send("$AMD 7/24 $120 calls $1.00")           # 第三笔必然超限
    total = sum(o.submitted_quantity * 100 for o in s.broker.orders.values() if o.side == "Buy")
    fails += expect_le(total, 1200.0,
                       f"在险权利金合计${total:.0f} 不得超过净值×MAX_GROSS_FRAC=$1200")
    return fails
