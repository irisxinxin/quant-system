#!/usr/bin/env python3
"""sim/scenarios/reduce_ambig.py — 攻击两条从未被仿真触及的实盘路径。

  ① mirror_reduce()    L895  站长 partial/vague 出场时的镜像减仓 (EXIT_MODE=mirror 才可达)
  ② resolve_direction() L327 信号缺 call/put 时靠同行权价双边报价消歧

⚠ EXIT_MODE 说明: launchd 实跑用 EXIT_MODE=mechanical, 该模式下 _handle 在 L1381 直接
  return, mirror_reduce 【当前不可达】。本文件的减仓场景显式把 B.EXIT_MODE 切成 "mirror"
  (mock.patch 在 Sim.__exit__ 会还原), 验证的是"回切 mirror 后会怎样"。

约定(见 sim/scenario_api.py): 场景只返回失败原因列表, 不 print 不 assert。
通用不变式(裸空/账实不符/孤儿单/弃仓/挂卖单超持仓)由 run_sim.py 自动跑。

命名: sc_bug_* = 已坐实的缺陷复现, 【预期为红】, 断言写的是"正确行为应该是什么";
      其余 sc_* = 攻击后认为该路径正确, 【预期为绿】, 是回归护栏。
"""
import discord_enrich_bot as B
from sim.scenario_api import (expect, expect_eq, expect_ge, expect_le,
                              expect_in, expect_not_in, osi)

TK = "HOOD"
OSI = osi("HOOD", "260724", "C", 120)
OSI_P = osi("HOOD", "260724", "P", 120)
BUY = "$HOOD 7/24 $120 calls $1.00"

FLAT = (1.00, 1.00, 1.00)
IDLE = (1.10, 1.10, 1.05)          # 高于入场限价, 低于止盈$2.00
TP_HIT = (2.10, 2.10, 2.00)        # 触 mirror 模式止盈 avg×TP_MULT(2.0)

# 缺方向的歧义信号: 单票 + 到期 + 两个$价(大的=行权价, 小的=权利金), 无 calls/puts
AMBIG = "$HOOD 7/24 $120 $1.00"


# ── 夹具 ──

def _isolate():
    B._rl_until = 0.0
    B._recent_exits.clear()


def _mirror_open(s, equity=1200.0, path=(FLAT,), oi=50000):
    """EXIT_MODE=mirror 下建仓并成交。6张 @$1.00, 止盈腿 3张 @$2.00 (TP_MULT=2.0)。"""
    _isolate()
    B.EXIT_MODE = "mirror"          # patch.multiple 会在 Sim.__exit__ 还原成 mechanical
    s.broker.equity = equity
    s.quotes.set_path(OSI, list(path))
    s.quotes.open_interest[OSI] = oi
    s.send(BUY)
    s.tick()
    return s.pos(OSI)


def _sell_orders(s, sym=OSI):
    return [o for o in s.broker.orders.values() if o.side == "Sell" and o.symbol == sym]


def _live_sell_qty(s, sym=OSI):
    return sum(o.submitted_quantity - o.executed_quantity
               for o in s.broker.live_orders(sym) if o.side == "Sell")


def _quote(s, sym, px, age=0, status="Normal"):
    s.quotes.set_path(sym, [(px, px, px)])
    s.quotes.quote_age_sec[sym] = age
    s.quotes.trade_status[sym] = status


# ══════════════════════════════════════════════════════════════════════════
# ① mirror_reduce
# ══════════════════════════════════════════════════════════════════════════

def sc_reduce_happy_path(s):
    """基线(预期绿): 首次 partial → 卖一半、剩一半继续跑; 二次 partial → 清 runner 全平。"""
    fails = []
    p = _mirror_open(s)
    fails += expect_eq(p.get("filled"), 6, "建仓应成交6张")
    fails += expect_eq(p.get("tp_qty"), 3, "mirror模式止盈腿应为一半(3张)")

    s.send("$HOOD - scaling out 1/2 here")
    s.run(ticks=2)
    fails += expect_eq(p.get("sold"), 3, "首次减仓应卖出一半")
    fails += expect_eq(p.get("reduced"), True, "全部成交后应置 reduced")
    fails += expect_eq(p.get("status"), "open", "减仓后仍应持有 runner")

    s.send("$HOOD scaling down here")
    s.run(ticks=3)
    fails += expect_eq(p.get("status"), "closed", "二次减仓应清掉 runner 全平")
    fails += expect_eq(s.broker_pos(OSI), 0, "全平后券商侧应无持仓")
    return fails


def sc_bug_reduce_submit_fail_signal_lost(s):
    """BUG: 减仓卖单提交失败时 mirror_reduce 丢弃 _start_exit 的返回值, 既不记 pending_action
    也不重试 —— 站长的 partial 是一次性信号, 就此永久丢失(close_position 同场景有待办队列)。"""
    fails = []
    p = _mirror_open(s)
    s.broker.fail_submit = 5          # 卖单提交连续失败
    s.send("$HOOD - scaling out 1/2 here")
    s.broker.fail_submit = 0          # 之后网络恢复, 有充分机会重试
    s.run(ticks=6)

    fails += expect_ge(len(s.evs("exit_submit_failed")), 1, "场景前提: 卖单提交应确实失败过")
    fails += expect(p.get("sold", 0) > 0 or p.get("pending_action"),
                    f"提交失败后必须留待办或后续重试成交, 否则站长减仓信号永久丢失: "
                    f"sold={p.get('sold')} pending_action={p.get('pending_action')} status={p.get('status')}")
    return fails


def sc_bug_reduce_dropped_while_closing(s):
    """BUG: 已有卖单在途(status=closing)时 mirror_reduce 直接 return, 不记待办 ——
    站长"再减一次/清runner"的意图被静默吞掉。close_position 对同一情形是记 pending_action 的。"""
    fails = []
    p = _mirror_open(s)
    s.broker.liquidity[OSI] = 0       # 卖单挂着永不成交 → 仓位卡在 closing
    s.send("$HOOD - scaling out 1/2 here")
    fails += expect_eq(p.get("status"), "closing", "场景前提: 首笔减仓卖单应在途")

    s.send("$HOOD scaling down here")   # 第二条: 意图=清runner
    fails += expect(p.get("pending_action") or len(s.evs("exit_deferred_closing")) >= 1,
                    "卖单在途时收到的第二条出场信号必须记为待办, 不能静默丢弃")
    s.broker.liquidity.pop(OSI, None)
    s.run(ticks=6)
    fails += expect_eq(s.broker_pos(OSI), 0,
                       f"第二条信号意图是清 runner, 最终应全平: 券商仍持 {s.broker_pos(OSI)} 张")
    return fails


def sc_bug_reduce_stacks_sell_on_live_tp_naked(s):
    """BUG(最严重): mirror_reduce 只撤止损腿, 【不撤止盈腿】; 且 _start_exit 只按券商净持仓封顶,
    不减去在挂卖单(_sell_budget)。减仓卖单部分成交时 reduced 又留在 False, 站长第二条 partial
    被当作"首次减仓"再卖一半 → 在挂卖单总量超过持仓 → 止盈腿与减仓单同时成交 = 券商净持仓为负(裸空期权)。
    对照: close_position 会把 tp/tp2/stop/entry 四条腿全撤净后才发卖单。"""
    fails = []
    p = _mirror_open(s)
    tp_oid = p.get("tp_order_id")
    fails += expect(bool(tp_oid), "场景前提: mirror 模式应挂出止盈腿")
    fails += expect_eq(p.get("filled"), 6, "场景前提: 应成交6张")

    # 第一条 partial → 市价卖3张, 此时止盈腿3张仍活着 (3+3=6=持仓, 尚未超卖)
    s.send("$HOOD - scaling out 1/2 here")
    fails += expect_eq(p.get("status"), "closing", "场景前提: 首笔减仓应发出卖单")
    # 发减仓单【之前】必须先撤净止盈腿, 否则 3(tp)+3(减仓)=6 张在挂, 后续任一腿部分成交
    # 都会让在挂量超过持仓。原实现只撤 stop 不撤 tp, 这里正是裸空的起点。
    fails += expect_le(_live_sell_qty(s), s.broker_pos(OSI),
                       f"发减仓单前应撤净止盈腿: 在挂{_live_sell_qty(s)}张 vs 持仓{s.broker_pos(OSI)}张")

    # 让减仓单只成交2张然后冻结, 收盘作废成 PartialWithdrawal → sold=2 而 reduced 保持 False
    s.broker.liquidity[OSI] = 2
    s.broker.tick(step_price=False)
    s.broker.liquidity[OSI] = 0
    s.clock.set_et(2026, 7, 20, 16, 30)          # Day 单作废 → 终态 PartialWithdrawal
    s.tick(seconds=1)
    fails += expect_eq(p.get("sold"), 2, "场景前提: 减仓单应只成交2张")
    fails += expect_eq(p.get("status"), "open", "部分成交后仓位应交回正常管理")

    # 第二条 partial: 语义上是"他在连续撤退→清runner", 却又被当成首次减仓再卖一半
    _isolate()
    s.clock.set_et(2026, 7, 20, 11, 0)
    s.broker.liquidity.pop(OSI, None)
    s.send("$HOOD scaling down here")
    over = _live_sell_qty(s) - s.broker_pos(OSI)
    fails += expect_le(_live_sell_qty(s), s.broker_pos(OSI),
                       f"在挂卖单总量不得超过持仓: 在挂{_live_sell_qty(s)}张 vs 持仓{s.broker_pos(OSI)}张 (超卖{over}张)")
    # 断言【行为】而非内部字段: 第一条 partial 已实际卖出过(sold=2), 所以第二条的语义是
    # "他在连续撤退→清 runner"=全平, 不是再减半。
    # (字段层面 bot 用 reduced_any 记"是否真减过仓" —— reduced 不能复用, 它同时是止盈通道的
    #  守卫, 部分成交时必须保持 False 好让剩余张数还有止盈出口。)
    _intent_full = (p.get("exit_intent") == "full" or p.get("pending_action") == "full_exit"
                    or p.get("status") == "closed")
    fails += expect(_intent_full,
                    f"第二条 partial 应走'二次减仓→清runner'(全平), 实际 exit_intent="
                    f"{p.get('exit_intent')} pending={p.get('pending_action')} status={p.get('status')}")

    # 价格摸到止盈价 → 止盈腿与减仓单同时成交
    s.quotes.set_path(OSI, [TP_HIT] * 6)
    s.run(ticks=4)
    fails += expect_ge(s.broker_pos(OSI), 0,
                       f"券商侧净持仓不得为负(裸空期权): {s.broker_pos(OSI)}")
    return fails


def sc_bug_reduce_vague_dropped_while_closing(s):
    """BUG: vague(模糊催促)在卖单在途时同样被静默丢弃。vague 的语义是"保守全平",
    是风险最高的一条 —— 丢了它等于站长喊撤退而 bot 装没听见。"""
    fails = []
    p = _mirror_open(s)
    s.broker.liquidity[OSI] = 0
    s.send("$HOOD - scaling out 1/2 here")
    fails += expect_eq(p.get("status"), "closing", "场景前提: 卖单应在途")

    s.send("$HOOD secure profits here")   # 无 full/partial 关键词 → vague
    fails += expect(p.get("pending_action") or len(s.evs("exit_deferred_closing")) >= 1,
                    "vague 出场在卖单在途时必须记待办(close_position 就是这么做的), 不能丢")
    s.broker.liquidity.pop(OSI, None)
    s.run(ticks=6)
    fails += expect_eq(s.broker_pos(OSI), 0, "vague 语义=保守全平, 最终不该还留着持仓")
    return fails


def sc_reduce_pending_entry_cancels(s):
    """攻击(预期绿): 入场单还没成交站长就减仓 → 应走 close_position 撤单, 不留在途买腿。"""
    fails = []
    _isolate()
    B.EXIT_MODE = "mirror"
    s.broker.equity = 1200.0
    s.quotes.set_path(OSI, [IDLE] * 8)           # 价格一直高于$1.00 → 入场限价单不成交
    s.quotes.open_interest[OSI] = 50000
    s.send(BUY)
    p = s.pos(OSI)
    fails += expect_eq(p.get("status"), "pending", "场景前提: 入场单应仍未成交")

    s.send("$HOOD - scaling out 1/2 here")
    s.run(ticks=3)
    fails += expect_not_in(p.get("status"), ("pending",), "站长已出场, 不该还挂着入场单")
    fails += expect_eq(s.broker_pos(OSI), 0, "未成交就撤退, 券商侧不该有持仓")
    fails += expect_eq(_live_sell_qty(s), 0, "不该留下任何在挂卖单")
    return fails


def sc_reduce_single_contract_full_exit(s):
    """攻击(预期绿): 只剩1张时 partial 也等于全出, 不该出现卖0张/卖超。"""
    fails = []
    _isolate()
    B.EXIT_MODE = "mirror"
    s.broker.equity = 200.0                      # 净值×0.5=$100 → 1张
    s.quotes.set_path(OSI, [FLAT] * 8)
    s.quotes.open_interest[OSI] = 50000
    s.send(BUY)
    s.tick()
    p = s.pos(OSI)
    fails += expect_eq(p.get("filled"), 1, "场景前提: 应只成交1张")

    s.send("$HOOD - scaling out 1/2 here")
    s.run(ticks=4)
    fails += expect_eq(p.get("sold"), 1, "只剩1张时 partial = 全出")
    fails += expect_eq(s.broker_pos(OSI), 0, "券商侧应清零")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# ② resolve_direction
# ══════════════════════════════════════════════════════════════════════════

def _ambig_order(s):
    """歧义信号发出后, 券商侧实际下的买单(symbol 揭示 bot 选了哪个方向)。"""
    return [o for o in s.broker.orders.values() if o.side == "Buy"]


def sc_ambig_both_in_window_refuses(s):
    """攻击(预期绿): C/P 都落在 0.4~2.2x 窗口 → 消歧失败, 只提醒不下单。"""
    fails = []
    _isolate()
    _quote(s, OSI, 1.00)
    _quote(s, OSI_P, 0.90)
    s.send(AMBIG)
    fails += expect_eq(len(_ambig_order(s)), 0, "两边都匹配时不该下任何单")
    fails += expect_eq(len(s.evs("disambig")), 0, "不该记 disambig 成功事件")
    return fails


def sc_ambig_quote_api_error_refuses(s):
    """攻击(预期绿): 报价接口抛异常 → 不下单, 只提醒。"""
    fails = []
    _isolate()
    s.quotes.fail.add(OSI)
    _quote(s, OSI_P, 1.00)
    s.send(AMBIG)
    fails += expect_eq(len(_ambig_order(s)), 0, "报价异常时不该下单")
    fails += expect_eq(len(s.evs("disambig")), 0, "不该记 disambig 成功事件")
    return fails


def sc_ambig_no_quotes_refuses(s):
    """攻击(预期绿): 两边都无报价 → 不下单。"""
    fails = []
    _isolate()
    s.send(AMBIG)
    fails += expect_eq(len(_ambig_order(s)), 0, "两边无报价时不该下单")
    return fails


def sc_ambig_osi_matches_quoted_symbol(s):
    """攻击(预期绿): 消歧改写 s.right 之后, 真正下单的 OSI 必须就是被报价的那一边。"""
    fails = []
    _isolate()
    _quote(s, OSI_P, 1.00)               # 只有 PUT 有报价 → 只能选 P
    s.send(AMBIG)
    orders = _ambig_order(s)
    fails += expect_eq(len(orders), 1, "应下一张买单")
    if orders:
        fails += expect_eq(orders[0].symbol, OSI_P,
                           "下单合约必须与消歧所依据的报价合约一致(否则下到了没查过价的合约上)")
        fails += expect_in(OSI_P, s.positions, "仓位账本的 key 应是同一个 OSI")
    return fails


def sc_bug_ambig_uses_halted_stale_quote(s):
    """BUG: resolve_direction 直接吃 last_done, 【不校验报价新鲜度, 也不校验 trade_status】。
    bot 自己的 _option_last() 明文规定这两项必查("停牌/熔断期间的价格不可据以交易",
    "低流动性期权的最新成交可能是几天前的旧价"), 而决定 call/put 方向这个更不可逆的判断
    反而完全没校验 —— 停牌 + 一天前的旧价照样能拍板方向并真下单。"""
    fails = []
    _isolate()
    _quote(s, OSI, 1.00, age=86400, status="Halted")   # 停牌 + 24小时前的陈旧成交价
    s.send(AMBIG)                                       # PUT 侧完全无报价
    orders = _ambig_order(s)
    fails += expect_eq(len(orders), 0,
                       f"停牌+陈旧报价不得作为方向判据下单: 实际下了 {[o.symbol for o in orders]}")
    return fails


def sc_bug_ambig_wrong_side_when_price_ran(s):
    """BUG: 窗口上界 2.2x 会在"信号方向是对的"时把正确一边排除掉, 反手买错方向。
    站长发单后价格急涨是【该方向正确】的证据, 而 resolve_direction 把 >2.2x 当作"不匹配"排除;
    同行权价的另一边此时正相应下跌, 恰好落进 0.4 下界 → 成为唯一匹配 → bot 买反方向。
    (0.4 下界同样有问题: 反方向腿只要还值信号价的四成就会被当成候选。)"""
    fails = []
    _isolate()
    _quote(s, OSI, 2.50)      # CALL 从$1.00涨到$2.50 = 站长看对了方向, 却因 2.5>2.2 被排除
    _quote(s, OSI_P, 0.45)    # 同行权价 PUT 相应下跌到 0.45x → 唯一"匹配"
    s.send(AMBIG)
    orders = _ambig_order(s)
    got = [o.symbol for o in orders]
    fails += expect_not_in(OSI_P, got,
                           f"正确一边只是涨出了窗口上界, 不能据此认定方向是反的: 实际下单 {got}")
    return fails


def sc_ambig_lotto_sizing(s):
    """攻击(预期绿): 消歧成功后必须走 lotto 小仓档(LOTTO_FRAC), 不能按常规档。"""
    fails = []
    _isolate()
    _quote(s, OSI_P, 1.00)
    s.broker.equity = 3000.0          # 常规档 0.5→15张; lotto档 0.3333→9张
    s.quotes.open_interest[OSI_P] = 500000
    s.send(AMBIG)
    orders = _ambig_order(s)
    fails += expect_eq(len(orders), 1, "应下一张买单")
    if orders:
        fails += expect_le(orders[0].submitted_quantity, 10,
                           f"歧义单应走 lotto 小仓(净值×0.3333≈9张), 实际 {orders[0].submitted_quantity} 张")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# ③ 顺带打到的: 出场词闸门与分级正则不一致
# ══════════════════════════════════════════════════════════════════════════

def sc_bug_exit_word_trimming_dropped(s):
    """BUG(附带发现): enrich_parser 的出场【闸门】EXIT_RE 里是 `trim`(带\\b, 匹配不到 trimming),
    而【分级】EXIT_PARTIAL_RE 里写的是 `\\btrim\\w*\\b` —— 分级正则明确预期 "trimming",
    闸门却先把整条判成 NOISE, 该消息连 EXIT 都不算, 永远走不到分级。
    "taking profits" 同理(闸门只有 take, 分级有 taking)。站长最常用的减仓措辞就此静默丢弃。"""
    fails = []
    p = _mirror_open(s)
    fails += expect_eq(p.get("filled"), 6, "场景前提: 应持有6张")
    s.send("Trimming $HOOD here")
    s.send("$HOOD taking profits here")
    fails += expect_ge(len(s.evs("exit_signal")), 1,
                       "'Trimming $HOOD' / '$HOOD taking profits' 应被识别为出场信号, "
                       "实际被判 NOISE 整条丢弃(分级正则里明写着支持 trim\\w*)")
    return fails
