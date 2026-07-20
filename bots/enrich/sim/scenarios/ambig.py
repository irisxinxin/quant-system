#!/usr/bin/env python3
"""sim/scenarios/ambig.py — 歧义单(缺 call/put)消歧链路的场景。

这条链路【不受出场模式影响, 一直是活的】: 站长发单没写 call/put 时, bot 拉同行权价
C/P 双边报价, 比对哪一边贴合信号权利金来定方向。判错 = 买反方向, 大概率全损。
历史 1473 条消息里 14 条缺方向(约占买入信号 8.6%), 真实触发过 2 次消歧。

(本文件原名 reduce_ambig.py, 另含 10 个 mirror_reduce 场景 —— 2026-07-20 随 mirror
 出场模式一并删除, 见 git commit。)
"""
import discord_enrich_bot as B
from sim.scenario_api import (expect, expect_eq, expect_ge, expect_le, expect_in,
                              expect_not_in, osi)

TK = "HOOD"
OSI = osi("HOOD", "260724", "C", 120.0)
OSI_P = osi("HOOD", "260724", "P", 120.0)
AMBIG = "$HOOD 7/24 $120 $1.00"          # 缺 call/put
FLAT = (1.00, 1.00, 1.00)


def _isolate():
    """清掉跨场景残留(去重表/退避), 保证每条场景独立。"""
    B._recent_exits.clear()
    B._rl_until = 0.0


def _quote(s, sym, px, age=0, status="Normal"):
    s.quotes.set_path(sym, [(px, px, px)])
    s.quotes.quote_age_sec[sym] = age
    s.quotes.trade_status[sym] = status


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
    _isolate()
    # 不需要持仓: 测的是 parser 的闸门, 看这些措辞能否走到"被识别为 EXIT"这一步。
    # (站长出场现在只播报不执行, 事件名是 exit_signal_mech_ignored)
    for msg in ("Trimming $HOOD here", "$HOOD taking profits here"):
        s.send(msg)
    got = len(s.evs("exit_signal_mech_ignored"))
    fails += expect_eq(got, 2,
                       f"'Trimming' / 'taking profits' 都应被识别为出场信号并播报, "
                       f"实际只识别了 {got}/2 条(闸门 EXIT_RE 用 `trim` 带\\b 匹配不到 "
                       f"trimming, 而分级 EXIT_PARTIAL_RE 明写着 trim\\w* —— 两者必须对齐)")
    return fails
