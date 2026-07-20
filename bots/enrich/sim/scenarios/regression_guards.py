#!/usr/bin/env python3
"""sim/scenarios/regression_guards.py — 给"本轮修复"配的守卫场景。

来历: 变异测试发现下面这几道防线【没有任何场景在守】——
拆掉它们, 全部 76 个场景照样全绿。绿不等于被验证, 只等于没人测。

每个场景都配了 mutation_check 里的变异, 拆掉对应防线必须让它变红。
"""
from sim.scenario_api import expect, expect_eq, expect_le, expect_ge
from types import SimpleNamespace

OSI = "HOOD260724C120000.US"


# ══════════════════════════════════════════════════════════════════════════
# ① I3 终极保障: 券商持仓查不到时【不卖】
#    守: _start_exit 的 `if bq is None: 拒卖`
#    变异: 券商持仓查不到时照常卖
# ══════════════════════════════════════════════════════════════════════════

def sc_guard_broker_qty_unknown_refuses_sell(s):
    """券商持仓查询失败时绝不卖 —— 账本偏高与查询失败常是同一次网络故障的两面。

    原实现 `if bq is not None:` 把封顶整段包住, 查询失败就跳过封顶按本地账本照常卖。
    触发链(单一根因, 不是两个独立巧合):
      网络抖动 → 卖单"券商已收单但客户端超时"(sold 未更新, 账本偏高)
               → 同一次抖动里 _broker_qty / _live_sell_order 两道防线一起失灵
               → 按偏高的账本再卖一次 → 券商净持仓 -6 = 裸空期权(无限风险)
    """
    import discord_enrich_bot as B
    fails = []
    s.quotes.set_path(OSI, [1.00] * 20)
    s.broker.position[OSI] = 6
    pos = {OSI: {"status": "open", "filled": 6, "sold": 0, "avg": 1.00, "qty": 6,
                 "right": "C", "ticker": "HOOD", "expiry": "2026-07-24", "strike": 120.0,
                 "entry_order_id": None, "opened_ts": s.clock.time()}}

    # 第一次卖出: 券商收单并成交, 客户端"超时" → 本地 sold 停在 0(账本偏高)
    s.broker.submit_but_lose_response = True
    B._start_exit(pos, OSI, pos[OSI], 6, "第一次出场", intent="full")
    s.broker.submit_but_lose_response = False
    s.broker.tick(step_price=False)
    fails += expect_eq(s.broker.net_position(OSI), 0, "前提: 第一笔卖出应真的成交")
    fails += expect_eq(pos[OSI].get("sold"), 0, "前提: 客户端超时 → 本地 sold 未更新(账本偏高)")

    # 同一次网络抖动: 持仓查询 + 在途单查询 都失败
    _bq, _ls = B._broker_qty, B._live_sell_order
    B._broker_qty = lambda osi: None
    B._live_sell_order = lambda osi: (None, None)
    try:
        pos[OSI]["status"] = "open"
        pos[OSI]["exit_order_id"] = None
        ok = B._start_exit(pos, OSI, pos[OSI], 6, "第二次出场", intent="full")
        s.broker.tick(step_price=False)
    finally:
        B._broker_qty, B._live_sell_order = _bq, _ls

    fails += expect_eq(ok, False, "查不到券商真值时 _start_exit 必须返回 False(未卖出)")
    fails += expect_ge(s.broker.net_position(OSI), 0,
                       f"绝不可裸空: 券商净持仓={s.broker.net_position(OSI)}")
    fails += expect_eq(pos[OSI].get("pending_action"), "full_exit",
                       "一次性出场意图必须留成待办, 否则下轮没人重试")
    return fails


def sc_guard_broker_qty_unknown_retries_next_round(s):
    """查不到券商持仓 → 记待办; 查询恢复后必须真的重试卖出(不能就此躺平)。"""
    import discord_enrich_bot as B
    fails = []
    s.quotes.set_path(OSI, [1.00] * 20)
    s.broker.position[OSI] = 6
    # 必须挂进 s.positions —— s.tick() 跑的是 manage_positions(s.positions), 用本地 dict
    # 的话 ⓪b 根本看不到这个仓位(第一版就这么写错, 场景假红了一轮)
    s.positions[OSI] = {"status": "open", "filled": 6, "sold": 0, "avg": 1.00, "qty": 6,
                        "right": "C", "ticker": "HOOD", "expiry": "2026-07-24", "strike": 120.0,
                        "entry_order_id": None, "opened_ts": s.clock.time()}
    p = s.positions[OSI]

    _bq = B._broker_qty
    B._broker_qty = lambda osi: None
    try:
        B._start_exit(s.positions, OSI, p, 6, "止损", intent="full")
    finally:
        B._broker_qty = _bq
    fails += expect_eq(p.get("pending_action"), "full_exit", "应记下待办")
    fails += expect_eq(p.get("sold", 0), 0, "查不到券商真值时不得卖出")

    # 查询恢复 → ⓪b 必须重试并真的卖出(否则"记待办"只是自我安慰)
    s.tick(n=2)
    fails += expect(p.get("sold", 0) > 0 or p.get("status") in ("closing", "closed"),
                    f"查询恢复后应重试卖出, 实际 sold={p.get('sold')} status={p.get('status')}")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# ② I3 硬闸 _sell_budget —— 纵深防御, 直接打 _start_exit 才能验到
#    (上游 mirror_reduce/close_position 都会先撤净卖腿, 正常路径到不了这道闸)
#    变异: I3硬闸失效(卖单预算无限)
# ══════════════════════════════════════════════════════════════════════════

def sc_guard_sell_budget_blocks_stacking(s):
    """已有在挂卖单占满剩仓时, 再来的卖出请求必须被 _sell_budget 挡下/削平。

    这是纵深防御: 正常路径(mirror_reduce/close_position)都会先撤净卖腿, 走不到这里。
    但 _sell_budget 的 docstring 写明"任何挂卖单的路径都必须先过这道闸" —— ensure_protection
    三处都过了, _start_exit 曾是唯一漏网的。直接调它来守住这个性质, 以后新增调用方自动受保护。
    """
    import discord_enrich_bot as B
    fails = []
    s.quotes.set_path(OSI, [1.00] * 20)
    s.broker.position[OSI] = 6
    # 账本: 持仓6张, 其中5张已挂在止盈腿上 → 还能再卖的预算只有1张
    pos = {OSI: {"status": "open", "filled": 6, "sold": 0, "avg": 1.00, "qty": 6,
                 "right": "C", "ticker": "HOOD", "expiry": "2026-07-24", "strike": 120.0,
                 "entry_order_id": None, "opened_ts": s.clock.time(),
                 "tp_order_id": "TP-LIVE", "tp_qty": 5}}
    fails += expect_eq(B._sell_budget(pos[OSI]), 1, "前提: 剩仓6 − 在挂5 = 预算1张")

    B._start_exit(pos, OSI, pos[OSI], 6, "叠加卖出", intent="full")
    sold_req = pos[OSI].get("exit_qty", 0)
    fails += expect_le(sold_req, 1,
                       f"卖出量必须被预算封顶到1张, 实际请求 {sold_req} 张")

    # 券商侧在挂卖单总量不得超过持仓
    live_sell = sum(o.submitted_quantity - o.executed_quantity
                    for o in s.broker.live_orders(OSI) if o.side == "Sell")
    fails += expect_le(live_sell, 6 + 5,   # 5 张止盈腿是场景伪造的账本, 券商侧并不存在
                       f"券商侧在挂卖单 {live_sell} 张")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# ③ 消歧: 两边贴合度接近时不猜方向
#    守: AMBIG_MIN_MARGIN
#    变异: 消歧模棱两可闸关闭
# ══════════════════════════════════════════════════════════════════════════

def sc_guard_ambig_close_deviation_refuses(s):
    """两边偏离度接近(都在窗口内)时必须拒绝消歧 —— 这时没有任何依据判方向。

    单靠 AMBIG_MAX_DEV 守不住: 它只管"最贴合的一边是否够贴合"。
    C=1.5(偏离0.41) / P=0.7(偏离0.36) 两边都很贴合, 差距仅0.05 —— MAX_DEV 全部放行,
    只有 MIN_MARGIN 能拦。拦不住就是 50/50 猜方向, 猜错 = 买反方向大概率全损。
    """
    import discord_enrich_bot as B
    fails = []
    syms = {}

    class _FQ:
        def option_quote(self, ss):
            return [SimpleNamespace(symbol=y, last_done=syms[y],
                                    timestamp=None, trade_status="Normal")
                    for y in ss if y in syms]

    _qc, _osi_fn = B._quote_ctx, B._osi
    B._quote_ctx = _FQ()
    B._osi = lambda t, e, r, k: f"X260724{r}100000.US"
    try:
        sig = SimpleNamespace(ticker="X", expiry="2026-07-24", strike=100.0, limit_price=1.0)
        # 两边都在 MAX_DEV(0.79) 内, 差距 0.05 → 无从判断
        syms.clear()
        syms.update({"X260724C100000.US": 1.5, "X260724P100000.US": 0.7})
        side, info = B.resolve_direction(sig)
        fails += expect_eq(side, None, f"两边贴合度接近时必须拒绝消歧, 实际判了 {side} ({info})")

        # 对照(阳性): 差距拉开就该判得出来, 否则这条场景等于"永远拒绝"的空断言
        syms.clear()
        syms.update({"X260724C100000.US": 1.0, "X260724P100000.US": 3.0})
        side2, info2 = B.resolve_direction(sig)
        fails += expect_eq(side2, "C", f"差距足够大时应能判定, 实际 {side2} ({info2})")
    finally:
        B._quote_ctx, B._osi = _qc, _osi_fn
    return fails


def sc_guard_ambig_far_deviation_refuses(s):
    """两边差距很大、但最贴合的一边也离信号价很远时, 同样不能判 —— 由 AMBIG_MAX_DEV 守。

    MIN_MARGIN 守不住这种: 差距足够大, 它直接放行。
    C=$12 / P=$0.02 对信号价 $1.00 —— 差距悬殊, 可最贴合的 C 也偏离了 12 倍,
    说明这条信号已经旧到价格面目全非(或行权价/到期解析错了), 此时"哪边更接近"没有意义。
    """
    import discord_enrich_bot as B
    fails = []
    syms = {}

    class _FQ:
        def option_quote(self, ss):
            return [SimpleNamespace(symbol=y, last_done=syms[y],
                                    timestamp=None, trade_status="Normal")
                    for y in ss if y in syms]

    _qc, _osi_fn = B._quote_ctx, B._osi
    B._quote_ctx = _FQ()
    B._osi = lambda t, e, r, k: f"X260724{r}100000.US"
    try:
        sig = SimpleNamespace(ticker="X", expiry="2026-07-24", strike=100.0, limit_price=1.0)
        syms.clear()
        syms.update({"X260724C100000.US": 12.0, "X260724P100000.US": 0.02})
        side, info = B.resolve_direction(sig)
        fails += expect_eq(side, None,
                           f"两边都远离信号价时必须拒绝, 实际判了 {side} ({info})")

        # 阳性对照: 把 C 拉回信号价附近就该判得出来
        syms.clear()
        syms.update({"X260724C100000.US": 1.05, "X260724P100000.US": 0.02})
        side2, info2 = B.resolve_direction(sig)
        fails += expect_eq(side2, "C", f"贴合信号价时应能判定, 实际 {side2} ({info2})")
    finally:
        B._quote_ctx, B._osi = _qc, _osi_fn
    return fails


def sc_guard_quote_age_uses_local_tz(s):
    """报价新鲜度必须按【本机时区】解释 SDK 的 naive timestamp —— 按 UTC 解释会偏 8 小时。

    LongPort 的 option_quote().timestamp 是 naive datetime, 值等于本机墙上时间
    (2026-07-20 真实API核对: SDK ts 23:29:27 / 本机SGT 23:29:28 / UTC 15:29:28)。
    曾经写成 replace(tzinfo=utc), 于是年龄恒等于【真实年龄 − 28800】, `age > 3600`
    的实际拒绝线从 1 小时变成 9 小时 —— 覆盖整个交易日加隔夜。
    后果是漏止损: 流动性差的合约今天没成交时, 拿昨天的 last_done 判今天的止损,
    明明已跌穿却判"没跌穿"。

    这类 bug 仿真本来抓不到(FakeQuotes 的 quote_age_sec 直接赋值, 不过时区),
    所以这条场景绕开 FakeQuotes, 直接构造带 naive timestamp 的报价对象打 _quote_usable。
    """
    import discord_enrich_bot as B
    from datetime import timedelta, timezone
    fails = []
    # 必须用【bot 自己的时钟】造 timestamp: Sim 把 B.datetime 换成了 FakeClock, 拿真实
    # datetime.now() 造数据会和假时钟差出好几小时, 场景就假失败了(第一版正是这么写错的)。
    now_local = B.datetime.now(timezone.utc).astimezone()
    for mins, should_accept in ((1, True), (30, True), (59, True), (90, False), (600, False)):
        q = SimpleNamespace(last_done=1.15, trade_status="Normal",
                            timestamp=(now_local - timedelta(minutes=mins)).replace(tzinfo=None))
        got = B._quote_usable(q, "GUARD.US")
        ok = (got is not None) if should_accept else (got is None)
        fails += expect(ok, f"{mins}分钟前的报价应{'接受' if should_accept else '拒绝'}, "
                            f"实际 {'接受' if got is not None else '拒绝'}"
                            f"(按UTC解释会把9小时内的全部放行)")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# ④ 锚点只进不退
#    守: _bump 取 max
#    变异: 锚点盲写(不取max)
# ══════════════════════════════════════════════════════════════════════════

def sc_guard_anchor_never_regresses(s):
    """锚点必须只进不退 —— 乱序写入较小的 id 时不得倒退。

    catch_up 顺序处理时 id 天然递增, 盲写和取 max 结果一样, 所以顺序场景守不住这个性质。
    真正会乱序的地方: catch_up 分页/重试与实时 on_message 交错, 前者手上是旧快照。
    锚点一旦倒退, 那段窗口的消息下次重连整段重放(出场信号重复执行)。
    """
    import discord_enrich_bot as B
    fails = []
    st = {}
    B._bump(st, "ch", 100)
    fails += expect_eq(st["ch"], "100", "首次写入")
    B._bump(st, "ch", 200)
    fails += expect_eq(st["ch"], "200", "更大的 id 应前移")
    B._bump(st, "ch", 150)                       # 乱序: 迟到的旧 id
    fails += expect_eq(st["ch"], "200", "较小的 id 绝不能让锚点倒退")
    B._bump(st, "ch", 200)
    fails += expect_eq(st["ch"], "200", "相同 id 不变")
    # 脏数据不能让它崩(旧版本可能写入非数字)
    st2 = {"ch": "not-a-number"}
    B._bump(st2, "ch", 300)
    fails += expect_eq(st2["ch"], "300", "旧值是脏数据时应直接用新值")
    return fails
