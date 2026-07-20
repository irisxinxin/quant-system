#!/usr/bin/env python3
"""sim/scenarios/discord_layer.py — Discord 接入层端到端场景。

攻击面: catch_up / on_message / bump_last / main() 启动流程。
这一段在 44 个旧场景里【一次都没跑过】—— 它们全是直接调 `_handle`, 从"Discord 收到消息"
到 `_handle` 之间的代码是纯盲区。

要证伪的三条业务铁律:
  L1 停机期间错过的 BUY 严禁按旧价补单(只能提醒)。
  L2 信号绝不能被静默丢弃 —— 锚点(LAST_MSG_JSON)越过某条消息就永不回看。
  L3 只认 频道ID + 作者ID, 昵称可仿冒。

写法约定见 sim/scenario_api.py: 场景返回失败原因列表, 空 = 通过。
异步部分自己用 asyncio.run 包好, 场景函数保持同步签名。
"""
import asyncio
from unittest import mock

import discord_enrich_bot as B
from sim.scenario_api import expect, expect_eq, expect_ge, expect_in, expect_not_in, osi
from sim.fake_discord import DiscordSim

TK = "HOOD"
OSI = osi("HOOD", "260724", "C", 120)
BUY = "$HOOD 7/24 $120 calls $1.00"
EXITMSG = "$HOOD all out"
FLAT = (1.00, 1.00, 1.00)


def _isolate():
    B._rl_until = 0.0
    B._recent_exits.clear()
    B._closing.clear()


def _exit_raises(times=10 ** 9):
    """让出场消息的处理抛异常 —— 给"处理失败 → 锚点保留 → 重试必须能真的重来"这类场景
    提供注入点。

    原来靠 EXIT_MODE=mirror 让 EXIT 真下单、再制造下单失败来触发异常。mirror 出场模式
    已于 2026-07-20 删除(站长出场只播报), 改成直接让播报失败 —— 被测的性质没变:
    处理抛异常时, 出场去重表不得留痕, 否则 10 分钟内的重试会被"重复出场消息跳过"静默吞掉,
    而调用方还以为成功、锚点照常前移 → 信号永久丢失。

    times: 失败多少次后恢复正常 —— "第一次失败、重试成功"才测得到完整重试链路。
    """
    orig = B.push_discord
    state = {"n": 0}

    def patched(msg):
        if "站长出场" in str(msg) and state["n"] < times:
            state["n"] += 1
            raise RuntimeError("模拟播报失败")
        return orig(msg)
    return mock.patch.object(B, "push_discord", patched)


def _sell_orders(s, sym=OSI):
    return [o for o in s.broker.orders.values() if o.side == "Sell" and o.symbol == sym]


def _open_position(s):
    """建一个真实持仓(走完整下单链路), 供出场类场景操作。"""
    s.broker.equity = 1200.0
    s.quotes.set_path(OSI, [FLAT])
    s.quotes.open_interest[OSI] = 50000
    s.send(BUY)
    s.tick()
    return s.pos(OSI)


# ══════════════════════════════════════════════════════════════════════════
# catch_up —— 停机追赶
# ══════════════════════════════════════════════════════════════════════════

def sc_catchup_missed_beyond_100_silently_lost(s):
    """L2: 停机漏了 >100 条时, 第101条之后的出场信号会被锚点永久跳过。

    catch_up 一次只取 history(limit=100) 且【没有翻页循环】; 处理完这100条后锚点停在第100条。
    只要随后来一条实时消息, on_message→bump_last 就把锚点【无条件覆盖】成这条最新消息的 id
    (bump_last 是盲写, 不是取 max, 也不管中间有没有缺口) → 101..N 永远不会再被回看。
    """
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        d.boot()
        base = d.enrich.post("noise 0")
        d.set_anchor(d.enrich.id, base.id)
        d.set_anchor(d.andy.id, 1)
        # 停机期间站长刷了 130 条; 第 120 条是一个出场信号
        buried = None
        for i in range(1, 131):
            if i == 120:
                buried = d.enrich.post(EXITMSG, at=d.ago(30))
            else:
                d.enrich.post(f"$SPY levels watchlist {i}", at=d.ago(30))
        d.catch_up()
        # (原来这里断言 "catch_up 一轮吃不完>100条" 作为复现 bug 的前提。修好翻页后该前提
        #  不再成立 —— 但那只是复现路径, 不是要守的性质。真正要守的是下面的 handled:
        #  埋在第120条的出场信号必须被处理, 无论 catch_up 是一轮吃完还是分页吃完。)
        # 重连后第一条实时消息到达
        live = d.enrich.post("$SPY levels watchlist live", at=d.now())
        d.deliver(live)
        anchor_now = int(d.anchor(d.enrich.id))
        replayed = False
        if anchor_now < buried.id:
            d.catch_up()                       # 再给它一次机会
            replayed = bool(d.logs_with("all out"))
        handled = bool(d.logs_with("all out")) or replayed
        fails += expect(handled,
                        f"[L2 信号丢失] 停机期间第120条出场信号从未被处理, "
                        f"而锚点已推到 {anchor_now} > 该消息 {buried.id} → 永不回看")
    return fails


def sc_catchup_anchor_regresses_on_race_with_live_msg(s):
    """catch_up 与实时 on_message 并发 → 锚点被回退, 已处理的消息会被重放。

    catch_up 在函数开头 `state = _load(...)` 拿一份快照, 中途 `await to_thread(handle)` 会把
    控制权交回事件循环(注释里明说这是为了不挂住心跳), 期间 on_message→bump_last 把新锚点落盘;
    catch_up 结束时 `_save(LAST_MSG_JSON, state)` 用【陈旧快照】整个覆盖回去 = 典型 lost update。
    后果: 锚点倒退 → 下次重连把已经实时处理过的消息当"错过的"重放一遍。
    """
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        d.boot()
        base = d.enrich.post("noise 0")
        d.set_anchor(d.enrich.id, base.id)
        d.set_anchor(d.andy.id, 1)
        d.andy.post("andy noise")
        missed = d.enrich.post("$SPY watchlist levels", at=d.ago(600))
        live_holder = {}

        async def _race():
            gate = asyncio.Event()
            # 卡在【第二个频道】的 history 上: 此时 enrich 频道的追赶已经做完, state 快照定型,
            # 但 catch_up 结尾的 _save 还没执行 —— 真实里这段窗口就是 to_thread(handle) 的耗时。
            d.andy.pre_history = gate.wait
            t = asyncio.create_task(B.catch_up(d.client, s.seen, s.positions))
            for _ in range(5):
                await asyncio.sleep(0)
            live = d.enrich.post(BUY, at=d.now())       # 重连后第一条实时消息
            live_holder["m"] = live
            await d.on_message(live)                    # 它把锚点推到 live.id 并落盘
            gate.set()
            await t

        asyncio.run(_race())
        live = live_holder["m"]
        final = int(d.anchor(d.enrich.id))
        fails += expect_ge(final, live.id,
                           f"[锚点回退] catch_up 的陈旧快照覆盖了 on_message 落的锚点: "
                           f"实时消息 {live.id} 已处理, 锚点却回到 {final} "
                           f"(missed={missed.id}) → 下次重连会重放它")
    return fails


def sc_catchup_andy_channel_anchor_never_advances(s):
    """andy 频道在 catch_up 里【锚点永不前移】→ 每次重连重复处理同一批消息。

    L1323-1325:
        if ch_id == ANDY_CHANNEL_ID:
            handle_andy(...)
            continue          ← 这个 continue 跳过了下面的 state[key] = str(m.id)
    enrich 分支靠 `state[key] = str(m.id)  # 处理成功才前移` 收尾, andy 分支直接 continue 走了。
    后果: 重连风暴里同一批 andy 消息被反复记账(重复 journal / 重复 push "📒 andy观察单"),
    且 andy 侧的追赶窗口永远不收敛。
    """
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        base = d.andy.post("noise 0")
        d.set_anchor(d.andy.id, base.id)
        d.set_anchor(d.enrich.id, 999999)
        m1 = d.andy.post("$NVDA trimming partial here", at=d.ago(600))
        m2 = d.andy.post("$NVDA out full", at=d.ago(300))
        d.catch_up()
        after = d.anchor(d.andy.id)
        fails += expect_eq(after, str(m2.id),
                           "[andy锚点不前移] catch_up 处理完 andy 消息后锚点应指向最后一条")
        n1 = len(d.andy.history_calls)
        d.catch_up()                       # 第二次重连
        replayed = d.andy.history_calls[-1]["got"] if len(d.andy.history_calls) > n1 else 0
        fails += expect_eq(replayed, 0,
                           f"[重复处理] 第二次 catch_up 又把 {replayed} 条 andy 消息回看了一遍 "
                           f"(m1={m1.id} m2={m2.id})")
    return fails


def sc_catchup_recent_exits_poison_defeats_retry(s):
    """出场去重表在【执行前】写入且失败不回滚 → catch_up 的"保留锚点待重试"是假的。

    _handle EXIT 分支顺序: 先 `_recent_exits[_norm] = _now` → 再真正平仓。平仓抛异常时
    catch_up 走 except 保留锚点(注释: "保留锚点: 下次重连会重新回看这条"), 但 10 分钟内的
    重试会被 `if _norm in _recent_exits: return` 直接静默吞掉 —— 而且这次【不抛异常】,
    catch_up 认为处理成功, 锚点照常前移 → 出场信号永久丢失(违反 L2)。

    异常源: 让出场播报抛一次异常(见 _exit_raises)。原来是靠 positions 里一条缺 "status"
    的脏数据在 EXIT 推导 held 时抛 KeyError —— mirror 出场模式删除后不再推导 held。
    """
    _isolate()
    fails = []
    with DiscordSim(s) as d, _exit_raises(times=1):    # 只失败一次, 重试应当成功
        _open_position(s)
        base = d.enrich.post("noise 0")
        d.set_anchor(d.enrich.id, base.id)
        d.set_anchor(d.andy.id, 1)
        d.enrich.post(EXITMSG, at=d.ago(600))

        d.catch_up()                                   # 第一次: 抛错 → 保留锚点
        fails += expect_eq(d.anchor(d.enrich.id), str(base.id),
                           "第一次追赶应因异常保留锚点")
        fails += expect(bool(d.logs_with("保留锚点待下次重试")),
                        "第一次追赶应记录'保留锚点待下次重试'")

        n_before = len(s.evs("exit_signal_mech_ignored"))
        d.catch_up()                                   # 第二次重连(10分钟内)
        skipped = bool(d.logs_with("重复出场消息跳过"))
        acted = len(s.evs("exit_signal_mech_ignored")) > n_before
        fails += expect(acted,
                        f"[L2 信号丢失] 重试被 _recent_exits 静默吞掉(skipped={skipped}), "
                        f"该出场信号从未被受理, 而锚点已前移到 {d.anchor(d.enrich.id)} → 永不回看")
    return fails


def sc_catchup_no_age_gate_on_exit(s):
    """catch_up 的 EXIT 必须有时效闸门 —— 陈年出场令不能当成刚发生的播报出去。

    BUY 有 STALE_BUY_SEC=180 兜底(且 catch_up 直接降级为提醒), EXIT 原本一条时效检查都没有:
    `await to_thread(handle, ...)` 连 msg_dt 都没传, 连 _handle 的迟到闸门都够不着。

    删除 mirror 出场模式之前, 后果是"三天前的全部清仓当场平掉今天刚建的仓"。现在站长出场
    只播报, 后果降级为误导: 把三天前的清仓令播成当前信号, 人看到会以为站长刚刚喊撤。
    锚点退回几天前是真实存在的路径 —— 见 sc_catchup_corrupt_anchor_falls_back_to_stale_bak
    与 sc_catchup_anchor_regresses_on_race_with_live_msg。
    """
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        _open_position(s)
        base = d.enrich.post("noise 0")
        d.set_anchor(d.enrich.id, base.id)
        d.set_anchor(d.andy.id, 1)
        d.enrich.post("all out everything", at=d.ago(3 * 86400))    # 三天前的全局清仓令
        d.catch_up()
        fails += expect_eq(len(s.evs("exit_signal_mech_ignored")), 0,
                           "[无时效闸门] 3天前的出场令被当成当前信号播报")
        fails += expect_eq(len(s.evs("stale_exit_skipped")), 1,
                           "陈年出场应记 stale_exit_skipped(标明已过期), 而不是静默丢弃")
        # 阳性对照: 刚发生的出场必须照常播报, 否则这条场景等于"永远不播报"的空断言
        d.enrich.post("all out now", at=d.ago(60))
        d.catch_up()
        fails += expect_eq(len(s.evs("exit_signal_mech_ignored")), 1,
                           "1分钟前的出场信号应正常播报")
    return fails


def sc_catchup_corrupt_anchor_falls_back_to_stale_bak(s):
    """锚点文件损坏 → _load 回落到 .bak; 若 .bak 是几天前的, 会重放一整个陈年窗口。

    _load 的降级链是为 positions/seen 设计的(宁可用旧账本也不空表), 但对"消息锚点"这种
    单调游标, 回落到旧备份 = 时光倒流。配合上一条(EXIT 无时效闸门), 后果是陈年出场令重放。
    """
    _isolate()
    fails = []
    with DiscordSim(s) as d, _exit_raises():
        _open_position(s)
        old = d.enrich.post("noise old", at=d.ago(3 * 86400))
        d.set_anchor(d.andy.id, 1)
        d.files["enrich_last_msg.json.bak"] = {str(d.enrich.id): str(old.id),
                                               str(d.andy.id): "1"}
        d.files["enrich_last_msg.json"] = {}
        d.corrupt["enrich_last_msg.json"] = ValueError("Expecting value: line 1 column 1")
        d.enrich.post("all out everything", at=d.ago(3 * 86400 - 60))
        n_sell_before = len(_sell_orders(s))
        d.catch_up()
        new_sells = _sell_orders(s)[n_sell_before:]
        fails += expect_eq(len(new_sells), 0,
                           f"[陈年重放] 锚点损坏后回落到3天前的 .bak, 陈年清仓令被执行, "
                           f"新发卖单 {new_sells}")
    return fails


def sc_catchup_break_preserves_anchor_and_retries(s):
    """✅ 正向: 中途异常应 break 并保留锚点, 已处理的前缀落盘, 未处理的下次重试。

    这条是攻击 catch_up 的 except/break 语义本身(_save 在循环外, break 之后到底落什么)。
    """
    _isolate()
    fails = []
    with DiscordSim(s) as d, _exit_raises():
        base = d.enrich.post("noise 0")
        d.set_anchor(d.enrich.id, base.id)
        d.set_anchor(d.andy.id, 1)
        ok = d.enrich.post("$SPY watchlist levels", at=d.ago(600))     # NOISE, 会前移锚点
        bad = d.enrich.post(EXITMSG, at=d.ago(500))
        later = d.enrich.post("$TSLA watchlist levels", at=d.ago(400))
        s.positions["LEGACY_BAD.US"] = {"ticker": "HOOD", "filled": 1, "sold": 0}
        d.catch_up()
        fails += expect(bool(d.logs_with("保留锚点待下次重试")),
                        "阳性对照失效: 根本没触发异常分支, 下面的锚点断言不可信")
        fails += expect_eq(d.anchor(d.enrich.id), str(ok.id),
                           f"[锚点越界] 出错那条({bad.id})之前的前缀应落盘, 出错的与其后的"
                           f"({later.id})不得越过")
        s.positions.pop("LEGACY_BAD.US", None)     # 这条脏数据只是异常源, 别让通用不变式误报
    return fails


def sc_catchup_first_run_anchors_without_replay(s):
    """✅ 正向 + L1: 首次运行(无锚点)必须只锚定到最新, 绝不回放历史、绝不补单。"""
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        for i in range(5):
            d.enrich.post(BUY, at=d.ago(86400))
        newest = d.enrich.post(EXITMSG, at=d.ago(3600))
        d.andy.post("$NVDA out full", at=d.ago(3600))
        d.catch_up()
        fails += expect_eq(d.anchor(d.enrich.id), str(newest.id), "首次运行应锚定到最新一条")
        fails += expect_eq(len(s.broker.orders), 0, "[L1] 首次运行绝不能下任何单")
        fails += expect_eq(len(d.journal_evs("missed_during_downtime")), 0,
                           "首次运行不该产生任何'错过的信号'提醒")
    return fails


def sc_catchup_missed_buy_never_orders(s):
    """✅ L1: 停机期间错过的 BUY 只能提醒, 绝不能按旧价补单 —— 各种 BUY 变体都要挡住。"""
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        base = d.enrich.post("noise 0")
        d.set_anchor(d.enrich.id, base.id)
        d.set_anchor(d.andy.id, 1)
        s.quotes.set_path(OSI, [FLAT])
        s.quotes.open_interest[OSI] = 50000
        for txt in (BUY,                                   # BUY
                    "$HOOD $120 calls $1.00",              # BUY_NOEXPIRY
                    "$HOOD 7/24 $120 lotto $1.00",         # BUY_AMBIG 之类
                    "grabbing some $HOOD 7/24 $120c here"):
            d.enrich.post(txt, at=d.ago(4 * 3600))
        d.catch_up()
        fails += expect_eq(len(s.broker.orders), 0,
                           f"[L1 迟到补单] 停机期间的 BUY 下了单: {s.broker.orders}")
        fails += expect_eq(len(s.positions), 0, "[L1] 不应建任何仓位")
        fails += expect(len(d.journal_evs("missed_during_downtime")) >= 1,
                        "错过的 BUY 至少要留一条 missed_during_downtime 记录")
    return fails


def sc_catchup_history_failure_keeps_anchor(s):
    """✅ 正向: Discord history 拉取失败时不得动锚点, 也不得吞掉后续频道。"""
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        base = d.enrich.post("noise 0")
        d.set_anchor(d.enrich.id, base.id)
        d.set_anchor(d.andy.id, 500)
        d.enrich.post(EXITMSG, at=d.ago(600))
        d.enrich.fail_history = 1
        d.catch_up()
        fails += expect_eq(d.anchor(d.enrich.id), str(base.id),
                           "history 失败后 enrich 锚点不得前移")
        fails += expect(bool(d.logs_with("追赶失败")), "history 失败应有日志")
        fails += expect_eq(d.anchor(d.andy.id), "500",
                           "一个频道失败不应影响另一个频道的锚点")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# on_message —— 实时路径
# ══════════════════════════════════════════════════════════════════════════

def sc_onmessage_anchor_held_when_handle_raises(s):
    """✅ 正向(L2): handle 抛异常时锚点必须保留, 重启后 catch_up 能补回来。"""
    _isolate()
    fails = []
    with DiscordSim(s) as d, _exit_raises():
        d.boot()
        base = d.enrich.post("noise 0")
        d.set_anchor(d.enrich.id, base.id)
        s.positions["LEGACY_BAD.US"] = {"ticker": "HOOD", "filled": 1, "sold": 0}
        m = d.enrich.post(EXITMSG, at=d.now())
        d.deliver(m)
        fails += expect_eq(d.anchor(d.enrich.id), str(base.id),
                           f"[L2] handle 抛异常后锚点不得前移到 {m.id}")
        fails += expect(bool(s.alerts), "处理异常应告警")
        s.positions.pop("LEGACY_BAD.US", None)
    return fails


def sc_onmessage_foreign_author_and_channel_rejected(s):
    """✅ L3: 只认 频道ID + 作者ID。仿冒昵称/别的频道/空正文都不许触发下单。"""
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        d.boot()
        d.set_anchor(d.enrich.id, 10)
        s.quotes.set_path(OSI, [FLAT])
        s.quotes.open_interest[OSI] = 50000
        impostor = d.enrich.post(BUY, at=d.now(), author_id=B.AUTHOR_ID + 1)
        d.deliver(impostor)
        fails += expect_eq(len(s.broker.orders), 0, "[L3] 冒名作者不得下单")
        fails += expect_eq(d.anchor(d.enrich.id), str(impostor.id),
                           "冒名消息属于'明确无需处理', 锚点应前移(否则它会被永久重放)")

        from sim.fake_discord import FakeMessage
        other = type("C", (), {"id": 424242})()
        fm = FakeMessage(77777, BUY, other, B.AUTHOR_ID, d.now())
        anchor_before = d.anchor(d.enrich.id)
        d.deliver(fm)
        fails += expect_eq(len(s.broker.orders), 0, "[L3] 非监听频道不得下单")
        fails += expect_eq(d.anchor(d.enrich.id), anchor_before,
                           "非监听频道不应改动任何锚点")
        fails += expect(str(other.id) not in (d.files.get("enrich_last_msg.json") or {}),
                        "非监听频道不应被写进锚点文件")

        empty = d.enrich.post("", at=d.now())
        d.deliver(empty)
        fails += expect_eq(d.anchor(d.enrich.id), str(empty.id),
                           "空正文属于'明确无需处理', 锚点应安全前移")

        # 阳性对照: 同样一条 BUY, 频道对+作者对 就必须真的下单。
        # 没有这一条, 上面三个"0 单"断言可能只是因为链路根本没通(假绿)。
        s.broker.equity = 1200.0
        real = d.enrich.post(BUY, at=d.now())
        d.deliver(real)
        fails += expect(len(s.broker.orders) >= 1,
                        "阳性对照失效: 合法频道+合法作者的 BUY 也没下单 → 上面的阴性断言不可信")
    return fails


def sc_onmessage_duplicate_delivery_no_double_entry(s):
    """✅ 正向: Discord 重发同一条消息, 不得重复建仓。"""
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        d.boot()
        d.set_anchor(d.enrich.id, 10)
        s.broker.equity = 1200.0
        s.quotes.set_path(OSI, [FLAT] * 6)
        s.quotes.open_interest[OSI] = 50000
        m = d.enrich.post(BUY, at=d.now())
        d.deliver(m)
        n_buy_1 = len([o for o in s.broker.orders.values() if o.side == "Buy"])
        fails += expect(n_buy_1 >= 1, "阳性对照失效: 第一次投递就没下单, 去重断言不可信")
        d.deliver(m)                       # 完全相同的一条(同 msg_id)再来一次
        n_buy_2 = len([o for o in s.broker.orders.values() if o.side == "Buy"])
        fails += expect_eq(n_buy_2, n_buy_1, "重复投递同一条消息不得再下一张买单")
    return fails


def sc_bumplast_corrupt_state_wipes_other_channel(s):
    """锚点文件损坏且无备份时, bump_last 会把【另一个频道】的锚点一并抹掉。

    bump_last = _load → 改一个键 → _save 整份覆盖。_load 对损坏文件走宽松分支返回 {},
    于是写回去的只剩当前频道一个键 → 另一频道退化成"首次运行", 停机窗口被静默丢弃。
    """
    _isolate()
    fails = []

    # ① 真实生命周期: bot 跑起来后锚点已进内存(catch_up/bump_last 都会填充), 再遇文件损坏
    #    → 必须用内存里的游标把另一频道救回来。
    with DiscordSim(s) as d:
        d.files["enrich_last_msg.json"] = {str(d.enrich.id): "10", str(d.andy.id): "20"}
        d.boot()
        import discord_enrich_bot as _B
        _B._anchor_mem.update({str(d.enrich.id): "10", str(d.andy.id): "20"})
        d.corrupt["enrich_last_msg.json"] = ValueError("half-written json")
        d.deliver(d.enrich.post("$SPY watchlist levels", at=d.now()))
        st = d.files.get("enrich_last_msg.json") or {}
        fails += expect_in(str(d.andy.id), st,
                           "[锚点被抹] 内存中有 andy 游标时, 落盘不得把它丢掉")
        fails += expect(str(st.get(str(d.enrich.id), "0")) >= "10",
                        "enrich 自身的游标应前移而非倒退")

    # ② 极端: 内存也没有(损坏发生在任何一次成功读取之前) → 游标物理上已不存在, 变不出来。
    #    这时唯一正确的行为是【告警】, 让人知道有个频道会退化成"首次运行"、丢掉停机窗口。
    _isolate()
    s.alerts.clear()
    with DiscordSim(s) as d:
        d.boot()
        d.files["enrich_last_msg.json"] = {str(d.enrich.id): "10", str(d.andy.id): "20"}
        d.corrupt["enrich_last_msg.json"] = ValueError("half-written json")
        d.deliver(d.enrich.post("$SPY watchlist levels", at=d.now()))
        fails += expect(any("锚点文件损坏" in a for a in s.alerts),
                        f"锚点损坏且内存无备份时必须告警(否则静默丢窗口), 实际告警={s.alerts}")
    return fails


# ══════════════════════════════════════════════════════════════════════════
# main() —— 启动流程
# ══════════════════════════════════════════════════════════════════════════

def sc_main_second_instance_refused(s):
    """✅ 正向: 并发启动第二个实例必须被文件锁挡住(否则两份进程抢写同一状态)。"""
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        rc1 = d.boot()
        fails += expect_eq(rc1, None, "第一个实例应正常启动")
        first_lock = B._lockf
        rc2 = d.boot()                 # 第二次 main(): 同一进程另开 fd, flock 应失败
        fails += expect_eq(rc2, 1, "[单实例] 第二个实例必须以退出码1拒启")
        try:
            first_lock.close()
        except Exception:
            pass
    return fails


def sc_main_stale_lock_from_crash_still_starts(s):
    """✅ 正向: 上次崩溃留下的锁文件(无人持锁)不应把 bot 永久挡在门外。"""
    _isolate()
    fails = []
    with DiscordSim(s) as d:
        (B.OUT).mkdir(parents=True, exist_ok=True)
        (B.OUT / "enrich_bot.lock").write_text("99999")     # 残留 pid, 无进程持锁
        rc = d.boot()
        fails += expect_eq(rc, None, "残留锁文件不应阻止启动")
    return fails


def sc_main_onready_reentrant_double_catchup(s):
    """重连风暴: on_ready 被重复触发时, catch_up 会不会把同一批消息处理两遍。

    discord.py 每次 RESUME/IDENTIFY 都会再发一次 on_ready, 而 on_ready 里对 catch_up
    没有任何重入保护。这里检查第二次 on_ready 是否会重放已经处理过的窗口。
    """
    _isolate()
    fails = []
    with DiscordSim(s) as d, _exit_raises():
        d.boot()
        base = d.enrich.post("noise 0")
        d.set_anchor(d.enrich.id, base.id)
        d.set_anchor(d.andy.id, 1)
        d.enrich.post("$SPY watchlist levels", at=d.ago(600))
        d.ready()
        n_hist = len(d.enrich.history_calls)
        d.ready()
        got = d.enrich.history_calls[-1]["got"] if len(d.enrich.history_calls) > n_hist else 0
        fails += expect_eq(got, 0, f"第二次 on_ready 又回看了 {got} 条已处理消息")
    return fails
