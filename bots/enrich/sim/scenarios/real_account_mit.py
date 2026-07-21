#!/usr/bin/env python3
"""sim/scenarios/real_account_mit.py — 真实盘 MIT 止损路径(切真钱后会走这条)。

痛点(用户 review 挖出): 模拟盘不支持 MIT 触价单 → 走轮询止损; 真实盘支持 MIT →
走券商侧触价单。这两条是【不同代码路径】, 而现有仿真的 fake 只模拟了模拟盘(拒 MIT),
所以真实盘的 MIT 路径从没被测过 —— 切真钱等于跑一段没测过的代码。

这里用 s.broker.mit_supported=True 让 fake 模拟真实盘(MIT 挂得成 + 触价成交),
把真实盘会走的 MIT 止损 + 首档后保本切换补测上。
写法约定见 scenario_api.py。通用不变式(裸空/账实/孤儿/弃仓)由 runner 自动跑。
"""
import discord_enrich_bot as B
from sim.scenario_api import expect, expect_eq, expect_ge, expect_le

OSI = "HOOD260724C120000.US"
BUY = "$HOOD 7/24 $120 calls $1.00"
FLAT = (1.00, 1.00, 1.00)


def _real(s):
    """把 fake 切成真实盘: MIT 挂得成。并复位 bot 的 MIT 探测标志。"""
    s.broker.mit_supported = True
    B._MIT_OK = None            # 让 bot 重新探测(会成功 → 真实盘路径)
    B._rl_until = 0.0
    B._recent_exits.clear()
    B._closing.clear()
    B._optfail.clear()


def _open_real(s, equity=1200.0, path=(FLAT,), oi=500_000):
    _real(s)
    s.broker.equity = equity
    s.quotes.set_path(OSI, list(path))
    s.quotes.open_interest[OSI] = oi
    s.send(BUY)
    s.tick()
    return s.pos(OSI)


def sc_mit_stop_placed_on_broker(s):
    """真实盘: 建仓成交后, 止损应挂成券商侧 MIT 触价单(不是靠轮询)。"""
    fails = []
    p = _open_real(s)
    fails += expect(p is not None and p.get("filled", 0) > 0, "应建仓成交")
    if p:
        fails += expect(bool(p.get("stop_order_id")),
                        "真实盘首档前止损应挂成券商侧 MIT(stop_order_id 有值), 而非空(轮询)")
        # 券商侧确有这张 MIT 卖单
        mits = [o for o in s.broker.live_orders(OSI)
                if o.side == "Sell" and o.order_type == "MIT"]
        fails += expect_ge(len(mits), 1, "券商侧应有一张 MIT 止损卖单")
        if mits:
            # 触发价 = 入场价×stop_mult(-50%)
            fails += expect(abs(mits[0].trigger - p["avg"] * 0.5) < 0.02,
                            f"MIT 触发价应=入场价×0.5(-50%): {mits[0].trigger} vs {p['avg']*0.5:.2f}")
    return fails


def sc_mit_stop_triggers_on_crash(s):
    """真实盘: 首档前崩盘 → 券商侧 MIT 触价成交平仓(不靠 bot 轮询)。"""
    fails = []
    p = _open_real(s)
    # 崩到 -55%(< -50% MIT 触发价)
    s.quotes.set_path(OSI, [(0.44, 0.44, 0.44)] * 3)
    s.tick(n=3)
    fails += expect_eq(s.broker_pos(OSI), 0, "MIT 应已触价平净, 券商侧无持仓")
    fails += expect_ge(s.broker_pos(OSI), 0, "不得裸空")
    return fails


def sc_mit_breakeven_replaces_old_stop(s):
    """真实盘: 首档止盈后, 券商侧止损从 -50% MIT 换成【入场价(保本) MIT】。

    真实 flow(调试确认): ③b轮询止盈先撤旧-50%MIT → 市价卖½ → ⓪对账 reduced=True → 回open →
    ensure_protection 按 _stop_price(现返回入场价)重挂保本价 MIT。全程自然完成, 无双卖。
    守住: 旧 MIT 必撤(不新旧并存), 新 MIT 触发价=入场价。
    """
    fails = []
    p = _open_real(s, path=(FLAT,))
    old_mit = [o for o in s.broker.live_orders(OSI) if o.order_type == "MIT" and o.side == "Sell"]
    fails += expect(len(old_mit) == 1 and abs(old_mit[0].trigger - p["avg"] * 0.5) < 0.02,
                    f"首档前应有一张 -50% MIT: {[(o.trigger) for o in old_mit]}")
    s.quotes.set_path(OSI, [1.00, 1.35, 1.35, 1.20, 1.20, 1.20])   # 摸+35%触首档
    s.tick(n=3)
    p = s.pos(OSI) or {}
    fails += expect(p.get("tp1_done") is True, f"首档止盈应成交: tp1_done={p.get('tp1_done')}")
    live_mit = [o for o in s.broker.live_orders(OSI) if o.order_type == "MIT" and o.side == "Sell"]
    fails += expect_le(len(live_mit), 1, "券商侧最多一张 MIT(旧-50%必须撤, 不能新旧并存)")
    if live_mit:
        fails += expect(abs(live_mit[0].trigger - p["avg"]) < 0.02,
                        f"新 MIT 触发价应=入场价(保本): {live_mit[0].trigger} vs {p['avg']}")
    else:
        fails += ["首档止盈后剩仓应有保本价 MIT 兜底"]
    return fails


def sc_mit_breakeven_triggers_flat(s):
    """真实盘: 首档止盈后保本 MIT 生效 → 价格跌回入场价触价平净, 剩仓不裸奔。"""
    fails = []
    p = _open_real(s, path=(FLAT,))
    s.quotes.set_path(OSI, [1.00, 1.35, 1.35, 1.20])   # 摸+35% 首档止盈 + 重挂保本MIT
    s.tick(n=3)
    fails += expect(bool(s.evs("tp_fill")) or (s.pos(OSI) or {}).get("tp1_done"),
                    "首档应止盈")
    s.quotes.set_path(OSI, [1.00, 0.99, 0.99, 0.99])   # 跌回入场价 → 保本MIT触价
    s.tick(n=4)
    fails += expect_ge(s.broker_pos(OSI), 0, "不得裸空")
    p = s.pos(OSI) or {}
    remain = p.get("filled", 0) - p.get("sold", 0)
    fails += expect(remain == 0 or p.get("status") in ("closing", "closed"),
                    f"跌回入场价 → 保本MIT应触价平剩仓, 实际剩 {remain} status={p.get('status')}")
    return fails
