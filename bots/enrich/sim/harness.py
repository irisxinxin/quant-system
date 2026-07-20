#!/usr/bin/env python3
"""sim/harness.py — 把假件接到 discord_enrich_bot 的【真实逻辑】上, 端到端驱动。

用法:
    from sim.harness import Sim
    s = Sim(equity=100_000)
    s.quotes.set_path("HOOD260724C120000.US", [0.83, 0.9, 1.1, 1.3])
    s.send("$HOOD 7/24 $120 calls $.83")     # 模拟站长发单
    s.run(ticks=10)                           # 推进行情 + 跑仓位管理
    s.assert_invariants()                     # 券商真值裁判

设计要点:
  · 时间: bot 的 time.time() / datetime.now(ET) 全部由 FakeClock 驱动。
  · time.sleep 会推进时钟【并推进券商订单状态机】(但不推进行情) —— 否则 bot 的
    cancel_and_reconcile 轮询等终态时券商永远不动, 每次撤单都"未确认", 测试就失真了。
  · 状态文件全在内存, 绝不碰 output/。
"""
import os, sys, types
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("EXIT_MODE", "mechanical")
os.environ.setdefault("ENRICH_LIVE", "true")      # 让 bot 真的走下单链路(下到假券商)
os.environ.setdefault("DISCORD_BOT_TOKEN", "sim")
os.environ.setdefault("POSITION_FRAC", "0.5")
os.environ.setdefault("LOTTO_FRAC", "0.3333")
os.environ.setdefault("ZERO_DTE_FRAC", "0.10")

import discord_enrich_bot as B
from sim.fakes import FakeClock, FakeQuotes, FakeBroker, SimError, ET, TERMINAL

_real_datetime = datetime


class Violation(Exception):
    pass


class Sim:
    def __init__(self, start_et=None, equity=100_000.0, exit_mode="mechanical"):
        start = start_et or _real_datetime(2026, 7, 20, 10, 0, tzinfo=ET)   # 周一 10:00 ET 盘中
        self.clock = FakeClock(start)
        self.quotes = FakeQuotes(self.clock)
        self.broker = FakeBroker(self.clock, self.quotes, equity=equity)
        self.positions = {}
        self.seen = {}
        self.events = []          # journal 事件
        self.logs = []
        self.pushes = []
        self.alerts = []
        self.saved = []           # 每次落盘的快照(用于崩溃重启测试)
        self.save_ok = True       # False 模拟落盘失败
        self._patches = []
        self.exit_mode = exit_mode

    # ── 打补丁: 把 bot 的外部依赖全换成假件 ──
    def __enter__(self):
        clock = self.clock

        class _TimeShim:
            @staticmethod
            def time():
                return clock.time()

            @staticmethod
            def sleep(sec):
                # 时间流逝 + 券商订单状态机推进(行情不动)。没有这一条, bot 轮询等终态时
                # 券商永远停在 PendingCancel, 所有撤单都会被判"未确认" → 测试全部失真。
                clock.advance(sec)
                self.broker.tick(step_price=False)

        class _DTShim:
            @staticmethod
            def now(tz=None):
                return clock.now(tz)

            def __getattr__(self, k):
                return getattr(_real_datetime, k)

        def _save(p, d):
            if not self.save_ok:
                return False
            self.saved.append((str(p.name), {k: dict(v) for k, v in d.items()}
                               if all(isinstance(v, dict) for v in d.values()) else dict(d)))
            return True

        def _journal(**kv):
            self.events.append(dict(kv))

        def _log(m):
            self.logs.append(str(m))

        def _push(m):
            self.pushes.append(str(m)); return True

        def _alert(m):
            self.alerts.append(str(m)); self.logs.append(str(m))

        p = mock.patch.multiple(
            B,
            _trade_ctx=self.broker,
            _quote_ctx=self.quotes,
            time=_TimeShim,
            datetime=_DTShim(),
            _save=_save,
            journal=_journal,
            log=_log,
            push_discord=_push,
            _alert=_alert,
            us_rth_now=lambda: True,
            EXIT_MODE=self.exit_mode,
            LIVE=True,
        )
        p.start(); self._patches.append(p)
        # 复位【全部】模块级可变全局。漏一个就会跨场景污染 —— 实测漏 _rl_until 时,
        # 任一场景注入429后, 按字母序排在它后面的所有场景都会因退避而假失败(曾污染12个)。
        B._optfail.clear(); B._closing.clear(); B._ema_cache.clear()
        B._recent_exits.clear()      # 出场文本10分钟去重表(按假时钟计时, 跨场景会静默吞消息)
        B._anchor_mem.clear()        # 消息锚点内存镜像(残留会让下个场景的锚点"已经很新"→不回看)
        B._rl_until = 0.0            # 429退避截止时刻
        B._MIT_OK = None
        B._last_equity[0] = None
        return self

    def __exit__(self, *a):
        for p in self._patches:
            p.stop()
        self._patches.clear()

    # ── 驱动 ──
    def send(self, text, msg_id=None, at_et=None):
        """模拟站长在 enrich 频道发一条消息, 走 bot 的真实 handle 链路。"""
        if at_et:
            self.clock.set_et(*at_et)
        mid = msg_id if msg_id is not None else 900000 + len(self.events) + len(self.logs)
        B._handle(text, self.clock.now(ET).date(), mid, self.seen, self.positions,
                  self.clock.now(ET))
        return self

    def tick(self, n=1, seconds=60):
        """推进 n 轮: 行情前进 + 券商撮合 + bot 仓位管理。"""
        for _ in range(n):
            self.clock.advance(seconds)
            self.broker.tick(step_price=True)
            B.manage_positions(self.positions)
        return self

    def run(self, ticks=10, seconds=60):
        return self.tick(ticks, seconds)

    # ── 查询 ──
    def pos(self, osi):
        return self.positions.get(osi)

    def bot_remain(self, osi):
        p = self.positions.get(osi) or {}
        return p.get("filled", 0) - p.get("sold", 0)

    def broker_pos(self, osi):
        return self.broker.net_position(osi)

    def evs(self, name):
        return [e for e in self.events if e.get("ev") == name]

    # ── 不变式裁判(用券商真值, 不用 bot 的账本) ──
    def check_invariants(self, allow_open=False):
        """返回违规列表。空 = 通过。"""
        v = []
        # I3: 券商侧净持仓永不为负 = 绝不裸空
        for sym, q in self.broker.position.items():
            if q < 0:
                v.append(f"[裸空] 券商侧 {sym} 净持仓 {q} < 0")
        if self.broker.rejected_naked:
            v.append(f"[裸空] 券商拒绝了卖超单: {self.broker.rejected_naked}")
        # 记账一致: bot 认为的剩仓 应等于 券商真值
        for osi, p in self.positions.items():
            bot_r = p.get("filled", 0) - p.get("sold", 0)
            brk = self.broker.net_position(osi)
            if p.get("status") == "closed":
                if brk > 0:
                    v.append(f"[弃仓] {osi} bot标closed但券商仍持有{brk}张")
            elif bot_r != brk:
                v.append(f"[账实不符] {osi} bot剩{bot_r}张 vs 券商{brk}张 (status={p.get('status')})")
            if p.get("sold", 0) > p.get("filled", 0):
                v.append(f"[超卖记账] {osi} sold={p.get('sold')} > filled={p.get('filled')}")
        # 已平仓的仓位不该还留活单
        for osi, p in self.positions.items():
            if p.get("status") == "closed":
                live = self.broker.live_orders(osi)
                if live:
                    v.append(f"[孤儿单] {osi} 已标closed但券商侧仍有活单 {live}")
        # 挂出的卖单总量不得超过持仓
        for osi, p in self.positions.items():
            if p.get("status") in ("pending", "open", "closing"):
                live_sell = sum(o.submitted_quantity - o.executed_quantity
                                for o in self.broker.live_orders(osi) if o.side == "Sell")
                brk = self.broker.net_position(osi)
                if live_sell > brk:
                    v.append(f"[潜在裸空] {osi} 在挂卖单{live_sell}张 > 持仓{brk}张")
        return v

    def assert_invariants(self):
        v = self.check_invariants()
        if v:
            raise Violation("; ".join(v))
        return True

    def summary(self):
        return {
            "bot_positions": {k: {kk: vv for kk, vv in p.items()
                                  if kk in ("status", "filled", "sold", "avg", "entry_order_id",
                                            "tp_order_id", "tp2_order_id", "exit_order_id",
                                            "pending_action", "fail_stop", "armed")}
                              for k, p in self.positions.items()},
            "broker_positions": dict(self.broker.position),
            "broker_live_orders": [repr(o) for o in self.broker.live_orders()],
            "fills": self.broker.fills,
            "events": [e.get("ev") for e in self.events],
        }
