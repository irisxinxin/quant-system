#!/usr/bin/env python3
"""sim/fake_discord.py — 假 Discord 接入层, 驱动 bot 的【真实】 catch_up / on_message / main。

现有 Sim(harness.py) 只覆盖到 `_handle` 以内; 从"Discord 收到消息"到 `_handle` 之间的
catch_up / on_message / bump_last / main() 启动流程 从未被仿真跑过。本文件补上这一段。

铁律(上一轮有 agent 漏 patch log(), 往真实 output/enrich_bot.log 写进 48 条假交易):
  · `_load` 在 harness 里【没有】被 patch —— catch_up/bump_last/handle_andy 都会用它读真实
    output/*.json。这里必须换成内存文件系统。
  · `_save` harness 已 patch 成"只记录不落盘", 但那样 _load 读不回自己写的东西, 锚点测试会失真。
    这里再包一层: 写进内存 FS, 同时仍旧 append 到 s.saved(保持旧行为)。
  · 所有状态文件路径(SEEN/POS/LAST_MSG/ANDY_TRACK)+ OUT 全部改指到临时目录, 保证即使某条
    路径绕过了 _load/_save(例如 main() 里的 .bak 播种、锁文件), 也落不到真实 output/。
  · verify_paper_trading() 会真连长桥 API, main() 必须把它换掉。

用法(必须在 `with Sim() as s:` 内部):

    from sim.fake_discord import DiscordSim
    with DiscordSim(s) as d:
        d.enrich.post("$HOOD all out", at=d.now())
        d.set_anchor(d.enrich.id, 1)
        d.catch_up()
        d.anchor(d.enrich.id)          # 锚点现在到哪了
"""
import asyncio
import tempfile
from datetime import timezone
from pathlib import Path
from unittest import mock

import discord_enrich_bot as B

UTC = timezone.utc


# ── 假 Discord 对象 ──────────────────────────────────────────────────────────

class FakeAuthor:
    def __init__(self, aid, bot=True):
        self.id = aid
        self.bot = bot
        self.name = f"user{aid}"

    def __repr__(self):
        return f"<author {self.id}>"


class FakeMessage:
    """只实现 bot 真正读到的字段面(已 grep 确认): id / content / author.id / created_at / channel.id"""

    def __init__(self, mid, content, channel, author_id, created_at):
        self.id = int(mid)
        self.content = content
        self.channel = channel
        self.author = FakeAuthor(author_id)
        self.created_at = created_at        # discord.py 给的是 aware UTC datetime

    def __repr__(self):
        return f"<msg {self.id} {self.content[:24]!r}>"


class FakeChannel:
    """假频道。history() 的语义照 discord.py:
         · 默认 newest-first; oldest_first=True 则 oldest-first
         · after=Object(id=X) 只返回 id > X 的消息
         · limit 是【取多少条】的上限
    """

    def __init__(self, cid, sim, default_author):
        self.id = int(cid)
        self._sim = sim
        self.default_author = default_author
        self.msgs = []                  # 按 id 升序
        self.fail_history = 0           # >0: 接下来 N 次 history 抛错
        self.pre_history = None         # 可选 awaitable 工厂, history 产出前先 await(用来构造竞态)
        self.history_calls = []

    # ── 场景配置 ──
    def post(self, content, at=None, msg_id=None, author_id=None):
        """往频道里"发"一条消息(不投递给 bot, 只进历史)。返回 FakeMessage。"""
        mid = msg_id if msg_id is not None else (self.msgs[-1].id + 1 if self.msgs else 1000)
        m = FakeMessage(mid, content, self,
                        self.default_author if author_id is None else author_id,
                        at or self._sim.now())
        self.msgs.append(m)
        self.msgs.sort(key=lambda x: x.id)
        return m

    # ── discord.TextChannel 接口 ──
    def history(self, limit=100, after=None, before=None, oldest_first=False):
        chan = self

        async def _gen():
            if chan.fail_history > 0:
                chan.fail_history -= 1
                raise RuntimeError("discord 5xx on history")
            if chan.pre_history is not None:
                hook, chan.pre_history = chan.pre_history, None
                await hook()
            ms = list(chan.msgs)
            if after is not None:
                ms = [m for m in ms if m.id > int(after.id)]
            if before is not None:
                ms = [m for m in ms if m.id < int(before.id)]
            if not oldest_first:
                ms.reverse()
            chan.history_calls.append(dict(limit=limit,
                                           after=(int(after.id) if after is not None else None),
                                           oldest_first=oldest_first, got=len(ms)))
            n = 0
            for m in ms:
                if limit is not None and n >= limit:
                    return
                n += 1
                yield m

        return _gen()


class FakeClient:
    """假 discord.Client。收集 @client.event 注册的处理函数, run() 不真连。"""

    def __init__(self, *a, **kw):
        self.handlers = {}
        self.channels = {}
        self.user = "sim-bot#0001"
        self.ran = False

    def event(self, fn):
        self.handlers[fn.__name__] = fn
        return fn

    def get_channel(self, cid):
        return self.channels.get(int(cid))

    def run(self, token, **kw):
        self.ran = True          # main() 走到这里就算启动完成


class _FakeIntents:
    def __init__(self):
        self.message_content = False

    @staticmethod
    def default():
        return _FakeIntents()


class _FakeLoop:
    """替 discord.ext.tasks.loop —— 真 Loop 会在 start() 时创建后台 task, 仿真里不要。"""

    def __init__(self, fn):
        self.fn = fn
        self.started = 0

    def is_running(self):
        return self.started > 0

    def start(self):
        self.started += 1


class _FakeTasks:
    @staticmethod
    def loop(**kw):
        def deco(fn):
            return _FakeLoop(fn)
        return deco


class _DiscordShim:
    """catch_up 只用 discord.Object; main() 用 Intents/Client。"""
    Object = None            # 在 __init__ 里绑真 discord.Object(纯 snowflake 容器, 无副作用)

    def __init__(self, client_factory):
        import discord as _real
        self.Object = _real.Object
        self.Intents = _FakeIntents
        self.Client = client_factory


# ── 主控 ────────────────────────────────────────────────────────────────────

class DiscordSim:
    """在 Sim 内部再补一层: 内存状态文件 + 假频道/客户端 + 真实 catch_up/on_message/main。"""

    def __init__(self, s, andy_author=None):
        self.s = s
        self.files = {}                 # 文件名 -> dict (内存状态文件系统)
        self.corrupt = {}               # 文件名 -> Exception, 模拟损坏(走 _load 的降级分支)
        self.load_calls = []
        self._patches = []
        self._tmp = None
        self.client = FakeClient()
        self.enrich = FakeChannel(B.CHANNEL_ID, self, B.AUTHOR_ID)
        # andy 频道的消息同样由站长转发 bot(AUTHOR_ID)发出 —— 见 enrich_archive.py 的抓取过滤
        self.andy = FakeChannel(B.ANDY_CHANNEL_ID, self,
                                B.AUTHOR_ID if andy_author is None else andy_author)
        self.client.channels = {self.enrich.id: self.enrich, self.andy.id: self.andy}
        self.on_message = None
        self.on_ready = None
        self.main_exit = None

    # ── 上下文 ──
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="enrich_sim_")
        tmp = Path(self._tmp.name)
        d = self

        def _load(p, strict=False):
            name = Path(p).name
            d.load_calls.append(name)
            # main() 会自己 _load 一份 positions/seen 并把它闭包进 on_message。若返回副本,
            # 由 boot() 抓来的 on_message 就跟 Sim 的账本脱钩 —— 场景往 s.positions 里放的
            # 东西 on_message 看不见, 断言会【假绿】。这里直接返回同一个对象, 保持端到端一致。
            if name not in d.corrupt:
                if name == Path(B.POS_JSON).name:
                    return d.s.positions
                if name == Path(B.SEEN_JSON).name:
                    return d.s.seen
            if name in d.corrupt:
                # 复刻真实 _load 的降级链: 损坏 → 试 .bak → strict 抛错 / 否则 {}
                B.log(f"🚨 状态文件损坏 {name}: {d.corrupt[name]}")
                bak = d.files.get(name + ".bak")
                if bak is not None:
                    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in bak.items()}
                if strict:
                    raise RuntimeError(f"状态文件 {name} 损坏且无可用备份")
                return {}
            cur = d.files.get(name)
            if cur is None:
                return {}
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in cur.items()}

        def _save(p, data):
            if not d.s.save_ok:
                return False
            name = Path(p).name
            if name in d.files:                       # 真 _save 会先播 .bak
                d.files[name + ".bak"] = d.files[name]
            d.files[name] = {k: (dict(v) if isinstance(v, dict) else v)
                             for k, v in data.items()}
            d.corrupt.pop(name, None)
            d.s.saved.append((name, dict(d.files[name])))
            return True

        pm = mock.patch.multiple(
            B,
            _load=_load,
            _save=_save,
            OUT=tmp,
            SEEN_JSON=tmp / "enrich_seen_live.json",
            POS_JSON=tmp / "enrich_positions.json",
            LAST_MSG_JSON=tmp / "enrich_last_msg.json",
            ANDY_TRACK=tmp / "andy_tracked.json",
            LOG=tmp / "enrich_bot.log",
            verify_paper_trading=lambda: True,
            discord=_DiscordShim(lambda *a, **kw: self.client),
            tasks=_FakeTasks,
        )
        pm.start()
        self._patches.append(pm)
        return self

    def __exit__(self, *a):
        lf = getattr(B, "_lockf", None)
        if lf is not None:
            try:
                lf.close()
            except Exception:
                pass
            B._lockf = None
        for p in self._patches:
            p.stop()
        self._patches.clear()
        if self._tmp:
            self._tmp.cleanup()
            self._tmp = None

    # ── 时间 ──
    def now(self):
        """当前假时钟的 aware-UTC 时刻(discord 的 created_at 就是这个形状)。"""
        return self.s.clock.now(UTC)

    def ago(self, seconds):
        from datetime import timedelta
        return self.now() - timedelta(seconds=seconds)

    # ── 锚点 ──
    def set_anchor(self, ch_id, msg_id):
        st = self.files.setdefault("enrich_last_msg.json", {})
        st[str(ch_id)] = str(msg_id)
        return self

    def anchor(self, ch_id):
        return (self.files.get("enrich_last_msg.json") or {}).get(str(ch_id))

    # ── 驱动真实入口 ──
    def catch_up(self):
        """跑真实 B.catch_up(client, seen, positions)。"""
        return asyncio.run(B.catch_up(self.client, self.s.seen, self.s.positions))

    def deliver(self, msg):
        """把一条消息投递给真实的 on_message 闭包(需先 boot())。"""
        if self.on_message is None:
            self.boot()
        return asyncio.run(self.on_message(msg))

    def ready(self):
        """跑真实的 on_ready(内含 catch_up + manager.start)。"""
        if self.on_ready is None:
            self.boot()
        return asyncio.run(self.on_ready())

    def boot(self):
        """跑真实的 main() 启动流程, 抓出 on_message / on_ready 闭包。
        返回 None = 正常启动完成; 返回 int = main() 调了 sys.exit(该码)。"""
        try:
            B.main()
        except SystemExit as e:
            self.main_exit = e.code
            return e.code
        self.on_message = self.client.handlers.get("on_message")
        self.on_ready = self.client.handlers.get("on_ready")
        self.main_exit = None
        return None

    # ── 断言辅助 ──
    def journal_evs(self, name):
        return [e for e in self.s.events if e.get("ev") == name]

    def logs_with(self, needle):
        return [l for l in self.s.logs if needle in l]
