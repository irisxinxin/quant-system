#!/usr/bin/env python3
"""replay/replay.py — 执行者: 用真实Discord时间线 + oracle价格, 回放【bot真实代码】。

按 enrich_history.json 时间线顺序: 推进时钟 → 喂消息给 bot._handle → 持仓期间每60s跑
manage_positions(仅RTH, 与bot真实loop门一致) → 每步查不变式(裸空/账实/孤儿/超卖) + 捕获异常。
引擎=bot本身, 无"另写引擎"分叉。跑法见 run_replay.py。
⚠ 用5分K近似last_done: 抓不到sub-5min针尖(PLTR式), 但隔夜/多bar/账实/竞态类都能抓。
"""
import json, sys, os, bisect, pickle, traceback
from datetime import datetime, timezone, date, time as dtime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bots" / "enrich"))
os.environ.setdefault("ENRICH_LIVE", "true")
os.environ.setdefault("DISCORD_BOT_TOKEN", "sim")
os.environ.setdefault("MECH_BE", "0")                  # 回放线上F配置(可用环境变量切)
# ⚠ POSITION_FRAC 必须在 import bot 前设 —— bot 在模块import时读死它; 默认"0"会退回固定1张,
#   导致 filled=1 → tp1 卖光 → runner/9ema/F阶梯路径【永不触发】(回放白跑那半条链路)。
os.environ.setdefault("POSITION_FRAC", "0.5")
os.environ.setdefault("LOTTO_FRAC", "0.3333")
os.environ.setdefault("ZERO_DTE_FRAC", "0.10")
from unittest import mock
import discord_enrich_bot as B
from sim.harness import Sim
from sim.fakes import FakeQuotes, SimError, ET
UTC = timezone.utc


class ReplayQuotes(FakeQuotes):
    """按 clock 从 oracle 查历史价的 FakeQuotes。cur()/candlesticks() 时间索引。"""
    def __init__(self, clock, opt, und15):
        super().__init__(clock)
        self.opt = opt
        self.und15 = und15
        self._ot = {k: [b[0] for b in v] for k, v in opt.items()}      # osi -> ts列(bisect)
        self._ut = {k: [b[0] for b in v] for k, v in und15.items()}

    def cur(self, symbol):
        bars = self.opt.get(symbol)
        if not bars:
            return None
        now = self.clock.now(UTC)
        i = bisect.bisect_right(self._ot[symbol], now) - 1
        if i < 0:
            return None
        _, o, h, l, c = bars[i]
        return (c, h, l)                                # (last≈收盘, high, low)

    def candlesticks(self, symbol, period, count, adjust):
        tk = symbol.replace(".US", "")
        bars = self.und15.get(tk)
        if not bars:
            raise SimError(f"no candles for {symbol}")
        now = self.clock.now(UTC)
        i = bisect.bisect_right(self._ut[tk], now)
        window = bars[max(0, i - count):i]
        out = []
        for ts, c in window:
            b = type("B", (), {})()
            b.close = c; b.timestamp = ts
            out.append(b)
        return out


def _rth(clock):
    """真实RTH判定(用假时钟ET), 替代harness强制True — 回放要还原bot只在盘中轮询。"""
    now = clock.now(ET)
    return now.weekday() < 5 and dtime(9, 31) <= now.time() <= dtime(15, 58)


class Replay:
    def __init__(self, oracle_path, equity=100_000.0):
        O = pickle.load(open(oracle_path, "rb"))
        self.opt, self.und15, self.meta = O["opt"], O["und15"], O["meta"]
        self.equity = equity
        self.violations = []       # [(ts, [viol...])]
        self.errors = []           # [(ts, context, traceback)]
        self.msg_stats = dict(total=0, buy=0, exit=0, entered=0, alerted=0)

    def run(self, lo, hi):
        msgs = []
        for m in json.load(open(ROOT / "output" / "enrich_history.json")):
            ts = datetime.fromisoformat(m["ts"]); ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
            if lo <= ts.date() <= hi:
                msgs.append((ts, m))
        msgs.sort(key=lambda x: x[0])
        if not msgs:
            print("窗口内无消息"); return self
        start = msgs[0][0].astimezone(ET)
        s = Sim(start_et=start.replace(tzinfo=ET), equity=self.equity)
        s.quotes = ReplayQuotes(s.clock, self.opt, self.und15)
        s.broker.quotes = s.quotes
        with s:
            mock.patch.object(B, "us_rth_now", lambda: _rth(s.clock)).start()   # 还原真实RTH门
            def _jrn(**kv):                                                     # journal补时钟ts(给每单重建生命周期)
                kv.setdefault("ts", str(s.clock.now(UTC)))
                s.events.append(dict(kv))
            mock.patch.object(B, "journal", _jrn).start()
            self.s = s
            for ts, m in msgs:
                self._advance_to(s, ts)                # 补跑持仓管理到这条消息时刻
                self.msg_stats["total"] += 1
                n_ev = len(s.events)
                try:
                    et = ts.astimezone(ET)
                    s.clock.set_et(et.year, et.month, et.day, et.hour, et.minute)
                    B._handle(m["text"], s.clock.now(ET).date(), int(m["id"]),
                              s.seen, s.positions, s.clock.now(ET))
                except Exception:
                    self.errors.append((str(ts), f"handle msg {m['id']}: {m['text'][:60]}", traceback.format_exc()))
                self._classify(s, n_ev, m)
                self._check(s, ts)
            # 收尾: 把所有持仓跑到窗口末+3天(到期强平/排空)
            end = datetime.combine(hi, dtime(20, 0), UTC) + timedelta(days=3)
            self._advance_to(s, end)
            self._check(s, end)
        return self

    def _advance_to(self, s, target):
        """把时钟推进到 target, 持仓期间每60s跑一轮 manage_positions(仅RTH)。无持仓则空转跳过。"""
        guard = 0
        while s.clock.now(UTC) < target and guard < 500_000:
            guard += 1
            active = any(p.get("status") in ("pending", "open", "closing") for p in s.positions.values())
            if not active:
                # 无活仓: 直接跳到 target(或下一个RTH开盘), 省时
                s.clock.advance(min(3600, (target - s.clock.now(UTC)).total_seconds() + 1))
                continue
            s.clock.advance(60)
            if _rth(s.clock):
                try:
                    s.broker.tick(step_price=True)
                    B.manage_positions(s.positions)
                except Exception:
                    self.errors.append((str(s.clock.now(UTC)), "manage_positions", traceback.format_exc()))

    def _classify(self, s, n_ev, m):
        new = [e.get("ev") for e in s.events[n_ev:]]
        logs_tail = " ".join(s.logs[-3:]) if s.logs else ""
        if any(e in ("entry_submit",) for e in new):
            self.msg_stats["entered"] += 1
        if "🟠" in logs_tail or "仅提醒" in logs_tail:
            self.msg_stats["alerted"] += 1

    def _check(self, s, ts):
        v = s.check_invariants(allow_open=True)
        if v:
            self.violations.append((str(ts), v))

    def trades(self):
        """从journal事件重建每个osi生命周期: 入场→各次卖出→是否过夜→粗P&L→是否还留仓。"""
        by = {}
        for e in self.s.events:
            osi = e.get("osi")
            if osi:
                by.setdefault(osi, []).append(e)
        out = []
        for osi, evs in by.items():
            ef = next((e for e in evs if e.get("ev") == "entry_fill"), None)
            if not ef:
                out.append(dict(osi=osi, entered=False,
                                note="入场单未成交(TTL撤或没穿价)" if any(e.get("ev") == "entry_ttl_cancel" for e in evs) else "未成交"))
                continue
            avg, fq = ef.get("avg"), ef.get("qty", 0)
            sells = [e for e in evs if e.get("ev") in ("tp_fill", "exit_fill")]
            sold_q = sum(e.get("qty", e.get("want", 0)) or 0 for e in sells)
            sold_val = sum((e.get("px") or 0) * (e.get("qty", e.get("want", 0)) or 0) for e in sells)
            reasons = [e.get("reason") or e.get("ev") for e in sells]
            edates = [e.get("ts","")[:10] for e in evs if e.get("ev") == "entry_fill"]
            xdates = [e.get("ts","")[:10] for e in sells]
            overnight = bool(xdates and edates and max(xdates) > min(edates))
            fin = self.s.positions.get(osi, {})
            remain = fin.get("filled", 0) - fin.get("sold", 0)
            realized = (sold_val / sold_q / avg - 1) * 100 if sold_q and avg else None
            out.append(dict(osi=osi, entered=True, avg=avg, filled=fq, sold=sold_q,
                            remain=remain, status=fin.get("status"), reduced=fin.get("reduced"),
                            exits=reasons, overnight=overnight,
                            realized_pct=round(realized) if realized is not None else None))
        return out

    def save_capture(self, path):
        from collections import Counter
        cap = dict(meta=self.meta, msg_stats=self.msg_stats,
                   event_counts=dict(Counter(e.get("ev") for e in self.s.events)),
                   violations=self.violations, errors=[(t, c, tb.strip().splitlines()[-1]) for t, c, tb in self.errors],
                   trades=self.trades(),
                   alerts_sample=self.s.alerts[:20])
        json.dump(cap, open(path, "w"), ensure_ascii=False, indent=1, default=str)
        return path

    def report(self):
        print(f"\n{'='*70}\n回放报告 {self.meta['lo']}~{self.meta['hi']} "
              f"(期权路径{self.meta['n_opt']}: 真实K{self.meta['n_real']}/BS{self.meta['n_bs']})")
        print(f"{'='*70}")
        print(f"消息处理: {self.msg_stats['total']} | 入场提交: {self.msg_stats['entered']}")
        from collections import Counter
        evc = Counter(e.get("ev") for e in self.s.events)
        print(f"journal事件分布: {dict(evc)}")
        print(f"\n🔴 不变式违规: {len(self.violations)} 次")
        seen_v = set()
        for ts, vs in self.violations:
            for x in vs:
                key = x.split("]")[0]
                if (key, x[:40]) not in seen_v:
                    seen_v.add((key, x[:40]))
                    print(f"  [{ts[:16]}] {x}")
        print(f"\n💥 异常: {len(self.errors)} 次")
        for ts, ctx, tb in self.errors[:10]:
            print(f"  [{ts[:16]}] {ctx}")
            print(f"     {tb.strip().splitlines()[-1]}")
        return self
