#!/usr/bin/env python3
"""test_enrich_bot_safety.py — enrich bot 订单生命周期安全测试。

背景: 2026-07-20 用户审查指出"修过很多局部竞态, 尚未把订单生命周期统一建模"。
本文件针对统一后的状态机 pending → open → closing → closed 断言核心不变式:

  I1  提交成功 ≠ 成交。sold / closed 只能由 manage_positions ⓪ 依据 executed_quantity 更新。
  I2  撤单 = 只是"请求撤销"。未查到终态就当挂单还活着, 【禁止】在其上再发卖单。
  I3  卖出总量绝不超过持仓(超卖=裸空期权=无限风险); 宁可少卖(亏损上限=权利金)。
  I4  入场单仍在途时绝不标 closed(否则后续成交变孤儿仓)。
  I5  价格不可用(0/陈旧/停牌)时不得据以判止损。

全程内存 mock: 不建真实 Quote/TradeContext, 不碰 output/, 不跑 main(), 不下真实单。
跑法: /usr/local/bin/python3 test_enrich_bot_safety.py
"""
import os, sys, json, time, tempfile
from pathlib import Path
from datetime import datetime, date, timedelta, timezone
from unittest import mock

os.environ["EXIT_MODE"] = "mechanical"
os.environ["ENRICH_LIVE"] = "false"
os.environ["DISCORD_BOT_TOKEN"] = "x"
sys.path.insert(0, str(Path(__file__).parent))

import discord_enrich_bot as B

PASS, FAIL = [], []
def chk(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'✅' if cond else '❌'} {name}" + (f"  — {detail}" if detail else ""))

def mkpos(**kw):
    d = dict(ticker="HOOD", entry_order_id=None, qty=6, limit=0.83,
             expiry=(date.today() + timedelta(days=4)).isoformat(),
             filled=6, sold=0, avg=1.00, tp_order_id=None, tp_qty=0,
             tp2_order_id=None, tp2_qty=0, tp1_done=False, tp2_done=False,
             armed=False, submitted_ts=time.time(), stop_mult=0.4,
             status="open", opened=str(date.today()))
    d.update(kw); return d

# 公共桩: 禁止任何真实IO
base = dict(_save=lambda *a: True, journal=lambda **k: None,
            push_discord=lambda *a, **k: False, log=lambda m: None,
            _alert=lambda m: None)
mgr = dict(ensure_protection=lambda *a: None, _option_last=lambda o: None,
           _ema15_break_count=lambda t: None, us_rth_now=lambda: True)

def hdr(t):
    print(); print("=" * 74); print(t); print("=" * 74)


hdr("I1  提交成功 ≠ 成交 — sold/closed 只能由 ⓪ 对账更新")

pos = {"A1": mkpos(filled=6, sold=0)}
with mock.patch.multiple(B, _submit=lambda *a, **k: (True, "X1"), **base):
    ok = B._start_exit(pos, "A1", pos["A1"], 6, "止损-60%")
chk("卖单受理后 status=closing(非closed)", ok and pos["A1"]["status"] == "closing",
    f"status={pos['A1']['status']}")
chk("卖单受理后 sold 不变(未成交)", pos["A1"]["sold"] == 0, f"sold={pos['A1']['sold']}")
chk("记录 exit_order_id 供对账", pos["A1"].get("exit_order_id") == "X1")

pos = {"A2": mkpos(status="closing", filled=6, sold=0, exit_order_id="X2",
                   exit_qty=6, exit_reason="止损-60%", exit_ts=time.time())}
with mock.patch.multiple(B, _order_state=lambda oid: ("Filled", 6, 0.4),
                         _cancel=lambda o: True, _submit=lambda *a, **k: (True, "Z"),
                         **mgr, **base):
    B.manage_positions(pos)
chk("卖单确认全成交 → sold=6 且 closed",
    pos["A2"]["sold"] == 6 and pos["A2"]["status"] == "closed",
    f"sold={pos['A2']['sold']} status={pos['A2']['status']}")

pos = {"A3": mkpos(status="closing", filled=6, sold=0, exit_order_id="X3",
                   exit_qty=6, exit_reason="止损-60%", exit_ts=time.time())}
with mock.patch.multiple(B, _order_state=lambda oid: ("Rejected", 0, 0.0),
                         _cancel=lambda o: True, _submit=lambda *a, **k: (True, "Z"),
                         **mgr, **base):
    B.manage_positions(pos)
chk("卖单被拒 → 退回open继续被管理", pos["A3"]["status"] == "open", f"status={pos['A3']['status']}")
chk("卖单被拒 → sold 仍为0", pos["A3"]["sold"] == 0)

pos = {"A4": mkpos(status="closing", filled=6, sold=0, exit_order_id="X4",
                   exit_qty=6, exit_reason="止损-60%", exit_ts=time.time())}
with mock.patch.multiple(B, _order_state=lambda oid: ("PartialWithdrawal", 2, 0.4),
                         _cancel=lambda o: True, _submit=lambda *a, **k: (True, "Z"),
                         **mgr, **base):
    B.manage_positions(pos)
chk("卖单部分成交 → sold=实际2张, 剩4张退回open",
    pos["A4"]["sold"] == 2 and pos["A4"]["status"] == "open",
    f"sold={pos['A4']['sold']} status={pos['A4']['status']}")

pos = {"A5": mkpos(status="closing", filled=6, sold=0, exit_order_id="X5",
                   exit_qty=6, exit_reason="止损", exit_ts=time.time())}
subs = []
with mock.patch.multiple(B, _order_state=lambda oid: ("New", 0, 0.0), _cancel=lambda o: True,
                         _submit=lambda osi, side_buy, qty, price, **k: (subs.append(qty), (True, "Z"))[1],
                         **mgr, **base):
    B.manage_positions(pos)
chk("卖单在途 → 不再发任何新单(防重复卖)", not subs and pos["A5"]["status"] == "closing",
    f"新单={subs} status={pos['A5']['status']}")

pos = {"A6": mkpos(filled=1, sold=0, avg=1.0, reduced=False)}
with mock.patch.multiple(B, _option_last=lambda o: 5.0, _submit=lambda *a, **k: (False, "券商拒单"),
                         _order_state=lambda oid: ("Filled", 0, 0.0), _cancel=lambda o: True,
                         ensure_protection=lambda *a: None, _ema15_break_count=lambda t: None,
                         us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("轮询止盈下单失败 → 不标closed(只剩1张时的原bug)", pos["A6"]["status"] != "closed",
    f"status={pos['A6']['status']}")
chk("轮询止盈下单失败 → sold不变", pos["A6"]["sold"] == 0)


hdr("I2  撤单必须确认终态 — 未确认就禁止再发卖单")

pos = {"B1": mkpos(filled=6, sold=0, stop_order_id="S1", stop_qty=6)}
subs = []
with mock.patch.multiple(B, _cancel=lambda oid: False, _order_state=lambda oid: (None, 0, 0.0),
                         _submit=lambda osi, side_buy, qty, price, **k: (subs.append(qty), (True, "Z"))[1],
                         **base):
    B.close_position(pos, "B1", "到期强平")
chk("腿未确认终态 → 不发卖单", not subs, f"卖单={subs}")
chk("腿未确认终态 → 保持open下轮重试", pos["B1"]["status"] == "open")
chk("腿未确认终态 → 不猜测sold", pos["B1"]["sold"] == 0)
chk("腿ID保留以便下轮继续对账", pos["B1"]["stop_order_id"] == "S1")

pos = {"B2": mkpos(filled=6, sold=0, tp_order_id="T1", tp_qty=2)}
subs = []
with mock.patch.multiple(B, _cancel=lambda oid: True, _order_state=lambda oid: ("New", 0, 0.0),
                         _submit=lambda osi, side_buy, qty, price, **k: (subs.append(qty), (True, "Z"))[1],
                         **base):
    B.close_position(pos, "B2", "止损")
chk("撤单请求成功但状态仍New → 视为未确认, 不发卖单", not subs, f"卖单={subs}")

pos = {"B3": mkpos(filled=6, sold=0, tp_order_id="T2", tp_qty=2)}
subs = []
with mock.patch.multiple(B, _cancel=lambda oid: True, _order_state=lambda oid: ("Canceled", 1, 1.3),
                         _submit=lambda osi, side_buy, qty, price, **k: (subs.append(qty), (True, "Z"))[1],
                         **base):
    B.close_position(pos, "B3", "止损")
chk("腿撤前成交1张已记账 → 只卖剩余5张", subs == [5], f"卖单={subs} sold={pos['B3']['sold']}")

p = mkpos(stop_order_id="S9", stop_qty=3)
with mock.patch.multiple(B, _cancel=lambda o: True,
                         _order_state=lambda o: ("PendingCancel", 0, 0.0), **base):
    r = B.cancel_stop(p)
chk("cancel_stop 未终态返回False且保留ID", r is False and p["stop_order_id"] == "S9",
    f"ret={r} id={p.get('stop_order_id')}")

pos = {"B4": mkpos(status="open", entry_order_id="E1", filled=1, sold=0, avg=1.0)}
subs = []
with mock.patch.multiple(B, _cancel=lambda oid: True, _order_state=lambda oid: ("Canceled", 3, 1.0),
                         _submit=lambda osi, side_buy, qty, price, **k: (subs.append(qty), (True, "Z"))[1],
                         **base):
    B.close_position(pos, "B4", "止损")
chk("撤入场单竞态成交回填 filled=3 并全部卖出", pos["B4"]["filled"] == 3 and subs == [3],
    f"filled={pos['B4']['filled']} 卖单={subs}")

p = mkpos(filled=6, sold=0)
B._credit_leg(p, "tp_order_id", 2)
B._credit_leg(p, "tp_order_id", 2)
B._credit_leg(p, "tp_order_id", 3)
chk("同一腿重复对账不重复计入sold", p["sold"] == 3, f"sold={p['sold']}")


hdr("I3 / I4  绝不超卖; 入场单在途时绝不标 closed")

# C1: 部分成交 → 撤单请求生效后订单转 Canceled(真实语义) → filled定型、余量撤净
cancelled = {"done": False}
def _os_c1(oid):
    return ("Canceled", 2, 1.0) if cancelled["done"] else ("PartialFilled", 2, 1.0)
def _cx_c1(oid):
    cancelled["done"] = True; return True
pos = {"C1": mkpos(status="pending", entry_order_id="E2", filled=0, sold=0, avg=0.0)}
prot = []
with mock.patch.multiple(B, _order_state=_os_c1, _cancel=_cx_c1,
                         ensure_protection=lambda po, o, pp: prot.append(o),
                         _option_last=lambda o: None, _ema15_break_count=lambda t: None,
                         _submit=lambda *a, **k: (True, "Z"), us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("入场部分成交 → 立刻撤余量(filled定型)", pos["C1"].get("entry_order_id") is None
    and pos["C1"]["filled"] == 2,
    f"entry={pos['C1'].get('entry_order_id')} filled={pos['C1']['filled']}")
chk("入场部分成交 → 转open并配保护腿", pos["C1"]["status"] == "open" and "C1" in prot)

# C1b: 撤单确认不了(订单一直非终态) → 保留ID下轮重试, 但【已成交部分必须立刻受保护】
pos = {"C1b": mkpos(status="pending", entry_order_id="E2b", filled=0, sold=0, avg=0.0)}
prot = []
with mock.patch.multiple(B, _order_state=lambda oid: ("PartialFilled", 2, 1.0),
                         _cancel=lambda oid: True,
                         ensure_protection=lambda po, o, pp: prot.append(o),
                         _option_last=lambda o: None, _ema15_break_count=lambda t: None,
                         _submit=lambda *a, **k: (True, "Z"), us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("撤单未确认 → 保留入场单ID下轮重试", pos["C1b"].get("entry_order_id") == "E2b")
chk("撤单未确认 → 已成交部分仍立即转open并保护(不裸奔)",
    pos["C1b"]["status"] == "open" and "C1b" in prot,
    f"status={pos['C1b']['status']} prot={prot}")

pos = {"C2": mkpos(status="open", entry_order_id="E3", filled=2, sold=0, avg=1.0,
                   tp_order_id="T3", tp_qty=2)}
def _os_c2(oid):
    return ("Filled", 2, 1.3) if oid == "T3" else ("PartialFilled", 2, 1.0)
with mock.patch.multiple(B, _order_state=_os_c2, _cancel=lambda o: False,
                         ensure_protection=lambda *a: None, _option_last=lambda o: None,
                         _ema15_break_count=lambda t: None, _submit=lambda *a, **k: (True, "Z"),
                         us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("入场单在途时净仓归零也不标closed(防孤儿仓)", pos["C2"]["status"] != "closed",
    f"status={pos['C2']['status']} entry={pos['C2'].get('entry_order_id')}")


hdr("I3 硬闸  卖单总量绝不超过剩仓 —— 用【真实】ensure_protection, 不 stub")
# 上一版测试把 ensure_protection stub 成 no-op, 导致 54项全绿却漏掉了真正的裸空路径。

B._MIT_OK = False           # 模拟盘: 不支持MIT, 走两档GTC止盈分支(即出问题的那条)
for filled, sold in ((6, 4), (9, 6), (12, 8), (30, 20)):
    p = mkpos(filled=filled, sold=sold, avg=1.0, tp_order_id=None, tp2_order_id=None,
              tp1_done=False, tp2_done=False)
    placed = []
    def _sub(osi, side_buy, qty, price=None, **k):
        placed.append(qty); return True, f"O{len(placed)}"
    with mock.patch.multiple(B, _submit=_sub, **base):
        B.ensure_protection({"N": p}, "N", p)
    remain = filled - sold
    chk(f"部分平仓后挂单总量≤剩仓 (filled={filled} sold={sold} remain={remain})",
        sum(placed) <= remain, f"挂出{placed}=共{sum(placed)}张 vs 剩{remain}张")

# 正常新建仓(sold=0)的档位必须与回测口径一致, 不被这次修复改变
p = mkpos(filled=9, sold=0, avg=1.0)
placed = []
with mock.patch.multiple(B, _submit=lambda osi, side_buy, qty, price=None, **k:
                         (placed.append(qty), (True, "O"))[1], **base):
    B.ensure_protection({"N2": p}, "N2", p)
chk("新建仓 sold=0 时档位不变(⅓+⅓, 与回测一致)", placed == [3, 3], f"挂出{placed}")

# _sell_budget: 已挂未成交卖单要占额度
p = mkpos(filled=9, sold=0, tp_order_id="T", tp_qty=3, tp2_order_id="T2", tp2_qty=3)
chk("_sell_budget 扣除已挂未成交卖单", B._sell_budget(p) == 3, f"budget={B._sell_budget(p)}")
p2 = mkpos(filled=9, sold=6, tp_order_id=None, tp2_order_id=None)
chk("_sell_budget 扣除已卖出", B._sell_budget(p2) == 3, f"budget={B._sell_budget(p2)}")

# tp1_done 只在全部成交时才落(部分成交置位会永久关闭该止盈通道)
pos = {"E1": mkpos(status="closing", filled=9, sold=0, exit_order_id="X",
                   exit_qty=3, exit_intent="tp1", exit_reason="轮询止盈", exit_ts=time.time())}
with mock.patch.multiple(B, _order_state=lambda oid: ("PartialWithdrawal", 1, 1.3),
                         _cancel=lambda o: True, _submit=lambda *a, **k: (True, "Z"),
                         **mgr, **base):
    B.manage_positions(pos)
chk("止盈单只成交1/3张 → 不置tp1_done(通道不被永久关闭)",
    not pos["E1"].get("tp1_done") and pos["E1"]["sold"] == 1,
    f"tp1_done={pos['E1'].get('tp1_done')} sold={pos['E1']['sold']}")

# closing 逃生舱: 卖单久挂不成 → 主动撤单对账退回 open, 不再永久卡死
pos = {"E2": mkpos(status="closing", filled=6, sold=0, exit_order_id="X2", exit_qty=6,
                   exit_reason="止损-60%", exit_ts=time.time() - 99999)}
with mock.patch.multiple(B, _order_state=lambda oid: ("Canceled", 0, 0.0),
                         _cancel=lambda o: True, _submit=lambda *a, **k: (True, "Z"),
                         **mgr, **base):
    B.manage_positions(pos)
chk("卖单久挂 → 撤单对账后退回open(不再永久卡closing)", pos["E2"]["status"] == "open",
    f"status={pos['E2']['status']}")

# TTL 竞态: 撤单请求成功但订单其实已全部成交 → 绝不能标 closed
pos = {"E3": mkpos(status="pending", entry_order_id="E9", filled=0, sold=0, avg=0.0,
                   submitted_ts=time.time() - 99999)}
with mock.patch.multiple(B, _order_state=lambda oid: ("Filled", 6, 0.83),
                         _cancel=lambda o: True, ensure_protection=lambda *a: None,
                         _option_last=lambda o: None, _ema15_break_count=lambda t: None,
                         _submit=lambda *a, **k: (True, "Z"), us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("TTL撤单竞态期已全成交 → 转open而非closed(防仓位失联)",
    pos["E3"]["status"] == "open" and pos["E3"]["filled"] == 6,
    f"status={pos['E3']['status']} filled={pos['E3']['filled']}")


hdr("P0 复审第三轮: 待办持久化 / 关键落盘 / closing覆盖 / 孤儿入场单")

# P0-1: 平仓被推迟必须留下【持久化的待办】, 且下轮真的重试
pos = {"F1": mkpos(filled=6, sold=0, tp_order_id="T", tp_qty=2)}
with mock.patch.multiple(B, _cancel=lambda o: False, _order_state=lambda o: (None, 0, 0.0),
                         _submit=lambda *a, **k: (True, "Z"), **base):
    B.close_position(pos, "F1", "站长all out")
chk("平仓推迟 → 写入 pending_action(不再静默丢弃一次性出场信号)",
    pos["F1"].get("pending_action") == "full_exit"
    and pos["F1"].get("pending_action_reason") == "站长all out",
    f"pending={pos['F1'].get('pending_action')} reason={pos['F1'].get('pending_action_reason')}")

# 下一轮: 挂单确认终态了 → ⓪b 必须真的重试并发出卖单
subs = []
with mock.patch.multiple(B, _cancel=lambda o: True, _order_state=lambda o: ("Canceled", 0, 0.0),
                         _submit=lambda osi, side_buy, qty, price, **k: (subs.append(qty), (True, "Z"))[1],
                         ensure_protection=lambda *a: None, _option_last=lambda o: None,
                         _ema15_break_count=lambda t: None, us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("下一轮 ⓪b 自动重试待办平仓并发出卖单", subs == [6], f"卖单={subs}")
chk("重试后进入 closing 等成交确认", pos["F1"]["status"] == "closing")

# 正在 closing 时收到全平信号 → 记为待办(动作升级), 不静默丢弃
pos = {"F2": mkpos(status="closing", filled=9, sold=0, exit_order_id="X", exit_qty=3,
                   exit_intent="tp1", exit_reason="轮询止盈", exit_ts=time.time())}
with mock.patch.multiple(B, _cancel=lambda o: True, _order_state=lambda o: ("New", 0, 0.0),
                         _submit=lambda *a, **k: (True, "Z"), **base):
    B.close_position(pos, "F2", "站长all out")
chk("closing 期间收到全平 → 记为待办(动作升级), 不丢弃",
    pos["F2"].get("pending_action") == "full_exit", f"pending={pos['F2'].get('pending_action')}")

# P0-2: 卖单已受理但落盘失败 → fail_stop, 停止该仓位自动交易
pos = {"F3": mkpos(filled=6, sold=0)}
with mock.patch.multiple(B, _submit=lambda *a, **k: (True, "OID"), _save=lambda *a: False,
                         journal=lambda **k: None, push_discord=lambda *a, **k: False,
                         log=lambda m: None, _alert=lambda m: None):
    B._start_exit(pos, "F3", pos["F3"], 6, "止损")
chk("卖单落盘失败 → 该仓位置 fail_stop", pos["F3"].get("fail_stop") is True,
    f"fail_stop={pos['F3'].get('fail_stop')}")
subs = []
with mock.patch.multiple(B, _order_state=lambda o: ("Filled", 6, 0.4), _cancel=lambda o: True,
                         _submit=lambda osi, side_buy, qty, price, **k: (subs.append(qty), (True, "Z"))[1],
                         **mgr, **base):
    B.manage_positions(pos)
chk("fail_stop 仓位不再被自动交易(等人工)", not subs, f"卖单={subs}")

# P0-3: closing 仓位不得被同合约新信号覆盖 / 必须计入敞口 / EXIT要能找到它
chk("ACTIVE_STATUSES 含 closing", "closing" in B.ACTIVE_STATUSES)
_pc = {"Z.US": mkpos(status="closing", ticker="AAPL")}
with mock.patch.object(B, "log", lambda m: None):
    chk("closing 仓位能被 EXIT(scope=all) 找到, 不会被当成无持仓",
        B._llm_exit_targets(_pc, {"action": "exit_full", "scope": "all"}) == ["Z.US"])

# P0-4: 卖单卡死恢复时, 入场单仍在途则不得标 closed
pos = {"F4": mkpos(status="closing", filled=1, sold=0, entry_order_id="E", exit_order_id="X",
                   exit_qty=1, exit_reason="止损", exit_ts=time.time() - 99999)}
with mock.patch.multiple(B, _order_state=lambda oid: ("Filled", 1, 0.4) if oid == "X"
                         else ("New", 1, 1.0),
                         _cancel=lambda o: True, _submit=lambda *a, **k: (True, "Z"),
                         **mgr, **base):
    B.manage_positions(pos)
chk("卡死恢复时入场单在途 → 不标closed(防孤儿仓)",
    pos["F4"]["status"] != "closed", f"status={pos['F4']['status']} entry={pos['F4'].get('entry_order_id')}")

# 敞口: 在途入场单按最大潜在张数计(部分成交后不得骤降)
src0 = Path(__file__).with_name("discord_enrich_bot.py").read_text()
chk("敞口计算含 closing 且按在途最大张数", "_exp_qty" in src0 and "ACTIVE_STATUSES" in src0)
chk("启动含券商对账(只读)", "def reconcile_with_broker" in src0
    and "reconcile_with_broker(positions)" in src0)
chk("catch_up 锚点改为处理成功后前移", "处理成功才前移锚点" in src0)


hdr("I5  价格不可用时不得判止损")

class Q:
    def __init__(self, last, age_sec=0, st="Normal"):
        self.last_done = last
        self.timestamp = datetime.now(timezone.utc) - timedelta(seconds=age_sec)
        self.trade_status = st
        self.open_interest = 5000

def _q(last, age=0, st="Normal"):
    return mock.Mock(option_quote=lambda syms: [Q(last, age, st)])

with mock.patch.multiple(B, log=lambda m: None):
    with mock.patch.object(B, "_quote_ctx", _q(0)):
        chk("last_done=0 → 价格不可用(原会立刻假止损全平)", B._option_last("X.US") is None)
    with mock.patch.object(B, "_quote_ctx", _q(1.5, age=7200)):
        chk("报价陈旧2小时 → 不可用", B._option_last("X.US") is None)
    with mock.patch.object(B, "_quote_ctx", _q(1.5, age=1200)):
        chk("报价20分钟(低流动性常态) → 仍可用, 止损不失效", B._option_last("X.US") == 1.5)
    class NaiveQ(Q):
        def __init__(s2):
            super().__init__(1.5); s2.timestamp = datetime.utcnow()   # naive UTC
    with mock.patch.object(B, "_quote_ctx", mock.Mock(option_quote=lambda syms: [NaiveQ()])):
        chk("naive时间戳按UTC解释(不被本机SGT当成8小时前致止损全灭)",
            B._option_last("X.US") == 1.5)
    with mock.patch.object(B, "_quote_ctx", _q(1.5, st="Halted")):
        chk("停牌 → 不可用", B._option_last("X.US") is None)
    with mock.patch.object(B, "_quote_ctx", _q(1.5)):
        chk("正常新鲜报价 → 可用", B._option_last("X.US") == 1.5)

pos = {"D1": mkpos(filled=6, sold=0, avg=1.0)}
subs = []
with mock.patch.multiple(B, _option_last=lambda o: None,
                         _order_state=lambda oid: ("Filled", 0, 0.0), _cancel=lambda o: True,
                         _submit=lambda osi, side_buy, qty, price, **k: (subs.append(qty), (True, "Z"))[1],
                         ensure_protection=lambda *a: None, _ema15_break_count=lambda t: None,
                         us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("价格不可用 → 不触发止损平仓", not subs and pos["D1"]["status"] == "open")

pos = {"D2": mkpos(filled=6, sold=0, avg=0.02, stop_mult=0.4)}
closed = []
with mock.patch.multiple(B, _option_last=lambda o: 0.01,
                         close_position=lambda po, o, r: closed.append(r),
                         _order_state=lambda oid: ("Filled", 0, 0.0), _cancel=lambda o: True,
                         ensure_protection=lambda *a: None, _ema15_break_count=lambda t: None,
                         _submit=lambda *a, **k: (True, "Z"), us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("低价期权止损抬到最小报价档后可触发", closed, f"closed={closed}")


hdr("其余审查项")

_p = {"X.US": mkpos(ticker="AAPL"), "Y.US": mkpos(ticker="MSFT")}
with mock.patch.object(B, "log", lambda m: None):
    chk("LLM scope=ticker 但 tickers=[] → 不出场(原会全仓平)",
        B._llm_exit_targets(_p, {"action": "exit_full", "scope": "ticker", "tickers": []}) == [])
    chk("LLM scope 不可识别 → 不出场",
        B._llm_exit_targets(_p, {"action": "exit_full", "scope": "???"}) == [])
    chk("LLM scope=all → 正常全平",
        len(B._llm_exit_targets(_p, {"action": "exit_full", "scope": "all"})) == 2)
    chk("LLM scope=ticker 指名AAPL → 只平AAPL",
        B._llm_exit_targets(_p, {"action": "exit_full", "scope": "ticker",
                                 "tickers": ["AAPL"]}) == ["X.US"])

with mock.patch.object(B, "log", lambda m: None):
    q, note = B.size_qty(0, 50000, "X.US", fallback=1)
chk("premium=0 不再除零崩溃", q == 0, f"qty={q} ({note})")
with mock.patch.multiple(B, log=lambda m: None):
    ok, err = B._submit("X.US", side_buy=True, qty=0, price=1.0)
chk("qty=0 被拒绝提交", ok is False, f"{err}")
with mock.patch.object(B, "_quote_ctx", mock.Mock(option_quote=lambda s: [])), \
     mock.patch.object(B, "log", lambda m: None):
    q, note = B.size_qty(0.05, 33330, "X.US", fallback=1)
chk("OI拿不到时保守封顶", q <= B.OI_UNKNOWN_CAP, f"{q}张 ({note})")

src = Path(__file__).with_name("discord_enrich_bot.py").read_text()
chk("#1 开锁文件前先 OUT.mkdir", "OUT.mkdir(parents=True, exist_ok=True)   # 全新部署" in src)
chk("#15 RTH窗口收紧到 9:31-15:58", "dtime(9, 31)" in src and "dtime(15, 58)" in src)
chk("#17 catch_up 覆盖 BUY_NOEXPIRY", '("BUY", "BUY_AMBIG", "BUY_NOEXPIRY")' in src)
chk("#18 通知不再谎称'首档后移保本'", "首档后移保本" not in src)
chk("#14 DRY_RUN 独立去重表", "enrich_seen_dry.json" in src)
chk("#8 入场落盘失败会撤单", "entry_persist_failed" in src)
chk("跨日重复建仓有守卫", "dup_open_skip" in src)
chk("总敞口闸存在", "gross_cap_skip" in src)
chk("manage_positions 置于 _handle_lock 下",
    "_managed_tick" in src and "with _handle_lock:\n            manage_positions" in src)

with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "s.json"
    with mock.patch.object(B, "OUT", Path(td)), mock.patch.object(B, "log", lambda m: None):
        chk("#8 _save 成功返回True", B._save(f, {"a": 1}) is True)
        chk("#8 _save 失败返回False", B._save(Path("/nonexistent-dir-xyz/s.json"), {"a": 1}) is False)

with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "pos.json"
    with mock.patch.object(B, "OUT", Path(td)), mock.patch.object(B, "log", lambda m: None):
        B._save(f, {"A": 1}); B._save(f, {"A": 1, "B": 2})
        chk("原子写: 主文件是新版", json.loads(f.read_text()) == {"A": 1, "B": 2})
        chk("原子写: .bak 是上一版",
            json.loads((Path(td) / "pos.json.bak").read_text()) == {"A": 1})
        chk("原子写: 无 .tmp 残留", not list(Path(td).glob("*.tmp")))
        f.write_text('{"A": 1, "B"')
        with mock.patch.object(B, "push_discord", lambda *a, **k: None):
            chk("损坏 → 从.bak恢复", B._load(f) == {"A": 1})
            (Path(td) / "pos.json.bak").unlink()
            chk("非关键状态损坏 → 返回{}不拒启", B._load(f) == {})
            raised = False
            try:
                B._load(f, strict=True)
            except RuntimeError:
                raised = True
            chk("关键状态损坏且无备份 → 抛错拒启", raised)

print()
print("=" * 74)
print(f"结果: {len(PASS)} 通过 / {len(FAIL)} 失败")
for f in FAIL:
    print("  ❌", f)
print("=" * 74)
sys.exit(1 if FAIL else 0)
