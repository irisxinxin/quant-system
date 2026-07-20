#!/usr/bin/env python3
"""test_enrich_bot_safety.py — enrich bot 资金安全回归测试 (2026-07-20 对抗性QA+复审)。
全程内存mock: 不建真实 Quote/TradeContext, 不碰 output/, 不跑 main(), 不下真实单。
跑法: /usr/local/bin/python3 qa_verify_fixes.py
"""
import os, sys, json, time, tempfile
from pathlib import Path
from datetime import datetime, date, timedelta
from unittest import mock

os.environ["EXIT_MODE"] = "mechanical"
os.environ["ENRICH_LIVE"] = "false"
os.environ["DISCORD_BOT_TOKEN"] = "x"
sys.path.insert(0, "/Users/xin/Documents/Claude/Projects/money/quant_system")

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

print("=" * 72)
print("① 状态文件: 原子写 + 损坏拒启 + .bak恢复")
print("=" * 72)
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "pos.json"
    with mock.patch.object(B, "OUT", Path(td)), mock.patch.object(B, "log", lambda m: None):
        B._save(p, {"A": 1})
        B._save(p, {"A": 1, "B": 2})                      # 第二次写会先备份上一版
        chk("原子写落盘正确", json.loads(p.read_text()) == {"A": 1, "B": 2})
        chk("保留了 .bak 上一版", (Path(td) / "pos.json.bak").exists()
            and json.loads((Path(td) / "pos.json.bak").read_text()) == {"A": 1})
        chk("不留 .tmp 残file", not (Path(td) / "pos.json.tmp").exists())
        p.write_text('{"A": 1, "B"')                       # 模拟写一半被kill
        with mock.patch.object(B, "push_discord", lambda *a, **k: None):
            got = B._load(p)
        chk("损坏时从.bak恢复而非静默清空", got == {"A": 1}, f"got={got}")
        (Path(td) / "pos.json.bak").unlink()
        raised = False
        try:
            with mock.patch.object(B, "push_discord", lambda *a, **k: None):
                B._load(p, strict=True)
        except RuntimeError:
            raised = True
        chk("关键状态损坏且无备份时【抛错拒启】(不返回{})", raised)

print()
print("=" * 72)
print("② close_position: 撤单失败/部分成交/卖失败 三种竞态")
print("=" * 72)
base = dict(_save=lambda *a: None, journal=lambda **k: None,
            push_discord=lambda *a, **k: None, log=lambda m: None)

# 2a: 撤单失败 + 查单失败 → 必须保守记账, 绝不重复卖
pos = {"O1": mkpos(filled=6, sold=0, tp_order_id="T1", tp_qty=2)}
sold_qty = []
with mock.patch.multiple(B, _cancel=lambda oid: False,
                         _order_state=lambda oid: (None, 0, 0.0),
                         _submit=lambda osi, side_buy, qty, price, **k: (sold_qty.append(qty), (True, "X"))[1],
                         **base):
    B.close_position(pos, "O1", "test")
chk("撤单+查单双失败时推迟平仓(既不重复卖也不假定已卖)",
    not sold_qty and pos["O1"]["status"] == "open" and pos["O1"]["sold"] == 0,
    f"卖单={sold_qty} status={pos['O1']['status']} sold={pos['O1']['sold']}")

# 2b: 止盈腿部分成交后被撤 → 成交量必须计入 sold
pos = {"O2": mkpos(filled=6, sold=0, tp_order_id="T2", tp_qty=2)}
sold_qty = []
with mock.patch.multiple(B, _cancel=lambda oid: True,
                         _order_state=lambda oid: ("Canceled", 1, 1.3),   # 撤前已成交1张
                         _submit=lambda osi, side_buy, qty, price, **k: (sold_qty.append(qty), (True, "X"))[1],
                         **base):
    B.close_position(pos, "O2", "test")
chk("止盈腿部分成交1张已记账(6-1=5, 非6)",
    sold_qty and sold_qty[0] == 5, f"实际市价卖 {sold_qty} 张")

# 2c: 市价卖失败 → 不得标记 closed(否则止损永久空转)
pos = {"O3": mkpos(filled=6, sold=0)}
with mock.patch.multiple(B, _cancel=lambda oid: True,
                         _order_state=lambda oid: ("Filled", 0, 0.0),
                         _submit=lambda osi, side_buy, qty, price, **k: (False, "网络错误"),
                         **base):
    B.close_position(pos, "O3", "test")
chk("市价平仓失败时保持open(不弃仓)", pos["O3"]["status"] == "open",
    f"status={pos['O3']['status']}, sold={pos['O3']['sold']}")
chk("平仓失败时不虚增sold", pos["O3"]["sold"] == 0)

# 2d: 重入保护
pos = {"O4": mkpos(filled=3, sold=0)}
calls = []
def _resub(osi, side_buy, qty, price, **k):
    calls.append(qty)
    B.close_position(pos, "O4", "重入")     # 在卖单里再次触发平仓
    return True, "X"
with mock.patch.multiple(B, _cancel=lambda oid: True,
                         _order_state=lambda oid: ("Filled", 0, 0.0),
                         _submit=_resub, **base):
    B.close_position(pos, "O4", "test")
chk("重入时不会双份市价卖", len(calls) == 1, f"提交了 {len(calls)} 次卖单 {calls}")

# 2e: 重入守卫绝不能落盘 — 否则崩溃重启后该仓位永远平不掉(自查发现的自造bug)
saved = []
pos = {"O5": mkpos(filled=6, sold=0)}
with mock.patch.multiple(B, _cancel=lambda oid: True,
                         _order_state=lambda oid: ("Filled", 0, 0.0),
                         _submit=lambda osi, side_buy, qty, price, **k: (False, "网络错误"),
                         _save=lambda p, d: saved.append(json.loads(json.dumps(d, default=str))),
                         journal=lambda **k: None, push_discord=lambda *a, **k: None,
                         log=lambda m: None):
    B.close_position(pos, "O5", "test")
chk("平仓失败后 status 仍为 open", pos["O5"]["status"] == "open")
chk("重入守卫不落盘(落盘=崩溃后永久平不掉)",
    all("closing" not in v for snap in saved for v in snap.values()),
    f"落盘快照数={len(saved)}")
chk("重入守卫在finally已释放", "O5" not in B._closing)

print()
print("=" * 72)
print("②′ 复审找出的【我自己引入的】bug — 回归")
print("=" * 72)

# R1: 429退避下撤单失败+查单失败 → 绝不"假定已卖"后标closed(否则一张没卖却认为已平)
pos = {"R1": mkpos(filled=3, sold=0, stop_order_id="S1", stop_qty=3)}
sold_qty = []
with mock.patch.multiple(B, _cancel=lambda oid: False,
                         _order_state=lambda oid: (None, 0, 0.0),
                         _submit=lambda osi, side_buy, qty, price, **k: (sold_qty.append(qty), (True, "X"))[1],
                         **base):
    B.close_position(pos, "R1", "到期强平")
chk("撤单+查单双失败时不再假定已卖", pos["R1"]["sold"] == 0, f"sold={pos['R1']['sold']}")
chk("此时【不得】标记closed(原会遗弃全部持仓)", pos["R1"]["status"] == "open",
    f"status={pos['R1']['status']}")
chk("此时不发市价卖单, 留待下轮重试", not sold_qty, f"卖单={sold_qty}")
chk("腿ID保留以便下轮继续对账", pos["R1"]["stop_order_id"] == "S1")

# R2: 部分成交(status已open)时, 在途入场买单也必须被撤
pos = {"R2": mkpos(status="open", entry_order_id="E9", filled=1, sold=0, avg=1.0)}
cancels = []
with mock.patch.multiple(B, _cancel=lambda oid: cancels.append(oid) or True,
                         _order_state=lambda oid: ("Filled", 0, 0.0),
                         _submit=lambda *a, **k: (True, "X"), **base):
    B.close_position(pos, "R2", "止损")
chk("平仓时撤在途入场买腿(原因status!=pending而漏撤)", "E9" in cancels, f"cancels={cancels}")
chk("入场腿已清空", pos["R2"].get("entry_order_id") is None)

# R3: executed_price 为空时不得把 avg 清零(会让保护腿+止损全灭)
pos = {"R3": mkpos(status="open", entry_order_id="E10", filled=2, sold=0, avg=1.00,
                   tp_order_id="T9")}
with mock.patch.multiple(B, _order_state=lambda oid: ("PartialFilled", 5, 0.0),  # 均价拿不到
                         ensure_protection=lambda *a: None, _option_last=lambda o: None,
                         _ema15_break_count=lambda t: None, _cancel=lambda o: True,
                         _submit=lambda *a, **k: (True, "X"), us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("成交均价拿不到时保留旧avg(不清零)", pos["R3"]["avg"] == 1.00, f"avg={pos['R3']['avg']}")
chk("filled 仍正常推进", pos["R3"]["filled"] == 5)

# R4: TTL 撤单失败必须保留 entry_order_id(否则filled冻结, 后续成交裸多无人管)
pos = {"R4": mkpos(status="pending", entry_order_id="E11", filled=1, sold=0, avg=1.0,
                   submitted_ts=time.time() - 99999)}
with mock.patch.multiple(B, _order_state=lambda oid: ("PartialFilled", 1, 1.0),
                         _cancel=lambda oid: False,          # 撤单失败
                         ensure_protection=lambda *a: None, _option_last=lambda o: None,
                         _ema15_break_count=lambda t: None, _submit=lambda *a, **k: (True, "X"),
                         us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("TTL撤单失败时保留入场单ID下轮重试", pos["R4"].get("entry_order_id") == "E11",
    f"entry_order_id={pos['R4'].get('entry_order_id')}")

# R5: PartialWithdrawal(长桥"部分成交后被撤"真实终态)必须当终态处理
pos = {"R5": mkpos(status="pending", entry_order_id="E12", filled=0, sold=0, avg=0.0)}
with mock.patch.multiple(B, _order_state=lambda oid: ("PartialWithdrawal", 2, 1.1),
                         ensure_protection=lambda *a: None, _option_last=lambda o: None,
                         _ema15_break_count=lambda t: None, _cancel=lambda o: True,
                         _submit=lambda *a, **k: (True, "X"), us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("PartialWithdrawal 被当作终态", pos["R5"].get("entry_order_id") is None
    and pos["R5"]["status"] == "open", f"status={pos['R5']['status']}")

# R6: 非strict的_load损坏时不抛错(否则on_message第一行bump_last就炸=全部信号丢失)
with tempfile.TemporaryDirectory() as td:
    q = Path(td) / "last.json"; q.write_text("{bad")
    with mock.patch.object(B, "log", lambda m: None), \
         mock.patch.object(B, "push_discord", lambda *a, **k: None):
        chk("非关键状态损坏→返回{}不抛错", B._load(q) == {})
        raised = False
        try:
            B._load(q, strict=True)
        except RuntimeError:
            raised = True
        chk("关键状态(strict)损坏→仍然抛错拒启", raised)

print()
print("=" * 72)
print("③ manage_positions: 入场部分成交 / 查单失败不吞TTL / 到期强平")
print("=" * 72)

# 3a: PartialFilled 必须转 open 并挂保护腿
pos = {"P1": mkpos(status="pending", entry_order_id="E1", filled=0, sold=0, avg=0.0)}
prot = []
with mock.patch.multiple(B, _order_state=lambda oid: ("PartialFilled", 2, 1.00),
                         ensure_protection=lambda po, o, p: prot.append(o),
                         _option_last=lambda o: None, _ema15_break_count=lambda t: None,
                         _cancel=lambda o: True, _submit=lambda *a, **k: (True, "X"),
                         us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("入场部分成交→转open(不再卡pending裸奔)", pos["P1"]["status"] == "open",
    f"status={pos['P1']['status']} filled={pos['P1']['filled']}")
chk("入场部分成交→立刻挂保护腿", "P1" in prot)

# 3b: 查单失败(429) 不得吞掉 TTL 撤单
pos = {"P2": mkpos(status="pending", entry_order_id="E2", filled=0, sold=0, avg=0.0,
                   submitted_ts=time.time() - 99999)}
cancels = []
with mock.patch.multiple(B, _order_state=lambda oid: (None, 0, 0.0),   # 查单全失败
                         _cancel=lambda oid: cancels.append(oid) or True,
                         ensure_protection=lambda *a: None, _option_last=lambda o: None,
                         _ema15_break_count=lambda t: None, _submit=lambda *a, **k: (True, "X"),
                         us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("查单失败时TTL仍然撤单(不再挂一天接刀)", "E2" in cancels, f"cancels={cancels}")

# 3c: 部分成交的入场单, TTL 要撤剩余买腿(而非永不撤)
pos = {"P3": mkpos(status="open", entry_order_id="E3", filled=2, sold=0, avg=1.0,
                   submitted_ts=time.time() - 99999)}
cancels = []
with mock.patch.multiple(B, _order_state=lambda oid: ("PartialFilled", 2, 1.0),
                         _cancel=lambda oid: cancels.append(oid) or True,
                         ensure_protection=lambda *a: None, _option_last=lambda o: None,
                         _ema15_break_count=lambda t: None, _submit=lambda *a, **k: (True, "X"),
                         us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("部分成交的入场单TTL到点撤剩余腿", "E3" in cancels, f"cancels={cancels}")

# 3d: 到期次日上午必须强平(元组比较陷阱)
yesterday = (date.today() - timedelta(days=1)).isoformat()
pos = {"P4": mkpos(status="open", expiry=yesterday, filled=3, sold=0, avg=1.0)}
closed = []
class FakeNow:
    @staticmethod
    def now(tz=None):
        return datetime.now(B.ET).replace(hour=9, minute=30)   # 到期次日 09:30 ET
with mock.patch.multiple(B, close_position=lambda po, o, r: closed.append(r),
                         _order_state=lambda oid: ("Filled", 0, 0.0),
                         ensure_protection=lambda *a: None, _option_last=lambda o: None,
                         _ema15_break_count=lambda t: None, _cancel=lambda o: True,
                         _submit=lambda *a, **k: (True, "X"), us_rth_now=lambda: True, **base), \
     mock.patch.object(B, "datetime", FakeNow):
    B.manage_positions(pos)
chk("到期次日上午即强平(原要等到次日15:40)", "到期强平" in closed, f"closed={closed}")

print()
print("=" * 72)
print("④ 定张与敞口: OI未知帽 / qty<=0 / 低价止损档")
print("=" * 72)

with mock.patch.multiple(B, log=lambda m: None):
    q, note = B.size_qty(0.05, 33330, "X.US", fallback=1)
    with mock.patch.object(B, "_quote_ctx", mock.Mock(option_quote=lambda s: [])):
        q, note = B.size_qty(0.05, 33330, "X.US", fallback=1)
chk("OI拿不到时保守封顶(不再是6666张)", q <= B.OI_UNKNOWN_CAP, f"{q}张 ({note})")

with mock.patch.multiple(B, log=lambda m: None):
    ok, err = B._submit("X.US", side_buy=True, qty=0, price=1.0)
chk("qty=0 被拒绝提交", ok is False, f"{err}")

# 低价期权止损档: avg=0.02 → 0.02*0.4=0.008 低于最小档, 必须抬到 0.01 才可能触发
pos = {"L1": mkpos(filled=3, sold=0, avg=0.02, stop_mult=0.4)}
closed = []
with mock.patch.multiple(B, _option_last=lambda o: 0.01,
                         close_position=lambda po, o, r: closed.append(r),
                         _order_state=lambda oid: ("Filled", 0, 0.0),
                         ensure_protection=lambda *a: None, _ema15_break_count=lambda t: None,
                         _cancel=lambda o: True, _submit=lambda *a, **k: (True, "X"),
                         us_rth_now=lambda: True, **base):
    B.manage_positions(pos)
chk("低价期权止损抬到最小档后可触发(原永不触发)", closed, f"closed={closed}")

print()
print("=" * 72)
print("⑤ 重复建仓与总敞口")
print("=" * 72)
print("(_handle 的跨日覆盖/软去重/敞口闸 依赖完整信号对象, 用源码断言核对)")
src = Path("/Users/xin/Documents/Claude/Projects/money/quant_system/discord_enrich_bot.py").read_text()
chk("跨日重复建仓有守卫(dup_open_skip)", "dup_open_skip" in src and 'old.get("status") in ("pending", "open")' in src)
chk("软去重键已登记(soft_key)", "soft_key" in src and "seen[soft_key]" in src)
chk("总敞口闸存在(gross_cap_skip)", "gross_cap_skip" in src and "MAX_GROSS_FRAC" in src)
chk("manage_positions 已置于 _handle_lock 下", "_managed_tick" in src and "with _handle_lock:\n            manage_positions" in src)

print()
print("=" * 72)
print(f"结果: {len(PASS)} 通过 / {len(FAIL)} 失败")
if FAIL:
    print("失败项:")
    for f in FAIL:
        print("  ❌", f)
print("=" * 72)
sys.exit(1 if FAIL else 0)
