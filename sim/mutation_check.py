#!/usr/bin/env python3
"""sim/mutation_check.py — 变异测试: 看守"测试本身"的测试。

问题: 场景全绿不等于场景有用 —— 断言可能根本没在验证东西(空转)。
本轮教训: 上一版单元测试把 ensure_protection stub 成 no-op, 54 项全绿却漏掉了真正的裸空路径。

做法: 故意把 bot 的关键参数改坏, 看本该抓住它的场景会不会失败。
      抓不住 = 那个场景是空转的, 必须重写。

跑法: /usr/local/bin/python3 sim/mutation_check.py
退出码非0 = 有场景空转。
"""
import sys, importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.harness import Sim
import discord_enrich_bot as B

# (变异描述, bot里的常量名, 坏值, 本应因此失败的场景名)
MUTATIONS = [
    ("止损倍数改到永不触发", "MECH_STOP_MULT", 0.0001, "sc_stop_loss_60pct"),
    ("入场TTL改到永不超时", "ENTRY_TTL_SEC", 10**9, "sc_entry_ttl_cancel"),
    ("迟到闸关闭", "STALE_BUY_SEC", 10**9, "sc_stale_signal_no_entry"),
    ("权利金上限放开", "MAX_PREMIUM", 100.0, "sc_premium_over_max_rejected"),
    ("OI流动性帽放开", "OI_CAP_PCT", 10.0, "sc_oi_cap_limits_qty"),
    ("总敞口闸关闭", "MAX_GROSS_FRAC", 999.0, "sc_gross_exposure_cap"),
    ("0DTE仓位档改成常规档", "ZERO_DTE_FRAC", 0.5, "sc_zero_dte_tenth_size"),
    ("一档止盈倍数改错", "MECH_TP_MULT", 9.99, "sc_full_mechanical_exit"),
    ("二档止盈倍数改错", "MECH_TP2_MULT", 9.99, "sc_full_mechanical_exit"),

    ("报价陈旧阈值放到无穷", "QUOTE_MAX_AGE_SEC", 10**9, None),   # 由对抗场景覆盖
    ("卖单卡死超时放到无穷", "EXIT_STUCK_SEC", 10**9, None),
]

# 逻辑变异: 常量改不动的行为(如"armed闸"), 用包装函数模拟"该守卫失效"。
# (MECH_EMA_N=0 抓不住 sc_unarmed_ema_no_exit 不是场景空转 —— armed 闸在 N 判断【之前】,
#  未武装时不出场是正确行为。要证伪它必须直接拆掉 armed 闸。)
def _mut_force_armed():
    """模拟 armed 闸失效: 每轮仓位管理前把所有持仓强制置为已武装。"""
    orig = B.manage_positions
    def patched(positions):
        for p in positions.values():
            if p.get("status") == "open":
                p["armed"] = True
        return orig(positions)
    B.manage_positions = patched
    return lambda: setattr(B, "manage_positions", orig)


LOGIC_MUTATIONS = [
    ("armed闸失效(未武装也拖尾)", _mut_force_armed, "sc_unarmed_ema_no_exit"),
]

MODULES = ["sim.scenarios.normal", "sim.scenarios.adversarial"]


def _all_scenarios():
    out = {}
    for m in MODULES:
        try:
            mod = importlib.import_module(m)
        except ModuleNotFoundError:
            continue
        for n in dir(mod):
            if n.startswith("sc_"):
                out[n] = getattr(mod, n)
    return out


def _run(fn):
    try:
        with Sim() as s:
            return list(fn(s) or []) + s.check_invariants()
    except Exception as e:
        return [f"{type(e).__name__}: {e}"]


def main():
    scen = _all_scenarios()
    if not scen:
        print("没有场景可测"); return 1
    print(f"{'变异':30}{'目标场景':32}结果")
    print("-" * 86)
    caught = tested = 0
    blind = []
    for desc, attr, bad, target in MUTATIONS:
        if target is None:
            continue
        if not hasattr(B, attr):
            print(f"{desc:30}{target:32}⚠️ bot 无常量 {attr}")
            continue
        if target not in scen:
            print(f"{desc:30}{target:32}⚠️ 场景不存在")
            continue
        tested += 1
        old = getattr(B, attr)
        setattr(B, attr, bad)
        try:
            fails = _run(scen[target])
        finally:
            setattr(B, attr, old)
        if fails:
            caught += 1
            print(f"{desc:30}{target:32}✅ 被抓住")
        else:
            blind.append((desc, target))
            print(f"{desc:30}{target:32}❌ 没抓住 — 该场景空转!")
    for desc, apply_fn, target in LOGIC_MUTATIONS:
        if target not in scen:
            print(f"{desc:30}{target:32}⚠️ 场景不存在"); continue
        tested += 1
        undo = apply_fn()
        try:
            fails = _run(scen[target])
        finally:
            undo()
        if fails:
            caught += 1
            print(f"{desc:30}{target:32}✅ 被抓住")
        else:
            blind.append((desc, target))
            print(f"{desc:30}{target:32}❌ 没抓住 — 该场景空转!")
    print("-" * 86)
    print(f"{caught}/{tested} 个人为缺陷被抓住")
    if blind:
        print("\n空转场景(必须重写):")
        for d, t in blind:
            print(f"  · {t}  抓不住「{d}」")
    return 1 if blind else 0


if __name__ == "__main__":
    sys.exit(main())
