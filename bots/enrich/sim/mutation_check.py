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

    # 本轮新增的防线, 每条都要能被对应场景抓住。
    # ⚠ 变异要指向【真正由它守住】的场景, 不是"看起来相关"的场景 —— 否则测的是别的防线:
    #   · MIN_MARGIN 曾指向 sc_bug_ambig_wrong_side_when_price_ran, 但那条是被 MAX_DEV
    #     拦住的(best_dev 0.80 > 0.79), 关掉 MIN_MARGIN 它照样绿 → 假的"空转"报告。
    #     真正只有 MIN_MARGIN 能拦的是"两边都贴合、差距很小"的局面。
    ("消歧模棱两可闸关闭", "AMBIG_MIN_MARGIN", 0.0, "sc_guard_ambig_close_deviation_refuses"),
    ("消歧偏离上限放开", "AMBIG_MAX_DEV", 99.0, "sc_guard_ambig_far_deviation_refuses"),
    ("停机出场时效闸关闭", "STALE_EXIT_SEC", 10**9, "sc_catchup_no_age_gate_on_exit"),
    ("追赶翻页退回单页100条", "CATCHUP_MAX", 100, "sc_catchup_missed_beyond_100_silently_lost"),
    ("心跳间隔放到无穷(等于没有心跳)", "HEARTBEAT_SEC", 10**9,
     "sc_guard_heartbeat_proves_polling_alive"),
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


def _mut_broker_qty_none_passes():
    """把 _broker_qty 查询失败的处理改回旧行为: 查不到就跳过封顶、按本地账本照常卖。
    (旧代码是 `if bq is not None:` 包住封顶段, 查询失败直接落到提交)"""
    orig = B._start_exit

    def patched(positions, osi, p, qty, reason, intent="full"):
        if B._broker_qty(osi) is None:
            # 模拟"没有这道防线": 伪造成查得到且量充足, 让它一路走到提交
            _bq = B._broker_qty
            B._broker_qty = lambda o: max(qty, p.get("filled", 0))
            try:
                return orig(positions, osi, p, qty, reason, intent)
            finally:
                B._broker_qty = _bq
        return orig(positions, osi, p, qty, reason, intent)
    B._start_exit = patched
    return lambda: setattr(B, "_start_exit", orig)


def _mut_sell_budget_unlimited():
    """I3 硬闸失效: _sell_budget 永远返回充足预算(不减去在挂卖单)。"""
    orig = B._sell_budget
    B._sell_budget = lambda p: 10 ** 6
    return lambda: setattr(B, "_sell_budget", orig)


def _mut_quote_age_as_utc():
    """把 naive timestamp 改回按 UTC 解释 —— 年龄偏 8 小时, 新鲜度阈值实际放宽到 9 小时。"""
    orig = B._quote_usable
    from datetime import timezone as _tz

    def patched(q, osi):
        ts = getattr(q, "timestamp", None)
        if ts is not None and ts.tzinfo is None:
            q = type(q)(**{**vars(q), "timestamp": ts.replace(tzinfo=_tz.utc)})
        return orig(q, osi)
    B._quote_usable = patched
    return lambda: setattr(B, "_quote_usable", orig)


def _mut_stop_fallback_uses_current_default():
    """把 _stop_price 的 stop_mult fallback 改回一律用当前 MECH_STOP_MULT(会随策略变)
    —— 老仓位丢 stop_mult 键就被新默认从-60%收紧到-50%, 破坏隔离。"""
    orig = B._stop_price

    def patched(p):
        avg = p.get("avg", 0) or 0.0
        if p.get("be") and (p.get("reduced") or p.get("tp1_done")):
            return max(B.MIN_TICK, avg)
        return max(B.MIN_TICK, avg * p.get("stop_mult", B.MECH_STOP_MULT))
    B._stop_price = patched
    return lambda: setattr(B, "_stop_price", orig)


def _mut_breakeven_off():
    """拆掉保本: _stop_price 无视 be 键, 一律走 avg×stop_mult(回到不保本)。"""
    orig = B._stop_price

    def patched(p):
        avg = p.get("avg", 0) or 0.0
        return max(B.MIN_TICK, avg * p.get("stop_mult", B.MECH_STOP_MULT))
    B._stop_price = patched
    return lambda: setattr(B, "_stop_price", orig)


def _mut_breakeven_ignores_be_flag():
    """拆掉隔离: _stop_price 无视 be 键, 只要 reduced 就保本(会误伤老仓位)。"""
    orig = B._stop_price

    def patched(p):
        avg = p.get("avg", 0) or 0.0
        if p.get("reduced") or p.get("tp1_done"):
            return max(B.MIN_TICK, avg)
        return max(B.MIN_TICK, avg * p.get("stop_mult", B.MECH_STOP_MULT))
    B._stop_price = patched
    return lambda: setattr(B, "_stop_price", orig)


def _mut_anchor_blind_write():
    """锚点回到盲写(不取 max) —— 分页/重试写入较小 id 时锚点会倒退。"""
    orig = B._bump

    def patched(state, key, msg_id):
        state[key] = str(msg_id)
    B._bump = patched
    return lambda: setattr(B, "_bump", orig)


def _mut_anchor_overwrite_whole():
    """锚点落盘回到整份覆盖(不与磁盘合并) —— 并发时冲掉别人刚写的锚点。"""
    orig = B._merge_save_anchor

    def patched(state):
        return B._save(B.LAST_MSG_JSON, dict(state))
    B._merge_save_anchor = patched
    return lambda: setattr(B, "_merge_save_anchor", orig)


LOGIC_MUTATIONS = [
    ("armed闸失效(未武装也拖尾)", _mut_force_armed, "sc_unarmed_ema_no_exit"),
    # 同样注意变异与场景的对应关系:
    #   · sc_submit_lost_response_no_double_sell 里持仓查询是【正常】的(_broker_qty 返回0
    #     走对账收口), 拆"查不到时的处理"根本触发不到 → 要用显式注入查询失败的守卫场景。
    #   · _sell_budget 是纵深防御, 上游 mirror_reduce/close_position 都会先撤净卖腿,
    #     正常路径走不到它 → 只能直接打 _start_exit 才验得到。
    #   · 锚点盲写在顺序处理下和取 max 结果相同(id 天然递增), 要乱序写入才证伪得了。
    ("券商持仓查不到时照常卖", _mut_broker_qty_none_passes,
     "sc_guard_broker_qty_unknown_refuses_sell"),
    ("I3硬闸失效(卖单预算无限)", _mut_sell_budget_unlimited,
     "sc_guard_sell_budget_blocks_stacking"),
    ("报价年龄按UTC解释(偏8小时)", _mut_quote_age_as_utc,
     "sc_guard_quote_age_uses_local_tz"),
    ("止损fallback用当前默认(破坏老仓位隔离)", _mut_stop_fallback_uses_current_default,
     "sc_v2_isolation_survives_stop_mult_loss"),
    ("保本关闭(回到不保本)", _mut_breakeven_off,
     "sc_guard_breakeven_after_tp1"),
    ("保本无视be键(误伤老仓位)", _mut_breakeven_ignores_be_flag,
     "sc_guard_breakeven_isolated_by_be_flag"),
    ("锚点盲写(不取max)", _mut_anchor_blind_write,
     "sc_guard_anchor_never_regresses"),
    ("锚点整份覆盖(不合并)", _mut_anchor_overwrite_whole,
     "sc_catchup_anchor_regresses_on_race_with_live_msg"),
]

MODULES = ["sim.scenarios.normal", "sim.scenarios.adversarial",
           "sim.scenarios.ambig", "sim.scenarios.discord_layer",
           "sim.scenarios.regression_guards", "sim.scenarios.strategy_v2"]


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
