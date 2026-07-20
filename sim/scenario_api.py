#!/usr/bin/env python3
"""sim/scenario_api.py — 场景契约。所有场景文件都按这个接口写, 由 run_sim.py 统一执行。

一个场景 = 一个函数:

    def sc_名字(s: Sim) -> list[str]:
        '''一句话说明这个场景验证什么'''
        ...  配置行情/故障 → s.send(...) → s.run(...) ...
        return check(...)        # 返回【失败原因列表】, 空列表 = 通过

规则:
  1. 场景【只返回失败原因】, 不要 print, 不要 assert(runner 统一汇总)。
  2. 不变式(裸空/账实不符/孤儿单/弃仓)由 runner 自动跑, 场景【不用重复写】——
     场景只写"这个场景特有的预期行为"。
  3. 场景之间必须独立: 每个场景拿到全新的 Sim, 不共享状态。
  4. 严禁碰真实 IO —— Sim 已把 output/ 与所有 API 隔离, 不要绕过它。

工具函数见下方 expect / expect_eq / expect_in。
"""


def expect(cond, msg):
    """cond 为假时返回一条失败原因。用法: fails += expect(x == 1, "x应为1")"""
    return [] if cond else [msg]


def expect_eq(actual, wanted, what):
    return [] if actual == wanted else [f"{what}: 实际={actual!r} 期望={wanted!r}"]


def expect_ge(actual, floor, what):
    return [] if actual >= floor else [f"{what}: 实际={actual!r} 应≥{floor!r}"]


def expect_le(actual, cap, what):
    return [] if actual <= cap else [f"{what}: 实际={actual!r} 应≤{cap!r}"]


def expect_in(needle, haystack, what):
    return [] if needle in haystack else [f"{what}: {needle!r} 不在 {haystack!r} 中"]


def expect_not_in(needle, haystack, what):
    return [] if needle not in haystack else [f"{what}: {needle!r} 不应出现在 {haystack!r}"]


def osi(ticker, yymmdd, right, strike):
    """构造 OSI 代码, 与 bot 的 _osi 一致。osi('HOOD','260724','C',120) →
    HOOD260724C120000.US"""
    return f"{ticker}{yymmdd}{right}{int(round(strike * 1000)):06d}.US"
