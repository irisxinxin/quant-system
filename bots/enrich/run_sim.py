#!/usr/bin/env python3
"""run_sim.py — enrich bot 端到端仿真回归。

每个场景拿一个全新的 Sim(假券商/假行情/假时钟), 驱动 bot 的真实逻辑, 然后:
  ① 跑该场景自己的预期断言
  ② 跑【通用不变式】(裸空/账实不符/孤儿单/弃仓/挂卖单超持仓) —— 用券商真值裁判

bot 每次改动后跑这个即可: /usr/local/bin/python3 run_sim.py [-v] [关键字]
退出码非0 = 有场景失败。
"""
import sys, traceback, importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim.harness import Sim, Violation

# 固定在前的两个基础模块, 其余 sim/scenarios/*.py 自动发现(新增场景文件不用再改这里)
_BASE_MODULES = ["sim.scenarios.normal", "sim.scenarios.adversarial"]
SCENARIO_MODULES = _BASE_MODULES + sorted(
    f"sim.scenarios.{p.stem}"
    for p in (Path(__file__).resolve().parent / "sim" / "scenarios").glob("*.py")
    if not p.stem.startswith("_") and f"sim.scenarios.{p.stem}" not in _BASE_MODULES)


def collect():
    out = []
    for mod_name in SCENARIO_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        for name in sorted(dir(mod)):
            if name.startswith("sc_"):
                fn = getattr(mod, name)
                if callable(fn):
                    out.append((mod_name.split(".")[-1], name, fn))
    return out


def main():
    verbose = "-v" in sys.argv
    kw = [a for a in sys.argv[1:] if not a.startswith("-")]
    scenarios = collect()
    if kw:
        scenarios = [x for x in scenarios if any(k in x[1] for k in kw)]
    if not scenarios:
        print("没有找到场景 (sim/scenarios/*.py 里以 sc_ 开头的函数)")
        return 1

    npass = 0
    failures = []
    cur_mod = None
    for mod, name, fn in scenarios:
        if mod != cur_mod:
            cur_mod = mod
            print(f"\n{'='*78}\n{mod}\n{'='*78}")
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        try:
            with Sim() as s:
                fails = fn(s) or []
                fails = list(fails) + s.check_invariants()
        except Exception as e:
            fails = [f"场景抛异常: {type(e).__name__}: {e}"]
            if verbose:
                fails.append(traceback.format_exc())
        if fails:
            failures.append((mod, name, doc, fails))
            print(f"❌ {name}  — {doc}")
            for f in fails:
                print(f"      · {f}")
        else:
            npass += 1
            print(f"✅ {name}  — {doc}")

    print(f"\n{'='*78}")
    print(f"结果: {npass} 通过 / {len(failures)} 失败  (共 {len(scenarios)} 个场景)")
    if failures:
        print("\n失败清单:")
        for mod, name, doc, fails in failures:
            print(f"  ❌ [{mod}] {name} — {doc}")
            for f in fails:
                print(f"       · {f}")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
