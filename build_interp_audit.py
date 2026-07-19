#!/usr/bin/env python3
"""
build_interp_audit.py — 生成6-7月enrich消息的"系统解读档案" (供reviewer审对错)。
逐条复现实盘bot仲裁路径: 规则解析 → hedge正则 → LLM层(买入否决/噪音捞漏) → 消歧 → 去重 → 分档。
输出 output/interp_audit.json。LLM判定复用/扩充 output/llm_ab_cache.json。
"""
import json, os, re, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("V2", "1")
import warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor

UTC = timezone.utc
OUT = Path(__file__).parent / "output"
CACHE = OUT / "llm_ab_cache.json"
LO, HI = "2026-05-01", "2026-07-18"


def main():
    import backtest_enrich as E
    from enrich_parser import parse_signal, to_longport_symbol
    from signal_history import _resolve_ambig_hist
    from llm_classifier import classify
    from longport.openapi import Config, QuoteContext
    q = QuoteContext(Config.from_env())

    msgs = [m for m in json.load(open(OUT / "enrich_history.json"))
            if LO <= m["ts"][:10] <= HI]
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    # ── 需要LLM判定的池 ──
    pool = {}
    pre = []
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        s = parse_signal(m["text"], ts.date())
        pre.append((m, ts, s))
        one = " ".join(m["text"].split())
        if s.kind in ("BUY", "BUY_AMBIG"):
            pool[m["id"]] = m["text"]
        elif (s.kind == "NOISE" or (s.kind == "EXIT" and s.exit_level == "alert")) \
                and len(one) > 15 and "http" not in m["text"]:
            pool[m["id"]] = m["text"]
    todo = {mid: t for mid, t in pool.items() if str(mid) not in cache}
    print(f"消息 {len(msgs)} 条 | LLM池 {len(pool)} | 需新判 {len(todo)}")
    if todo:
        held_ctx = ["HOOD", "MSFT", "XOM", "GOOGL", "DELL", "IBM", "LLY", "NVDA", "TSLA"]
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(classify, t, held_ctx): mid for mid, t in todo.items()}
            done = 0
            for f in futs:
                cache[str(futs[f])] = f.result()
                done += 1
                if done % 25 == 0:
                    print(f"  LLM判定 {done}/{len(todo)}")
                    CACHE.write_text(json.dumps(cache, ensure_ascii=False))
        CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    fails = sum(1 for mid in pool if cache.get(str(mid)) is None)
    print(f"LLM判定完成 (失败{fails})")

    # ── 逐条仲裁 (镜像实盘bot逻辑) ──
    rows = []
    seen_contract = set()
    for m, ts, s in pre:
        one = " ".join(m["text"].split())[:200]
        v = cache.get(str(m["id"]))
        rule = dict(kind=s.kind)
        if s.kind in ("BUY", "BUY_AMBIG"):
            rule.update(ticker=s.ticker, right=s.right or "?", strike=s.strike,
                        expiry=str(s.expiry), premium=s.limit_price, size_tag=s.size_tag)
        elif s.kind == "EXIT":
            rule.update(ticker=s.ticker, level=s.exit_level)

        act = ""
        if s.kind == "NOISE":
            if v and v["action"] in ("exit_full", "exit_partial") and v["confidence"] >= 0.75:
                act = f"🤖LLM捞漏出场[{v['action']}] {v.get('ticker') or 'ALL'}" \
                      + (f" 豁免{v['except']}" if v.get("except") else "") + " (有持仓才执行)"
            else:
                act = "忽略"
        elif s.kind == "EXIT":
            if s.exit_level == "alert":
                act = f"仅提醒(多票/豁免词) [{s.ticker}]"
            else:
                act = f"出场指令[{s.exit_level}] {s.ticker}" + ("(全仓位)" if s.ticker == "*" else "")
        else:  # BUY / BUY_AMBIG
            is_lotto = ("lotto" in (s.size_tag or "").lower() or "scalp" in one.lower()
                        or s.expiry == ts.date())
            if re.search(r"\bhedge\b", one, re.I):
                act = "🛡️hedge跳过, 仅提醒"
            elif s.kind == "BUY_AMBIG":
                side = _resolve_ambig_hist(q, E, s, ts)
                if side is None:
                    act = "歧义无法消歧, 仅提醒"
                else:
                    s.right = side
                    key = f"{s.ticker}{s.expiry}{s.strike}:{ts.date()}"
                    if key in seen_contract:
                        act = "重复信号跳过"
                    elif v and v["action"] != "buy" and v["confidence"] >= 0.7:
                        act = f"🤖LLM否决[{v['action']}], 仅提醒"
                    else:
                        seen_contract.add(key)
                        tier = "0DTE档⅒" if s.expiry == ts.date() else "lotto档⅓"
                        act = f"下单: 买 {s.ticker} {s.expiry} ${s.strike:g}{side}(消歧) 限价{s.limit_price} [{tier}]"
            else:
                key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
                if key in seen_contract:
                    act = "重复信号跳过"
                elif s.limit_price > 5.0:
                    act = f"权利金{s.limit_price}>上限$5, 拒绝"
                elif v and v["action"] != "buy" and v["confidence"] >= 0.7:
                    act = f"🤖LLM否决[{v['action']}], 仅提醒"
                else:
                    seen_contract.add(key)
                    tier = ("0DTE档⅒" if s.expiry == ts.date() else
                            ("lotto档⅓" if is_lotto else "波段档½"))
                    stop = "-30%止损" if is_lotto else "-50%兜底"
                    act = (f"下单: 买 {s.ticker} {s.expiry} ${s.strike:g}{s.right} "
                           f"限价{s.limit_price} [{tier}, 止盈+60%, {stop}]")

        rows.append(dict(idx=len(rows), id=m["id"], ts=m["ts"], text=one,
                         rule=rule, llm=v, final_action=act))

    (OUT / "interp_audit.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    from collections import Counter
    c = Counter(r["final_action"].split(":")[0].split("[")[0].strip() for r in rows)
    print(f"档案完成 {len(rows)} 条 → output/interp_audit.json")
    print("动作分布:", dict(c.most_common(12)))


if __name__ == "__main__":
    main()
