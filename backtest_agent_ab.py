#!/usr/bin/env python3
"""
backtest_agent_ab.py — A/B: 纯规则 vs 规则+LLM仲裁 (真实期权5分K, 与实盘同规则)。

A组: 现行规则引擎 (V2镜像+止损-30%+2张, 歧义单1张)
B组: A + LLM层:
     · 买入否决: LLM判非buy(conf≥0.7) → 该笔不进场
     · 出场捞漏: 规则判NOISE/alert的消息, LLM判exit(conf≥0.75) → 加入出场时间线
LLM判定缓存 output/llm_ab_cache.json (按消息id), 重跑免费。
⚠️诚实声明: 现行正则是照本段历史事故补的, 本测会【低估】agent对未来新说法的价值。
"""
import json, os, sys
from datetime import datetime, timezone, time as dtime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("V2", "1")
os.environ.setdefault("V3_STOP", "0.7")
os.environ.setdefault("N", "2")
import warnings; warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor

UTC = timezone.utc
OUT = Path(__file__).parent / "output"
CACHE = OUT / "llm_ab_cache.json"


def main():
    import backtest_enrich as E
    from enrich_parser import parse_signal, to_longport_symbol
    from llm_classifier import classify
    from longport.openapi import Config, QuoteContext
    from signal_history import _resolve_ambig_hist
    q = QuoteContext(Config.from_env())

    msgs = json.load(open(OUT / "enrich_history.json"))
    # ── 解析全部消息 ──
    parsed = []
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        s = parse_signal(m["text"], ts.date())
        parsed.append((m["id"], ts, m["text"], s))

    # 规则出场时间线 (A/B共用基础)
    rule_exits = [dict(ts=ts, ticker=s.ticker, level=s.exit_level)
                  for _, ts, _, s in parsed if s.kind == "EXIT"]

    # 可测买入 (常规BUY 2张 + 歧义单消歧后1张)
    buys = []
    seen = set()
    for mid, ts, text, s in parsed:
        if s.kind == "BUY" and s.limit_price <= 5.0:
            key = f"{to_longport_symbol(s)}:{ts.date()}"
            if key in seen: continue
            seen.add(key)
            buys.append(dict(mid=mid, ts=ts, text=text, sig=s, qty=2, ambig=False))
        elif s.kind == "BUY_AMBIG" and s.limit_price <= 5.0:
            key = f"{s.ticker}{s.expiry}{s.strike}:{ts.date()}"
            if key in seen: continue
            seen.add(key)
            side = _resolve_ambig_hist(q, E, s, ts)
            if side is None: continue
            s.kind, s.right = "BUY", side
            buys.append(dict(mid=mid, ts=ts, text=text, sig=s, qty=1, ambig=True))
    # 只留有K线的
    testable = [b for b in buys if E.bars(q, to_longport_symbol(b["sig"]))]
    tickers = sorted({b["sig"].ticker for b in testable})
    t_lo = min(b["ts"] for b in testable)
    print(f"可测买入 {len(testable)}/{len(buys)} 笔 ({t_lo:%m-%d}起) 票: {tickers}")

    # ── LLM 判定池: 全部可测买入 + 交易窗口内的 NOISE/alert 消息 ──
    pool = {b["mid"]: b["text"] for b in testable}
    n_noise = 0
    for mid, ts, text, s in parsed:
        if ts < t_lo: continue
        one = " ".join(text.split())
        if len(one) < 16 or "http" in text: continue
        if s.kind == "NOISE" or (s.kind == "EXIT" and s.exit_level == "alert"):
            pool[mid] = text; n_noise += 1
    print(f"LLM判定池: {len(pool)} 条 (含噪音/alert {n_noise}) — 并行分类中(缓存复用)...")

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = {mid: t for mid, t in pool.items() if str(mid) not in cache}
    if todo:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(classify, t, tickers): mid for mid, t in todo.items()}
            done = 0
            for f in futs:
                cache[str(futs[f])] = f.result()
                done += 1
                if done % 20 == 0: print(f"  ...{done}/{len(todo)}")
        CACHE.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"LLM判定完成 (新增{len(todo)}, 失败{sum(1 for m in todo if cache.get(str(m)) is None)})")

    # ── B组增量: 否决集 + 捞漏出场 ──
    vetoed = {}
    for b in testable:
        v = cache.get(str(b["mid"]))
        if v and v["action"] != "buy" and v["confidence"] >= 0.7:
            vetoed[b["mid"]] = v
    llm_exits = []
    for mid, ts, text, s in parsed:
        if str(mid) not in cache or mid in pool and pool[mid] and (s.kind == "NOISE" or (s.kind == "EXIT" and s.exit_level == "alert")):
            v = cache.get(str(mid))
            if not v or v["action"] not in ("exit_full", "exit_partial") or v["confidence"] < 0.75:
                continue
            level = "full" if v["action"] == "exit_full" else "partial"
            targets = ([v["ticker"]] if v.get("scope") == "ticker" and v.get("ticker")
                       else [t for t in tickers if t != v.get("except")])
            for tk in targets:
                llm_exits.append(dict(ts=ts, ticker=tk, level=level))
    print(f"B组: 否决买入 {len(vetoed)} 笔 | LLM捞漏出场事件 {len(llm_exits)} 条(展开后)")

    # ── 跑两组 ──
    def run(arm, exits, veto):
        rows = []
        for b in testable:
            if veto and b["mid"] in vetoed:
                rows.append((b, None, "LLM否决"))
                continue
            old = E.CONTRACTS
            E.CONTRACTS = b["qty"]
            try:
                r = E.simulate(q, dict(ts=b["ts"], osi=to_longport_symbol(b["sig"]), sig=b["sig"]), exits)
            finally:
                E.CONTRACTS = old
            rows.append((b, r, ""))
        return rows

    A = run("A", rule_exits, veto=False)
    B = run("B", sorted(rule_exits + llm_exits, key=lambda e: e["ts"]), veto=True)

    def stat(rows):
        tr = [r for _, r, note in rows if r and r["status"] == "traded"]
        pnl = sum(r["pnl"] for r in tr)
        cost = sum(r["cost"] for r in tr) or 1
        w = sum(1 for r in tr if r["pnl"] > 0)
        return len(tr), w, pnl, cost

    print("\n" + "═" * 96)
    print(f"{'笔':26} {'A组(纯规则)':>20} {'B组(规则+LLM)':>20}  差异说明")
    print("─" * 96)
    delta_notes = []
    for (ba, ra, _), (bb, rb, noteb) in zip(A, B):
        s = ba["sig"]
        tag = f"{s.ticker} {s.expiry:%m/%d} ${s.strike:g}{s.right}" + ("·歧义" if ba["ambig"] else "")
        pa = f"{ra['pnl']:+,.0f} ({ra['pnl']/ra['cost']*100:+.0f}%)" if ra and ra["status"] == "traded" else (ra["status"] if ra else "?")
        pb = ("否决⛔" if noteb else (f"{rb['pnl']:+,.0f} ({rb['pnl']/rb['cost']*100:+.0f}%)" if rb and rb["status"] == "traded" else (rb["status"] if rb else "?")))
        diff = ""
        if noteb and ra and ra["status"] == "traded":
            diff = f"LLM否决避免{'亏' if ra['pnl']<0 else '赚'}{abs(ra['pnl']):,.0f}"
            delta_notes.append((tag, "veto", ra["pnl"]))
        elif ra and rb and ra.get("pnl") != rb.get("pnl") and ra["status"] == rb["status"] == "traded":
            diff = f"出场路径变化 Δ{rb['pnl']-ra['pnl']:+,.0f}"
            delta_notes.append((tag, "exit", rb["pnl"] - ra["pnl"]))
        print(f"{tag:26} {pa:>20} {pb:>20}  {diff}")
    na, wa, pa_, ca = stat(A)
    nb, wb, pb_, cb = stat(B)
    print("─" * 96)
    print(f"{'A组(纯规则)':14} 成交{na}笔 胜{wa} 总P&L ${pa_:+,.0f} ({pa_/ca*100:+.1f}%)")
    print(f"{'B组(规则+LLM)':13} 成交{nb}笔 胜{wb} 总P&L ${pb_:+,.0f} ({pb_/cb*100:+.1f}%)")
    print(f"\n结论差值: B-A = ${pb_-pa_:+,.0f}")
    if vetoed:
        print("\n否决明细:")
        for mid, v in vetoed.items():
            txt = " ".join(pool[mid].split())[:70]
            print(f"  ⛔ [{v['action']} conf={v['confidence']}] {txt}")


if __name__ == "__main__":
    main()
