#!/usr/bin/env python3
"""
bt_style_grid.py — 止盈×止损参数网格 (B组配置=规则+LLM, 与实盘一致)。
用户方向: 按enrich风格走 = 止盈不贪(他+40~70%就减) + 抗回撤(止损放宽)。
网格找最优档, 并给"单笔最大账户回撤"风险联动表。
⚠️同池调参警告: 10笔样本, 结果只作定档参考, 需前向确认。
"""
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("V2", "1")
os.environ.setdefault("N", "2")
import warnings; warnings.filterwarnings("ignore")

UTC = timezone.utc
OUT = Path(__file__).parent / "output"


def main():
    import backtest_enrich as E
    from enrich_parser import parse_signal, to_longport_symbol
    from signal_history import _resolve_ambig_hist
    from longport.openapi import Config, QuoteContext
    q = QuoteContext(Config.from_env())

    msgs = json.load(open(OUT / "enrich_history.json"))
    parsed = []
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        s = parse_signal(m["text"], ts.date())
        parsed.append((m["id"], ts, m["text"], s))
    rule_exits = [dict(ts=ts, ticker=s.ticker, level=s.exit_level)
                  for _, ts, _, s in parsed if s.kind == "EXIT"]

    buys, seen = [], set()
    for mid, ts, text, s in parsed:
        if s.kind == "BUY" and s.limit_price <= 5.0:
            key = f"{to_longport_symbol(s)}:{ts.date()}"
            if key in seen: continue
            seen.add(key); buys.append(dict(mid=mid, ts=ts, sig=s, qty=2))
        elif s.kind == "BUY_AMBIG" and s.limit_price <= 5.0:
            key = f"{s.ticker}{s.expiry}{s.strike}:{ts.date()}"
            if key in seen: continue
            seen.add(key)
            side = _resolve_ambig_hist(q, E, s, ts)
            if side is None: continue
            s.kind, s.right = "BUY", side
            buys.append(dict(mid=mid, ts=ts, sig=s, qty=1))
    testable = [b for b in buys if E.bars(q, to_longport_symbol(b["sig"]))]
    tickers = sorted({b["sig"].ticker for b in testable})

    # B组配置: LLM否决 + LLM捞漏出场 (缓存)
    cache = json.loads((OUT / "llm_ab_cache.json").read_text())
    vetoed = {b["mid"] for b in testable
              if (v := cache.get(str(b["mid"]))) and v["action"] != "buy" and v["confidence"] >= 0.7}
    llm_exits = []
    for mid, ts, text, s in parsed:
        v = cache.get(str(mid))
        if not v or v["action"] not in ("exit_full", "exit_partial") or v["confidence"] < 0.75:
            continue
        if not (s.kind == "NOISE" or (s.kind == "EXIT" and s.exit_level == "alert")):
            continue
        level = "full" if v["action"] == "exit_full" else "partial"
        targets = ([v["ticker"]] if v.get("scope") == "ticker" and v.get("ticker")
                   else [t for t in tickers if t != v.get("except")])
        for tk in targets:
            llm_exits.append(dict(ts=ts, ticker=tk, level=level))
    exits = sorted(rule_exits + llm_exits, key=lambda e: e["ts"])
    active = [b for b in testable if b["mid"] not in vetoed]
    print(f"可测{len(testable)}笔 - LLM否决{len(testable)-len(active)} = 参赛{len(active)}笔 | 出场事件{len(exits)}")

    # ── 网格 ──
    TPS = [1.4, 1.5, 1.6, 1.8, 2.0]
    STOPS = [(0.7, "-30%"), (0.6, "-40%"), (0.5, "-50%"), (0.0, "无止损(到期兜底)")]
    header = "止盈/止损"
    print(f"\n{header:12}" + "".join(f"{lab:>16}" for _, lab in STOPS))
    best = (None, -1e9)
    for tp in TPS:
        row = f"+{(tp-1)*100:.0f}%卖半仓 "
        for stop, lab in STOPS:
            E.TP_MULT = tp
            E.RUNNER_STOP = stop
            tot = cost = 0.0
            wins = 0
            worst = 0.0
            for b in active:
                old = E.CONTRACTS; E.CONTRACTS = b["qty"]
                try:
                    r = E.simulate(q, dict(ts=b["ts"], osi=to_longport_symbol(b["sig"]), sig=b["sig"]), exits)
                finally:
                    E.CONTRACTS = old
                if r and r["status"] == "traded":
                    tot += r["pnl"]; cost += r["cost"]
                    wins += r["pnl"] > 0
                    worst = min(worst, r["pnl"] / r["cost"] * 100)
            ret = tot / cost * 100 if cost else 0
            if tot > best[1]:
                best = ((tp, stop, lab), tot)
            row += f"{ret:>+9.1f}%({worst:>+4.0f})"
        print(row)
    (tp, stop, lab), tot = best
    print(f"\n网格最优: 止盈+{(tp-1)*100:.0f}% × 止损{lab} → 总P&L ${tot:+,.0f}")
    print("(括号内=该档最差单笔%; 无止损档的风险看这里)")


if __name__ == "__main__":
    main()
