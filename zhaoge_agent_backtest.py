#!/usr/bin/env python3
"""
zhaoge_agent_backtest.py — A/B: 纯正则解析 vs Opus agent 批次语义解析 (真实5分K, ALL时段)。

A组: zhaoge_backtest 现行引擎 — 卖出frac相对【总持仓】, 部分卖出后批次合并为均价(批次身份丢失)。
B组: agent逐日解析的结构化op (output/zhaoge_llm_ops.json) + 批次记账:
     · 买入批次以他报的价格命名; "23出一半21.7的pl"=卖21.7批次的一半(非总仓一半)
     · rebuy(加回)数量 = 被引用卖单的数量
     · 无票名上下文单已由agent用当日前文归属 (正则只能丢弃)
成交规则两组一致: 买=限价单(信号后至次一交易日20:00 ET), Open≤px按Open/Low≤px按px; 不追高。
                  卖=下根bar开盘市价; 成本0.1%/边; 每批$10k, 每票≤3批。
正确性度量: 覆盖率 / 悬空批次引用 / 无持仓卖出 / 上下文单救回数; 差异明细逐条列出。
窗口: 2026-05-29~07-17 (5月末2天只作建仓种子, 月度成绩报6月/7月)。
"""
import json, os, sys
from datetime import datetime, date, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ["ALL"] = "1"
import warnings; warnings.filterwarnings("ignore")
from zoneinfo import ZoneInfo

import zhaoge_backtest as Z

ROOT = Path(__file__).parent
OPS_FILE = ROOT / "output" / "zhaoge_llm_ops.json"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
W_START, W_END = date(2026, 5, 29), date(2026, 7, 17)
MIN_CONF = 0.5
REF_TOL = 0.02      # 批次引用价匹配容差 2%


def build_day_index():
    """重建与agent输入完全一致的 逐日消息索引 → (day, idx) -> ts。"""
    msgs = json.load(open(ROOT / "output" / "zhaoge_history.json"))
    days = {}
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        d = ts.astimezone(ET).date()
        if not (W_START <= d <= W_END):
            continue
        raw = " ".join(Z.strip_prefix(m["text"]).split())
        if not raw or "http" in raw or len(raw) > 250:
            continue
        days.setdefault(d.isoformat(), []).append((ts, raw))
    return days


def load_ops():
    """workflow 结果 → 时间排序的op流; 过滤非stock/低conf/缺字段, 全部计数不静默丢。"""
    data = json.loads(OPS_FILE.read_text())
    days = build_day_index()
    ops, drop = [], {"agent_failed_day": 0, "bad_idx": 0, "non_stock": 0,
                     "no_ticker": 0, "no_px": 0, "low_conf": 0}
    for rec in data:
        if rec.get("ops") is None:
            drop["agent_failed_day"] += 1
            continue
        idx_map = days.get(rec["day"], [])
        for seq, o in enumerate(rec["ops"]):
            i = o.get("idx", -1)
            if not (0 <= i < len(idx_map)):
                drop["bad_idx"] += 1; continue
            ts, raw = idx_map[i]
            if o.get("kind") != "stock":
                drop["non_stock"] += 1; continue
            if not o.get("ticker"):
                drop["no_ticker"] += 1; continue
            if o.get("px") is None:
                drop["no_px"] += 1; continue
            if float(o.get("conf") or 0) < MIN_CONF:
                drop["low_conf"] += 1; continue
            ops.append(dict(ts=ts, seq=seq, day=rec["day"], raw=raw,
                            action=o["action"], tk=o["ticker"].lower().strip(),
                            px=float(o["px"]),
                            frac=(None if o.get("frac") is None else float(o["frac"])),
                            batch_px=(None if o.get("batch_px") is None else float(o["batch_px"])),
                            conf=float(o["conf"]), quote=o.get("quote", "")))
    ops.sort(key=lambda x: (x["ts"], x["seq"]))
    return ops, drop


def fill_buy(B, ts, px):
    """与A组完全一致的限价买成交规则。返回 fill 价(未含成本) 或 None。"""
    ref = next((b for b in B if b["ts"] > ts), None)
    if ref and px < ref["o"] * 0.3:       # 期权权利金误标为stock的保底闸
        return "OPT_GUARD"
    deadline = Z.next_trading_day_end(ts)
    for b in B:
        if b["ts"] <= ts or b["ts"] > deadline:
            continue
        if b["o"] <= px:
            return b["o"]
        if b["l"] <= px:
            return px
    return None


def simulate_b(ops):
    from longport.openapi import Config, QuoteContext
    q = QuoteContext(Config.from_env())
    tickers = sorted({o["tk"] for o in ops})
    bars = {}
    for tk in tickers:
        b = Z.fetch_bars(q, tk)
        if b:
            bars[tk] = b
        else:
            print(f"  ⚠️ B组 {tk}: 无K线, 排除")
    batches = {}      # tk -> [dict(sh, cost, label)]
    sells_hist = {}   # tk -> [dict(label, sh)]  供加回引用
    trades, diffs = [], []
    n = dict(nofill=0, cap=0, nopos=0, dangling_sell=0, dangling_rebuy=0,
             opt_guard=0, recovered=0)
    nopos_dates, recov_pnl = {}, []
    deploy = peak = 0.0
    # 正则口径的信号ts集合 (用于统计agent救回的上下文单)
    Z.START, Z.END = W_START, W_END
    regex_ts = {t for t, *_ in Z.parse_stream()}

    for o in ops:
        B = bars.get(o["tk"])
        if not B:
            continue
        tk, ts = o["tk"], o["ts"]
        if ts not in regex_ts:
            n["recovered"] += 1        # 正则流里没有这条消息的信号 → agent救回的
        if o["action"] in ("buy", "rebuy"):
            if len(batches.get(tk, [])) >= Z.MAX_LOTS:
                n["cap"] += 1; continue
            fill = fill_buy(B, ts, o["px"])
            if fill == "OPT_GUARD":
                n["opt_guard"] += 1; continue
            if fill is None:
                n["nofill"] += 1; continue
            fill *= (1 + Z.COST)
            sh = None
            if o["action"] == "rebuy" and o["batch_px"]:
                ref = next((s for s in reversed(sells_hist.get(tk, []))
                            if abs(s["label"] - o["batch_px"]) / o["batch_px"] <= REF_TOL), None)
                if ref:
                    sh = ref["sh"]
                else:
                    n["dangling_rebuy"] += 1
            if sh is None:
                sh = Z.LOT_USD / fill
            batches.setdefault(tk, []).append(dict(sh=sh, cost=fill, label=o["px"]))
            deploy += sh * fill; peak = max(peak, deploy)
        else:  # sell
            lots = batches.get(tk) or []
            if not lots:
                n["nopos"] += 1
                d = ts.astimezone(ET).date().isoformat()
                nopos_dates[d] = nopos_dates.get(d, 0) + 1
                continue
            nxt = next((b for b in B if b["ts"] > ts), None)
            if nxt is None:
                continue
            sell_px = nxt["o"] * (1 - Z.COST)
            frac = 1.0 if o["frac"] is None else max(0.0, min(1.0, o["frac"]))
            picks = []    # (lot, 卖出sh)
            if o["batch_px"] is not None:
                cand = min(lots, key=lambda l: abs(l["label"] - o["batch_px"]))
                if abs(cand["label"] - o["batch_px"]) / o["batch_px"] <= REF_TOL:
                    picks = [(cand, cand["sh"] * frac)]
                else:
                    n["dangling_sell"] += 1
            if not picks:  # 未点名批次 或 引用悬空回退 → 按比例摊到全部批次
                picks = [(l, l["sh"] * frac) for l in lots]
            sold_sh = sum(s for _, s in picks)
            if sold_sh <= 1e-9:
                continue
            cost_sold = sum(l["cost"] * s for l, s in picks) / sold_sh
            pnl = (sell_px - cost_sold) * sold_sh
            trades.append(dict(date=str(ts.astimezone(ET).date()), ticker=tk,
                               sell=round(sell_px, 3), cost=round(cost_sold, 3),
                               shares=round(sold_sh), pnl=round(pnl),
                               pct=round((sell_px / cost_sold - 1) * 100, 2),
                               batch_ref=o["batch_px"], quote=o["quote"][:50]))
            if ts not in regex_ts:
                recov_pnl.append(round(pnl))
            deploy = max(0.0, deploy - sold_sh * cost_sold)
            for l, s in picks:
                l["sh"] -= s
            batches[tk] = [l for l in lots if l["sh"] > 1e-6]
            sells_hist.setdefault(tk, []).append(dict(label=o["px"], sh=sold_sh))
    # 期末mark
    inv = []
    for tk, lots in batches.items():
        B = bars.get(tk)
        last = B[-1]["c"] if B else None
        for l in lots:
            if last is None:
                continue
            inv.append(dict(ticker=tk, cost=round(l["cost"], 3), label=l["label"], last=last,
                            usd=round(l["sh"] * l["cost"]),
                            upnl=round((last - l["cost"]) * l["sh"]),
                            upct=round((last / l["cost"] - 1) * 100, 1)))
    n["nopos_dates"] = nopos_dates
    n["recov_win"] = sum(1 for p in recov_pnl if p > 0)
    n["recov_n"] = len(recov_pnl)
    n["recov_pnl"] = sum(recov_pnl)
    return trades, inv, n, peak


def stat(trades, inv, peak, months=("2026-06", "2026-07")):
    out = {}
    for mth in months:
        tr = [t for t in trades if t["date"][:7] == mth]
        w = [t for t in tr if t["pnl"] > 0]
        out[mth] = dict(n=len(tr), win=len(w),
                        wr=round(len(w) / len(tr) * 100) if tr else None,
                        pnl=round(sum(t["pnl"] for t in tr)),
                        ret=round(sum(t["pnl"] for t in tr) / peak * 100, 1) if peak else None)
    tr = [t for t in trades if t["date"][:7] in months]
    w = [t for t in tr if t["pnl"] > 0]
    upnl = sum(i["upnl"] for i in inv)
    out["total"] = dict(n=len(tr), win=len(w),
                        wr=round(len(w) / len(tr) * 100) if tr else None,
                        realized=round(sum(t["pnl"] for t in tr)),
                        unrealized=round(upnl), peak=round(peak),
                        ret=round((sum(t["pnl"] for t in tr) + upnl) / peak * 100, 1) if peak else None)
    return out


def main():
    # ── A组 ──
    Z.START, Z.END = W_START, W_END
    stream = Z.parse_stream()
    tA, invA, nofillA = Z.simulate(stream)
    peakA = Z._PEAK
    sA = stat(tA, invA, peakA)
    # ── B组 ──
    ops, drop = load_ops()
    print(f"\nagent op流: {len(ops)} 条可交易op | 丢弃: {drop}")
    tB, invB, nB, peakB = simulate_b(ops)
    sB = stat(tB, invB, peakB)

    print("\n" + "═" * 78)
    print(f"{'':16}{'A组·纯正则':>22}{'B组·agent批次记账':>26}")
    print("─" * 78)
    for mth, lab in (("2026-06", "6月"), ("2026-07", "7月(至17日)")):
        a, b = sA[mth], sB[mth]
        print(f"{lab:14} {a['n']:>4}笔 胜率{a['wr'] or 0:>3}% ${a['pnl']:>+8,}"
              f"   {b['n']:>4}笔 胜率{b['wr'] or 0:>3}% ${b['pnl']:>+8,}")
    a, b = sA["total"], sB["total"]
    print("─" * 78)
    print(f"{'合计已实现':13} {a['n']:>4}笔 胜率{a['wr']}% ${a['realized']:>+8,}"
          f"   {b['n']:>4}笔 胜率{b['wr']}% ${b['realized']:>+8,}")
    print(f"{'期末浮动':14} {'':>4}     ${a['unrealized']:>+8,}   {'':>10} ${b['unrealized']:>+8,}")
    print(f"{'峰值占用':14} {'':>9}${a['peak']:>9,}   {'':>10} ${b['peak']:>9,}")
    print(f"{'总收益率':14} {'':>10}{a['ret']:>+8.1f}%   {'':>11}{b['ret']:>+8.1f}%")
    print(f"\nA组: 买入未成交{nofillA} | B组: 未成交{nB['nofill']} 满3批拒{nB['cap']} "
          f"无持仓卖{nB['nopos']} 悬空卖引用{nB['dangling_sell']} 悬空加回{nB['dangling_rebuy']} "
          f"期权闸{nB['opt_guard']}")
    print(f"无持仓卖出按日: " + " ".join(f"{d[5:]}×{c}" for d, c in sorted(nB['nopos_dates'].items())))
    print(f"agent救回正则漏单: 触发{nB['recovered']}条 → 成交平仓{nB['recov_n']}笔 "
          f"胜{nB['recov_win']} 贡献P&L ${nB['recov_pnl']:+,.0f}")
    out = dict(generated=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds"),
               window=f"{W_START}~{W_END}", armA=sA, armB=sB,
               tradesA=tA, tradesB=tB, invA=invA, invB=invB, drop=drop, nB=nB)
    (ROOT / "output" / "zhaoge_agent_ab.json").write_text(json.dumps(out, ensure_ascii=False))
    print("存 → output/zhaoge_agent_ab.json")


if __name__ == "__main__":
    main()
