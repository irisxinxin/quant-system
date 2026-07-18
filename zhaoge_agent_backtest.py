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
    """与A组完全一致的限价买成交规则。返回 (fill价, 成交bar) / ("OPT_GUARD", None) / (None, None)。"""
    ref = next((b for b in B if b["ts"] > ts), None)
    if ref and px < ref["o"] * 0.3:       # 期权权利金误标为stock的保底闸
        return "OPT_GUARD", None
    deadline = Z.next_trading_day_end(ts)
    for b in B:
        if b["ts"] <= ts or b["ts"] > deadline:
            continue
        if b["o"] <= px:
            return b["o"], b
        if b["l"] <= px:
            return px, b
    return None, None


def simulate_b(ops, bars, stop_pct=None):
    """agent批次记账回测。stop_pct=None→无止损; 否则每批成本×(1-stop_pct)为止损线,
    买入成交bar之后任意bar.Low≤止损→平该批(跳空低开按Open, 否则按止损价)。
    止损用'懒结算': 该票下次有op前 / 期末, 扫新区间bar找首次跌破。"""
    batches = {}      # tk -> [dict(sh, cost, label, entry_ts, last_ts, stop_px)]
    sells_hist = {}   # tk -> [dict(label, sh)]  供加回引用
    trades = []
    n = dict(nofill=0, cap=0, nopos=0, dangling_sell=0, dangling_rebuy=0,
             opt_guard=0, recovered=0, stopped=0, stop_pnl=0.0)
    nopos_dates, recov_pnl = {}, []
    deploy = peak = 0.0
    Z.START, Z.END = W_START, W_END
    regex_ts = {t for t, *_ in Z.parse_stream()}

    def settle(tk, until_ts):
        """结算tk所有批次在(last_ts, until_ts]内的止损, 平掉首次跌破的批次。"""
        nonlocal deploy
        if stop_pct is None:
            return
        B = bars[tk]
        for lot in list(batches.get(tk, [])):
            hit = None
            for b in B:
                if b["ts"] <= lot["last_ts"] or b["ts"] > until_ts:
                    continue
                if b["l"] <= lot["stop_px"]:
                    hit = b; break
                lot["last_ts"] = b["ts"]
            if hit is None:
                lot["last_ts"] = until_ts
                continue
            fill = min(hit["o"], lot["stop_px"])       # 跳空低开→开盘, 否则→止损价
            sell_px = fill * (1 - Z.COST)
            pnl = (sell_px - lot["cost"]) * lot["sh"]
            trades.append(dict(date=str(hit["ts"].astimezone(ET).date()), ticker=tk,
                               sell=round(sell_px, 3), cost=round(lot["cost"], 3),
                               shares=round(lot["sh"]), pnl=round(pnl),
                               pct=round((sell_px / lot["cost"] - 1) * 100, 2),
                               batch_ref=None, quote=f"⛑强止损-{stop_pct*100:g}%"))
            n["stopped"] += 1; n["stop_pnl"] += pnl
            deploy = max(0.0, deploy - lot["sh"] * lot["cost"])
            batches[tk].remove(lot)

    for o in ops:
        B = bars.get(o["tk"])
        if not B:
            continue
        tk, ts = o["tk"], o["ts"]
        settle(tk, ts)                     # 处理他的op前, 先结算已触发的止损
        if ts not in regex_ts:
            n["recovered"] += 1
        if o["action"] in ("buy", "rebuy"):
            if len(batches.get(tk, [])) >= Z.MAX_LOTS:
                n["cap"] += 1; continue
            fill, fbar = fill_buy(B, ts, o["px"])
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
            batches.setdefault(tk, []).append(dict(
                sh=sh, cost=fill, label=o["px"], entry_ts=fbar["ts"], last_ts=fbar["ts"],
                stop_px=(fill * (1 - stop_pct) if stop_pct is not None else 0.0)))
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
            picks = []
            if o["batch_px"] is not None:
                cand = min(lots, key=lambda l: abs(l["label"] - o["batch_px"]))
                if abs(cand["label"] - o["batch_px"]) / o["batch_px"] <= REF_TOL:
                    picks = [(cand, cand["sh"] * frac)]
                else:
                    n["dangling_sell"] += 1
            if not picks:
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
    # 期末: 先把止损结算到最后一根bar, 剩余进库存
    for tk in list(batches):
        settle(tk, bars[tk][-1]["ts"])
    inv = []
    for tk, lots in batches.items():
        last = bars[tk][-1]["c"]
        for l in lots:
            inv.append(dict(ticker=tk, cost=round(l["cost"], 3), label=l["label"], last=last,
                            usd=round(l["sh"] * l["cost"]),
                            upnl=round((last - l["cost"]) * l["sh"]),
                            upct=round((last / l["cost"] - 1) * 100, 1)))
    n["nopos_dates"] = nopos_dates
    n["recov_win"] = sum(1 for p in recov_pnl if p > 0)
    n["recov_n"] = len(recov_pnl)
    n["recov_pnl"] = sum(recov_pnl)
    n["stop_pnl"] = round(n["stop_pnl"])
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
    from longport.openapi import Config, QuoteContext
    ops, drop = load_ops()
    print(f"agent op流: {len(ops)} 条可交易op | 丢弃: {drop}")
    # K线只拉一次, 各止损档共用
    q = QuoteContext(Config.from_env())
    bars = {}
    for tk in sorted({o["tk"] for o in ops}):
        b = Z.fetch_bars(q, tk)
        if b:
            bars[tk] = b
        else:
            print(f"  ⚠️ {tk}: 无K线, 排除")

    LEVELS = [(None, "无止损"), (0.02, "止损2%"), (0.03, "止损3%"), (0.05, "止损5%")]
    runs = {}
    for sp, lab in LEVELS:
        t, inv, nn, pk = simulate_b(ops, bars, stop_pct=sp)
        runs[lab] = dict(s=stat(t, inv, pk), n=nn, peak=pk, trades=t, inv=inv)

    print("\n" + "═" * 92)
    print(f"{'档位':12}{'已实现':>13}{'期末浮动':>13}{'总账':>13}{'峰值占用':>13}{'总收益率':>11}{'胜率':>7}{'止损次数':>9}")
    print("─" * 92)
    base = runs["无止损"]["s"]["total"]
    for _, lab in LEVELS:
        r = runs[lab]; tot = r["s"]["total"]; nn = r["n"]
        net = tot["realized"] + tot["unrealized"]
        stp = f"{nn['stopped']}(${nn['stop_pnl']:+,})" if nn["stopped"] else "—"
        print(f"{lab:12}{tot['realized']:>+12,}{tot['unrealized']:>+13,}{net:>+13,}"
              f"{tot['peak']:>13,}{tot['ret']:>+10.1f}%{str(tot['wr'])+'%':>7}{stp:>13}")
    print("─" * 92)
    # 月度分解 (各档)
    print("\n月度已实现 (6月 / 7月至17日):")
    for _, lab in LEVELS:
        s = runs[lab]["s"]
        print(f"  {lab:10} 6月 {s['2026-06']['n']:>3}笔 ${s['2026-06']['pnl']:>+7,} 胜{s['2026-06']['wr'] or 0}%"
              f"   |  7月 {s['2026-07']['n']:>3}笔 ${s['2026-07']['pnl']:>+7,} 胜{s['2026-07']['wr'] or 0}%")

    out = dict(generated=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds"),
               window=f"{W_START}~{W_END}", drop=drop,
               runs={lab: dict(stat=runs[lab]["s"], n={k: v for k, v in runs[lab]["n"].items()
                                                       if k != "nopos_dates"},
                               trades=runs[lab]["trades"], inv=runs[lab]["inv"])
                     for _, lab in LEVELS})
    (ROOT / "output" / "zhaoge_stop_ab.json").write_text(json.dumps(out, ensure_ascii=False))
    print("\n存 → output/zhaoge_stop_ab.json")


if __name__ == "__main__":
    main()
