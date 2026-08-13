#!/usr/bin/env python3
"""
backtest_elite_casey.py — elite & casey 胜率/盈亏审计回测 (2026-07-27 ~ 08-12)。

elite (#elite-alert, BOUGHT/SOLD带价):
  A. 宣称口径 = 只算他报了SOLD的单, 用他自己的价格
  B. 真实口径 = A + 沉默死单(BOUGHT后无SOLD且已到期 → 到期内在价值, 无K线则0)
     + runner死亡(部分仓位无尾单) ; 在途单按最后bar价mark
  C. 跟单口径 = 有K线的单子: 入场=警报后下一根5分bar开盘, 出场腿同理 → 滑点后真实收益
casey (#指数-casey, 不报价格为主):
  跟单口径 = "I'm taking X"时间戳后下一根bar开盘进, 各trim/out时间戳后下一根bar开盘出
  8/5及以前到期合约K线已被长桥清除(保留窗~7天) → 只能记宣称结果(已过标的级验证)
诚实边界: casey出场份额部分为叙述近似(trim=半仓); 5分bar开盘≈延迟0-5分钟的跟单者。
"""
import csv, json, re, sys
from datetime import datetime, date, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent
BARS = ROOT / "data" / "enrich_bars"
UTC = timezone.utc
ET = ZoneInfo("America/New_York")


def load_bars(osi):
    f = BARS / f"{osi}.csv"
    if not f.exists():
        return None
    out = []
    with open(f) as fh:
        for r in csv.DictReader(fh):
            out.append(dict(ts=datetime.fromisoformat(r["ts"]), o=float(r["o"]),
                            h=float(r["h"]), l=float(r["l"]), c=float(r["c"])))
    return sorted(out, key=lambda x: x["ts"])


def next_open(bars, ts):
    for b in bars:
        if b["ts"] > ts:
            return b["o"], b["ts"]
    return None, None


def bar_at(bars, ts):
    """含ts的那根bar (查他报价是否在bar范围内)。"""
    cand = [b for b in bars if b["ts"] <= ts]
    return cand[-1] if cand else None


# ═══════════════ ELITE ═══════════════
MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
B_RE = re.compile(r"\*\*BOUGHT\*\*\s*\|\s*([A-Z]+)\s+([A-Z]+)\s+(\d+)\s+(\d+(?:\.\d+)?)([CP])\s+\$(\d+(?:\.\d+)?)", re.I)
S_RE = re.compile(r"\*\*SOLD\*\*\s*\|\s*([A-Z]+)\s+([A-Z]+)\s+(\d+)\s+(\d+(?:\.\d+)?)([CP])\s+\$(\d+(?:\.\d+)?)\s*(.*)", re.I)
FRAC_RE = re.compile(r"1/2|1/4|1/3|ALL OUT", re.I)


def osi_of(tk, mon, day, strike, right, ref_date):
    mo = MON.get(mon[:3].upper())
    if mon.upper() == "JULY": mo = 7
    y = ref_date.year
    exp = date(y, mo, int(day))
    if exp < ref_date:                     # 到期早于买入日=他写错月份(如AUG写成JULY) → 顺延一个月
        mo2 = mo + 1 if mo < 12 else 1
        exp = date(y + (1 if mo2 == 1 else 0), mo2, int(day))
    return f"{tk}{exp:%y%m%d}{right.upper()}{int(float(strike)*1000):06d}.US", exp


def elite_ledger():
    msgs = json.load(open(ROOT/"output"/"elite_alert_history.json"))
    seen, events = set(), []
    for m in msgs:
        t = " ".join(m["text"].split())
        ts = datetime.fromisoformat(m["ts"]);  ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        key = (m["ts"][:16], t[:60])
        if key in seen: continue
        seen.add(key)
        mb, ms_ = B_RE.search(t), S_RE.search(t)
        if mb:
            tk, mon, day, st, r, px = mb.groups()
            if mon.upper() in MON or mon.upper() == "JULY":
                osi, exp = osi_of(tk, mon, day, st, r, ts.date())
                events.append(dict(ts=ts, kind="B", osi=osi, exp=exp, px=float(px),
                                   label=f"{tk} {exp:%m/%d} {st}{r.upper()}"))
        elif ms_:
            tk, mon, day, st, r, px, tail = ms_.groups()
            if mon.upper() in MON or mon.upper() == "JULY":
                osi, exp = osi_of(tk, mon, day, st, r, ts.date())
                fm = FRAC_RE.search(tail)
                frac = {"1/2": .5, "1/4": .25, "1/3": 1/3}.get(fm.group(0).upper() if fm else "ALL OUT", None) \
                       if fm and fm.group(0).upper() != "ALL OUT" else None
                events.append(dict(ts=ts, kind="S", osi=osi, px=float(px), frac=frac))
    # 组装 trade (同合约按时间: B开仓 → S腿; 同分钟同价的S+B=中继口误, 丢S)
    events.sort(key=lambda e: e["ts"])
    trades = []
    open_by_osi = {}
    for e in events:
        if e["kind"] == "B":
            tr = dict(osi=e["osi"], label=e["label"], exp=e["exp"], entry=e["px"],
                      ts=e["ts"], legs=[], remain=1.0)
            open_by_osi.setdefault(e["osi"], []).append(tr)
            trades.append(tr)
        else:
            lst = open_by_osi.get(e["osi"]) or []
            tr = next((t_ for t_ in reversed(lst) if t_["remain"] > 1e-6), None)
            if tr is None:
                continue
            if abs(e["px"] - tr["entry"]) < 1e-9 and abs((e["ts"] - tr["ts"]).total_seconds()) < 120:
                continue    # 中继把买单重发成SOLD
            f = e["frac"] if e["frac"] is not None else tr["remain"]
            f = min(f, tr["remain"])
            tr["legs"].append(dict(ts=e["ts"], px=e["px"], frac=f))
            tr["remain"] = round(tr["remain"] - f, 6)
    return trades


def settle_elite(trades, today):
    rows = []
    for tr in trades:
        bars = load_bars(tr["osi"])
        sold_val = sum(l["px"] * l["frac"] for l in tr["legs"])
        sold_frac = 1.0 - tr["remain"]
        status, tail_val = "closed", 0.0
        if tr["remain"] > 1e-6:
            if tr["exp"] < today:                       # 沉默死亡/runner死亡 → 到期价值
                tail_val = (bars[-1]["c"] if bars and bars[-1]["ts"].date() >= tr["exp"] else 0.0) * tr["remain"]
                status = "expired_silent" if sold_frac < 1e-6 else "runner_died"
            else:
                tail_val = (bars[-1]["c"] * tr["remain"]) if bars else None
                status = "open"
        adv = (sold_val / sold_frac / tr["entry"] - 1) * 100 if sold_frac > 1e-6 else None
        real = ((sold_val + (tail_val or 0)) / tr["entry"] - 1) * 100 if status != "open" else None
        # 跟单口径
        fol = None
        if bars:
            fe, _ = next_open(bars, tr["ts"])
            if fe:
                fv, done = 0.0, True
                for l in tr["legs"]:
                    fx, _ = next_open(bars, l["ts"])
                    if fx is None: done = False; break
                    fv += fx * l["frac"]
                if done and status != "open":
                    fol = ((fv + tail_val) / fe - 1) * 100
        rows.append(dict(label=tr["label"], entry=tr["entry"], status=status,
                         adv=adv, real=real, fol=fol,
                         legs=[(l["px"], round(l["frac"], 2)) for l in tr["legs"]],
                         tail=round(tail_val, 2) if tail_val is not None else None))
    return rows


# ═══════════════ CASEY (8月手工时间线, 出场份额=叙述近似) ═══════════════
def cts(s):
    return datetime.fromisoformat(s + "+00:00")

CASEY = [
 # (标注, osi或None, 入场ts, [(出场ts, frac)], 宣称结果%或None, 备注)
 ("SPY 8/3 753c", None, "2026-08-03T13:47", [("2026-08-03T14:04", 1.0)], 130, "宣称+130%全出(K线已清)"),
 ("QQQ 8/4 719c", None, "2026-08-04T14:04", [("2026-08-04T14:07", .5), ("2026-08-04T15:23", .5)], 170, "宣称尾仓+170%(K线已清)"),
 ("QQQ 8/5 738c", None, "2026-08-05T13:46", [("2026-08-05T14:02", 1.0)], None, "结果未明说(K线已清), 不计胜负"),
 ("QQQ 8/5 735c", None, "2026-08-05T14:09", [("2026-08-05T14:22", 1.0)], -100, "stopped out, 幅度未说→计败不计均值"),
 ("IWM 8/5 300p", None, "2026-08-05T14:50", [("2026-08-05T16:04", .5), ("2026-08-05T16:22", .5)], 100, "宣称+100%后runner止损(K线已清)"),
 ("QQQ 8/6 722c", "QQQ260806C722000.US", "2026-08-06T14:59", [("2026-08-06T15:11", .5), ("2026-08-06T15:18", .5)], None, "小scalp"),
 ("SPY 8/6 767p", "SPY260806P767000.US", "2026-08-06T15:25", [("2026-08-06T15:47", 1.0)], None, ""),
 ("QQQ 8/10 719p", "QQQ260810P719000.US", "2026-08-10T16:50", [("2026-08-10T17:08", 1.0)], None, "他报fill .40"),
 ("QQQ 8/10 720p", "QQQ260810P720000.US", "2026-08-10T17:28", [("2026-08-10T17:38", .5), ("2026-08-10T18:48", .5)], None, "17:52曾+100%"),
 ("SPY 8/11 772p", "SPY260811P772000.US", "2026-08-11T13:44", [("2026-08-11T13:46", .5), ("2026-08-11T16:07", .5)], None, "尾仓出场时点近似"),
 ("QQQ 8/11 718p", "QQQ260811P718000.US", "2026-08-11T15:34", [("2026-08-11T16:07", 1.0)], None, ""),
 ("SPY 8/12 770p#1", "SPY260812P770000.US", "2026-08-12T14:02", [("2026-08-12T14:18", 1.0)], None, "止损"),
 ("SPY 8/12 770p#2", "SPY260812P770000.US", "2026-08-12T14:20", [("2026-08-12T14:33", .6), ("2026-08-12T16:26", .4)], None, "多数+50%卖, 尾仓被磨"),
 ("SPY 8/12 772p", "SPY260812P772000.US", "2026-08-12T16:33", [("2026-08-12T16:39", 1.0)], None, "他报入.56, 快损"),
 ("IWM 8/12 302c", "IWM260812C302000.US", "2026-08-12T18:11", [("2026-08-12T18:18", .33), ("2026-08-12T18:29", .33), ("2026-08-12T19:21", .34)], 170, "宣称+170%"),
 ("IWM 8/12 302.5c", "IWM260812C302500.US", "2026-08-12T18:12", [("2026-08-12T18:29", .5), ("2026-08-12T19:21", .5)], 350, "宣称+350%"),
]


def casey_ledger():
    rows = []
    for label, osi, ein, exits, claimed, note in CASEY:
        fol = None
        if osi:
            bars = load_bars(osi)
            if bars:
                fe, feta = next_open(bars, cts(ein))
                if fe and fe > 0:
                    fv, ok = 0.0, True
                    for xts, fr in exits:
                        fx, _ = next_open(bars, cts(xts))
                        if fx is None:
                            fx = bars[-1]["c"]
                        fv += fx * fr
                    fol = (fv / fe - 1) * 100
        rows.append(dict(label=label, fol=fol, claimed=claimed, note=note))
    return rows


def pstats(name, vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f"  {name}: 无样本"); return
    w = sum(1 for v in vals if v > 0)
    print(f"  {name}: {len(vals)}笔 胜{w} ({w/len(vals)*100:.0f}%) 均{sum(vals)/len(vals):+.1f}%/笔 等权累计{sum(vals):+.0f}%")


def main():
    today = date(2026, 8, 13)
    print("═"*74); print("ELITE (#elite-alert)  7/27~8/12"); print("═"*74)
    rows = settle_elite(elite_ledger(), today)
    for r in rows:
        legs = " ".join(f"{p}x{f}" for p, f in r["legs"]) or "—"
        a = f"{r['adv']:+.0f}%" if r["adv"] is not None else "—"
        rl = f"{r['real']:+.0f}%" if r["real"] is not None else "在途"
        fo = f"{r['fol']:+.0f}%" if r["fol"] is not None else "—"
        print(f"  {r['label']:22} 入{r['entry']:<6} 腿[{legs}] 宣称{a:>7} 真实{rl:>7} 跟单{fo:>7} {r['status']}")
    closed = [r for r in rows if r["status"] != "open"]
    print()
    pstats("宣称口径(只算他报SOLD的)", [r["adv"] for r in closed])
    pstats("真实口径(含沉默死单/runner死亡)", [r["real"] for r in closed])
    pstats("跟单口径(有K线, bar开盘成交)", [r["fol"] for r in closed])
    openp = [r for r in rows if r["status"] == "open"]
    for r in openp:
        print(f"  在途: {r['label']} 入{r['entry']} 现值mark {r['tail']}")

    print(); print("═"*74); print("CASEY (#指数-casey)  8月 (7月K线已清, 仅宣称口径)"); print("═"*74)
    rows = casey_ledger()
    for r in rows:
        fo = f"{r['fol']:+.0f}%" if r["fol"] is not None else "无K线"
        cl = f"{r['claimed']:+.0f}%" if r["claimed"] is not None else "未宣称"
        print(f"  {r['label']:18} 跟单{fo:>8}  宣称{cl:>8}  {r['note']}")
    print()
    pstats("跟单口径(11笔有K线)", [r["fol"] for r in rows])
    pstats("宣称口径(他给了%的)", [r["claimed"] for r in rows if r["claimed"] is not None and r["claimed"] != -100])
    out = dict(generated=datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds"),
               elite=settle_elite(elite_ledger(), today), casey=casey_ledger())
    (ROOT/"output"/"elite_casey_backtest.json").write_text(json.dumps(out, ensure_ascii=False, default=str))
    print("\n存 → output/elite_casey_backtest.json")


if __name__ == "__main__":
    main()
