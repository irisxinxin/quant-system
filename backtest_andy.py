#!/usr/bin/env python3
"""
backtest_andy.py — #andy-option 频道信号 fill-realistic 回测 (真实期权5分K)。

andy 格式: `:RedAlert: TICKER - $STRIKE CALLS/PUTS <到期> $权利金 [STOP LOSS AT $X]`
  到期: EXPIRATION THIS WEEK / NEXT WEEK / 0DTE / M/D / EXPIRATION M/D
  ~60% 信号自带止损位 → 回测两种止损: (A)统一-30% (B)他给的止损优先, 没给用-30%

模拟规则 (同 enrich 引擎语义):
  入场: 信号价限价 Day 单, 下一根完整bar起, Open≤限价按Open / Low≤限价按限价; 不追高
  止盈: +100% 卖半仓 (券商侧挂单语义: bar.High≥tp)
  止损: bar.Low≤stop → 按stop (跳空按Open); 触发全平
  出场跟随: 他的出场消息按票分级 (trim/PT HIT/lock=partial→卖半; all out/stopped/cut=full→全平;
            二次partial→清仓; 多票消息→忽略)
  到期强平: 到期日 15:40 ET
  仓位: 统一名义 $10,000/信号 (张数=预算//权利金/100, 上限50张), lotto减半$2,000
数据诚实: 合约K线拉不到→如实排除; SPX指数期权符号不确定→拉不到就排除。
"""
import json, re, sys
from datetime import datetime, date, time as dtime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
from zoneinfo import ZoneInfo
from longport.openapi import Config, QuoteContext, Period, AdjustType

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
TP_MULT, DEF_STOP, FEE = 2.0, 0.7, 0.7
BUDGET, LOTTO_BUDGET, MAX_QTY = 10000, 2000, 50

ENTRY_RE = re.compile(
    r"RedAlert:?\s*([A-Z]{1,5}(?:\.[A-Z])?)\s*-\s*\$?([\d.]+)\s+(?:ITM\s+)?(CALLS?|PUTS?)", re.I)
PREM_RE = re.compile(r"(?:CALLS?|PUTS?)[^$]*?\$?(\d*\.\d+|\d+\.?\d*)")
STOP_RE = re.compile(r"STOP\s*LOSS\s*AT\s*\$?(\d*\.?\d+)", re.I)
DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
LOTTO_RE = re.compile(r"\bLOTTO|HERO OR ZERO|SUPER LOTTO", re.I)

EXIT_FULL_RE = re.compile(r"\ball\s+out\b|\bstopped\s*(?:out)?\b|\bcut(?:ting)?\b|\bclosed?\s+(?:out|it|all)\b|\bi'?m\s+out\b", re.I)
EXIT_PART_RE = re.compile(r"\btrim\w*\b|\bPT\s*HIT\b|\block(?:ed|ing)?\s+(?:in\s+)?(?:profits?|gains?|some)\b|\btak(?:e|ing)\s+(?:some\s+)?profits?\b|\bsold\s+(?:some|half)\b|\bscal(?:e|ing)\s+out\b|\bsell(?:ing)?\s+(?:some|half|into)\b", re.I)
TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
NOT_TICKER = {"THE", "AND", "ALL", "OUT", "PT", "HIT", "SL", "AT", "CAN", "FOR", "NOW", "HERE",
              "THIS", "WEEK", "NEXT", "DTE", "ITM", "OTM", "HIGH", "RISK", "STOP", "LOSS", "ROUND",
              "SET", "ON", "REST", "OF", "MY", "IF", "YOU", "WANT", "TO", "BE", "UP", "PER", "WOW",
              "WHAT", "LOW", "NEW", "ATH", "EOD", "GAP", "FILL", "CHEAP", "PLAY", "SIZE", "ONLY", "SMALL"}


def _friday(d: date, weeks=0) -> date:
    return d + timedelta(days=(4 - d.weekday()) % 7 + 7 * weeks)


def parse_entry(text: str, d: date):
    m = ENTRY_RE.search(text)
    if not m:
        return None
    ticker = m.group(1).replace(".", "")           # BRK.B → BRKB
    strike = float(m.group(2))
    right = "C" if m.group(3).upper().startswith("CALL") else "P"
    # 到期
    up = text.upper()
    if "0DTE" in up:
        expiry = d
    elif re.search(r"EXPIRATION\s+THIS\s+WEEK", up):
        expiry = _friday(d)
    elif re.search(r"EXPIRATION\s+NEXT\s+WEEK", up):
        expiry = _friday(d, 1)
    else:
        md = DATE_RE.search(text[m.end():])
        if md and 1 <= int(md.group(1)) <= 12:
            y = d.year
            expiry = date(y, int(md.group(1)), int(md.group(2)))
            if expiry < d:
                expiry = date(y + 1, int(md.group(1)), int(md.group(2)))
        else:
            expiry = _friday(d)                    # 默认本周 (他99%是周内)
    # 权利金: CALLS/PUTS 之后第一个价格数 (跳过到期日 M/D 数字)
    tail = text[m.end():]
    tail_wo_stop = STOP_RE.sub(" ", tail)
    tail_wo_date = DATE_RE.sub(" ", tail_wo_stop)
    pm = re.search(r"\$?(\d*\.\d+|\d+\.\d*|\d+)\b", tail_wo_date.replace("$", " $"))
    prem = None
    for cand in re.findall(r"(?<![\d/])(\d*\.\d+|\d+)(?![\d/])", tail_wo_date):
        v = float(cand)
        if 0.05 <= v <= 60 and v != strike:
            prem = v; break
    if prem is None:
        return None
    ms = STOP_RE.search(text)
    stop = float(ms.group(1)) if ms else None
    if stop and stop >= prem:                      # 异常(20% SL文字等) → 弃用他的
        stop = None
    return dict(ticker=ticker, strike=strike, right=right, expiry=expiry,
                prem=prem, stop=stop, lotto=bool(LOTTO_RE.search(text)))


def classify_exit(text: str):
    """返回 (tickers, level) level: full/partial/None"""
    up = " ".join(text.split())
    tks = [t for t in TICKER_RE.findall(up) if t not in NOT_TICKER]
    lv = "full" if EXIT_FULL_RE.search(up) else ("partial" if EXIT_PART_RE.search(up) else None)
    return list(dict.fromkeys(tks)), lv


def osi(e):
    s = int(round(e["strike"] * 1000))
    return f"{e['ticker']}{e['expiry']:%y%m%d}{e['right']}{s:06d}.US"


def load_events():
    msgs = json.load(open("output/andy_history.json"))
    buys, exits = [], []
    seen = set()
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        t = m["text"]
        if "RedAlert" in t:
            e = parse_entry(t, ts.date())
            if e:
                key = f"{osi(e)}:{ts.date()}"
                if key not in seen:
                    seen.add(key)
                    buys.append(dict(ts=ts, e=e, raw=" ".join(t.split())[:90]))
                continue
        tks, lv = classify_exit(t)
        if lv and len(tks) == 1:
            exits.append(dict(ts=ts, ticker=tks[0], level=lv))
    return buys, exits


_cache = {}
def bars(q, sym):
    if sym not in _cache:
        try:
            b = q.candlesticks(sym, Period.Min_5, 1000, AdjustType.NoAdjust)
            _cache[sym] = sorted([dict(ts=x.timestamp.astimezone(UTC), o=float(x.open),
                                       h=float(x.high), l=float(x.low), c=float(x.close)) for x in b],
                                 key=lambda r: r["ts"])
        except Exception:
            _cache[sym] = []
    return _cache[sym]


def et_dt(d, t):
    return datetime.combine(d, t, tzinfo=ET).astimezone(UTC)


def simulate(q, buy, all_exits, use_his_stop):
    e, t0 = buy["e"], buy["ts"]
    sym = osi(e)
    B = bars(q, sym)
    if not B:
        return None
    qty = max(1, min(MAX_QTY, int((LOTTO_BUDGET if e["lotto"] else BUDGET) // (e["prem"] * 100))))
    day_end = et_dt(t0.astimezone(ET).date(), dtime(16, 0))
    entry_px = entry_ts = None
    for b in B:
        if b["ts"] <= t0:
            continue
        if b["ts"] > day_end:
            break
        if b["o"] <= e["prem"]:
            entry_px, entry_ts = b["o"], b["ts"]; break
        if b["l"] <= e["prem"]:
            entry_px, entry_ts = e["prem"], b["ts"]; break
    if entry_px is None:
        return dict(status="no_fill", qty=qty)
    stop = (e["stop"] if (use_his_stop and e["stop"]) else round(entry_px * DEF_STOP, 2))
    tp = round(entry_px * TP_MULT, 2)
    remain, reduced = qty, False
    sells = []
    ex = sorted([x for x in all_exits if x["ticker"] == e["ticker"] and x["ts"] > entry_ts],
                key=lambda x: x["ts"])
    force_ts = et_dt(e["expiry"], dtime(15, 40))
    post = [b for b in B if b["ts"] > entry_ts]

    def sell_open_after(ts, n, why):
        nonlocal remain
        for b in post:
            if b["ts"] > ts:
                sells.append((b["o"], n, why)); remain -= n; return
        sells.append((post[-1]["c"] if post else 0.0, n, why)); remain -= n

    events = [(x["ts"], x["level"]) for x in ex] + [(force_ts, "force")]
    events.sort(key=lambda x: x[0])
    ei = 0
    for b in post:
        if remain <= 0:
            break
        while ei < len(events) and events[ei][0] <= b["ts"]:
            ts_, lv = events[ei]; ei += 1
            if remain <= 0:
                break
            if lv == "force":
                sell_open_after(ts_, remain, "到期强平")
            elif lv == "full":
                sell_open_after(ts_, remain, "他清仓")
            elif lv == "partial":
                if not reduced and remain >= 2:
                    sell_open_after(ts_, max(1, remain // 2), "跟随减仓"); reduced = True
                elif not reduced:
                    sell_open_after(ts_, remain, "跟随减仓(全)"); reduced = True
                else:
                    sell_open_after(ts_, remain, "二次减仓清")
        if remain <= 0:
            break
        if b["o"] <= stop:
            sells.append((b["o"], remain, "止损")); remain = 0; break
        if b["l"] <= stop:
            sells.append((stop, remain, "止损")); remain = 0; break
        if not reduced and remain >= 1:
            if b["o"] >= tp:
                n = max(1, remain // 2); sells.append((b["o"], n, "止盈2x")); remain -= n; reduced = True
            elif b["h"] >= tp:
                n = max(1, remain // 2); sells.append((tp, n, "止盈2x")); remain -= n; reduced = True
    if remain > 0:
        if e["expiry"] >= date.today():
            sells.append((B[-1]["c"], remain, "持仓中mark"))
        else:
            sells.append((0.0, remain, "归零"))
        remain = 0
    gross = sum(px * 100 * n for px, n, _ in sells) - entry_px * 100 * qty
    fees = FEE * qty + FEE * sum(n for _, n, _ in sells)
    return dict(status="traded", qty=qty, entry=entry_px, sells=sells,
                pnl=gross - fees, cost=entry_px * 100 * qty)


def run(use_his_stop):
    q = QuoteContext(Config.from_env())
    buys, exits = load_events()
    rows, skipped, nofill = [], [], 0
    for b in buys:
        r = simulate(q, b, exits, use_his_stop)
        if r is None:
            skipped.append(b)
        elif r["status"] == "no_fill":
            nofill += 1
        else:
            rows.append((b, r))
    total = sum(r["pnl"] for _, r in rows)
    invested = sum(r["cost"] for _, r in rows)
    wins = [r["pnl"] for _, r in rows if r["pnl"] > 0]
    losses = [r["pnl"] for _, r in rows if r["pnl"] <= 0]
    tag = "他的止损优先" if use_his_stop else "统一-30%"
    print(f"\n════ 止损策略: {tag} ════")
    for b, r in sorted(rows, key=lambda x: x[0]["ts"]):
        e = b["e"]
        s = "; ".join(f"{n}@{px:.2f}[{w[:5]}]" for px, n, w in r["sells"][:3])
        print(f"[{b['ts']:%m-%d %H:%M}] {e['ticker']:5}{e['expiry']:%m/%d} {e['strike']:>7}{e['right']} "
              f"入{r['entry']:<5}x{r['qty']:<3} {s[:58]:58} ${r['pnl']:+8,.0f} ({r['pnl']/r['cost']*100:+5.0f}%)")
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) else float("inf")
    n = len(rows)
    print(f"—— 成交{n} 未成交{nofill} 无数据{len(skipped)} | 赢{len(wins)}/输{len(losses)} "
          f"胜率{len(wins)/n*100 if n else 0:.0f}% | 总P&L ${total:+,.0f} / 投入 ${invested:,.0f} "
          f"= {total/invested*100 if invested else 0:+.0f}% | PF {pf:.2f}")
    if skipped:
        print("无数据合约: " + ", ".join(sorted({osi(b['e']) for b in skipped})))
    return total, invested


if __name__ == "__main__":
    buys, exits = load_events()
    print(f"解析: {len(buys)} 笔入场(去重) | {len(exits)} 条单票出场 | 预算$10k/lotto$2k 上限50张")
    run(use_his_stop=False)
    run(use_his_stop=True)
