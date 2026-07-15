#!/usr/bin/env python3
"""
backtest_enrich.py — 用真实期权5分钟K, fill-realistic 回测"跟 enrich 信号"策略 (与 discord_enrich_bot 同规则)。

规则 (严格镜像 bot):
  入场: 信号价限价×2张, Day单(当日16:00 ET未成交作废)。成交判定: 从信号后【下一根完整bar】起,
        bar.Open≤限价→按Open成交(更优), 否则 bar.Low≤限价→按限价成交。(信号所在的进行中bar不用=防lookahead)
  止盈: 成交后挂 GTC 卖1张 @ 2×成交价。bar.Open≥tp→按Open, 否则 bar.High≥tp→按tp。成交即记"已减仓"。
  出场跟随 (站长EXIT分级, 与enrich_parser一致):
    partial/vague(未减)→下一根bar开盘价卖1张记已减; partial(已减)→不动; vague(已减)→下根bar开盘清剩仓;
    full→撤止盈+下根bar开盘清剩仓; alert(多票/豁免词)→不动; 入场未成交时任何出场→撤单放弃。
  到期强平: 到期日 15:40 ET 后第一根bar开盘清剩仓; 若之后无bar→按当日最后bar收盘; 全天无数据→按0(归零)。
  摩擦: 每张每边 $0.7 (佣金+平台费近似)。
数据诚实: 合约历史K拉不到(长桥只留约1个月)→整笔标"无法回测"排除, 绝不用模型价近似。
"""
import json, sys
from datetime import datetime, date, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
from longport.openapi import Config, QuoteContext, Period, AdjustType
from enrich_parser import parse_signal, to_longport_symbol

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
CONTRACTS, TP_MULT, MAX_PREMIUM, FEE = 2, 2.0, 5.0, 0.7
import os
SECOND_PARTIAL_CLOSES = os.environ.get("V2","")=="1"   # 二次partial清runner
RUNNER_STOP = float(os.environ.get("V3_STOP","0") or 0)
LADDER = os.environ.get("LADDER","")=="1"
CONTRACTS = int(os.environ.get("N", CONTRACTS))  # >0: 权利金止损倍数(0.5=-50%), 需期权行情


def load_events():
    msgs = json.load(open("output/enrich_history.json"))
    buys, exits = [], []
    seen = set()
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        s = parse_signal(m["text"], ts.date())
        if s.kind == "BUY":
            osi = to_longport_symbol(s)
            key = f"{osi}:{ts.date()}"
            if key in seen:            # bot同款去重(站长重发)
                continue
            if s.limit_price > MAX_PREMIUM:
                continue
            seen.add(key)
            buys.append(dict(ts=ts, osi=osi, sig=s))
        elif s.kind == "EXIT":
            exits.append(dict(ts=ts, ticker=s.ticker, level=s.exit_level,
                              raw=" ".join(s.raw.split())[:70]))
    return buys, exits



def _bars_from_csv(sym):
    import csv as _csv
    from pathlib import Path as _P
    f = _P(__file__).parent / "data" / "enrich_bars" / f"{sym}.csv"
    if not f.exists():
        return []
    out = []
    try:
        with open(f, encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                out.append(dict(ts=datetime.fromisoformat(r["ts"]), o=float(r["o"]),
                                h=float(r["h"]), l=float(r["l"]), c=float(r["c"])))
    except Exception:
        return []
    return sorted(out, key=lambda x: x["ts"])


_bars_cache = {}
def bars(q, osi):
    if osi in _bars_cache:
        return _bars_cache[osi]
    try:
        b = q.candlesticks(osi, Period.Min_5, 1000, AdjustType.NoAdjust)
        out = [dict(ts=x.timestamp.astimezone(UTC), o=float(x.open), h=float(x.high),
                    l=float(x.low), c=float(x.close)) for x in b]
        out.sort(key=lambda x: x["ts"])
    except Exception:
        out = []
    if not out:
        out = _bars_from_csv(osi)   # 长桥清库后用归档K线
    _bars_cache[osi] = out
    return out


def et_dt(d: date, t: dtime) -> datetime:
    return datetime.combine(d, t, tzinfo=ET).astimezone(UTC)


def simulate(q, buy, all_exits):
    """回测一笔。返回 dict(状态, 成交价, 各次卖出[价,张,原因], pnl$) 或 None=无数据。"""
    osi, sig, t0 = buy["osi"], buy["sig"], buy["ts"]
    B = bars(q, osi)
    if not B:
        return None
    expiry = sig.expiry
    day_end = et_dt(t0.astimezone(ET).date(), dtime(16, 0))
    # ── 入场: 信号后下一根完整bar起 ──
    entry_px, entry_ts = None, None
    for b in B:
        if b["ts"] <= t0:            # 进行中/历史bar跳过 (防lookahead)
            continue
        if b["ts"] > day_end:
            break                     # Day单作废
        # 出场信号先于成交 → 撤单放弃
        pre_exit = [e for e in all_exits if e["ticker"] == sig.ticker
                    and t0 < e["ts"] < b["ts"] and e["level"] in ("partial", "vague", "full")]
        if pre_exit:
            return dict(status="cancelled", note="站长先出场,未成交撤单", sells=[], pnl=0.0)
        if b["o"] <= sig.limit_price:
            entry_px, entry_ts = b["o"], b["ts"]; break
        if b["l"] <= sig.limit_price:
            entry_px, entry_ts = sig.limit_price, b["ts"]; break
    if entry_px is None:
        return dict(status="no_fill", note="限价未触及(不追高)", sells=[], pnl=0.0)

    # ── 持仓管理 ──
    remain, reduced = CONTRACTS, False
    tp = round(entry_px * TP_MULT, 2)
    sells = []
    # 该票入场后的出场信号(时间序)
    ex = sorted([e for e in all_exits if e["ticker"] == sig.ticker and e["ts"] > entry_ts],
                key=lambda e: e["ts"])
    force_ts = et_dt(expiry, dtime(15, 40))
    post = [b for b in B if b["ts"] > entry_ts]

    def sell_at_open_after(ts, qty, why):
        nonlocal remain
        for b in post:
            if b["ts"] > ts:
                sells.append((b["o"], qty, why)); remain -= qty; return True
        # 之后无bar → 当日最后bar收盘, 再没有 → 0(归零)
        last = post[-1]["c"] if post else 0.0
        sells.append((last if ts < force_ts else 0.0, qty, why + "(尾盘无bar)"))
        remain -= qty
        return True

    events = [(e["ts"], "exit", e) for e in ex] + [(force_ts, "force", None)]
    events.sort(key=lambda x: x[0])
    ei = 0
    for b in post:
        if remain <= 0:
            break
        # 先处理该bar前的事件
        while ei < len(events) and events[ei][0] <= b["ts"]:
            ts_, kind, e = events[ei]; ei += 1
            if remain <= 0:
                break
            if kind == "force":
                sell_at_open_after(ts_, remain, "到期强平")
            else:
                lv = e["level"]
                if lv == "alert":
                    continue
                if lv == "full":
                    sell_at_open_after(ts_, remain, "站长清仓")
                elif lv in ("partial", "vague"):
                    if not reduced and remain >= 2:
                        sell_at_open_after(ts_, 1, f"镜像减仓({lv})"); reduced = True
                    elif not reduced:
                        sell_at_open_after(ts_, remain, f"站长减仓(仅剩{remain})"); reduced = True
                    elif lv == "vague":
                        sell_at_open_after(ts_, remain, "模糊催促清仓")
                    elif lv == "partial" and LADDER:
                        sell_at_open_after(ts_, min(1, remain) if remain >= 2 else remain,
                                           f"阶梯减仓(剩{remain})") if remain >= 2 else                             sell_at_open_after(ts_, remain, "阶梯末张清仓")
                    elif lv == "partial" and SECOND_PARTIAL_CLOSES:
                        sell_at_open_after(ts_, remain, "二次减仓→清runner")
        if remain <= 0:
            break
        # 止损 (V3: 权利金跌破 entry×倍数 → 全平; bar.Open跳空按Open, 否则按止损价 — MIT触发单语义)
        if RUNNER_STOP > 0 and remain > 0:
            stop = round(entry_px * RUNNER_STOP, 2)
            if b["o"] <= stop:
                sells.append((b["o"], remain, "止损-50%")); remain = 0; break
            if b["l"] <= stop:
                sells.append((stop, remain, "止损-50%")); remain = 0; break
        # 止盈单 (GTC, 券商侧)
        if not reduced and remain >= 1 and b["ts"] > entry_ts:
            if b["o"] >= tp:
                sells.append((b["o"], 1, "止盈+100%")); remain -= 1; reduced = True
            elif b["h"] >= tp:
                sells.append((tp, 1, "止盈+100%")); remain -= 1; reduced = True
    if remain > 0:
        # 到期与否按【美东日期】判 (Mac在SGT, 到期日ET下午盘=SGT翌日凌晨, 用本地日期会把仍在交易的持仓假归零)
        if expiry >= datetime.now(ET).date():    # 还没到期 → 持仓中, 按最后bar收盘mark(未实现)
            sells.append((B[-1]["c"], remain, "持仓中(市价mark)"))
        else:
            sells.append((0.0, remain, "到期归零兜底"))
        remain = 0

    gross = sum(px * 100 * qty for px, qty, _ in sells) - entry_px * 100 * CONTRACTS
    fees = FEE * CONTRACTS + FEE * sum(q_ for _, q_, _ in sells)
    return dict(status="traded", entry=entry_px, entry_ts=entry_ts, tp=tp,
                sells=sells, pnl=gross - fees,
                cost=entry_px * 100 * CONTRACTS)


def main():
    q = QuoteContext(Config.from_env())
    buys, exits = load_events()
    print(f"信号: {len(buys)} 笔BUY (去重后) | {len(exits)} 条EXIT | 每笔{CONTRACTS}张 止盈{TP_MULT}x 摩擦${FEE}/张/边")
    print("=" * 100)
    rows, skipped = [], []
    for b in buys:
        r = simulate(q, b, exits)
        if r is None:
            skipped.append(b); continue
        rows.append((b, r))
    total, wins, losses = 0.0, [], []
    invested = 0.0
    for b, r in rows:
        s = b["sig"]
        tag = f"{s.ticker} {s.expiry:%m/%d} ${s.strike}{s.right}"
        if r["status"] == "traded":
            total += r["pnl"]; invested += r["cost"]
            (wins if r["pnl"] > 0 else losses).append(r["pnl"])
            ret = r["pnl"] / r["cost"] * 100
            sell_s = "; ".join(f"{q_}张@{px:.2f}[{w[:6]}]" for px, q_, w in r["sells"])
            print(f"{b['ts']:%m-%d %H:%M} {tag:22} 入{r['entry']:.2f}x2  {sell_s:52} P&L ${r['pnl']:+8.0f} ({ret:+6.0f}%)")
        else:
            print(f"{b['ts']:%m-%d %H:%M} {tag:22} ✗ {r['note']}")
    n = len(wins) + len(losses)
    print("=" * 100)
    if n:
        pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")
        print(f"成交 {n} 笔: 赢{len(wins)}/输{len(losses)} 胜率{len(wins)/n*100:.0f}% | "
              f"总盈亏 ${total:+,.0f} | 总投入 ${invested:,.0f} | 收益率 {total/invested*100:+.0f}% | PF {pf:.2f}")
        print(f"平均每笔 ${total/n:+,.0f} | 最大单笔赢 ${max(wins) if wins else 0:+,.0f} / 输 ${min(losses) if losses else 0:+,.0f}")
    unfilled = sum(1 for _, r in rows if r["status"] == "no_fill")
    cancelled = sum(1 for _, r in rows if r["status"] == "cancelled")
    print(f"未成交(不追高) {unfilled} 笔 | 未成交先撤 {cancelled} 笔")
    if skipped:
        print(f"\n⚠️ 无法回测 {len(skipped)} 笔 (合约历史K已被清, 长桥只留约1月) — 如实排除:")
        for b in skipped:
            s = b["sig"]
            print(f"   {b['ts']:%m-%d} {s.ticker} {s.expiry:%m/%d} ${s.strike}{s.right} @{s.limit_price}")


if __name__ == "__main__":
    main()
