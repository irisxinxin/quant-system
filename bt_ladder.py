#!/usr/bin/env python3
"""
bt_ladder.py — 阶梯止盈 vs 纯跟单 对照实验 (fill-realistic, 真实期权5分K)。

用户提案: "+40%止盈一批 +60%止盈一批 最后留一小批(不断太高止损)" vs "就是跟单"。

实验设计 (关键: 归一到同样张数, 只改出场结构, 隔离"加不加机械止盈"的效果):
  所有配置 contracts=3, 同一批信号, 同一套站长出场信号, 同一到期强平。
  唯一变量 = 机械止盈梯队 tp_peels:
    纯跟单      tp_peels=[]           无机械止盈, 全靠站长buy/sell + 宽止损兜底
    单档+60     tp_peels=[1.6]        现行冠军配置(等比放大到3张: +60卖1张, 剩2张跟他)
    阶梯40/60   tp_peels=[1.4,1.6]    +40卖1 +60卖1 最后1张跑
    阶梯40/60/80 tp_peels=[1.4,1.6,1.8]
  runner宽止损子网格: -30% / -50% / -60% / 无(到期兜底)

fill语义 (严格镜像 backtest_enrich + bot, 保守方向防美化):
  入场: 信号后下一根完整bar, Open≤限价→Open 否则 Low≤限价→限价 (不追高)
  每根post-entry bar, 处理顺序 = 站长出场事件 → runner止损 → 机械止盈梯队 (止损先于止盈=保守)
  机械止盈: 挂GTC限价卖, 每档 bar.Open≥档价→按Open(更优) 否则 bar.High≥档价→按档价; 一根bar可连破多档
  站长出场: full/vague(已减)→清剩仓; 首个partial/vague→卖1张; (与现引擎reduced状态机一致, 机械止盈也算已减)
  runner止损: bar.Open≤止损→按Open(跳空) 否则 bar.Low≤止损→按止损价 (MIT语义) → 清剩仓
  到期: 到期日15:40 ET后首根bar开盘清剩仓; 之后无bar→末bar收盘; 全天无数据→0
  摩擦: $0.7/张/边

跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 bt_ladder.py
"""
import json, os, sys
from datetime import datetime, date, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
FEE = 0.7
OUT = Path(__file__).parent / "output"


def et_dt(d, t):
    return datetime.combine(d, t, tzinfo=ET).astimezone(UTC)


def simulate(q, E, osi, sig, t0, all_exits, cfg):
    """回测一笔, 返回 dict 或 None(无K线)。cfg: contracts/tp_peels/runner_stop。"""
    B = E.bars(q, osi)
    if not B:
        return None
    C = cfg["contracts"]
    day_end = et_dt(t0.astimezone(ET).date(), dtime(16, 0))

    # ── 入场 (信号后下一根完整bar, 不追高) ──
    entry_px = entry_ts = None
    for b in B:
        if b["ts"] <= t0:
            continue
        if b["ts"] > day_end:
            break
        pre_exit = [e for e in all_exits if e["ticker"] == sig.ticker
                    and t0 < e["ts"] < b["ts"] and e["level"] in ("partial", "vague", "full")]
        if pre_exit:
            return dict(status="cancelled", sells=[], pnl=0.0, cost=0.0)
        if b["o"] <= sig.limit_price:
            entry_px, entry_ts = b["o"], b["ts"]; break
        if b["l"] <= sig.limit_price:
            entry_px, entry_ts = sig.limit_price, b["ts"]; break
    if entry_px is None:
        return dict(status="no_fill", sells=[], pnl=0.0, cost=0.0)

    remain, reduced = C, False
    peaks = 0.0
    sells = []
    peels = sorted(cfg["tp_peels"])          # 升序档价倍数
    peel_done = [False] * len(peels)
    rstop = cfg["runner_stop"]               # 0=无
    expiry = sig.expiry
    force_ts = et_dt(expiry, dtime(15, 40))
    post = [b for b in B if b["ts"] > entry_ts]
    ex = sorted([e for e in all_exits if e["ticker"] == sig.ticker and e["ts"] > entry_ts],
                key=lambda e: e["ts"])
    events = [(e["ts"], "exit", e) for e in ex] + [(force_ts, "force", None)]
    events.sort(key=lambda x: x[0])
    ei = 0

    def sell_after(ts, qty, why):
        nonlocal remain
        for b in post:
            if b["ts"] > ts:
                sells.append((b["o"], qty, why)); remain -= qty; return
        last = post[-1]["c"] if post else 0.0
        sells.append((last if ts < force_ts else 0.0, qty, why)); remain -= qty

    for b in post:
        if remain <= 0:
            break
        peaks = max(peaks, b["h"])
        # 1) 站长出场事件 (含到期强平)
        while ei < len(events) and events[ei][0] <= b["ts"]:
            ts_, kind, e = events[ei]; ei += 1
            if remain <= 0:
                break
            if kind == "force":
                sell_after(ts_, remain, "到期强平")
            else:
                lv = e["level"]
                if lv == "alert":
                    continue
                if lv == "full":
                    sell_after(ts_, remain, "站长清仓"); reduced = True
                elif lv in ("partial", "vague"):
                    if not reduced and remain >= 2:
                        sell_after(ts_, 1, f"镜像减仓({lv})"); reduced = True
                    elif not reduced:
                        sell_after(ts_, remain, "站长减仓(剩1)"); reduced = True
                    elif lv == "vague":
                        sell_after(ts_, remain, "模糊催促清仓")
        if remain <= 0:
            break
        # 2) runner止损 (保守: 先于止盈)
        if rstop > 0:
            stop = round(entry_px * rstop, 2)
            if b["o"] <= stop:
                sells.append((b["o"], remain, f"止损{(rstop-1)*100:.0f}%")); remain = 0; break
            if b["l"] <= stop:
                sells.append((stop, remain, f"止损{(rstop-1)*100:.0f}%")); remain = 0; break
        # 3) 机械止盈梯队 (每档卖1张; 一根bar可连破多档)
        for i, p in enumerate(peels):
            if peel_done[i] or remain <= 0:
                continue
            tp = round(entry_px * p, 2)
            if b["o"] >= tp:
                sells.append((b["o"], 1, f"止盈+{(p-1)*100:.0f}%")); remain -= 1
                peel_done[i] = True; reduced = True
            elif b["h"] >= tp:
                sells.append((tp, 1, f"止盈+{(p-1)*100:.0f}%")); remain -= 1
                peel_done[i] = True; reduced = True

    if remain > 0:
        if expiry >= datetime.now(ET).date():
            sells.append((B[-1]["c"], remain, "持仓中mark"))
        else:
            sells.append((0.0, remain, "到期归零"))
        remain = 0

    gross = sum(px * 100 * q_ for px, q_, _ in sells) - entry_px * 100 * C
    fees = FEE * C + FEE * sum(q_ for _, q_, _ in sells)
    peak_mult = peaks / entry_px if entry_px else 0
    return dict(status="traded", entry=entry_px, sells=sells, pnl=gross - fees,
                cost=entry_px * 100 * C, peak_mult=peak_mult)


def load_signals(q, E):
    from enrich_parser import parse_signal, to_longport_symbol
    from signal_history import _resolve_ambig_hist
    msgs = json.load(open(OUT / "enrich_history.json"))
    parsed = []
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        parsed.append((m["id"], ts, m["text"], parse_signal(m["text"], ts.date())))
    exits = [dict(ts=ts, ticker=s.ticker, level=s.exit_level)
             for _, ts, _, s in parsed if s.kind == "EXIT"]
    buys, seen = [], set()
    for mid, ts, text, s in parsed:
        if s.kind == "BUY" and s.limit_price <= 5.0:
            osi = to_longport_symbol(s)
            key = f"{osi}:{ts.date()}"
            if key in seen:
                continue
            seen.add(key); buys.append(dict(ts=ts, osi=osi, sig=s))
        elif s.kind == "BUY_AMBIG" and s.limit_price <= 5.0:
            key = f"{s.ticker}{s.expiry}{s.strike}:{ts.date()}"
            if key in seen:
                continue
            seen.add(key)
            side = _resolve_ambig_hist(q, E, s, ts)
            if side is None:
                continue
            s.kind, s.right = "BUY", side
            buys.append(dict(ts=ts, osi=to_longport_symbol(s), sig=s))
    testable = [b for b in buys if E.bars(q, b["osi"])]
    return testable, exits


CONFIGS = [
    ("纯跟单",       dict(contracts=3, tp_peels=[],            runner_stop=0.0)),
    ("单档+60",      dict(contracts=3, tp_peels=[1.6],         runner_stop=0.0)),
    ("阶梯40/60",    dict(contracts=3, tp_peels=[1.4, 1.6],    runner_stop=0.0)),
    ("阶梯40/60/80", dict(contracts=3, tp_peels=[1.4, 1.6, 1.8], runner_stop=0.0)),
]
STOPS = [(0.0, "无止损"), (0.7, "-30%"), (0.5, "-50%"), (0.4, "-60%")]


def main():
    os.environ.setdefault("V2", "1")
    import backtest_enrich as E
    from longport.openapi import Config, QuoteContext
    q = QuoteContext(Config.from_env())
    testable, exits = load_signals(q, E)
    print(f"可回测 {len(testable)} 笔 (真实期权5分K) | 出场事件 {len(exits)} 条 | 每笔3张 | 摩擦${FEE}/张/边")
    print("=" * 92)
    print(f"{'配置':14}{'runner止损':>10}" + "".join(f"{s:>12}" for s in ["收益率", "胜率", "PF", "最差单笔", "均峰值x"]))
    print("-" * 92)
    results = {}
    for name, base in CONFIGS:
        for rs, slab in STOPS:
            cfg = dict(base); cfg["runner_stop"] = rs
            tot = cost = 0.0; wins = losses = 0; worst = 0.0; peaks = []
            details = []
            for b in testable:
                r = simulate(q, E, b["osi"], b["sig"], b["ts"], exits, cfg)
                if not r or r["status"] != "traded":
                    continue
                tot += r["pnl"]; cost += r["cost"]
                if r["pnl"] > 0: wins += 1
                else: losses += 1
                worst = min(worst, r["pnl"] / r["cost"] * 100)
                peaks.append(r["peak_mult"])
                details.append((b, r))
            n = wins + losses
            ret = tot / cost * 100 if cost else 0
            pf = wins and (sum(max(0, r["pnl"]) for _, r in details) /
                           (abs(sum(min(0, r["pnl"]) for _, r in details)) or 1))
            import statistics as st
            mp = st.median(peaks) if peaks else 0
            results[(name, slab)] = dict(ret=ret, pnl=tot, win=wins/n if n else 0,
                                         pf=pf, worst=worst, n=n)
            print(f"{name:14}{slab:>10}{ret:>+11.1f}%{wins}/{n:>10}{pf:>11.2f}"
                  f"{worst:>+11.0f}%{mp:>11.1f}")
        print("-" * 92)
    # 最优
    best = max(results.items(), key=lambda kv: kv[1]["pnl"])
    (name, slab), r = best
    print(f"\n最优: 【{name}】+ runner{slab} → 收益率{r['ret']:+.1f}% 胜率{r['win']*100:.0f}% "
          f"PF{r['pf']:.2f} 最差单笔{r['worst']:+.0f}% (n={r['n']})")
    json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()},
              open(OUT / "bt_ladder.json", "w"), ensure_ascii=False, indent=1)
    print("存 → output/bt_ladder.json")


if __name__ == "__main__":
    main()
