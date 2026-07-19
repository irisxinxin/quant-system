#!/usr/bin/env python3
"""
bt_mechanical_intraday.py — 机械出场回测, 用真实期权5分K按【真实时序】触发止盈/止损 (修日线whipsaw)。
只测七月(有真实期权5分K的可信子集), 标的9ema用日线拖尾runner。与镜像跟单(+33%加权)apples对比。
跑法: source ~/.longport_creds.env && /usr/local/opt/python@3.13/bin/python3.13 bt_mechanical_intraday.py
"""
import json, math, statistics as st
import warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone, date
from pathlib import Path
from enrich_parser import parse_signal
from signal_history import _resolve_ambig_hist
import backtest_enrich as E
from longport.openapi import Config, QuoteContext, Period, AdjustType

UTC = timezone.utc
OUT = Path(__file__).parent / "output"


def ema(vals, period=9):
    k = 2 / (period + 1); e = vals[0]; out = []
    for v in vals:
        e = v * k + e * (1 - k); out.append(e)
    return out


CONFIGS = {
    "用户提案(20/40/60各20%+9ema×2+-60硬止损)":
        dict(ladder=[(.2,.2),(.4,.2),(.6,.2)], init_stop=0.4, be=False, ema_n=2),
    "用户提案但止损改-30(手册数)":
        dict(ladder=[(.2,.2),(.4,.2),(.6,.2)], init_stop=0.7, be=False, ema_n=2),
    "阶梯20/40/60+首档保本+9ema×2":
        dict(ladder=[(.2,.2),(.4,.2),(.6,.2)], init_stop=0.7, be=True, ema_n=2),
    "手册忠实(先回本50%@+30, 25%@+60, 首档保本, 9ema×2)":
        dict(ladder=[(.3,.5),(.6,.25)], init_stop=0.7, be=True, ema_n=2),
    "手册忠实-9ema×1(更快出runner)":
        dict(ladder=[(.3,.5),(.6,.25)], init_stop=0.7, be=True, ema_n=1),
    "末期期权规则(+60半仓+9ema×2 runner, -30)":
        dict(ladder=[(.6,.5)], init_stop=0.7, be=True, ema_n=2),
    "纯9ema拖尾(不阶梯, -30初始)":
        dict(ladder=[], init_stop=0.7, be=False, ema_n=2),
    "纯TP+保本无9ema(对照)":
        dict(ladder=[(.3,.5),(.6,.25)], init_stop=0.7, be=True, ema_n=0),
}


def main():
    q = QuoteContext(Config.from_env())
    msgs = json.load(open(OUT / "enrich_history.json"))
    parsed = []
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"]); ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        parsed.append((m["id"], ts, m["text"], parse_signal(m["text"], ts.date())))
    buys, seen = [], set()
    for mid, ts, text, s in parsed:
        if s.kind in ("BUY", "BUY_NOEXPIRY") and s.limit_price <= 8.0: pass
        elif s.kind == "BUY_AMBIG" and s.limit_price <= 8.0:
            side = _resolve_ambig_hist(q, E, s, ts)
            if side is None: continue
            s.kind, s.right = "BUY", side
        else: continue
        if not (date(2026, 5, 1) <= s.expiry <= date(2026, 7, 17)): continue
        key = f"{s.ticker}{s.expiry}{s.strike}{s.right}:{ts.date()}"
        if key in seen: continue
        seen.add(key); buys.append(dict(ts=ts, sig=s))

    daily = {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 300, AdjustType.ForwardAdjust)
                daily[tk] = [(x.timestamp.date(), float(x.close)) for x in b]
            except Exception: daily[tk] = None
        return daily[tk]

    def osi_of(s): return f"{s.ticker}{s.expiry:%y%m%d}{s.right}{int(round(s.strike*1000)):06d}.US"

    def simulate(s, entry_ts, cfg):
        B = E.bars(q, osi_of(s))                    # 真实期权5分K
        if not B: return None
        post = [x for x in B if x["ts"] > entry_ts]
        if len(post) < 3: return None
        ud = gd(s.ticker)
        if not ud: return None
        closes = [c for _, c in ud]; e9 = ema(closes, 9)
        ema_by_day = {ud[i][0]: (closes[i], e9[i]) for i in range(len(ud))}
        entry = s.limit_price
        pos, first_trim, val = 1.0, False, 0.0
        ladder = list(cfg["ladder"]); done = [False] * len(ladder)
        ema_break = 0; last_day = None
        for x in post:
            if pos <= 1e-9: break
            d = x["ts"].astimezone(UTC).date()
            # 1) 止损 (期权低, 真实时序)
            if not first_trim and cfg["init_stop"] > 0 and x["l"] <= entry * cfg["init_stop"]:
                val += pos * entry * cfg["init_stop"]; pos = 0; break
            if first_trim and cfg["be"] and x["l"] <= entry:
                val += pos * entry; pos = 0; break
            # 2) 阶梯止盈 (期权高)
            for j, (thr, frac) in enumerate(ladder):
                if done[j] or pos <= 1e-9: continue
                if x["h"] >= entry * (1 + thr):
                    f = min(frac, pos); val += f * entry * (1 + thr); pos -= f; done[j] = True; first_trim = True
            if pos <= 1e-9: break
            # 3) 9ema拖尾 (标的日线收盘, 每到新交易日结束判一次)
            if cfg["ema_n"] > 0 and d != last_day and last_day is not None:
                if last_day in ema_by_day:
                    c, e = ema_by_day[last_day]
                    if c < e:
                        ema_break += 1
                        if ema_break >= cfg["ema_n"]:
                            val += pos * x["o"]; pos = 0; break
                    else:
                        ema_break = 0
            last_day = d
        if pos > 1e-9:
            val += pos * post[-1]["c"]
        return val / entry - 1

    testable = [b for b in buys if E.bars(q, osi_of(b["sig"]))]
    print(f"真实期权5分K可测 {len(testable)} 笔 (主要7月, 盘中真实时序止盈止损)")
    print("=" * 92)
    print(f"{'出场配置':48}{'仓位加权':>9}{'等额':>8}{'中位':>8}{'胜率':>7}{'归零%':>7}")
    print("-" * 92)
    def wt(l): return 0.3333 if l else 0.5
    results = {}
    for name, cfg in CONFIGS.items():
        rows = []
        for b in testable:
            s = b["sig"]
            lotto = ("lotto" in (s.size_tag or "").lower() or "scalp" in " ".join(s.raw.split()).lower() or s.expiry == b["ts"].date())
            r = simulate(s, b["ts"], cfg)
            if r is not None: rows.append((r, lotto))
        n = len(rows)
        if not n: continue
        W = sum(wt(l) for _, l in rows)
        wavg = sum(wt(l) * r for r, l in rows) / W * 100
        eq = sum(r for r, _ in rows) / n * 100
        med = st.median([r for r, _ in rows]) * 100
        win = sum(1 for r, _ in rows if r > 0) / n * 100
        zero = sum(1 for r, _ in rows if r <= -0.99) / n * 100
        results[name] = dict(wavg=wavg, eq=eq, med=med, win=win, zero=zero, n=n)
        print(f"{name:48}{wavg:>+8.0f}%{eq:>+7.0f}%{med:>+7.0f}%{win:>6.0f}%{zero:>6.0f}%")
    print("-" * 92)
    best = max(results.items(), key=lambda kv: kv[1]["wavg"])
    print(f"\n最优: 【{best[0]}】 {best[1]['wavg']:+.0f}%加权 胜率{best[1]['win']:.0f}% 归零{best[1]['zero']:.0f}% (n={best[1]['n']})")
    print("对照: 镜像跟他出场(七月真实K) 纯镜像+23% / 镜像+止损兜底+33%")
    json.dump(results, open(OUT / "bt_mechanical_intraday.json", "w"), ensure_ascii=False, indent=1)
    print("存 → output/bt_mechanical_intraday.json")


if __name__ == "__main__":
    main()
