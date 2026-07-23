#!/usr/bin/env python3
"""replay/build_oracle.py — 构造回放用【时间索引价格 oracle】。

产出 pickle:
  opt[osi]     = sorted [(ts_utc, o,h,l,c)]   期权5分K: 优先真实 data/enrich_bars, 否则 BS 模拟
  und15[tk]    = sorted [(ts_utc, close)]      标的15分K(bot 9ema 用)
  meta         = {lo, hi, n_real, n_bs, tickers}

回放时 ReplayQuotes 按 clock 查这两张表 → 喂 bot 真实代码。
跑法: source ~/.longport_creds.env && /usr/local/bin/python3 bots/enrich/replay/build_oracle.py 2026-05-01 2026-07-31
⚠ 5-6月无真实期权K → BS(固定入场IV, 偏悲观~28pp), 只看结构性bug别抠P&L; 7月真实K高保真。
"""
import json, csv, sys, pickle, os
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]          # repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bots" / "enrich"))
os.environ.setdefault("DISCORD_BOT_TOKEN", "x")
from enrich_parser import parse_signal
from signal_history import _resolve_ambig_hist
from bt_be_grid import bs, solve_iv
import backtest_enrich as E
UTC = timezone.utc
BARS = ROOT / "data" / "enrich_bars"
OUT = ROOT / "bots" / "enrich" / "replay"


def osi_of(s):
    return f"{s.ticker}{s.expiry:%y%m%d}{s.right}{int(round(s.strike*1000)):06d}.US"


def load_real_bars(osi):
    f = BARS / f"{osi}.csv"
    if not f.exists():
        return None
    rows = []
    for r in csv.DictReader(open(f)):
        t = datetime.fromisoformat(r["ts"])
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        rows.append((t.astimezone(UTC), float(r["o"]), float(r["h"]), float(r["l"]), float(r["c"])))
    return sorted(rows) or None


def main():
    lo = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 5, 1)
    hi = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 7, 31)
    from longport.openapi import Config, QuoteContext, Period, AdjustType
    q = QuoteContext(Config.from_env())
    msgs = json.load(open(ROOT / "output" / "enrich_history.json"))

    # ── 解析出该窗口内所有 BUY 的 (osi, ticker, expiry, entry_ts) ──
    buys, tickers = [], set()
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"]); ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        if not (lo <= ts.date() <= hi):
            continue
        try:
            s = parse_signal(m["text"], ts.date())
        except Exception:
            continue
        if s.kind == "BUY_AMBIG" and s.limit_price and s.limit_price <= 8.0:
            side = _resolve_ambig_hist(q, E, s, ts)
            if side is None:
                continue
            s.kind, s.right = "BUY", side
        if s.kind not in ("BUY", "BUY_NOEXPIRY") or not (s.right and s.strike and s.expiry):
            continue
        if s.expiry < ts.date():
            continue
        buys.append((ts, s))
        tickers.add(s.ticker)

    # ── 标的日K(BS要标的现价+反解IV) + 标的5分K→合成期权BS路径 + 15分K(9ema) ──
    daily, i5 = {}, {}
    def gd(tk):
        if tk not in daily:
            try:
                b = q.candlesticks(f"{tk}.US", Period.Day, 500, AdjustType.ForwardAdjust)
                daily[tk] = [(x.timestamp.date(), float(x.close)) for x in b]
            except Exception:
                daily[tk] = None
        return daily[tk]
    def gi5(tk, d0, d1):
        key = (tk, d0, d1)
        if key in i5:
            return i5[key]
        out, cur = [], d0
        while cur <= d1:
            end = min(cur + timedelta(days=12), d1)
            try:
                b = q.history_candlesticks_by_date(f"{tk}.US", Period.Min_5, AdjustType.ForwardAdjust, cur, end)
                out += [(x.timestamp.astimezone(UTC), float(x.open), float(x.high), float(x.low), float(x.close)) for x in b]
            except Exception:
                pass
            cur = end + timedelta(days=1)
        out = sorted(set(out)); i5[key] = out; return out

    def bs_path(s, entry_ts):
        """无真实K时用 标的5分K + 入场反解IV 逐根BS 合成期权5分K。"""
        ud = gd(s.ticker)
        if not ud:
            return None
        ed = entry_ts.date()
        S0 = next((c for d, c in ud if d >= ed), None)
        if S0 is None:
            return None
        intr = max(0.0, (S0 - s.strike) if s.right == "C" else (s.strike - S0))
        if s.limit_price <= intr + 0.05:
            return None
        T0 = max(0.5, (s.expiry - ed).days + 0.5) / 365
        try:
            iv = solve_iv(s.limit_price, S0, s.strike, T0, s.right)
        except Exception:
            return None
        if iv >= 5.9:
            return None
        us = gi5(s.ticker, ed, s.expiry)
        if not us:
            return None
        out = []
        for (ts_, o, h, l, c) in us:
            Tr = max(0.0007, (s.expiry - ts_.date()).days + 0.4) / 365
            out.append((ts_, bs(o, s.strike, Tr, iv, s.right),
                        bs(h if s.right == "C" else l, s.strike, Tr, iv, s.right),
                        bs(l if s.right == "C" else h, s.strike, Tr, iv, s.right),
                        bs(c, s.strike, Tr, iv, s.right)))
        return sorted(out) or None

    # ── 组装 opt oracle ──
    opt, n_real, n_bs = {}, 0, 0
    seen = set()
    for ts, s in buys:
        osi = osi_of(s)
        if osi in seen:
            continue
        seen.add(osi)
        real = load_real_bars(osi)
        if real:
            opt[osi] = real; n_real += 1
        else:
            bp = bs_path(s, ts)
            if bp:
                opt[osi] = bp; n_bs += 1

    # ── 组装 und15: 标的5分K → 15分K收盘 ──
    und15 = {}
    for tk in sorted(tickers):
        u5 = gi5(tk, lo, hi)
        if not u5:
            continue
        buckets = {}
        for (ts_, o, h, l, c) in u5:
            b0 = ts_.replace(minute=(ts_.minute // 15) * 15, second=0, microsecond=0)
            # 每个15分桶取桶内最后一根5分K的收盘 = 该15分bar收盘
            if b0 not in buckets or ts_ > buckets[b0][0]:
                buckets[b0] = (ts_, c)
        und15[tk] = sorted((b0 + timedelta(minutes=15), cc) for b0, (t_, cc) in buckets.items())

    OUT.mkdir(exist_ok=True)
    path = OUT / f"oracle_{lo:%y%m%d}_{hi:%y%m%d}.pkl"
    pickle.dump(dict(opt=opt, und15=und15,
                     meta=dict(lo=str(lo), hi=str(hi), n_real=n_real, n_bs=n_bs,
                               n_opt=len(opt), n_tickers=len(und15))), open(path, "wb"))
    print(f"[oracle] BUY去重 {len(seen)} → 期权路径 {len(opt)} (真实K {n_real} / BS {n_bs}); 标的15分K {len(und15)}只")
    print(f"存 → {path}")


if __name__ == "__main__":
    main()
