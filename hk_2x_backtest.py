"""hk_2x_backtest.py — 港股2x三星/海力士杠杆ETF日内回测(7747.HK/7709.HK)。
⚠注意: 策略在美股上校准; 港股有午休/不同时段, 机械跑; 实盘系统只做美股(不能下港股单), 仅供参考。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from longport.openapi import Config, QuoteContext
from longport_history_backtest import fetch_history
from data.save_intraday_data import convert_one
from backtest_engine import run_backtest, compute_metrics, load_5m_data, KLINES_5M_DIR

# tv_strategies/intraday_pool 的 8 策略
from signals.strategies.intraday_pool import STRATEGIES

HK = {"7747.HK": "三星2x", "7709.HK": "海力士2x"}
HIST = Path(__file__).parent / "cache" / "longport_history"
ctx = QuoteContext(Config.from_env())

from longport.openapi import Period, AdjustType

def fetch_5m(ctx, sym, max_batches=25):
    """直接分页抓 5m (绕过fetch_history在港股上的bug); 港股时间戳=HKT墙钟。"""
    seen, bars = set(), []
    batch = ctx.candlesticks(sym, Period.Min_5, 1000, AdjustType.NoAdjust)
    for c in batch:
        if c.timestamp not in seen:
            seen.add(c.timestamp); bars.append(c)
    oldest = min((c.timestamp for c in batch), default=None)
    n = 1
    while oldest and n < max_batches:
        try:
            older = ctx.history_candlesticks_by_offset(sym, Period.Min_5, AdjustType.NoAdjust,
                                                       forward=False, count=1000, time=oldest)
        except Exception:
            break
        if not older:
            break
        new = sum(1 for c in older if c.timestamp not in seen)
        for c in older:
            if c.timestamp not in seen:
                seen.add(c.timestamp); bars.append(c)
        if new == 0:
            break
        oldest = min(c.timestamp for c in older); n += 1
    bars.sort(key=lambda c: c.timestamp)
    return bars

print("📥 抓港股 5m + 存 JSON")
for sym in HK:
    bars = fetch_5m(ctx, sym)
    if not bars:
        print(f"   {sym} ❌ 无数据"); continue
    recs = [{"datetime": c.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "open": round(float(c.open), 4),
             "high": round(float(c.high), 4), "low": round(float(c.low), 4),
             "close": round(float(c.close), 4), "volume": int(c.volume)} for c in bars]
    payload = {"symbol": sym, "interval": "5m", "timezone": "Asia/Hong_Kong", "bars": len(recs),
               "first": recs[0]["datetime"], "last": recs[-1]["datetime"], "data": recs}
    json.dump(payload, open(KLINES_5M_DIR / f"{sym}.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"   {sym} ({HK[sym]}) → {len(recs)}根 {payload['first'][:10]}→{payload['last'][:10]}")

print("\n📊 8 策略矩阵")
def pfclean(s):  # 干净最优 + 近期评级
    c = s[(s.phantom_rate_pct < 25) & (s.filled >= 60) & (s.pf >= 1.0)]
    if not len(c):
        return None
    t = c.sort_values("composite_score", ascending=False).iloc[0]
    rec, pf, n90 = t.recency_pf, t.pf, t.last_90d_n
    g = ("⚠️近期样本少" if n90 < 8 else "✅强(近期)" if rec >= 1.5 else "⚠️中等" if rec >= 1.2 else "⚠️弱" if rec >= 1.0 else "⚠️近期转弱")
    return t, g

for sym in HK:
    if load_5m_data(sym).empty:
        print(f"\n=== {sym} 无数据 ==="); continue
    de = load_5m_data(sym).index.max().date()
    rows = []
    for n, fn in STRATEGIES.items():
        try:
            m = compute_metrics(run_backtest(sym, fn, n), data_end_date=de)
            rows.append({"strategy": n, **{k: m.get(k) for k in
                ["filled", "phantom_rate_pct", "win_rate", "pf", "recency_pf",
                 "recency_win_rate", "recency_avg_pnl", "last_90d_n", "last_90d_cum_pct", "cum_pct", "composite_score"]}})
        except Exception as e:
            print(f"   {n} ERR {e}")
    d = pd.DataFrame(rows).sort_values("recency_pf", ascending=False)
    print(f"\n=== {sym} ({HK[sym]}) ===")
    print(f"{'策略':12}{'成交':>5}{'phantom':>8}{'胜率':>5}{'全PF':>6}{'近期PF':>7}{'近90笔':>7}{'近90%':>7}")
    for _, r in d.iterrows():
        fl = "⚠ph" if r.phantom_rate_pct >= 25 else ""
        print(f"{r.strategy:12}{int(r.filled):>5}{r.phantom_rate_pct:>7.0f}%{r.win_rate:>4.0f}%{r.pf:>6.2f}{r.recency_pf:>7.2f}{int(r.last_90d_n):>6}{r.last_90d_cum_pct:>6.1f}%")
    res = pfclean(d)
    if res:
        t, g = res
        print(f"  ✅ 干净最优: {t.strategy} | 近期PF {t.recency_pf:.2f} 全PF {t.pf:.2f} 胜率 {t.recency_win_rate:.0f}% 每笔 {t.recency_avg_pnl:+.2f}% 近90 {int(t.last_90d_n)}笔 → {g}")
    else:
        print(f"  ❌ 无干净策略(phantom<25%+成交≥60+PF≥1全不满足) → 不建议")
