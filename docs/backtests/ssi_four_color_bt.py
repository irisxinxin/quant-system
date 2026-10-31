# -*- coding: utf-8 -*-
"""
SSI 四色图回测 —— 严格按通达信语义复刻
  SMA(X,N,M): Y = (M*X + (N-M)*Y_prev)/N, 首值=X[0]
  EMA(X,N):   Y = (2*X + (N-1)*Y_prev)/(N+1), 首值=X[0]
  LLV/HHV:    含当根的 N 根滚动极值
信号在 T 日收盘确定，T+1 开盘成交；收益按 open→open 计。
"""
import sys, pathlib, numpy as np, pandas as pd
D = pathlib.Path(sys.argv[1])
TK = ['SOXL','TQQQ','GOOG','MSFT','CRWD','LITE','NVDA','NET']
COST = 0.0005      # 单边 0.05%
WARM = 60

def tdx_sma(x, n, m):
    y = np.empty(len(x)); y[:] = np.nan
    s = None
    for i, v in enumerate(x):
        if np.isnan(v): y[i] = np.nan if s is None else s; continue
        s = v if s is None else (m*v + (n-m)*s)/n
        y[i] = s
    return y

def tdx_ema(x, n):
    y = np.empty(len(x)); s = None
    for i, v in enumerate(x):
        s = v if s is None else (2*v + (n-1)*s)/(n+1)
        y[i] = s
    return y

def states(df):
    c, h, l = df.close.values, df.high.values, df.low.values
    llv = pd.Series(l).rolling(14, min_periods=14).min().values
    hhv = pd.Series(h).rolling(14, min_periods=14).max().values
    with np.errstate(divide='ignore', invalid='ignore'):
        rsv = (c - llv) / (hhv - llv) * 100
    rsv = pd.Series(rsv).ffill().values
    k = tdx_sma(rsv, 3, 1); d = tdx_sma(k, 3, 1)
    dif = tdx_ema(c, 12) - tdx_ema(c, 26); dea = tdx_ema(dif, 9); m = 2*(dif - dea)
    st = np.full(len(c), '', dtype=object)
    st[(k > d) & (m > 0)] = 'G'
    st[(k > d) & (m < 0)] = 'Y'
    st[(k < d) & (m > 0)] = 'B'
    st[(k < d) & (m < 0)] = 'R'
    return st, k, d, m

RULES = {
    '只拿绿':      ({'G'},     {'Y','B','R'}),
    'KDJ金叉(绿黄)': ({'G','Y'}, {'B','R'}),
    'MACD红柱(绿蓝)': ({'G','B'}, {'Y','R'}),
    '绿进蓝出':    ({'G'},     {'B','R'}),
    '绿进红出':    ({'G'},     {'R'}),
    '黄进红出':    ({'Y','G'}, {'R'}),
}

def run_rule(st, enter, exit_):
    pos = np.zeros(len(st), dtype=int); long_ = 0
    for i, s in enumerate(st):
        if s == '': pos[i] = long_; continue
        if long_ == 0 and s in enter: long_ = 1
        elif long_ == 1 and s in exit_: long_ = 0
        pos[i] = long_
    return pos

def metrics(df, pos_signal, label):
    # pos_signal[t] 由 t 日收盘状态决定；实际持仓区间 = [open[t+1], open[t+2])
    o = df.open.values
    n = len(o)
    ret_oo = np.zeros(n); ret_oo[:-1] = o[1:]/o[:-1] - 1          # open[t]→open[t+1]
    pos = np.zeros(n, dtype=int); pos[1:] = pos_signal[:-1]        # 延后一天执行
    chg = np.abs(np.diff(np.r_[0, pos]))
    daily = pos*ret_oo - chg*COST
    daily[:WARM] = 0
    eq = np.cumprod(1 + daily)
    yrs = (n - WARM)/252
    tot = eq[-1] - 1; cagr = (eq[-1])**(1/yrs) - 1
    peak = np.maximum.accumulate(eq); mdd = ((eq/peak) - 1).min()
    sh = daily[WARM:].mean()/daily[WARM:].std()*np.sqrt(252) if daily[WARM:].std() > 0 else 0
    # 逐笔
    trades = []; e = None
    for i in range(WARM, n):
        if pos[i] == 1 and (i == 0 or pos[i-1] == 0): e = i
        if pos[i] == 0 and i > 0 and pos[i-1] == 1 and e is not None:
            trades.append(o[i]/o[e] - 1 - 2*COST); e = None
    if e is not None: trades.append(o[-1]/o[e] - 1 - 2*COST)
    tr = np.array(trades)
    wins = tr[tr > 0]; loss = tr[tr <= 0]
    pf = wins.sum()/abs(loss.sum()) if len(loss) and loss.sum() != 0 else np.inf
    return dict(规则=label, 总收益=tot, 年化=cagr, 最大回撤=mdd, Calmar=cagr/abs(mdd) if mdd else np.nan,
                Sharpe=sh, 交易次数=len(tr), 胜率=(tr > 0).mean() if len(tr) else np.nan,
                盈亏比=pf, 在场比例=pos[WARM:].mean())

def fwd_by_state(df, st):
    c = df.close.values; rows = []
    base = {}
    for k in (1, 5, 10, 20):
        f = np.full(len(c), np.nan); f[:-k] = c[k:]/c[:-k] - 1
        base[k] = f
    unc = {k: np.nanmean(base[k][WARM:]) for k in base}
    for s in 'GYBR':
        mask = (st == s); mask[:WARM] = False
        r = {'状态': s, '天数占比': mask.mean()}
        for k in base:
            v = base[k][mask]; v = v[~np.isnan(v)]
            r[f'均值{k}d'] = v.mean(); r[f'胜率{k}d'] = (v > 0).mean()
            r[f't{k}d'] = (v.mean() - unc[k])/(v.std()/np.sqrt(len(v))) if len(v) > 1 else np.nan
        rows.append(r)
    return pd.DataFrame(rows), unc

all_m, all_f = [], []
for t in TK:
    df = pd.read_csv(D/f'{t}.csv', parse_dates=['date']).reset_index(drop=True)
    st, k, d, m = states(df)
    fw, unc = fwd_by_state(df, st); fw.insert(0, '标的', t); all_f.append(fw)
    bh = metrics(df, np.ones(len(df), dtype=int), '买入持有'); bh['标的'] = t; all_m.append(bh)
    for name, (en, ex) in RULES.items():
        r = metrics(df, run_rule(st, en, ex), name); r['标的'] = t; all_m.append(r)
    print(f'\n{"="*96}\n{t}  {df.date.iloc[WARM].date()} → {df.date.iloc[-1].date()}   无条件均值: 5d {unc[5]:+.2%}  20d {unc[20]:+.2%}')
    print('-'*96)
    print(f'{"状态":4s} {"占比":>6s} | {"5d均值":>7s} {"5d胜率":>6s} {"t":>5s} | {"20d均值":>7s} {"20d胜率":>6s} {"t":>5s}')
    for _, r in fw.iterrows():
        print(f'{r.状态:4s} {r.天数占比:>6.0%} | {r.均值5d:>+7.2%} {r.胜率5d:>6.0%} {r.t5d:>+5.1f} | {r.均值20d:>+7.2%} {r.胜率20d:>6.0%} {r.t20d:>+5.1f}')
    print('-'*96)
    print(f'{"规则":14s} {"总收益":>8s} {"年化":>7s} {"最大回撤":>7s} {"Calmar":>6s} {"Sharpe":>6s} {"笔数":>4s} {"胜率":>5s} {"盈亏比":>5s} {"在场":>4s}')
    for r in [x for x in all_m if x['标的'] == t]:
        print(f'{r["规则"]:14s} {r["总收益"]:>+8.0%} {r["年化"]:>+7.1%} {r["最大回撤"]:>7.0%} {r["Calmar"]:>6.2f} {r["Sharpe"]:>6.2f} {r["交易次数"]:>4d} {r["胜率"]:>5.0%} {r["盈亏比"]:>5.2f} {r["在场比例"]:>4.0%}')

M = pd.DataFrame(all_m); F = pd.concat(all_f)
M.to_csv(D/'ssi_metrics.csv', index=False); F.to_csv(D/'ssi_fwd_by_state.csv', index=False)
print(f'\n{"="*96}\n跨标的汇总：每个规则在 8 只票上的中位数')
g = M.groupby('规则').agg(年化=('年化','median'), 最大回撤=('最大回撤','median'), Calmar=('Calmar','median'),
                           Sharpe=('Sharpe','median'), 胜率=('胜率','median'), 在场=('在场比例','median'))
print(g.sort_values('Calmar', ascending=False).to_string(float_format=lambda x: f'{x:.2f}'))
print('\n每个规则跑赢买入持有(按 Calmar)的标的数:')
for rule in RULES:
    w = sum(M[(M.规则==rule)].set_index('标的').Calmar > M[M.规则=='买入持有'].set_index('标的').Calmar)
    print(f'  {rule:14s} {w}/8')
