#!/usr/bin/env python3
"""
backtest_biancheng_suoya.py — 边城 & 索亚 期权台账回测。

方法(与小鱼/蛋挞不同, 他们是期权+自报成交价):
  1. source=self  : 他本人报了开仓价+平仓价 → 直接记账 (最可信)
  2. source=bars  : 持有到期未报平仓 → 用正股日线在到期日的收盘算内在价值结算
     sell put: 收盘>行权价 → 权利金全收; 否则按行权价接货, 再用现价算浮动
     spread/蝶: 到期收盘的内在价值
  3. source=claim : 只报盈亏结论(入场在归档开始前) → 单列, 不进胜率主表
  4. source=pending: 仍持仓 → 只标状态
台账: output/biancheng_suoya_ledger.json (人工逐条读全部222条消息建立)
"""
import json, os, sys, time, urllib.request, statistics as st
from datetime import datetime, timezone

PROXY = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7897")
L = json.load(open("output/biancheng_suoya_ledger.json"))
CACHE = {}
YMAP = {"SPX": "^GSPC", "BRKB": "BRK-B"}


def bars(sym):
    sym = YMAP.get(sym, sym)
    if sym in CACHE:
        return CACHE[sym]
    out = {}
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=6mo"
        op = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY}))
        d = json.loads(op.open(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read())
        r = d["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        for i, t in enumerate(r["timestamp"]):
            if q["close"][i]:
                out[str(datetime.fromtimestamp(t, timezone.utc).date())] = round(q["close"][i], 2)
        time.sleep(0.05)
    except Exception as e:
        print(f"  ⚠️ {sym}: {type(e).__name__}", file=sys.stderr)
    CACHE[sym] = out
    return out


def close_on(tk, d):
    B = bars(tk)
    if d in B:
        return B[d]
    ks = sorted(k for k in B if k <= d)
    return B[ks[-1]] if ks else None


def cur(tk):
    B = bars(tk)
    return B[max(B)] if B else None


# ═════════ 边城 ═════════
print("═" * 100)
print("【边城】期权台账结算  (07-16 ~ 08-13, 46条记录)")
print("═" * 100)
rows = []
for t in L["biancheng"]:
    r = dict(t)
    pnl = t.get("pnl")
    if t["source"] == "self":
        if pnl is None:
            if "credit" in t and t.get("close_cost") is not None and t["credit"]:
                pnl = t["credit"] - t["close_cost"]
            elif "debit" in t and t.get("close_px") is not None:
                pnl = t["close_px"] - t["debit"]
        r["pnl"] = pnl
    elif t["source"] == "bars" and pnl is None:
        exp, S = t.get("exp"), t.get("strikes")
        C = close_on(t["tk"], exp) if exp else None
        if C is None or S is None:
            r["pnl"] = None
        elif t["kind"] == "sps":          # sell put spread
            intr = max(0, S[0] - C) - max(0, S[1] - C)
            r["pnl"] = round((t["credit"] or 0) - intr * 100)
        elif t["kind"] == "cs":            # call spread (debit)
            intr = max(0, C - S[0]) - max(0, C - S[1] if len(S) > 1 else 0)
            r["pnl"] = round(intr * 100 - t["debit"])
        elif t["kind"] == "bps":           # buy put spread
            intr = max(0, S[0] - C) - max(0, S[1] - C)
            r["pnl"] = round(intr * 100 - t["debit"])
        elif t["kind"] == "fly_call":
            a, b, c = S
            intr = max(0, C - a) - 2 * max(0, C - b) + max(0, C - c)
            r["pnl"] = round(intr * 100 - t["debit"])
        elif t["kind"] == "sp":            # naked sell put
            if t.get("credit") is None:
                r["pnl"] = None
                r["outcome"] = "胜(权利金未报)" if C > S[0] else "接货"
            elif C > S[0]:
                r["pnl"] = t["credit"]
            else:
                r["pnl"] = round(t["credit"] - (S[0] - cur(t["tk"])) * 100)
                r["outcome"] = f"接货@{S[0]}, 现价{cur(t['tk'])}"
        elif t["kind"] == "calendar":
            r["pnl"] = pnl
        r["settle"] = C
    rows.append(r)

for grp, name in [("self", "① 自报开平价（记账, 最可信）"), ("bars", "② 到期结算（正股K线判定）"),
                  ("claim", "③ 只报结论（入场在归档前, 单列）"), ("pending", "④ 仍持仓")]:
    sub = [r for r in rows if r["source"] == grp]
    if not sub:
        continue
    print(f"\n{name}  {len(sub)}笔")
    for r in sub:
        p = r.get("pnl")
        ps = f"{p:+6.0f}" if p is not None else "  n/a "
        flag = " ⚠️成交未确认" if r.get("fill") == "assumed" else ""
        settle = f" 结算{r.get('settle')}" if r.get("settle") else ""
        print(f"    {r['d']} {r['tk']:5} {ps}  {r['desc'][:46]:48}{settle}{flag} {r.get('note','')[:40]}")
    v = [r["pnl"] for r in sub if r.get("pnl") is not None]
    if v and grp in ("self", "bars"):
        w = [x for x in v if x > 0]
        print(f"    小计: {len(v)}笔可算 | 胜 {len(w)} 负 {len(v)-len(w)} | 净盈亏 ${sum(v):+,.0f} | 均笔 ${st.mean(v):+,.0f}")

conf = [r for r in rows if r["source"] in ("self", "bars") and r.get("pnl") is not None and r.get("fill") != "assumed"]
v = [r["pnl"] for r in conf]
w = [x for x in v if x > 0]
print(f"\n  ★ 确认成交且可结算 {len(v)} 笔: 胜率 {len(w)/len(v)*100:.0f}% | 总盈亏 ${sum(v):+,.0f} | "
      f"盈利笔均 ${st.mean(w):+,.0f} | 亏损笔均 ${st.mean([x for x in v if x<=0]):+,.0f}")

# 按策略类型
print("\n  按策略类型 (确认成交):")
KINDMAP = {"sps": "QQQ/个股 sell put spread", "sp": "单腿 sell put", "cs": "call spread(多为赌财报)",
           "fly_call": "蝴蝶(赌财报)", "calendar": "calendar(赌财报/短空)", "diag": "SPX对角(对冲)",
           "buy_put": "买put(方向空)", "buy_call": "买call", "sc": "sell call(对冲)", "bps": "买put spread(赌空)", "cs_mix": "组合"}
bykind = {}
for r in conf:
    bykind.setdefault(KINDMAP.get(r["kind"], r["kind"]), []).append(r["pnl"])
for k, vs in sorted(bykind.items(), key=lambda kv: -sum(kv[1])):
    ws = [x for x in vs if x > 0]
    print(f"    {k:28} {len(vs):2d}笔 | 胜{len(ws)}/负{len(vs)-len(ws)} | ${sum(vs):+,.0f}")

# ═════════ 索亚 ═════════
print("\n" + "═" * 100)
print("【索亚】sell put / covered call / 正股 三本账  (07-13 ~ 08-14)")
print("═" * 100)

print("\n▸ Sell Put 台账")
sp_done, sp_pend = [], []
for t in L["suoya_sp"]:
    r = dict(t)
    if t["source"] == "self":
        r["res"] = f"+{t['pnl']}"
        sp_done.append((t["tk"], t["pnl"], "self"))
    elif t["source"] == "bars":
        strike, exp = (t.get("roll_to") or [t["strike"], t["exp"]])
        C = close_on(t["tk"], exp)
        total_cr = t["credit"] + t.get("roll_gain", 0)
        if C is None:
            r["res"] = "无数据"
        elif C > strike:
            r["res"] = f"+{total_cr} (到期{C}>{strike})"
            sp_done.append((t["tk"], total_cr, "bars"))
        else:
            eq = round(total_cr - (strike - cur(t["tk"])) * 100)
            r["res"] = f"接货@{strike} 现{cur(t['tk'])} 等效{eq:+}"
            sp_done.append((t["tk"], eq, "bars-assigned"))
    else:
        strike = (t.get("roll_to") or [t["strike"]])[0]
        c = cur(t["tk"])
        r["res"] = f"持仓中 现{c} vs {strike} ({'价外✓' if c and c > strike else '⚠️价内'})"
        sp_pend.append(r)
    exp_s = t.get("exp") or "-"
    print(f"    {t['tk']:5} {str(t['strike']):>5} exp{exp_s[5:] if exp_s!='-' else '  - '} 权利金${t['credit']:>5}  → {r['res']}")
v = [p for _, p, _ in sp_done]
w = [x for x in v if x > 0]
print(f"  已了结 {len(v)} 笔: 胜 {len(w)} / 负 {len(v)-len(w)} = 胜率 {len(w)/len(v)*100:.0f}% | 合计 ${sum(v):+,.0f} | 持仓中 {len(sp_pend)} 笔")

print("\n▸ Covered Call 台账 (对冲性质: '亏'=被行权即正股止盈, 不是真亏)")
cc_done, cc_pend, cc_called = [], [], []
for t in L["suoya_cc"]:
    if t["source"] == "pending":
        cc_pend.append(t)
        continue
    if t["source"] == "self":
        cc_done.append(t["pnl"])
        print(f"    {t['tk']:5} {t['strike']:>5} exp{t['exp'][5:]} +${t['pnl']} (自报)")
        continue
    C = close_on(t["tk"], t["exp"])
    if C is None:
        continue
    if C < t["strike"]:
        cc_done.append(t["credit"])
        print(f"    {t['tk']:5} {t['strike']:>5} exp{t['exp'][5:]} +${t['credit']:>5} (到期{C}<{t['strike']}, 全收)")
    else:
        cc_called.append(t)
        print(f"    {t['tk']:5} {t['strike']:>5} exp{t['exp'][5:]} +${t['credit']:>5} +被行权(正股在{t['strike']}止盈, 收{C})")
print(f"  已到期 {len(cc_done)+len(cc_called)} 笔: 权利金全收 {len(cc_done)} / 被行权止盈 {len(cc_called)} | "
      f"权利金合计 ${sum(cc_done)+sum(t['credit'] for t in cc_called):+,.0f} | 挂着 {len(cc_pend)} 笔")

print("\n▸ 正股 DCA 台账 (vs 最新收盘)")
stk = []
for t in L["suoya_stock"]:
    c = cur(t["tk"])
    if c is None:
        continue
    pct = (c / t["px"] - 1) * 100
    stk.append(pct)
    print(f"    {t['tk']:5} {t['d'][5:]} 买{t['px']:>7} → 现{c:>8}  {pct:+6.1f}%")
w = [x for x in stk if x > 0]
print(f"  {len(stk)} 笔: 胜率 {len(w)/len(stk)*100:.0f}% | 均笔 {st.mean(stk):+.1f}% | 中位 {st.median(stk):+.1f}%")

json.dump(dict(biancheng=rows), open("output/biancheng_suoya_backtest.json", "w"), ensure_ascii=False, indent=1, default=str)
print("\n→ output/biancheng_suoya_backtest.json")
