#!/usr/bin/env python3
"""
backtest_xiaoyu_vip.py — 小鱼(鱼哥) VIP频道 2026-06-08 ~ 08-14 喊单质量回测。

台账: output/xiaoyu_vip_calls.json (人工逐条读 350 条原始消息抽取, 每条带原话)
数据: Yahoo 日线 (含盘中 High/Low, 限价单按 Low 检验能否成交)

四个口径:
  A  无脑跟单     喊单日收盘买入, 持有到最后交易日 (次日开盘买入做敏感性)
  B  他自己的纪律 收盘买入 + 321止盈(+30%走1/3, +50%走剩下一半, +100%清仓)
                  B2 = 321 再叠加 -20% 止损 (他本人没有止损, 纯对照)
  C  点位限价单   他给了具体价格的单, Low<=价才算成交, 未成交=踏空(单独计)
  D  择时/仓位    按他公开的现金比例曲线, 对比 100% QQQ

所有收益率都同时给出 same-period QQQ 基准 → alpha 才是真信息量。
"""
import json, os, sys, time, urllib.request, statistics as st
from datetime import datetime, timedelta, timezone

PROXY = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:7897")
CALLS = json.load(open("output/xiaoyu_vip_calls.json"))
CACHE = {}


def bars(sym):
    """→ [(date, o, h, l, c)] 升序"""
    if sym in CACHE:
        return CACHE[sym]
    out = []
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=6mo"
        op = urllib.request.build_opener(urllib.request.ProxyHandler({"https": PROXY}))
        d = json.loads(op.open(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read())
        r = d["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        for i, t in enumerate(r["timestamp"]):
            if q["close"][i] is None:
                continue
            dt = datetime.fromtimestamp(t, timezone.utc).date()
            out.append((dt, q["open"][i], q["high"][i], q["low"][i], q["close"][i]))
        time.sleep(0.06)
    except Exception as e:
        print(f"  ⚠️ {sym} 取数失败: {type(e).__name__}", file=sys.stderr)
    CACHE[sym] = out
    return out


def trade_date(ts):
    """消息UTC时间 → 可交易日. ET=UTC-4; 收盘(16:00 ET)后喊的算次日。"""
    dt = datetime.strptime(ts[:16], "%Y-%m-%dT%H:%M") - timedelta(hours=4)
    if dt.hour >= 16:
        dt += timedelta(days=1)
    return dt.date()


def bar_on_or_after(B, d):
    for i, b in enumerate(B):
        if b[0] >= d:
            return i, b
    return None, None


QQQ = bars("QQQ")
LAST = QQQ[-1][0]


def qqq_ret(d0):
    i, b = bar_on_or_after(QQQ, d0)
    return None if b is None else (QQQ[-1][4] / b[4] - 1) * 100


# ───────────────────────── 口径 A / B ─────────────────────────
def run_AB(entries):
    rows = []
    for e in entries:
        tk, d0 = e["tk"], trade_date(e["ts"])
        B = bars(tk)
        i, b = bar_on_or_after(B, d0)
        if b is None or i >= len(B) - 1:
            rows.append(dict(tk=tk, d0=str(d0), st="无数据/太晚"))
            continue
        ent = b[4]                       # A: 喊单日收盘
        ent_open = B[i + 1][1]           # A': 次日开盘
        fwd = B[i + 1:]
        pctA = (B[-1][4] / ent - 1) * 100
        pctA2 = (B[-1][4] / ent_open - 1) * 100

        # B: 321 止盈 (+30% 走1/3, +50% 走剩余一半, +100% 清仓)
        def ladder(stop_pct=None):
            left, real = 1.0, 0.0
            tiers = [(1.30, 1 / 3), (1.50, 0.5), (2.00, 1.0)]  # (触发倍数, 卖出比例(占当时剩余))
            ti, exit_note = 0, "持有"
            for bb in fwd:
                if stop_pct is not None and bb[3] <= ent * (1 + stop_pct):
                    px = min(bb[1], ent * (1 + stop_pct))
                    real += left * (px / ent - 1)
                    left, exit_note = 0.0, "止损"
                    break
                while ti < len(tiers) and bb[2] >= ent * tiers[ti][0]:
                    mult, frac = tiers[ti]
                    sold = left * frac
                    real += sold * (mult - 1)
                    left -= sold
                    ti += 1
                    exit_note = f"止盈{ti}档"
                if left <= 1e-9:
                    break
            if left > 1e-9:
                real += left * (B[-1][4] / ent - 1)
            return real * 100, exit_note

        pctB, noteB = ladder(None)
        pctB2, noteB2 = ladder(-0.20)
        rows.append(dict(tk=tk, d0=str(d0), st="ok", entry=round(ent, 2),
                         A=round(pctA, 1), A2=round(pctA2, 1),
                         B=round(pctB, 1), Bnote=noteB, B2=round(pctB2, 1), B2note=noteB2,
                         qqq=round(qqq_ret(d0), 1), note=e.get("note", "")))
    return rows


def stats(rows, key):
    v = [r[key] for r in rows if r.get("st") == "ok"]
    q = [r["qqq"] for r in rows if r.get("st") == "ok"]
    if not v:
        return None
    w = [x for x in v if x > 0]
    beat = sum(1 for r in rows if r.get("st") == "ok" and r[key] > r["qqq"])
    return dict(n=len(v), win=len(w) / len(v) * 100, mean=st.mean(v), med=st.median(v),
                qqq=st.mean(q), alpha=st.mean(v) - st.mean(q), beat=beat / len(v) * 100)


def show(title, s):
    if not s:
        print(f"{title}: 无样本"); return
    print(f"{title:22s} {s['n']:3d}笔 | 胜率 {s['win']:5.1f}% | 均笔 {s['mean']:+6.1f}% | "
          f"中位 {s['med']:+6.1f}% | 同期QQQ {s['qqq']:+5.1f}% | alpha {s['alpha']:+6.1f}% | 跑赢大盘 {s['beat']:4.1f}%")


print("═" * 108)
print(f"小鱼(鱼哥) VIP 喊单回测   2026-06-08 ~ {LAST}   最后交易日 QQQ={QQQ[-1][4]:.2f}")
print("═" * 108)

rows = run_AB(CALLS["buys"])
bad = [r for r in rows if r.get("st") != "ok"]
print(f"\n台账 {len(CALLS['buys'])} 条喊单, 可测 {len(rows)-len(bad)} 条" + (f", 跳过 {len(bad)}" if bad else ""))
print("\n【口径A 无脑跟单】喊单日收盘买入, 持有到今天 —— 没有止盈没有止损")
show("  全部", stats(rows, "A"))
show("  (敏感性)次日开盘买", stats(rows, "A2"))
print("\n【口径B 他自己的321止盈纪律】+30%走1/3 → +50%走剩下一半 → +100%清仓")
show("  321止盈", stats(rows, "B"))
show("  321+(-20%止损)对照", stats(rows, "B2"))

# 分时段
print("\n【按他的市场阶段切】")
for lo, hi, name in [("2026-06-01", "2026-07-01", "6月 防御期(现金40-50%)"),
                     ("2026-07-01", "2026-08-01", "7月 等待抄底条件"),
                     ("2026-08-01", "2026-09-01", "8月 三批加仓→满仓")]:
    sub = [r for r in rows if r.get("st") == "ok" and lo <= r["d0"] < hi]
    show(f"  {name}", stats(sub, "A"))

# 最好/最差
ok = [r for r in rows if r.get("st") == "ok"]
ok.sort(key=lambda r: r["A"])
print("\n  最差10笔:", "  ".join(f"{r['tk']}{r['A']:+.0f}%" for r in ok[:10]))
print("  最好10笔:", "  ".join(f"{r['tk']}{r['A']:+.0f}%" for r in ok[-10:][::-1]))

# ───────────────────────── 口径 C 点位限价单 ─────────────────────────
print("\n【口径C 他给的具体点位挂限价单】Low<=点位才算成交")
cres = []
for e in CALLS["levels"]:
    tk, d0, px = e["tk"], trade_date(e["ts"]), e["px"]
    B = bars(tk)
    i, b = bar_on_or_after(B, d0)
    if b is None:
        continue
    filled = None
    for bb in B[i:]:
        if bb[3] <= px:
            filled = (bb[0], min(px, bb[1]))
            break
    if filled:
        pct = (B[-1][4] / filled[1] - 1) * 100
        cres.append(dict(tk=tk, ts=e["ts"][:10], px=px, fill=str(filled[0]),
                         entry=round(filled[1], 2), pct=round(pct, 1),
                         qqq=round(qqq_ret(filled[0]), 1), st="成交"))
    else:
        cur = B[-1][4]
        cres.append(dict(tk=tk, ts=e["ts"][:10], px=px, st="未成交(踏空)",
                         gap=round((cur / px - 1) * 100, 1), cur=round(cur, 2)))
f = [r for r in cres if r["st"] == "成交"]
nf = [r for r in cres if r["st"] != "成交"]
print(f"  {len(cres)} 个点位: 成交 {len(f)}, 未成交(踏空) {len(nf)}")
if f:
    v = [r["pct"] for r in f]
    q = [r["qqq"] for r in f]
    print(f"  成交单: 胜率 {sum(1 for x in v if x>0)/len(v)*100:.0f}% | 均笔 {st.mean(v):+.1f}% | "
          f"中位 {st.median(v):+.1f}% | 同期QQQ {st.mean(q):+.1f}% | alpha {st.mean(v)-st.mean(q):+.1f}%")
    for r in sorted(f, key=lambda x: -x["pct"]):
        print(f"    ✅ {r['tk']:6} {r['ts']} 挂{r['px']:>7} 实成{r['entry']:>8} @{r['fill']} → {r['pct']:+7.1f}% (QQQ{r['qqq']:+5.1f}%)")
for r in sorted(nf, key=lambda x: -x["gap"]):
    print(f"    ❌ {r['tk']:6} {r['ts']} 挂{r['px']:>7} 从未触及, 现价{r['cur']:>8} (比挂单价高 {r['gap']:+.0f}%)")

# ───────────────────────── 口径 D 择时/仓位 ─────────────────────────
print("\n【口径D 择时: 跟他的现金比例 vs 一直满仓QQQ】")
reg = sorted([(trade_date(r["ts"]), 1 - r["cash"]) for r in CALLS["regime"]])
start = reg[0][0]
qb = [b for b in QQQ if b[0] >= start]


def pos_on(d):
    p = reg[0][1]
    for dd, w in reg:
        if dd <= d:
            p = w
    return p


# 跟单组合(等权, 每笔喊单当天收盘入场, 一直持有) 的日收益序列
basket_ret = {}
for e in CALLS["buys"]:
    tk, d0 = e["tk"], trade_date(e["ts"])
    B = bars(tk)
    i, b = bar_on_or_after(B, d0)
    if b is None or i >= len(B) - 1:
        continue
    for j in range(i + 1, len(B)):
        r = B[j][4] / B[j - 1][4] - 1
        basket_ret.setdefault(B[j][0], []).append(r)

eq_qqq = eq_timing = eq_full = 1.0
mdd_t = mdd_f = 0.0
pk_t = pk_f = 1.0
for k in range(1, len(qb)):
    d = qb[k][0]
    rq = qb[k][4] / qb[k - 1][4] - 1
    w = pos_on(qb[k - 1][0])
    rb = st.mean(basket_ret[d]) if basket_ret.get(d) else 0.0
    eq_qqq *= (1 + rq)
    eq_timing *= (1 + w * rq)      # D1 择时 x QQQ
    eq_full *= (1 + w * rb)        # D2 择时 x 他的等权组合
    pk_t = max(pk_t, eq_timing); mdd_t = min(mdd_t, eq_timing / pk_t - 1)
    pk_f = max(pk_f, eq_full);   mdd_f = min(mdd_f, eq_full / pk_f - 1)

print(f"  区间 {start} → {LAST} ({len(qb)} 个交易日)")
print(f"  ① 一直满仓 QQQ          {(eq_qqq-1)*100:+6.2f}%")
print(f"  ② 跟他仓位 × QQQ        {(eq_timing-1)*100:+6.2f}%   (最大回撤 {mdd_t*100:.1f}%)  ← 纯择时贡献")
print(f"  ③ 跟他仓位 × 他的等权组合 {(eq_full-1)*100:+6.2f}%   (最大回撤 {mdd_f*100:.1f}%)  ← 完整跟单")
print(f"  他当前公开仓位: {pos_on(LAST)*100:.0f}% (现金 {(1-pos_on(LAST))*100:.0f}%)")

# ───────────────────────── 补充1: 事件研究 (喊单后N日超额) ─────────────────────────
SECTOR = {
    "PL": "太空", "ASTS": "太空", "LUNR": "太空", "SATL": "太空", "BKSY": "太空", "FLY": "太空",
    "SPCL": "太空", "RKLB": "太空", "RDW": "太空", "NASA": "太空", "MNTS": "太空",
    "AXTI": "光电存", "POET": "光电存", "LITE": "光电存", "COHR": "光电存", "MRVL": "光电存",
    "GLW": "光电存", "ALMU": "光电存", "MU": "光电存", "SNDK": "光电存", "RMBS": "光电存",
    "INTC": "光电存", "ARM": "光电存", "ON": "光电存", "WOLF": "光电存", "SMCI": "光电存",
    "VSH": "光电存", "TTMI": "光电存", "AMKR": "光电存", "MCHP": "光电存", "MXL": "光电存",
    "QUBT": "量子", "IBM": "量子", "QBTS": "量子", "RGTI": "量子", "QNT": "量子",
    "MSFT": "M7大科技", "AMZN": "M7大科技", "META": "M7大科技", "GOOG": "M7大科技", "NFLX": "M7大科技",
    "ADBE": "软件", "NOW": "软件", "UBER": "软件", "MDB": "软件", "NET": "软件", "DDOG": "软件",
    "SNPS": "软件", "PLTR": "软件", "APP": "软件", "PYPL": "软件", "FSLY": "软件",
    "COIN": "加密", "SOFI": "加密", "PURR": "加密", "BTBT": "加密", "CRCL": "加密", "FIGR": "加密",
    "CIFR": "矿/数据中心", "WULF": "矿/数据中心", "APLD": "矿/数据中心", "IREN": "矿/数据中心",
    "CRWV": "矿/数据中心", "MP": "矿/数据中心", "USAR": "矿/数据中心", "CRML": "矿/数据中心", "UUUU": "矿/数据中心",
    "ANNA": "油/大宗", "XOM": "油/大宗", "USO": "油/大宗", "KOS": "油/大宗", "AA": "油/大宗", "UAMY": "油/大宗",
    "HIMS": "健康消费", "BRBR": "健康消费", "NKE": "健康消费",
    "ONDS": "无人机", "RCAT": "无人机", "BOT": "无人机", "AMPX": "无人机",
    "SMR": "核电储能", "OKLO": "核电储能", "EOSE": "核电储能", "FCEL": "核电储能", "VRT": "核电储能",
    "AMSC": "核电储能", "BE": "核电储能", "FLNC": "核电储能",
    "ORCL": "M7大科技", "XPEV": "其他小盘", "CBRS": "其他小盘", "CBRG": "其他小盘", "CRCG": "其他小盘",
    "HYPG": "其他小盘", "XE": "其他小盘", "LASE": "其他小盘", "MRAM": "其他小盘", "CPSH": "其他小盘",
    "TE": "其他小盘", "VCX": "其他小盘", "DXYZ": "其他小盘", "FGRU": "其他小盘", "RVI": "其他小盘",
    "KEEL": "其他小盘", "CIEN": "光电存", "FPS": "其他小盘", "SIVEF": "其他小盘", "HTZ": "其他小盘",
}
print("\n【事件研究: 喊单后N个交易日, 相对QQQ的超额】负数=跟进就被套")
horizons = [1, 3, 5, 10, 20]
ev = {h: [] for h in horizons}
chase = []
for e in CALLS["buys"]:
    tk, d0 = e["tk"], trade_date(e["ts"])
    B, i = bars(tk), None
    i, b = bar_on_or_after(B, d0)
    if b is None:
        continue
    qi, _ = bar_on_or_after(QQQ, b[0])
    if i >= 5:
        lo5 = min(x[3] for x in B[i - 5:i])
        chase.append((b[4] / lo5 - 1) * 100)   # 喊单价比前5日最低点高多少 = 追高幅度
    for h in horizons:
        if i + h < len(B) and qi + h < len(QQQ):
            ev[h].append((B[i + h][4] / b[4] - 1) * 100 - (QQQ[qi + h][4] / QQQ[qi][4] - 1) * 100)
for h in horizons:
    v = ev[h]
    print(f"  T+{h:<3d} n={len(v):3d}  平均超额 {st.mean(v):+6.2f}%  中位 {st.median(v):+6.2f}%  "
          f"跑赢比例 {sum(1 for x in v if x>0)/len(v)*100:4.1f}%")
print(f"  追高幅度: 喊单收盘价平均比前5日最低点高 {st.mean(chase):+.1f}% (中位 {st.median(chase):+.1f}%)")
print("  ⚠️ T+20 只有 7/17 之前的喊单够长度, 天然偏向6月样本 → 按月再切一次:")
for lo, hi, name in [("2026-06", "2026-07", "6月喊的"), ("2026-07", "2026-08", "7月喊的"), ("2026-08", "2026-09", "8月喊的")]:
    for h in (5, 10):
        v = []
        for e in CALLS["buys"]:
            d0 = trade_date(e["ts"])
            if not (lo <= str(d0)[:7] < hi):
                continue
            B = bars(e["tk"])
            i, b = bar_on_or_after(B, d0)
            if b is None:
                continue
            qi, _ = bar_on_or_after(QQQ, b[0])
            if i + h < len(B) and qi + h < len(QQQ):
                v.append((B[i + h][4] / b[4] - 1) * 100 - (QQQ[qi + h][4] / QQQ[qi][4] - 1) * 100)
        if v:
            print(f"     {name} T+{h:<3d} n={len(v):3d} 平均超额 {st.mean(v):+6.2f}%  跑赢 {sum(1 for x in v if x>0)/len(v)*100:4.1f}%")

print("\n【按板块】(口径A, 买入持有到今天)")
sec = {}
for r in rows:
    if r.get("st") != "ok":
        continue
    sec.setdefault(SECTOR.get(r["tk"], "未分类"), []).append(r)
for name, rs in sorted(sec.items(), key=lambda kv: -st.mean([x["A"] for x in kv[1]])):
    s = stats(rs, "A")
    print(f"  {name:10s} {s['n']:3d}笔 | 胜率 {s['win']:5.1f}% | 均笔 {s['mean']:+6.1f}% | "
          f"alpha {s['alpha']:+6.1f}% | {' '.join(sorted(set(x['tk'] for x in rs)))[:60]}")

json.dump(dict(generated=str(LAST), A=rows, C=cres,
               event={f"T+{h}": st.mean(ev[h]) for h in horizons},
               D=dict(qqq=(eq_qqq - 1) * 100, timing=(eq_timing - 1) * 100,
                      full=(eq_full - 1) * 100, mdd_timing=mdd_t * 100, mdd_full=mdd_f * 100)),
          open("output/xiaoyu_vip_backtest.json", "w"), ensure_ascii=False, indent=1)
print("\n→ output/xiaoyu_vip_backtest.json")
