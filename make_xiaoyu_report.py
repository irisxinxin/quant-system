#!/usr/bin/env python3
"""
make_xiaoyu_report.py — 把小鱼(鱼哥)喊单回测整理成表格报告 (Markdown + HTML)。
输入: output/xiaoyu_vip_calls.json + output/xiaoyu_vip_backtest.json
输出: output/xiaoyu_report.md / output/xiaoyu_report.html
"""
import json, statistics as st
from collections import defaultdict

BT = json.load(open("output/xiaoyu_vip_backtest.json"))
CALLS = json.load(open("output/xiaoyu_vip_calls.json"))
LAST = BT["generated"]

SECTOR = {
    "PL": "太空", "ASTS": "太空", "LUNR": "太空", "SATL": "太空", "BKSY": "太空", "FLY": "太空",
    "SPCL": "太空", "RKLB": "太空", "RDW": "太空", "NASA": "太空", "MNTS": "太空", "SPCX": "太空",
    "AXTI": "光电存/半导体", "POET": "光电存/半导体", "LITE": "光电存/半导体", "COHR": "光电存/半导体",
    "MRVL": "光电存/半导体", "GLW": "光电存/半导体", "ALMU": "光电存/半导体", "MU": "光电存/半导体",
    "SNDK": "光电存/半导体", "RMBS": "光电存/半导体", "INTC": "光电存/半导体", "ARM": "光电存/半导体",
    "ON": "光电存/半导体", "WOLF": "光电存/半导体", "SMCI": "光电存/半导体", "VSH": "光电存/半导体",
    "TTMI": "光电存/半导体", "AMKR": "光电存/半导体", "MCHP": "光电存/半导体", "MXL": "光电存/半导体",
    "CIEN": "光电存/半导体",
    "QUBT": "量子", "IBM": "量子", "QBTS": "量子", "RGTI": "量子", "QNT": "量子",
    "MSFT": "M7大科技", "AMZN": "M7大科技", "META": "M7大科技", "GOOG": "M7大科技",
    "NFLX": "M7大科技", "ORCL": "M7大科技", "AAPL": "M7大科技", "TSLA": "M7大科技",
    "ADBE": "软件", "NOW": "软件", "UBER": "软件", "MDB": "软件", "NET": "软件", "DDOG": "软件",
    "SNPS": "软件", "PLTR": "软件", "APP": "软件", "PYPL": "软件", "FSLY": "软件",
    "COIN": "加密", "SOFI": "加密", "PURR": "加密", "BTBT": "加密", "CRCL": "加密", "FIGR": "加密",
    "CIFR": "矿/数据中心", "WULF": "矿/数据中心", "APLD": "矿/数据中心", "IREN": "矿/数据中心",
    "CRWV": "矿/数据中心", "MP": "矿/数据中心", "USAR": "矿/数据中心", "CRML": "矿/数据中心",
    "UUUU": "矿/数据中心",
    "ANNA": "油/大宗", "XOM": "油/大宗", "USO": "油/大宗", "KOS": "油/大宗", "AA": "油/大宗", "UAMY": "油/大宗",
    "HIMS": "健康消费", "BRBR": "健康消费", "NKE": "健康消费",
    "ONDS": "无人机", "RCAT": "无人机", "BOT": "无人机", "AMPX": "无人机",
    "SMR": "核电储能", "OKLO": "核电储能", "EOSE": "核电储能", "FCEL": "核电储能", "VRT": "核电储能",
    "AMSC": "核电储能", "BE": "核电储能", "FLNC": "核电储能",
    "XPEV": "其他小盘", "CBRS": "其他小盘", "CBRG": "其他小盘", "CRCG": "其他小盘",
    "HYPG": "其他小盘", "XE": "其他小盘", "LASE": "其他小盘", "MRAM": "其他小盘", "CPSH": "其他小盘",
    "TE": "其他小盘", "VCX": "其他小盘", "DXYZ": "其他小盘", "FGRU": "其他小盘", "RVI": "其他小盘",
    "KEEL": "其他小盘", "FPS": "其他小盘", "SIVEF": "其他小盘", "HTZ": "其他小盘",
}
ok = [r for r in BT["A"] if r.get("st") == "ok"]
for r in ok:
    r["sec"] = SECTOR.get(r["tk"], "未分类")
    r["alpha"] = round(r["A"] - r["qqq"], 1)

lvl_by_tk = defaultdict(list)
for e in CALLS["levels"]:
    lvl_by_tk[e["tk"]].append(e["px"])
fill_by_tk = {}
for r in BT["C"]:
    fill_by_tk.setdefault(r["tk"], []).append(r)

L = []
A = L.append
A(f"# 小鱼(鱼哥) 喊单 & 买点回测总表\n")
A(f"**区间** 2026-06-08 ~ {LAST}（他 VIP 频道 350 条消息全量人工建台账） "
  f"**基准** 同期 QQQ **+2.1%**　**最后交易日** {LAST}\n")

# ── 总览 ──
A("\n## 一、总账\n")
A("| 跟单方式 | 笔数 | 胜率 | 均笔收益 | 中位 | 同期QQQ | **超额(alpha)** | 跑赢大盘比例 |")
A("|---|---:|---:|---:|---:|---:|---:|---:|")


def row(label, rs, key="A"):
    v = [r[key] for r in rs]
    q = [r["qqq"] for r in rs]
    w = sum(1 for x in v if x > 0) / len(v) * 100
    beat = sum(1 for r in rs if r[key] > r["qqq"]) / len(rs) * 100
    A(f"| {label} | {len(v)} | {w:.1f}% | **{st.mean(v):+.1f}%** | {st.median(v):+.1f}% | "
      f"{st.mean(q):+.1f}% | **{st.mean(v)-st.mean(q):+.1f}%** | {beat:.1f}% |")


row("① 无脑跟喊单（喊单日收盘买，持有至今）", ok, "A")
row("② 用他的321止盈（+30%走⅓ / +50%走一半 / +100%清）", ok, "B")
row("③ 321止盈 + 机械-20%止损（对照，他本人不设损）", ok, "B2")
fc = [r for r in BT["C"] if r["st"] == "成交"]
v = [r["pct"] for r in fc]
q = [r["qqq"] for r in fc]
A(f"| ④ **只挂他给的具体点位**（成交{len(fc)}/{len(BT['C'])}） | {len(fc)} | "
  f"{sum(1 for x in v if x>0)/len(v)*100:.0f}% | **{st.mean(v):+.1f}%** | {st.median(v):+.1f}% | "
  f"{st.mean(q):+.1f}% | **{st.mean(v)-st.mean(q):+.1f}%** | — |")

A("\n| 跟他的仓位曲线（择时） | 收益 | 最大回撤 |")
A("|---|---:|---:|")
A(f"| 一直满仓 QQQ（躺平基准） | **{BT['D']['qqq']:+.2f}%** | — |")
A(f"| 跟他的现金比例 × QQQ（纯择时贡献） | {BT['D']['timing']:+.2f}% | {BT['D']['mdd_timing']:.1f}% |")
A(f"| 跟他仓位 × 他的等权组合（完整跟单） | **{BT['D']['full']:+.2f}%** | {BT['D']['mdd_full']:.1f}% |")

# ── 点位单 ──
A("\n## 二、他给了具体买点的票（31 个点位）\n")
A("### ✅ 已成交的 12 个 —— 这是他最值钱的东西\n")
A("| 票 | 给点位日期 | 他的挂单价 | 实际成交 | 成交日 | 至今收益 | 同期QQQ | 超额 |")
A("|---|---|---:|---:|---|---:|---:|---:|")
for r in sorted(fc, key=lambda x: -x["pct"]):
    A(f"| **{r['tk']}** | {r['ts']} | {r['px']} | {r['entry']} | {r['fill']} | "
      f"**{r['pct']:+.1f}%** | {r['qqq']:+.1f}% | {r['pct']-r['qqq']:+.1f}% |")
A(f"\n> 12 笔成交：胜率 **{sum(1 for x in v if x>0)/len(v)*100:.0f}%**，均笔 **{st.mean(v):+.1f}%**，"
  f"超额 **{st.mean(v)-st.mean(q):+.1f}%**。但成交集中在 7/17–7/31（本段最低点区），"
  + "有「市场刚好给了机会」的成分。\n")

nf = [r for r in BT["C"] if r["st"] != "成交"]
A("### ❌ 从未触及的 19 个 —— 踏空率 61%，这是跟他的最大成本\n")
A("| 票 | 给点位日期 | 他的挂单价 | 现价 | 现价比挂单高 |")
A("|---|---|---:|---:|---:|")
for r in sorted(nf, key=lambda x: -x["gap"]):
    A(f"| {r['tk']} | {r['ts']} | {r['px']} | {r['cur']} | **+{r['gap']:.0f}%** |")

# ── 全量喊单 ──
A("\n## 三、108 笔喊单全表（无点位的口径①，按收益排序）\n")
A("| # | 票 | 板块 | 喊单日 | 入场价 | 现价 | 收益 | 同期QQQ | 超额 | 用321止盈后 | 他的原话 |")
A("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|")
for i, r in enumerate(sorted(ok, key=lambda x: -x["A"]), 1):
    cur = round(r["entry"] * (1 + r["A"] / 100), 2)
    lv = f"（他给过点位 {'/'.join(str(x) for x in lvl_by_tk[r['tk']])}）" if r["tk"] in lvl_by_tk else ""
    A(f"| {i} | **{r['tk']}** | {r['sec']} | {r['d0']} | {r['entry']} | {cur} | "
      f"**{r['A']:+.1f}%** | {r['qqq']:+.1f}% | {r['alpha']:+.1f}% | {r['B']:+.1f}% | {r['note']}{lv} |")

# ── 板块 ──
A("\n## 四、按板块（口径①）\n")
A("| 板块 | 笔数 | 胜率 | 均笔 | 超额 | 票 |")
A("|---|---:|---:|---:|---:|---|")
sec = defaultdict(list)
for r in ok:
    sec[r["sec"]].append(r)
for name, rs in sorted(sec.items(), key=lambda kv: -st.mean([x["A"] for x in kv[1]])):
    vv = [x["A"] for x in rs]
    qq = [x["qqq"] for x in rs]
    A(f"| **{name}** | {len(rs)} | {sum(1 for x in vv if x>0)/len(vv)*100:.0f}% | "
      f"**{st.mean(vv):+.1f}%** | {st.mean(vv)-st.mean(qq):+.1f}% | {' '.join(sorted(set(x['tk'] for x in rs)))} |")

# ── 分月 ──
A("\n## 五、按他自己划的市场阶段\n")
A("| 阶段 | 笔数 | 胜率 | 均笔 | 同期QQQ | 超额 |")
A("|---|---:|---:|---:|---:|---:|")
for lo, hi, name in [("2026-06", "2026-07", "6月 防御期（他压 40–50% 现金）"),
                     ("2026-07", "2026-08", "7月 等抄底条件（QQQ650/恐慌20/油110）"),
                     ("2026-08", "2026-09", "8月 三批加仓 → 8/13 喊满仓")]:
    rs = [r for r in ok if lo <= r["d0"][:7] < hi]
    vv = [x["A"] for x in rs]
    qq = [x["qqq"] for x in rs]
    A(f"| {name} | {len(rs)} | {sum(1 for x in vv if x>0)/len(vv)*100:.0f}% | "
      f"**{st.mean(vv):+.1f}%** | {st.mean(qq):+.1f}% | {st.mean(vv)-st.mean(qq):+.1f}% |")

A("\n## 六、事件研究：跟进之后被套多久\n")
A("| 跟进后 | 平均超额(vs QQQ) | 跑赢比例 |")
A("|---|---:|---:|")
for k, vv in BT.get("event", {}).items():
    A(f"| {k} 个交易日 | **{vv:+.2f}%** | — |")
A("\n> 他喊单当天的收盘价，平均比**前 5 日最低点高 13.4%**（中位 +9.4%）——习惯在票已经弹起来一截之后才喊。\n")

A("\n## 七、口径与诚实性说明\n")
A("- 台账由**人工逐条读 350 条原始消息**建立（`output/xiaoyu_vip_calls.json`，每条带原话），不是正则扫的。\n")
A("- 口径①按**喊单日收盘价**买入。他强调「到点位才买」，所以这个口径对他偏严苛——这正是单列口径④的原因。\n")
A("- 限价单按**日线 Low ≤ 挂单价**才算成交，成交价取 min(挂单价, 当日开盘)，不存在「最高价穿过就算成交」的自欺。\n")
A("- 他多次说「这些视频都说了」，**视频里的口径没有进台账**，属于已知缺口。\n")
A("- 板块分组最小的只有 3–5 笔；择时组合为日频等权再平衡近似。\n")
A("- 全部价格取自 Yahoo 日线；VCX/MNTS/ON/CRCG 等异常跌幅已单独查过拆股事件（CRCG 的 1:10 拆股在 5/5，早于喊单日，不影响）。\n")

md = "\n".join(L)
open("output/xiaoyu_report.md", "w").write(md)
print(md)

# ── 同时输出一份自包含 HTML（表格好读, 深浅色都行）──
def md2html(md):
    out, in_tbl = [], False
    for line in md.split("\n"):
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue
            if not in_tbl:
                out.append("<div class='w'><table>"); in_tbl = "head"
            tag = "th" if in_tbl == "head" else "td"
            row = "".join(f"<{tag}>{fmt(c)}</{tag}>" for c in cells)
            out.append(f"<tr>{row}</tr>")
            if in_tbl == "head":
                in_tbl = "body"
            continue
        if in_tbl:
            out.append("</table></div>"); in_tbl = False
        if s.startswith("### "):   out.append(f"<h3>{fmt(s[4:])}</h3>")
        elif s.startswith("## "):  out.append(f"<h2>{fmt(s[3:])}</h2>")
        elif s.startswith("# "):   out.append(f"<h1>{fmt(s[2:])}</h1>")
        elif s.startswith("> "):   out.append(f"<blockquote>{fmt(s[2:])}</blockquote>")
        elif s.startswith("- "):   out.append(f"<p class='li'>• {fmt(s[2:])}</p>")
        elif s:                    out.append(f"<p>{fmt(s)}</p>")
    if in_tbl:
        out.append("</table></div>")
    return "\n".join(out)


def fmt(t):
    import re as _r
    t = t.replace("&", "&amp;").replace("<", "&lt;")
    t = _r.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = _r.sub(r"`(.+?)`", r"<code>\1</code>", t)
    t = _r.sub(r"(?<![\w>])([+-]\d+(?:\.\d+)?%)", lambda m: f"<span class='{'up' if m.group(1)[0]=='+' else 'dn'}'>{m.group(1)}</span>", t)
    return t


HTML = """<meta charset="utf-8"><title>小鱼喊单回测总表</title><style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--head:#f6f7f9;--zeb:#fafbfc;--up:#0a8f52;--dn:#d1344e;--acc:#2563eb}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e6e8eb;--mut:#9aa3af;--line:#262b33;--head:#171a20;--zeb:#13161b;--up:#35d48b;--dn:#ff6b81;--acc:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;padding:28px 20px 60px;background:var(--bg);color:var(--fg);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"PingFang SC","Helvetica Neue",Arial,sans-serif;max-width:1180px;margin-inline:auto}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:38px 0 12px;padding-top:16px;border-top:2px solid var(--line)}
h3{font-size:16px;margin:24px 0 10px;color:var(--mut);font-weight:600}
p{margin:8px 0}.li{margin:4px 0;color:var(--mut);font-size:14px}
blockquote{margin:12px 0;padding:10px 14px;border-left:3px solid var(--acc);background:var(--head);border-radius:0 6px 6px 0;font-size:14px;color:var(--mut)}
blockquote b{color:var(--fg)}
.w{overflow-x:auto;margin:12px 0;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th{background:var(--head);text-align:left;font-weight:600;font-size:12.5px;color:var(--mut);
padding:10px 12px;white-space:nowrap;position:sticky;top:0}
td{padding:8px 12px;border-top:1px solid var(--line);vertical-align:top}
tr:nth-child(even) td{background:var(--zeb)}
td:nth-child(n+4){white-space:nowrap}
table tr td:last-child{white-space:normal;color:var(--mut);font-size:12.5px;min-width:220px}
.up{color:var(--up);font-weight:600}.dn{color:var(--dn);font-weight:600}
code{background:var(--head);padding:1px 5px;border-radius:4px;font-size:12.5px}
b{font-weight:650}
</style>
""" + md2html(md)
open("output/xiaoyu_report.html", "w").write(HTML)
print("\n→ output/xiaoyu_report.md  /  output/xiaoyu_report.html")
