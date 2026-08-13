#!/usr/bin/env python3
"""
vip_buypoints_report.py — 小鱼vip / 蛋挞vip / 张张 三频道买点汇总页 → output/vip_buypoints.html

数据: output/vip_buypoints_raw.json (小鱼/张张=Opus逐条精读提取; 蛋挞=27条人工整理内嵌)。
仅为"他们说了什么"的目录, 不构成对其战绩的验证 (与elite/casey的审计是两回事)。
"""
import html, json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "vip_buypoints.html"
LV_ORDER = ["核心主推", "看好", "波段机会", "观察", "已减仓/离场", "提到但无买点"]
LV_COLOR = {"核心主推": "#f85149", "看好": "#e3b341", "波段机会": "#58a6ff",
            "观察": "#8b949e", "已减仓/离场": "#3fb950", "提到但无买点": "#484f58"}

DANTA = dict(style="GEX/Gamma点位派: SPY/QQQ每日期权点位(30分钟K确认机制) + 中长线watchlist(支撑入/压力止盈/放量突破转长线) + 深夜短线小票。每单必带买点/止损/目标, 满口'不包赢/风险自担', 结构最工整。", stocks=[
 dict(ticker="AMC", sector="娱乐/影院", level="核心主推", buy_points="2.39已建一仓; 2.25-2.3加仓", stop="2.15", targets="2.7→3.0→3.2 (15-30%)", latest="08-12", status="当前有效", evidence="106年来最高单周末收入记录…利好还被增发消息盖着; 2H/4H止跌金叉"),
 dict(ticker="LRCX", sector="半导体设备", level="核心主推", buy_points="$307上下已买(中长线仓位)", stop=None, targets=None, latest="08-10", status="当前有效", evidence="今天买了点, 刚好进了中长线仓的位置"),
 dict(ticker="BTDR", sector="加密矿", level="波段机会", buy_points="财报dip三仓: 6.9-7.1 / 6.3-6.5 / 5.9-6.1(恐惧深dip)", stop=None, targets="10-12 / 12-14 / 16+", latest="08-11", status="当前有效", evidence="过去4次财报3跌1涨, 平均回撤~32%; 盈亏比非常好 upside 50-100%"),
 dict(ticker="INTC", sector="半导体", level="看好", buy_points="90/95入 (中长线watchlist)", stop=None, targets="压力100, 108", latest="08-12", status="当前有效", evidence="突破前做支撑到压力波段, 放量突破后长线开飞"),
 dict(ticker="SOFI", sector="金融科技", level="看好", buy_points="16-17.5入", stop=None, targets="压力19-20.5", latest="08-12", status="当前有效", evidence="中长线watchlist"),
 dict(ticker="HOOD", sector="券商", level="看好", buy_points="85-92入", stop=None, targets="压力108, 115", latest="08-12", status="当前有效", evidence="中长线watchlist"),
 dict(ticker="NFLX", sector="流媒体", level="看好", buy_points="69-72入", stop=None, targets="压力78/82/85", latest="08-12", status="当前有效", evidence="中长线watchlist"),
 dict(ticker="OPTX", sector="AI+太空+国防+光通讯", level="波段机会", buy_points="7.25-7.55买点玩", stop="7.1", targets="8, 8.3", latest="08-10", status="当前有效", evidence="筑底+跳高回调, 有支撑"),
 dict(ticker="ENPH", sector="光伏", level="波段机会", buy_points="38上下、36上下分批", stop="34", targets="44 / 50 / 64", latest="08-06", status="当前有效", evidence="FSLR盘后265带动的关联便宜票"),
 dict(ticker="CELH", sector="消费饮料", level="看好", buy_points="27.85已进(期权), 正股也可以", stop="参考前日分享", targets=None, latest="08-06", status="当前有效", evidence="vip频道分享自行搜索"),
 dict(ticker="OPEN", sector="地产科技", level="波段机会", buy_points="$4已进(财报小仓)", stop="3.8", targets=None, latest="08-04", status="可能过时", evidence="财报我会小参与的"),
 dict(ticker="FCX", sector="物料/铜", level="看好", buy_points="XLB板块回踩做波段", stop=None, targets=None, latest="08-11", status="当前有效", evidence="物料里面铜最近很好, 我比较喜欢的"),
 dict(ticker="CF", sector="物料/氮肥", level="看好", buy_points="回踩波段", stop=None, targets=None, latest="08-11", status="当前有效", evidence="XLB篮子, 比较喜欢"),
 dict(ticker="STLD", sector="物料/钢铁", level="看好", buy_points="回踩波段", stop=None, targets=None, latest="08-11", status="当前有效", evidence="基本面强"),
 dict(ticker="MOS", sector="物料/磷钾肥", level="观察", buy_points="回踩波段", stop=None, targets=None, latest="08-11", status="当前有效", evidence="便宜但基本面一般"),
 dict(ticker="NUE", sector="物料/钢铁", level="观察", buy_points="XLB篮子", stop=None, targets=None, latest="08-11", status="当前有效", evidence="周线走的都不错"),
 dict(ticker="XLU", sector="公用事业ETF", level="观察", buy_points="41-43上下(防御仓)", stop=None, targets=None, latest="08-10", status="当前有效", evidence="大盘新高时加少量防御仓; utilities被卖到超卖"),
 dict(ticker="RZLV", sector="AI高弹性小票", level="看好", buy_points="回踩买(未给价)", stop=None, targets=None, latest="08-10", status="当前有效", evidence="大弹性里最喜欢"),
 dict(ticker="AUR", sector="AI自驾/trucking", level="看好", buy_points="回踩买(未给价)", stop=None, targets=None, latest="08-10", status="当前有效", evidence="商业化有了, 第二喜欢"),
 dict(ticker="PDYN", sector="AI小票", level="观察", buy_points="—", stop=None, targets=None, latest="08-10", status="当前有效", evidence="大弹性名单"),
 dict(ticker="BBAI", sector="AI小票", level="观察", buy_points="—", stop=None, targets=None, latest="08-10", status="当前有效", evidence="指引不好增长有点慢"),
 dict(ticker="GRRR", sector="视频/安防AI", level="观察", buy_points="—", stop=None, targets=None, latest="08-10", status="当前有效", evidence="大弹性名单"),
 dict(ticker="SOUN", sector="语音AI", level="观察", buy_points="—", stop=None, targets=None, latest="08-10", status="当前有效", evidence="过财报了"),
 dict(ticker="MITK", sector="身份验证AI", level="观察", buy_points="—", stop=None, targets=None, latest="08-10", status="当前有效", evidence="现金流这几个里好点"),
 dict(ticker="PLTR/NOW/CRWD/PANW/SNOW/DDOG/PATH/NET", sector="AI软件观察池", level="观察", buy_points="回踩才健康, 不建议追涨", stop=None, targets=None, latest="08-10", status="当前有效", evidence="AI给盈利增长高+最大化改善商业模式的财报赢家"),
 dict(ticker="LUNR / RDW", sector="太空", level="观察", buy_points="回调深度最香(未给价)", stop=None, targets=None, latest="08-10", status="当前有效", evidence="按回调深度来说lunr和rdw还是最香的"),
 dict(ticker="ASTS / RKLB", sector="太空", level="观察", buy_points="—(RKLB财报风险自担)", stop=None, targets=None, latest="08-10", status="当前有效", evidence="太空名单"),
 dict(ticker="FSLR", sector="光伏", level="已减仓/离场", buy_points="6/30提219-225好价→盘后265", stop=None, targets="240/250已达", latest="08-06", status="已了结", evidence="战绩回顾, 关联替代=ENPH"),
])


def sec(ch_key, title, data):
    rows_by_lv = {}
    for s in data["stocks"]:
        rows_by_lv.setdefault(s["level"], []).append(s)
    n_act = sum(1 for s in data["stocks"] if s["status"] == "当前有效")
    out = [f'<h2>{title} <small>{len(data["stocks"])}只 · 当前有效{n_act}</small></h2>',
           f'<div class="note">{html.escape(data["style"])}</div>']
    for lv in LV_ORDER:
        ss = rows_by_lv.get(lv)
        if not ss:
            continue
        out.append(f'<h3><span class="dot" style="background:{LV_COLOR[lv]}"></span>{lv} ({len(ss)})</h3>')
        out.append('<table><thead><tr><th>票</th><th>板块</th><th>买点</th><th>止损</th>'
                   '<th>目标</th><th>最新</th><th>状态</th><th>依据(原话)</th></tr></thead><tbody>')
        for s in sorted(ss, key=lambda x: x["latest"], reverse=True):
            stale = ' class="stale"' if s["status"] != "当前有效" else ""
            out.append(f'<tr{stale}><td class="tk">{html.escape(s["ticker"])}</td>'
                       f'<td>{html.escape(s["sector"][:14])}</td>'
                       f'<td class="bp">{html.escape((s["buy_points"] or "—")[:150])}</td>'
                       f'<td>{html.escape(s.get("stop") or "—")}</td>'
                       f'<td>{html.escape((s.get("targets") or "—")[:40])}</td>'
                       f'<td>{s["latest"]}</td><td>{s["status"]}</td>'
                       f'<td class="ev">{html.escape(s["evidence"][:130])}</td></tr>')
        out.append('</tbody></table>')
    return "\n".join(out)


def main():
    raw = json.loads((ROOT / "output" / "vip_buypoints_raw.json").read_text())
    raw.setdefault("danta", DANTA)   # 已有(含图片补录)则不覆盖
    now = datetime.now(ZoneInfo("Asia/Singapore"))
    parts = [
        sec("xiaoyu", "小鱼vip (鱼哥 · 板块轮动/仓位管理型)", raw["xiaoyu"]),
        sec("danta", "蛋挞vip (UnstoppableEggtart · GEX点位派)", raw["danta"]),
        sec("zhangzhang", "张张 (zzlucky · 指数杠杆ETF为主)", raw["zhangzhang"]),
    ]
    for key, title in [("kova", "Kova (#信号 · 实时喊单带止损, 买强势+死止损)"),
                       ("cm", "CM (实盘持仓表 · 用户手动供图)"),
                       ("tangzhuren", "唐主任 (指数点位+快进快出报单)"),
                       ("biancheng", "边城 (期权卖方策略流 · sell put/spread)"),
                       ("suoya", "索亚财经 (期权现金流稳健派 · DCA+sell put接货)")]:
        if key in raw:
            parts.append(sec(key, title, raw[key]))
    body = "\n".join(parts)
    OUT.write_text(f"""<meta charset="utf-8"><title>三频道买点汇总 · {now:%m-%d}</title>
<style>
body{{background:#0d1117;color:#e6edf3;font:14px/1.6 -apple-system,"PingFang SC",sans-serif;
margin:0;padding:24px 16px;max-width:1080px;margin-inline:auto}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#8b949e;font-size:13px;margin-bottom:8px}}
h2{{font-size:16px;border-bottom:1px solid #21262d;padding-bottom:6px;margin:30px 0 8px}}
h2 small{{color:#8b949e;font-weight:400;font-size:12px;margin-left:8px}}
h3{{font-size:13.5px;margin:16px 0 6px;color:#e6edf3}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}}
.note{{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:9px 14px;
color:#8b949e;font-size:12.5px}}
.warn{{background:#1c1500;border:1px solid #6a5300;border-radius:10px;padding:9px 14px;
color:#e3b341;font-size:12.5px;margin:10px 0}}
table{{width:100%;border-collapse:collapse;font-size:12.5px}}
th{{text-align:left;color:#8b949e;font-weight:500;padding:3px 7px;white-space:nowrap}}
td{{padding:4px 7px;border-top:1px solid #21262d;vertical-align:top}}
.tk{{font-weight:600;white-space:nowrap}} .bp{{min-width:180px}}
.ev{{color:#8b949e;font-size:11.5px}}
tr.stale{{opacity:.45}}
</style>
<h1>小鱼vip · 蛋挞vip · 张张 — 每票买点/分类/推荐级别</h1>
<div class="sub">生成 {now:%Y-%m-%d %H:%M} SGT · 数据: 三频道全量消息(小鱼343条自6/8 · 蛋挞27条自8/4 · 张张2130条自8/7), 小鱼/张张由Opus逐条精读提取, 蛋挞人工整理</div>
<div class="warn">⚠️ 这是"他们说了什么"的目录, 推荐级别按其话术强度归档 — 不是对其战绩的验证。三位博主的历史胜率均未经独立核验(对照: elite已被审出只报赢)。变灰行=可能过时/已了结。</div>
{body}""", encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
