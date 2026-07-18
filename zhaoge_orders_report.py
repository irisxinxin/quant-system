#!/usr/bin/env python3
"""
zhaoge_orders_report.py — 赵哥(#股票赵哥-日内)全量单子清单 → output/zhaoge_orders.html

数据源 output/zhaoge_history.json (每日05:10归档), 解析语法完全复用 zhaoge_backtest
(同一套正则, 保证与回测口径一致)。仅展示他喊的单, 不算P&L(回测另有 zhaoge_backtest.py)。
诚实边界: 多票指令/中文名(双倍三倍ETF)无法自动归属 → 原文列在页尾"未解析"区, 不静默丢弃。
"""
import html, json, re, sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from zoneinfo import ZoneInfo

import zhaoge_backtest as Z

ROOT = Path(__file__).parent
OUT = ROOT / "output" / "zhaoge_orders.html"
ET = ZoneInfo("America/New_York")
UTC = timezone.utc
WD = "一二三四五六日"
FRAC_LABEL = {1.0: "全部", 0.5: "1/2", 1 / 3: "1/3", 2 / 3: "2/3"}


def load_all():
    Z.START, Z.END = date(2026, 1, 1), date(2026, 12, 31)
    stream = Z.parse_stream()
    msgs = json.load(open(ROOT / "output" / "zhaoge_history.json"))
    raw_by_ts = {}
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        raw_by_ts[ts] = " ".join(Z.strip_prefix(m["text"]).split())
    # 未解析但像单子的消息 — 复刻 parse_stream 的去重口径(站长重发不重复列), 加分类标签
    handled = {ts for ts, *_ in stream}
    skipped = []
    seen = set()
    for m in msgs:
        ts = datetime.fromisoformat(m["ts"])
        ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts
        raw = Z.strip_prefix(m["text"])
        if len(raw) > 200 or "http" in raw:
            continue
        line = " ".join(raw.split())
        key = (ts.date().isoformat(), line)
        if key in seen:
            continue
        seen.add(key)
        if ts in handled:
            continue
        m_hit = Z.BUY_RE.search(line) or Z.SELL_PX_RE.search(line) or Z.SELL_TAIL_RE.search(line)
        if not m_hit and not re.search(r"出掉|出一半|平出", line):
            continue
        low = line.lower()
        tks = list(dict.fromkeys(w.lower() for w in Z.TK_RE.findall(line) if w.lower() not in Z.STOP))
        if re.search(r"\b(spy|qqq|spx|ndx)\b", low):
            cat = "指数/期权"
        elif not tks and m_hit and m_hit.start() < 12:
            cat = "无票名(上下文单)"
        elif len(tks) > 1:
            cat = "多票"
        elif re.search(r"双倍|三倍|谷歌|苹果|微软|英伟达|特斯拉", line):
            cat = "中文名"
        elif not tks:
            cat = "疑似说明文"
        else:
            cat = "其他"
        skipped.append((ts, cat, line))
    return stream, raw_by_ts, skipped


def render(stream, raw_by_ts, skipped):
    n_buy = sum(1 for _, s, *_ in stream if s == "buy")
    n_sell = len(stream) - n_buy
    tks = Counter(tk for _, _, tk, _, _ in stream)
    lo, hi = stream[0][0].astimezone(ET), stream[-1][0].astimezone(ET)
    mon = Counter(ts.astimezone(ET).strftime("%Y-%m") for ts, *_ in stream)

    rows_by_month = {}
    for ts, side, tk, px, frac in stream:
        t = ts.astimezone(ET)
        rows_by_month.setdefault(t.strftime("%Y-%m"), []).append((t, side, tk, px, frac, raw_by_ts.get(ts, "")))

    top = "".join(f'<button class="tkbtn" onclick="filt(this,\'{t.upper()}\')">{t.upper()}<span>{n}</span></button>'
                  for t, n in tks.most_common(12))
    body = []
    for m in sorted(rows_by_month, reverse=True):
        rows = rows_by_month[m]
        nb = sum(1 for r in rows if r[1] == "buy")
        body.append(f'<h2>{m} <small>{len(rows)}单 · 买{nb} 卖{len(rows)-nb}</small></h2>')
        body.append('<table><thead><tr><th>美东时间</th><th>动作</th><th>票</th>'
                    '<th class="r">价格</th><th class="r">份额</th><th>原文</th></tr></thead><tbody>')
        for t, side, tk, px, frac, raw in reversed(rows):
            act = ('<span class="b">买入</span>' if side == "buy"
                   else f'<span class="s">卖出</span>')
            fr = "—" if side == "buy" else FRAC_LABEL.get(frac, f"{frac:.0%}")
            pxs = f"{px:g}" if px is not None else "—"
            body.append(f'<tr data-tk="{tk.upper()}"><td>{t:%m-%d} 周{WD[t.weekday()]} {t:%H:%M}</td>'
                        f'<td>{act}</td><td class="tk">{tk.upper()}</td><td class="r">{pxs}</td>'
                        f'<td class="r">{fr}</td><td class="raw">{html.escape(raw[:90])}</td></tr>')
        body.append('</tbody></table>')

    cat_n = Counter(c for _, c, _ in skipped)
    cat_sum = " · ".join(f"{c}×{n}" for c, n in cat_n.most_common())
    skip_rows = "".join(
        f'<tr><td>{ts.astimezone(ET):%m-%d %H:%M}</td><td>{cat}</td>'
        f'<td class="raw">{html.escape(line[:160])}</td></tr>'
        for ts, cat, line in reversed(skipped))

    return f"""<meta charset="utf-8"><title>赵哥全量单子 · {len(stream)}条</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--line:#21262d;--fg:#e6edf3;--dim:#8b949e;
--red:#f85149;--green:#3fb950;--acc:#58a6ff}}
body{{background:var(--bg);color:var(--fg);font:14px/1.6 -apple-system,"PingFang SC",sans-serif;
margin:0;padding:24px 16px;max-width:980px;margin-inline:auto}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--dim);font-size:13px;margin-bottom:16px}}
.tiles{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 16px}}
.tile b{{display:block;font-size:20px}} .tile span{{color:var(--dim);font-size:12px}}
.tkbtn{{background:var(--card);border:1px solid var(--line);border-radius:999px;color:var(--fg);
padding:4px 12px;margin:2px;cursor:pointer;font-size:13px}}
.tkbtn span{{color:var(--dim);margin-left:5px;font-size:11px}}
.tkbtn.on{{border-color:var(--acc);color:var(--acc)}}
h2{{font-size:15px;border-bottom:1px solid var(--line);padding-bottom:6px;margin:26px 0 8px}}
h2 small{{color:var(--dim);font-weight:400;font-size:12px;margin-left:8px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;color:var(--dim);font-weight:500;padding:4px 8px;white-space:nowrap}}
td{{padding:4px 8px;border-top:1px solid var(--line);white-space:nowrap}}
td.raw{{color:var(--dim);white-space:normal;font-size:12px}}
.r{{text-align:right}} .tk{{font-weight:600}}
.b{{color:var(--red);font-weight:600}} .s{{color:var(--green);font-weight:600}}
.note{{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;color:var(--dim);font-size:12.5px;margin:18px 0}}
</style>
<h1>赵哥 · #股票赵哥-日内 全量单子</h1>
<div class="sub">{lo:%Y-%m-%d} ~ {hi:%Y-%m-%d} (美东时间) · 数据源: 站长转发全量存档 {len(raw_by_ts)} 条消息 · 生成 {datetime.now(ET):%Y-%m-%d %H:%M} ET</div>
<div class="tiles">
<div class="tile"><b>{len(stream)}</b><span>可解析单子</span></div>
<div class="tile"><b class="b">{n_buy}</b><span>买入</span></div>
<div class="tile"><b class="s">{n_sell}</b><span>卖出/减仓</span></div>
<div class="tile"><b>{len(tks)}</b><span>只票</span></div>
<div class="tile"><b>{len(skipped)}</b><span>未解析(见页尾)</span></div>
</div>
<div>{top}<button class="tkbtn" onclick="filt(this,'')">全部</button></div>
{''.join(body)}
<h2>未解析单据 <small>{cat_sum}</small></h2>
<div class="note">这些消息像单子但无法自动归属到唯一正股: "无票名(上下文单)"=只报价格不报票,
靠前文才知道卖哪只(多为他的期权腿); "指数/期权"=SPY/QQQ彩票单(非正股, 股票回测不计);
"多票"=一条消息带多只票; "中文名"=谷歌A/双倍ETF等。回测与上表均未计入 — 不静默丢弃, 供人工核对。</div>
<table><thead><tr><th>美东时间</th><th>类别</th><th>原文</th></tr></thead><tbody>{skip_rows}</tbody></table>
<script>
function filt(btn,tk){{
 document.querySelectorAll('.tkbtn').forEach(b=>b.classList.remove('on'));
 btn.classList.add('on');
 document.querySelectorAll('tr[data-tk]').forEach(r=>
   r.style.display=(!tk||r.dataset.tk===tk)?'':'none');
}}
</script>"""


def main():
    stream, raw_by_ts, skipped = load_all()
    OUT.write_text(render(stream, raw_by_ts, skipped), encoding="utf-8")
    print(f"→ {OUT} ({len(stream)}单, 未解析{len(skipped)}条)")


if __name__ == "__main__":
    main()
