#!/usr/bin/env python3
"""
signal_history.py — 信号源历史逐笔成绩 (按我们的跟单规则, fill-realistic 真实5分K模拟)。
输出 output/signal_history.json, 供 trade_report.py 渲染【历史成绩】分区。
每日归档 job 自动刷新; K线优先长桥API, 清库后回退 data/enrich_bars/ 归档。

口径:
  enrich: 2张 / 止盈2x卖半 / 镜像出场(二次partial清runner) / 止损-30% / 到期强平 (=实盘bot规则)
  andy:   波段+明确止损子集 / 他的止损 + BE + 减仓后自动保本 / $1万预算 (=观察评估口径)
无数据/未成交的笔如实标注, 不猜价。
"""
import os, sys, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("V2", "1")        # enrich: 二次partial清runner
os.environ.setdefault("V3_STOP", "0.7")  # enrich: 止损-30%
os.environ.setdefault("N", "2")          # enrich: 2张
os.environ.setdefault("AUTO_BE", "1")    # andy: 减仓后自动保本
import warnings; warnings.filterwarnings("ignore")
from zoneinfo import ZoneInfo

SGT = ZoneInfo("Asia/Singapore")
OUT = Path(__file__).parent / "output"


def _pack(row, r, qty):
    if r is None:
        row.update(status="no_data")
        return row
    if r["status"] != "traded":
        row.update(status=r["status"], note=r.get("note", ""))
        return row
    open_mark = any("持仓中" in w or "mark" in w for _, _, w in r["sells"])
    row.update(status="open" if open_mark else "closed",
               entry=r["entry"], qty=qty,
               sells=[dict(px=px, qty=q_, why=w) for px, q_, w in r["sells"]],
               pnl=round(r["pnl"]), pct=round(r["pnl"] / r["cost"] * 100, 1),
               cost=round(r["cost"]))
    return row


def enrich_rows(q):
    import backtest_enrich as E
    buys, exits = E.load_events()
    rows = []
    for b in buys:
        s = b["sig"]
        row = dict(src="enrich", ts=b["ts"].isoformat(), ticker=s.ticker,
                   label=f"{s.ticker} {s.expiry:%m/%d} ${s.strike:g}{s.right}",
                   signal_px=s.limit_price, expiry=str(s.expiry))
        rows.append(_pack(row, E.simulate(q, b, exits), E.CONTRACTS))
    return rows


def andy_rows(q):
    import backtest_andy as A
    buys, exits = A.load_events()
    buys = [b for b in buys
            if not b["e"]["lotto"] and b["e"]["expiry"] > b["ts"].date() and b["e"]["stop"]]
    rows = []
    for b in buys:
        e = b["e"]
        row = dict(src="andy", ts=b["ts"].isoformat(), ticker=e["ticker"],
                   label=f"{e['ticker']} {e['expiry']:%m/%d} ${e['strike']:g}{e['right']}",
                   signal_px=e["prem"], his_stop=e["stop"], expiry=str(e["expiry"]))
        r = A.simulate(q, b, exits, use_his_stop=True)
        rows.append(_pack(row, r, (r or {}).get("qty")))
    return rows


def main():
    from longport.openapi import Config, QuoteContext
    q = QuoteContext(Config.from_env())
    out = dict(
        generated=datetime.now(SGT).isoformat(timespec="seconds"),
        note=("跟单规则模拟(真实5分K, fill-realistic): enrich=2张·止盈2x卖半·镜像出场·止损-30%·到期强平; "
              "andy=波段+他的止损+BE+自动保本·$1万预算。窗口受长桥期权K线保留期限制(归档K线可延展), "
              "无数据/未成交如实标注。"),
        enrich=enrich_rows(q), andy=andy_rows(q))
    OUT.mkdir(exist_ok=True)
    (OUT / "signal_history.json").write_text(json.dumps(out, ensure_ascii=False))
    for k in ("enrich", "andy"):
        rs = out[k]
        done = [r for r in rs if r["status"] in ("closed", "open")]
        pnl = sum(r.get("pnl", 0) for r in done)
        print(f"{k}: {len(done)}/{len(rs)} 笔可测, 合计 ${pnl:+,.0f}")


if __name__ == "__main__":
    main()
