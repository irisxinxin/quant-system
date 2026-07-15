#!/usr/bin/env python3
"""
enrich_archive.py — enrich 跟单数据每日归档 (回测原料抢救; launchd 每日 05:10 SGT = 美股收盘后)。

三样原料:
  1. output/enrich_history.json   频道消息全量存档 (Discord 重新拉取覆盖)
  2. data/enrich_bars/<OSI>.csv   相关期权合约的5分K (⚠长桥只留约1月, 必须及时抢救; 增量合并)
     — 含 BUY 的合约 + BUY_AMBIG 的 C/P 两腿 (未来做消歧回测)
  3. output/enrich_orders.csv     模拟盘订单流水 (真实成交价 ground truth; 按 order_id 去重)
归档后若有变化: git add + commit + pull --rebase + push。
"""
import csv, json, subprocess, sys
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
BARS = ROOT / "data" / "enrich_bars"
HIST = ROOT / "output" / "enrich_history.json"
ORDERS = ROOT / "output" / "enrich_orders.csv"
UTC = timezone.utc


def refresh_history():
    """Discord 一次性拉取, 覆盖存档。失败保留旧档。"""
    import os, discord
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("⚠️ 无DISCORD_BOT_TOKEN, 跳过消息存档"); return
    intents = discord.Intents.default(); intents.message_content = True
    client = discord.Client(intents=intents)
    AUTHOR_ID = 1392020997393088542
    CHANS = {"期权-波段-enrich": HIST, "andy-option": ROOT/"output"/"andy_history.json",
             "股票赵哥-日内": ROOT/"output"/"zhaoge_history.json"}

    @client.event
    async def on_ready():
        for g in client.guilds:
            for c in g.text_channels:
                for key, path in CHANS.items():
                    if key in c.name:
                        msgs = []
                        async for m in c.history(limit=3000):
                            if m.author.id == AUTHOR_ID and m.content:
                                msgs.append(dict(id=m.id, ts=m.created_at.isoformat(), text=m.content))
                        msgs.reverse()
                        path.write_text(json.dumps(msgs, ensure_ascii=False))
                        print(f"① 消息存档 {key}: {len(msgs)} 条")
        await client.close()

    client.run(token, log_handler=None)


def wanted_contracts():
    """要归档K线的合约: enrich 近35天 BUY/BUY_AMBIG两腿 + andy 近35天入场合约。"""
    from enrich_parser import parse_signal, to_longport_symbol
    out = set()
    cutoff = date.today() - timedelta(days=35)
    # andy 频道
    try:
        from backtest_andy import parse_entry, osi as andy_osi
        for m in json.load(open(ROOT/"output"/"andy_history.json")):
            d = datetime.fromisoformat(m["ts"]).date()
            if d < cutoff: continue
            e = parse_entry(m["text"], d)
            if e: out.add(andy_osi(e))
    except Exception as ex:
        print(f"⚠️ andy合约收集失败: {ex}")
    for m in json.load(open(HIST)):
        ts = datetime.fromisoformat(m["ts"])
        d = ts.date()
        if d < cutoff:
            continue
        s = parse_signal(m["text"], d)
        if s.kind == "BUY":
            out.add(to_longport_symbol(s))
        elif s.kind == "BUY_AMBIG":
            for r in ("C", "P"):
                out.add(f"{s.ticker}{s.expiry:%y%m%d}{r}{int(round(s.strike*1000)):06d}.US")
    return sorted(out)


def archive_bars(ctx):
    from longport.openapi import Period, AdjustType
    BARS.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for osi in wanted_contracts():
        try:
            b = ctx.candlesticks(osi, Period.Min_5, 1000, AdjustType.NoAdjust)
        except Exception:
            continue
        if not b:
            continue
        f = BARS / f"{osi}.csv"
        rows = {}
        if f.exists():
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    rows[r["ts"]] = r
        for x in b:
            ts = x.timestamp.astimezone(UTC).isoformat()
            rows[ts] = dict(ts=ts, o=float(x.open), h=float(x.high), l=float(x.low),
                            c=float(x.close), v=int(x.volume))
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ts", "o", "h", "l", "c", "v"])
            w.writeheader()
            for ts in sorted(rows):
                w.writerow(rows[ts])
        n_new += 1
    print(f"② K线归档: {n_new} 个合约 → data/enrich_bars/")


def archive_orders(tctx):
    """模拟盘订单流水 (近30天), 按 order_id 去重合并。"""
    rows = {}
    if ORDERS.exists():
        with open(ORDERS) as fh:
            for r in csv.DictReader(fh):
                rows[r["order_id"]] = r
    try:
        ods = tctx.history_orders(start_at=datetime.now(UTC) - timedelta(days=30),
                                  end_at=datetime.now(UTC))
    except Exception as e:
        print(f"⚠️ 拉订单失败: {e}"); ods = []
    for o in ods:
        rows[str(o.order_id)] = dict(
            order_id=str(o.order_id), symbol=o.symbol,
            side=str(o.side).split(".")[-1], status=str(o.status).split(".")[-1],
            qty=str(o.quantity), executed_qty=str(o.executed_quantity or 0),
            price=str(o.price or ""), executed_price=str(o.executed_price or ""),
            submitted_at=str(o.submitted_at), updated_at=str(o.updated_at or ""),
            remark=getattr(o, "remark", "") or "")
    with open(ORDERS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["order_id", "symbol", "side", "status", "qty",
                                           "executed_qty", "price", "executed_price",
                                           "submitted_at", "updated_at", "remark"])
        w.writeheader()
        for k in sorted(rows, key=lambda k: rows[k]["submitted_at"]):
            w.writerow(rows[k])
    print(f"③ 订单流水: 共 {len(rows)} 条 → output/enrich_orders.csv")


def git_commit_push():
    def run(*a):
        return subprocess.run(a, cwd=ROOT, capture_output=True, text=True)
    run("git", "add", "-f", "output/enrich_history.json", "output/enrich_journal.jsonl",
        "output/enrich_positions.json", "output/enrich_orders.csv",
        "output/andy_history.json", "output/zhaoge_history.json")
    run("git", "add", "data/enrich_bars")
    run("git", "add", "-f", "output/trade_report.html")
    if not run("git", "diff", "--cached", "--quiet").returncode:
        print("④ 无变化, 不提交"); return
    run("git", "commit", "-m", f"data(enrich): 归档 {date.today()}\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>")
    run("git", "pull", "--rebase")
    r = run("git", "push")
    print(f"④ 已提交推送 {'✅' if r.returncode == 0 else '⚠️push失败:' + r.stderr[:100]}")


def main():
    from longport.openapi import Config, QuoteContext, TradeContext
    print(f"=== enrich归档 {datetime.now():%Y-%m-%d %H:%M} ===")
    try:
        refresh_history()
    except Exception as e:
        print(f"⚠️ 消息存档失败(保留旧档): {e}")
    cfg = Config.from_env()
    archive_bars(QuoteContext(cfg))
    archive_orders(TradeContext(cfg))
    try:
        import subprocess as sp
        sp.run([sys.executable, str(ROOT / "trade_report.py")], capture_output=True, timeout=120)
        print("⑤ 战报HTML已刷新")
    except Exception as e:
        print(f"⑤ 战报刷新失败: {e}")
    git_commit_push()


if __name__ == "__main__":
    main()
