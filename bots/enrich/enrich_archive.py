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
# 仓库根进 path: 复用根目录的 backtest_andy 解析器。
ROOT = Path(__file__).resolve().parent.parent.parent      # 仓库根 — data/ output/ git 都在这
sys.path.insert(1, str(ROOT))
import warnings; warnings.filterwarnings("ignore")

BARS = ROOT / "data" / "enrich_bars"
HIST = ROOT / "output" / "enrich_history.json"
ORDERS = ROOT / "output" / "enrich_orders.csv"
UTC = timezone.utc


def refresh_history():
    """Discord 一次性拉取, 覆盖存档。失败保留旧档。"""
    import os, socket
    import aiohttp
    _orig = aiohttp.TCPConnector.__init__      # 2026-07-31起: 本机网络IPv6半残且Discord需走本地代理
    def _v4(self, *a, **kw):
        kw["family"] = socket.AF_INET
        _orig(self, *a, **kw)
    aiohttp.TCPConnector.__init__ = _v4
    import discord
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("⚠️ 无DISCORD_BOT_TOKEN, 跳过消息存档"); return
    proxy = os.environ.get("HTTPS_PROXY")
    if not proxy:
        # 2026-08-17: 代理环境会变(Clash 7897 ↔ SSX-NG 1087, 且1087的privoxy拒绝Discord CONNECT)。
        # 探测顺序: 直连Discord可用→不走代理; 否则依次试本地代理端口。
        import urllib.request as _ur
        try:
            _ur.urlopen("https://discord.com/api/v10/gateway", timeout=6)
            proxy = None
        except Exception:
            proxy = None
            for port in (7897, 7890, 1087):
                try:
                    s = socket.create_connection(("127.0.0.1", port), timeout=1); s.close()
                    proxy = f"http://127.0.0.1:{port}"
                    break
                except OSError:
                    continue
    intents = discord.Intents.default(); intents.message_content = True
    client = discord.Client(intents=intents, proxy=proxy)
    ZWZF = 1392020997393088542        # 站长转发#2054
    ZWZF3 = 1486283682753937488       # 站长转发3#7191
    ZZ = 1535470978660827196          # zhangzhanglucky/zzlucky (张张本人)
    CHANS = {"期权-波段-enrich": HIST, "andy-option": ROOT/"output"/"andy_history.json",
             "股票赵哥-日内": ROOT/"output"/"zhaoge_history.json",
             "elite-alert": ROOT/"output"/"elite_alert_history.json",
             "elite-commentary": ROOT/"output"/"elite_commentary_history.json",
             "指数-casey": ROOT/"output"/"casey_history.json",
             "小鱼vip": ROOT/"output"/"xiaoyu_vip_history.json",
             "蛋挞vip": ROOT/"output"/"danta_vip_history.json",
             "张张": ROOT/"output"/"zhangzhang_history.json",
             "唐主任": ROOT/"output"/"tangzhuren_history.json",
             "边城": ROOT/"output"/"biancheng_history.json",
             "索亚": ROOT/"output"/"suoya_history.json",
             "信号": ROOT/"output"/"kova_signal_history.json",
             "潜力形态-多": ROOT/"output"/"qianli_duo_history.json",
             "小鱼日内vip": ROOT/"output"/"xiaoyu_intraday_history.json",
             "所有突發信息": ROOT/"output"/"samlam_history.json",   # Sam lam投資筆記(港, Minervini流)
             "华尔街观察-正股": ROOT/"output"/"wallst_history.json",  # 第九源: 大类资产轮动(黄金主仓+行业ETF)
             "seek-vip": ROOT/"output"/"seek_vip_history.json"}  # 第十二源候选: ~Seeker~英文技术流(站长转发3中继), 深回撤埋伏+阻力位, 8/19接入观察
    KOVA = 1520803125647380640        # Kova本人
    KOVA_TR = 1511035459709702314     # 懂王翻译2
    TTT = 1350502142997434582         # ttt2023(群主, 华尔街观察频道发布人)
    WSGC = 1538784726234693652        # 华尔街观察官方号
    ALLOW = {ZWZF, ZWZF3, ZZ, KOVA, KOVA_TR, TTT, WSGC}  # 站长中继×2+张张+Kova+翻译+华尔街观察×2

    IMG_SAVE = {"蛋挞vip": ROOT / "data" / "danta_img",   # 点位表以图片发布的频道 → 落盘抢救(CDN链接会过期)
                "潜力形态-多": ROOT / "data" / "qianli_img",   # 形态派全靠图
                "小鱼vip": ROOT / "data" / "xiaoyu_img",       # 鱼哥每天两次的 radar10.png(雷达十票 P10/P50/P90 预测图)
                "小鱼日内vip": ROOT / "data" / "danta_intraday_img",  # 蛋挞在日内频道发的点位图/持仓图
                "所有突發信息": ROOT / "data" / "samlam_img",          # Sam lam 的图表标注(base/breakout 划线图)
                "华尔街观察-正股": ROOT / "data" / "wallst_img",          # 华尔街观察周报配图
                "seek-vip": ROOT / "data" / "seek_img"}                  # Seeker 的点位全画在图上(箱体zone)

    @client.event
    async def on_ready():
        for g in client.guilds:
            for c in g.text_channels:
                for key, path in CHANS.items():
                    if key in c.name:
                      try:
                        msgs = []
                        async for m in c.history(limit=3000):
                            if m.author.id in ALLOW and (m.content or m.attachments):
                                row = dict(id=m.id, ts=m.created_at.isoformat(), text=m.content or "")
                                if m.attachments:
                                    row["att"] = [a.filename for a in m.attachments]
                                msgs.append(row)
                                d = IMG_SAVE.get(key)
                                if d:
                                    d.mkdir(parents=True, exist_ok=True)
                                    # 中继号把图片贴成正文CDN链接(非附件), 从文本抠链接下载; 链接会过期→趁新鲜每天抢救
                                    import re as _re
                                    urls = [a.url for a in m.attachments] + \
                                        _re.findall(r"https://cdn\.discordapp\.com/attachments/\S+", m.content or "")
                                    for e in m.embeds:      # 形态频道的图挂在embed上
                                        if e.image and e.image.url:
                                            urls.append(e.image.url)
                                        elif e.thumbnail and e.thumbnail.url:
                                            urls.append(e.thumbnail.url)
                                    for i, u in enumerate(urls):
                                        f = d / f"{m.id}_{i}.png"
                                        if f.exists():
                                            continue
                                        try:
                                            async with aiohttp.ClientSession() as sess:
                                                async with sess.get(u, proxy=proxy, timeout=aiohttp.ClientTimeout(total=30)) as r:
                                                    if r.status == 200:
                                                        f.write_bytes(await r.read())
                                        except Exception:
                                            pass
                        msgs.reverse()
                        path.write_text(json.dumps(msgs, ensure_ascii=False))
                        print(f"① 消息存档 {key}: {len(msgs)} 条")
                      except Exception as e:
                        print(f"⚠️ {c.name} 跳过: {str(e)[:60]}")  # 如边城聊天区403, 不拖垮其他频道
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



def casey_elite_contracts():
    """casey(QQQ/SPY 0DTE '683p'口径) + elite('QQQ JULY 31 685C'口径) 近32天合约。"""
    import re
    out = set()
    cutoff = date.today() - timedelta(days=32)
    MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
    try:
        for m in json.load(open(ROOT/"output"/"casey_history.json")):
            d = datetime.fromisoformat(m["ts"]).date()
            if d < cutoff: continue
            for tk, st, r in re.findall(r"\b(QQQ|SPY|IWM)\s+(\d{2,4})\s*([cpCP])\b", m["text"]):
                out.add(f"{tk}{d:%y%m%d}{r.upper()}{int(st)*1000:06d}.US")
    except Exception as e:
        print(f"⚠️ casey合约收集失败: {e}")
    try:
        for m in json.load(open(ROOT/"output"/"elite_alert_history.json")):
            d = datetime.fromisoformat(m["ts"]).date()
            if d < cutoff: continue
            for tk, mon, day, st, r in re.findall(r"\b([A-Z]{1,5})\s+([A-Z]{3,9})\s+(\d{1,2})\s+(\d{2,4})([CP])\b", m["text"].upper()):
                mo = MON.get(mon[:3])
                if not mo: continue
                y = d.year + (1 if mo < d.month - 6 else 0)
                try:
                    exp = date(y, mo, int(day))
                except ValueError:
                    continue
                out.add(f"{tk}{exp:%y%m%d}{r}{int(st)*1000:06d}.US")
    except Exception as e:
        print(f"⚠️ elite合约收集失败: {e}")
    return out


def archive_bars(ctx):
    from longport.openapi import Period, AdjustType
    BARS.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for osi in sorted(set(wanted_contracts()) | casey_elite_contracts()):
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
    run("git", "add", "-f", "output/trade_report.html", "output/signal_history.json")
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
        sp.run([sys.executable, str(ROOT / "signal_history.py")], capture_output=True, timeout=1200)
        sp.run([sys.executable, str(ROOT / "trade_report.py")], capture_output=True, timeout=120)
        print("⑤ 战报HTML已刷新")
    except Exception as e:
        print(f"⑤ 战报刷新失败: {e}")
    git_commit_push()


if __name__ == "__main__":
    main()
