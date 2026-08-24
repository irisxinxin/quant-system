#!/usr/bin/env python3
"""
notion_board_sync.py — 12 个 Notion 看板的批量刷价工具。

为什么要有这个脚本(2026-08-24 立):
  之前靠 MCP 一行一个 update 调用刷 300+ 行, 一轮要几十次工具往返, 而且靠"我改过哪些行"
  记账 → 连续两次报告"刷完了"却漏了 11 行/60 行(用户两次抓到)。
  这里改成**全表遍历**: 拉全部行 → 比对 → 只 PATCH 真正变化的 → 再拉一次验证。
  漏行在结构上不可能发生, 因为遍历的是数据库返回的行, 不是我的记忆。

用法:
  export NOTION_TOKEN=...            # 或写进 ~/.notion_creds.env
  python3 notion_board_sync.py            # 全部库, 只刷现价
  python3 notion_board_sync.py --dry      # 干跑, 只报告要改什么不实际写
  python3 notion_board_sync.py --db 聚焦 蛋挞板   # 只刷指定库

前置(用户需自己做一次):
  1. https://www.notion.so/my-integrations 建 internal integration, 拿 secret
  2. 每个数据库页面 → 右上 ⋯ → Connections → 添加该 integration
     (12 个库都要加, 漏了的库脚本会报 404 并列出来)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
API = "https://api.notion.com/v1"
VERSION = "2022-06-28"          # 稳定版; 行为与当前看板结构匹配
RATE_SLEEP = 0.34               # Notion 官方约 3 req/s, 留一点余量


# ── 12 个库: 名字 → (database_id, 标题字段名) ───────────────────────────────
DATABASES = {
    "聚焦":        ("fad27d9ceb074982aa478cd700651d4e", "票"),
    "蛋挞板":      ("e97e1f42c08145d7b3e92c628e1b4c5e", "票"),
    "共振表":      ("5354fc00b43444428b10cbdb40d64a4f", "票"),
    "Kova":        ("8b0bd46ebb724721816d2ad94cdebafe", "票"),
    "索亚":        ("bb4046e7f3224c1bb071881465b00d60", "票"),
    "边城":        ("db09cca6ba44461cb7f5e26fd22e04ed", "票"),
    "CM唐主任":    ("ded79a509b3942e5b88f333fee463e67", "票"),
    "Sam":         ("81e10a74adcb408cbf783eb01ee0d9d9", "票"),
    "形态多":      ("ce3dc83cdea54ef797285e065625267b", "票"),
    "天哥":        ("b5bb590a5140473e937c0a85c6a97789", "标的"),
    "AbTrades":    ("4df653638ab34a50b28cacc8175c91d8", "标的"),
    "Seeker":      ("beec20e6bca64c00a47555d07cc87e2e", "标的"),
    "华尔街观察":  ("5c414a7de51a461a948f851e56753c11", "标的"),
}

# 标题里带修饰的行 → 实际可查价的代码
ALIAS = {
    "DRAM(美股海力士)": "DRAM", "GOOG/GOOGL": "GOOGL", "GOOG": "GOOGL",
    "黄金主题(GLD/金股)": "GLD", "云计算主题(SKYY)": "SKYY", "IGV (软件ETF)": "IGV",
    "UGL (黄金2倍ETF)": "UGL", "ETH / IBIT (加密ETF)": "IBIT",
    "QTUM(量子篮子)": "QTUM", "MU (7月单)": "MU",
    "SOXL(一仓)": "SOXL", "SOXL(二仓)": "SOXL", "BRK-B": "BRK.B",
}

# 非个股行(方法论/汇总/A股港股ETF/加密现货) — 不刷价, 但要在报告里点名, 避免"静默跳过"
SKIP_PREFIX = ("📐", "📊", "🎩", "🅰️", "🔭")
SKIP_EXACT = {
    "BTC/黄金", "纳指期货点位", "存储板块", "支付宝黄金ETF", "SIVE", "CLBK", "BMO",
    "512100/510500 中证1000+500", "512400 有色金属", "517520 黄金股ETF",
    "513120 港股创新药", "159131 港股通信息技术", "518880 黄金ETF",
    "513100/159516 半导体两ETF",
}


def load_token() -> str:
    tok = os.environ.get("NOTION_TOKEN")
    if not tok:
        env = Path.home() / ".notion_creds.env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("NOTION_TOKEN"):
                    tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not tok:
        sys.exit("❌ 未找到 NOTION_TOKEN。先建 internal integration(见文件头说明), "
                 "然后 echo 'NOTION_TOKEN=secret_xxx' >> ~/.notion_creds.env && chmod 600 ~/.notion_creds.env")
    return tok


def headers(tok):
    return {"Authorization": f"Bearer {tok}", "Notion-Version": VERSION,
            "Content-Type": "application/json"}


def fetch_rows(tok, db_id, title_prop):
    """拉一个库的全部行(自动翻页) → [(page_id, 标题, 现价)]"""
    rows, cursor = [], None
    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        r = requests.post(f"{API}/databases/{db_id}/query", headers=headers(tok),
                          json=body, timeout=30)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.json().get('message', '')[:120]}"
        d = r.json()
        for p in d["results"]:
            props = p.get("properties", {})
            t = props.get(title_prop, {}).get("title", [])
            name = "".join(x.get("plain_text", "") for x in t).strip()
            cur = props.get("现价", {}).get("number")
            rows.append((p["id"], name, cur))
        if not d.get("has_more"):
            break
        cursor = d["next_cursor"]
        time.sleep(RATE_SLEEP)
    return rows, None


def quote_all(tickers):
    """长桥批量取价 → {ticker: 最新价}"""
    from longport.openapi import Config, QuoteContext
    ctx = QuoteContext(Config.from_env())
    out, ts = {}, sorted(tickers)
    for i in range(0, len(ts), 40):
        batch = [t + ".US" for t in ts[i:i + 40]]
        try:
            for q in ctx.quote(batch):
                out[q.symbol.replace(".US", "")] = round(float(q.last_done), 2)
        except Exception as e:                       # 单批失败就逐个重试, 不让整批丢
            for t in ts[i:i + 40]:
                try:
                    q = ctx.quote([t + ".US"])[0]
                    out[t] = round(float(q.last_done), 2)
                except Exception:
                    pass
    return out


def patch_price(tok, page_id, price):
    r = requests.patch(f"{API}/pages/{page_id}", headers=headers(tok),
                       json={"properties": {"现价": {"number": price}}}, timeout=30)
    return r.status_code == 200, (r.json().get("message", "")[:80] if r.status_code != 200 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只报告不写入")
    ap.add_argument("--db", nargs="*", help="只处理指定库(默认全部)")
    ap.add_argument("--tol", type=float, default=0.005, help="价格差异阈值(默认0.5%%内不改)")
    args = ap.parse_args()

    tok = load_token()
    targets = {k: v for k, v in DATABASES.items() if not args.db or k in args.db}

    # ① 全表遍历: 拉所有库的所有行
    print("① 拉取各库全部行…")
    all_rows, failed = {}, {}
    for name, (db_id, tp) in targets.items():
        rows, err = fetch_rows(tok, db_id, tp)
        if err:
            failed[name] = err
            print(f"   ⚠️ {name:10} {err}")
            continue
        all_rows[name] = rows
        print(f"   {name:10} {len(rows):>3} 行")
        time.sleep(RATE_SLEEP)
    if failed:
        print("\n⚠️ 以下库拉取失败(多半是没在 Notion 里把库分享给 integration):")
        for k, v in failed.items():
            print(f"   {k}: {v}")

    # ② 收集需要报价的 ticker
    need, skipped = set(), []
    for name, rows in all_rows.items():
        for _, title, _ in rows:
            if not title or title.startswith(SKIP_PREFIX) or title in SKIP_EXACT:
                skipped.append(f"{name}/{title}")
                continue
            need.add(ALIAS.get(title, title))
    print(f"\n② 需报价 {len(need)} 个代码, 跳过非个股行 {len(skipped)} 条")

    px = quote_all(need)
    miss = sorted(need - set(px))
    if miss:
        print(f"   ⚠️ 取不到价({len(miss)}): {' '.join(miss)}")

    # ③ 比对 → 只改真正变了的
    print("\n③ 比对并更新…")
    changed = same = err_n = 0
    for name, rows in all_rows.items():
        hits = []
        for pid, title, cur in rows:
            if not title or title.startswith(SKIP_PREFIX) or title in SKIP_EXACT:
                continue
            new = px.get(ALIAS.get(title, title))
            if new is None:
                continue
            if cur is not None and abs(cur - new) / max(new, 1e-9) < args.tol:
                same += 1
                continue
            hits.append((pid, title, cur, new))
        if not hits:
            print(f"   {name:10} 无需更新")
            continue
        print(f"   {name:10} {len(hits)} 行待改: " +
              ", ".join(f"{t} {c}→{n}" for _, t, c, n in hits[:6]) +
              (" …" if len(hits) > 6 else ""))
        if args.dry:
            changed += len(hits)
            continue
        for pid, title, cur, new in hits:
            ok, msg = patch_price(tok, pid, new)
            if ok:
                changed += 1
            else:
                err_n += 1
                print(f"      ❌ {title}: {msg}")
            time.sleep(RATE_SLEEP)

    print(f"\n{'[干跑] ' if args.dry else ''}完成: 更新 {changed} 行 | 已是最新 {same} 行 | 失败 {err_n} 行")

    # ④ 验证: 重新拉一遍确认真的写进去了(这一步是防"报告说改了其实没改")
    if not args.dry and changed:
        print("\n④ 回读验证…")
        bad = []
        for name, (db_id, tp) in targets.items():
            if name in failed:
                continue
            rows, err = fetch_rows(tok, db_id, tp)
            if err:
                continue
            for _, title, cur in rows:
                if not title or title.startswith(SKIP_PREFIX) or title in SKIP_EXACT:
                    continue
                new = px.get(ALIAS.get(title, title))
                if new is not None and cur is not None and abs(cur - new) / max(new, 1e-9) >= args.tol:
                    bad.append(f"{name}/{title}: 库里{cur} vs 应为{new}")
            time.sleep(RATE_SLEEP)
        if bad:
            print(f"   ⚠️ 仍有 {len(bad)} 行不一致:")
            for b in bad[:20]:
                print(f"      {b}")
        else:
            print("   ✅ 全部一致")

    json.dump({"px": px, "skipped": skipped, "missing": miss, "failed": failed},
              open(ROOT / "output" / "notion_sync_last.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
