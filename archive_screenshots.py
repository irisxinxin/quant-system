"""
archive_screenshots.py — 把 ~/Desktop/投研 里的 Kova 截图自动同步进仓库归档。

用法:
    python3 archive_screenshots.py            # 同步新图到 kova_screenshots/inbox/
    python3 archive_screenshots.py --src DIR  # 指定其它来源目录

机制(幂等):
- 按内容 SHA8 去重: 已归档过的(无论改没改名)不重复拷。
- 新图拷进 kova_screenshots/inbox/, 文件名保留原名(image-<ts>.webp)。
- 维护 kova_screenshots/manifest.csv: sha8,orig_name,size,src_mtime,ticker,date,note
  (ticker/date 留空, 由 Claude 读图识别后回填, 再 mv 到 <date>/<ticker>.webp)

这就是"自动存"的核心: 用户把图丢进 投研, 跑一次本脚本即全部入库, 不丢任何一张。
"""
import sys, os, shutil, hashlib, csv
from pathlib import Path
from datetime import datetime, timezone

SRC = Path.home() / "Desktop" / "投研"
ARCHIVE = Path(__file__).parent / "kova_screenshots"
INBOX = ARCHIVE / "inbox"
MANIFEST = ARCHIVE / "manifest.csv"
EXTS = {".webp", ".png", ".jpg", ".jpeg"}


def sha8(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def load_manifest():
    rows = {}
    if MANIFEST.exists():
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["sha8"]] = r
    return rows


def save_manifest(rows):
    cols = ["sha8", "orig_name", "size", "src_mtime", "ticker", "date", "note"]
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows.values(), key=lambda x: x.get("src_mtime", "")):
            w.writerow({c: r.get(c, "") for c in cols})


def main():
    src = SRC
    if "--src" in sys.argv:
        src = Path(sys.argv[sys.argv.index("--src") + 1]).expanduser()
    if not src.exists():
        print(f"❌ 来源不存在: {src}")
        sys.exit(1)
    INBOX.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    known = set(manifest.keys())

    imgs = [p for p in src.iterdir() if p.suffix.lower() in EXTS]
    new = 0
    for p in sorted(imgs, key=lambda x: x.stat().st_mtime):
        h = sha8(p)
        if h in known:
            continue
        dest = INBOX / p.name
        if dest.exists():                       # 同名不同内容, 加 sha 防覆盖
            dest = INBOX / f"{p.stem}-{h}{p.suffix}"
        shutil.copy2(p, dest)
        manifest[h] = {
            "sha8": h, "orig_name": p.name, "size": str(p.stat().st_size),
            "src_mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "ticker": "", "date": "", "note": "",
        }
        known.add(h)
        new += 1
    save_manifest(manifest)
    print(f"✅ 同步完成: 新增 {new} 张, 归档总计 {len(manifest)} 张 → {INBOX}")
    unlabeled = sum(1 for r in manifest.values() if not r.get("ticker"))
    if unlabeled:
        print(f"   待识别(ticker未填): {unlabeled} 张 — 由 Claude 读图回填 manifest.csv")


if __name__ == "__main__":
    main()
