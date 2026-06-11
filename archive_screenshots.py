"""
archive_screenshots.py — 把 ~/Desktop/投研 里的 Kova 截图自动同步进仓库归档。

用法:
    python3 archive_screenshots.py            # 同步新图到 kova_screenshots/inbox/
    python3 archive_screenshots.py --src DIR  # 指定其它来源目录

机制(自愈式去重, 不依赖索引文件):
- 扫 kova_screenshots/** 下所有已归档图(inbox + 各日期目录)的内容 SHA8 → 集合 known。
- 来源目录里 sha8 已在 known 的跳过; 新的拷进 kova_screenshots/inbox/(保留原名)。
- 之后由 Claude 读图识别 → mv 到 <date>/<ticker>.webp + 写 manifest.csv。

这就是"自动存"的核心: 用户把图丢进 投研, 跑一次本脚本即新图入库, 已归档的不重复, 不丢任何一张。
"""
import sys, shutil, hashlib
from pathlib import Path

SRC = Path.home() / "Desktop" / "投研"
ARCHIVE = Path(__file__).parent / "kova_screenshots"
INBOX = ARCHIVE / "inbox"
EXTS = {".webp", ".png", ".jpg", ".jpeg"}


def sha8(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def main():
    src = SRC
    if "--src" in sys.argv:
        src = Path(sys.argv[sys.argv.index("--src") + 1]).expanduser()
    if not src.exists():
        print(f"❌ 来源不存在: {src}")
        sys.exit(1)
    INBOX.mkdir(parents=True, exist_ok=True)

    # 已归档内容指纹(inbox + 所有日期目录递归)
    known = {}
    for p in ARCHIVE.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            known[sha8(p)] = p.relative_to(ARCHIVE)

    imgs = [p for p in src.iterdir() if p.suffix.lower() in EXTS]
    new = 0
    for p in sorted(imgs, key=lambda x: x.stat().st_mtime):
        h = sha8(p)
        if h in known:
            continue
        dest = INBOX / p.name
        if dest.exists():
            dest = INBOX / f"{p.stem}-{h}{p.suffix}"
        shutil.copy2(p, dest)
        known[h] = dest.relative_to(ARCHIVE)
        new += 1
    inbox_n = len(list(INBOX.glob("*")))
    total = len(known)
    print(f"✅ 同步完成: 新增 {new} 张 → inbox/  (仓库已归档总计 {total} 张)")
    if inbox_n:
        print(f"   inbox 待识别归位: {inbox_n} 张 — 由 Claude 读图 → mv 到 <date>/<ticker>.webp")


if __name__ == "__main__":
    main()
