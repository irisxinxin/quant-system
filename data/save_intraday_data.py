"""
data/save_intraday_data.py — 把 LongPort 拉的 5m 数据从 cache 转成持久化 JSON
存到 output/klines_5m/{TICKER}.json (与 daily 的 output/klines 同一规范)

调用:
  python3 -m data.save_intraday_data
"""
import json
import pickle
from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).parent.parent / "cache" / "longport_history"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "klines_5m"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_one(pkl_path: Path) -> dict:
    """把一个 pkl 转成 JSON 可写记录"""
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    if df.empty:
        return None

    # 移除时区 (统一保存为 ET naive 字符串)
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    # 输出格式 (和现有 klines 一致, 加 datetime)
    records = []
    for idx, row in df.iterrows():
        records.append({
            "datetime": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "open":   round(float(row["Open"]), 4),
            "high":   round(float(row["High"]), 4),
            "low":    round(float(row["Low"]), 4),
            "close":  round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        })

    return {
        "symbol": pkl_path.stem.replace("_5m", "").replace("_", "."),
        "interval": "5m",
        "timezone": "US/Eastern",
        "bars": len(records),
        "first": records[0]["datetime"] if records else None,
        "last": records[-1]["datetime"] if records else None,
        "data": records,
    }


def main():
    pkl_files = sorted(CACHE_DIR.glob("*_5m.pkl"))
    if not pkl_files:
        print("⚠️ cache/longport_history 没有找到 pkl 文件，先跑 longport_history_backtest.py")
        return

    print(f"📦 转换 {len(pkl_files)} 个 LongPort 5m 数据文件 → JSON")
    print(f"   源: {CACHE_DIR}")
    print(f"   目标: {OUTPUT_DIR}\n")

    for pkl_path in pkl_files:
        print(f"  {pkl_path.name:<30}", end=" ")
        try:
            payload = convert_one(pkl_path)
            if payload is None:
                print("空数据")
                continue

            symbol = payload["symbol"]
            out_path = OUTPUT_DIR / f"{symbol}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

            size_kb = out_path.stat().st_size / 1024
            print(f"→ {symbol}.json  ({payload['bars']} 根, {size_kb:.0f} KB)")
        except Exception as e:
            print(f"❌ {e}")

    # 汇总 manifest
    manifest = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "interval": "5m",
        "source": "LongPort OpenAPI",
        "tickers": [],
    }
    for json_file in sorted(OUTPUT_DIR.glob("*.json")):
        if json_file.name == "_manifest.json":
            continue
        with open(json_file) as f:
            data = json.load(f)
        manifest["tickers"].append({
            "symbol": data["symbol"],
            "bars": data["bars"],
            "first": data["first"],
            "last": data["last"],
            "file": json_file.name,
        })

    manifest_path = OUTPUT_DIR / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完成。Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
