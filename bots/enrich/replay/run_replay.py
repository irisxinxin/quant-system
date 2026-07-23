#!/usr/bin/env python3
"""replay/run_replay.py — 跑回放并出报告。
用法: DISCORD_BOT_TOKEN=x /usr/local/bin/python3 bots/enrich/replay/run_replay.py oracle_260701_260731.pkl 2026-07-01 2026-07-31
"""
import sys
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import Replay

oracle = sys.argv[1] if len(sys.argv) > 1 else "oracle_260701_260731.pkl"
lo = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date(2026, 7, 1)
hi = date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3 else date(2026, 7, 31)
op = Path(__file__).resolve().parent / oracle
r = Replay(str(op)).run(lo, hi)
r.report()
cap = sys.argv[4] if len(sys.argv) > 4 else "/tmp/replay_capture.json"
r.save_capture(cap)
print(f"\n捕获存 → {cap}")
