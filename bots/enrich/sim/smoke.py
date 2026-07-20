#!/usr/bin/env python3
"""sim/smoke.py — 冒烟: 证明 harness 能驱动 bot 真实链路走完一笔完整交易。
跑法: /usr/local/bin/python3 sim/smoke.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sim.harness import Sim

OSI = "HOOD260724C120000.US"

with Sim(equity=100_000) as s:
    # 期权价格路径: 0.83 建仓 → 涨到 1.09(+30%触一档) → 1.35(+60%触二档) → 回落
    s.quotes.set_path(OSI, [0.83, 0.83, 0.95, 1.10, 1.20, 1.35, 1.40, 1.20, 1.00])
    s.quotes.open_interest[OSI] = 50000
    s.send("$HOOD 7/24 $120 calls $.83")
    print("① 发单后 bot 视角:", s.pos(OSI) and
          {k: s.pos(OSI)[k] for k in ("status", "qty", "filled")})
    print("   券商侧活单:", s.broker.live_orders())

    s.run(ticks=8)

    p = s.pos(OSI)
    print("\n② 跑完 8 轮:")
    print("   bot:    status=%s filled=%s sold=%s avg=%s armed=%s"
          % (p["status"], p["filled"], p["sold"], p["avg"], p.get("armed")))
    print("   券商:   净持仓=%s" % s.broker_pos(OSI))
    print("   成交:   %s 笔" % len(s.broker.fills))
    for f in s.broker.fills:
        print("           %s %s×%s @ $%s" % (f[3], f[2][:18], f[4], f[5]))
    print("   事件:   %s" % [e.get("ev") for e in s.events])

    v = s.check_invariants()
    print("\n③ 不变式:", "✅ 全部通过" if not v else "❌ " + " | ".join(v))
    sys.exit(1 if v else 0)
