"""
kova_stops.py — 终端查看持仓止损（按 Kova A–E 状态机）。
计算逻辑在 stops_core.py；网页版见 /stops（app.py 的 /api/stops 实时计算）。
用法: source ~/.longport_creds.env && python3 kova_stops.py
"""
from stops_core import compute_stops


def main():
    res = compute_stops()
    rows = res["rows"]
    print(f"数据源 {res['source']} · 截至 {res['data_date']} 收盘 · {len(rows)} 只\n")
    for r in rows:
        dis = f"{r['disaster']:.2f}" if r["disaster"] is not None else "—"
        print(f"  {r['tk']:5s} {r['state']:8s} 盈亏{r['gain']:+6.1f}%  "
              f"主止损 {r['follow']:8.2f} ({r['follow_label']})  灾难 {dis}")


if __name__ == "__main__":
    main()
