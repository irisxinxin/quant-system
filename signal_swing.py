#!/usr/bin/env python3
"""
signal_swing.py — 波段择时信号 (日线; 吃隔夜+趋势; 仅 Telegram 提醒, 不自动下单)

针对杠杆ETF + 强趋势票: 日内只吃盘中一段, 波段才吃得到整段趋势(三星2x: 日内+19% vs 波段+725%)。
规则(Oliver Kell 10/20EMA trail + 趋势模板, long-only):
  🟢 波段买入: 收盘"站回"20EMA(昨破今站) 且 上升趋势(>50S>200S)
  🟠 波段减仓: 持有中 且 涨过头(距21EMA ≥ 2.93 ATR)  → 减 1/3~1/2
  🔴 波段退出: 收盘跌破 20EMA×0.99 (趋势破, 离场; 杠杆ETF尤其要守纪律)
每日收盘后跑一次即可(日线信号, 非盘中轮询)。

⚠ 仅提醒不下单(港股+杠杆下单路径未接); 纯只读行情 + Telegram。

用法:
  python3 signal_swing.py          # 跑一次, 推当天发生转档的票
  python3 signal_swing.py --status # 不只转档, 打印所有票当前波段状态
"""
import os, sys
import urllib.request, urllib.parse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from longport.openapi import Config, QuoteContext, Period, AdjustType

REDUCE_DEV = 2.93
# 票池: symbol -> (名称, 波段历史收益备注)  波段=20EMA trail
SWING_POOL = {
    "7747.HK": "三星2x(波段20E回测+725%)", "7709.HK": "海力士2x(+413%)",
    "SOXL.US": "半导体3x(+474%)", "NVDL.US": "英伟达2x", "MSFL.US": "微软2x", "TSLL.US": "特斯拉2x",
    "MXL.US": "MaxLinear(+308%)", "NBIS.US": "Nebius(+253%)", "AAOI.US": "AAOI(+161%)",
    "DELL.US": "Dell", "SNDK.US": "SanDisk", "MU.US": "美光",
}


def send_telegram(title, body):
    token = os.environ.get("TELEGRAM_BOT_TOKEN"); chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print(f"\n[无TG·本地打印]\n{title}\n{body}\n"); return
    data = urllib.parse.urlencode({"chat_id": chat, "text": f"{title}\n{body}"}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, method="POST"), timeout=10)
        print(f"   ✅ TG已发: {title}")
    except Exception as e:
        print(f"   ⚠️ TG失败: {e}")


def pull(ctx, sym, n=260):
    adj = AdjustType.ForwardAdjust if sym.endswith(".US") else AdjustType.NoAdjust
    b = list(ctx.candlesticks(sym, Period.Day, n, adj))
    if not b:
        return None
    df = pd.DataFrame({"H": [float(x.high) for x in b], "L": [float(x.low) for x in b],
                       "C": [float(x.close) for x in b]}, index=[x.timestamp for x in b]).sort_index()
    return df


def analyze(df):
    """返回最新状态 dict 或 None。检测当日转档信号。"""
    c, h, l = df.C, df.H, df.L
    if len(c) < 60:
        return None
    e10 = c.ewm(span=10).mean(); e20 = c.ewm(span=20).mean(); e21 = c.ewm(span=21).mean()
    s50 = c.rolling(50).mean(); s200 = c.rolling(200).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    px = c.iloc[-1]; e20n = e20.iloc[-1]; e20p = e20.iloc[-2]; pxp = c.iloc[-2]
    dev21 = (px - e21.iloc[-1]) / atr.iloc[-1]
    uptrend = (px > s50.iloc[-1]) and (pd.isna(s200.iloc[-1]) or px > s200.iloc[-1])
    in_trend = px > e20n * 0.99
    # 转档检测
    buy = (pxp <= e20p) and (px > e20n) and uptrend
    exit_ = (pxp >= e20p * 0.99) and (px < e20n * 0.99)
    reduce = in_trend and dev21 >= REDUCE_DEV
    sig = "买入" if buy else ("退出" if exit_ else ("减仓" if reduce else ("持有" if in_trend else "观望")))
    return dict(px=px, e20=e20n, e10=e10.iloc[-1], s50=s50.iloc[-1], dev21=dev21,
                stop=e20n * 0.99, sig=sig, in_trend=in_trend)


def main():
    status = "--status" in sys.argv
    ctx = QuoteContext(Config.from_env())
    print("📊 波段择时信号 (日线·仅提醒不下单)")
    fired = 0
    for sym, note in SWING_POOL.items():
        df = pull(ctx, sym)
        if df is None:
            print(f"   {sym} 无数据"); continue
        a = analyze(df)
        if a is None:
            continue
        emoji = {"买入": "🟢🆕", "退出": "🔴", "减仓": "🟠", "持有": "🔵", "观望": "⚪"}[a["sig"]]
        line = (f"{emoji} {sym} {note.split('(')[0]} | {a['sig']} | 价{a['px']:.2f} "
                f"20E止损{a['stop']:.2f} 距21E {a['dev21']:+.1f}ATR")
        if status:
            print("  " + line)
        # 只对转档(买入/退出/减仓)发 Telegram
        if a["sig"] in ("买入", "退出", "减仓"):
            fired += 1
            risk = (a["px"] - a["stop"]) / a["px"] * 100
            title = f"📈 波段{a['sig']}: {sym} {note.split('(')[0]}"
            body = (f"信号: 波段{a['sig']}  | {note}\n"
                    f"现价 {a['px']:.2f} · 20EMA跟踪止损 {a['stop']:.2f} ({-risk:.1f}%)\n"
                    f"距21EMA {a['dev21']:+.1f} ATR" + (" (涨过头, 减1/3~1/2)" if a['sig'] == '减仓' else "") + "\n"
                    f"⚠️ 波段·请手动下单(本系统不自动执行)")
            send_telegram(title, body)
    if not status and fired == 0:
        print("   今日无波段转档信号 (用 --status 看所有票当前状态)")


if __name__ == "__main__":
    main()
