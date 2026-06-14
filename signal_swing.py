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
import os, sys, json
import urllib.request, urllib.parse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from longport.openapi import Config, QuoteContext, Period, AdjustType

REDUCE_DEV = 2.93
RANK_CSV = Path(__file__).parent / "output" / "swing_ranked.csv"
STATE_JSON = Path(__file__).parent / "output" / "swing_alerted.json"   # 去重: {sym: "barDate:信号"}


def load_state():
    try:
        return json.loads(STATE_JSON.read_text())
    except Exception:
        return {}


def save_state(s):
    try:
        STATE_JSON.write_text(json.dumps(s, ensure_ascii=False))
    except Exception:
        pass


def load_pool():
    """从 swing_ranked.csv 读全票最优跟踪线; 只收 ✅波段好 + ⚠️一般 (❌不适合的剔除)。
       返回 {symbol: dict(name, ma, ret, calmar, verdict)}。"""
    if not RANK_CSV.exists():
        return {}
    df = pd.read_csv(RANK_CSV)
    pool = {}
    for _, r in df.iterrows():
        if str(r["verdict"]).startswith("❌"):
            continue
        pool[r["symbol"]] = dict(name=r["name"], ma=r["best_ma"], ret=r["swing_ret"],
                                 calmar=r["calmar"], verdict=r["verdict"])
    return pool


SWING_POOL = load_pool()


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


def analyze(df, ma_choice="20E"):
    """用该票最优跟踪线(10E/20E/50S)检测当日波段转档信号。返回状态 dict 或 None。"""
    c, h, l = df.C, df.H, df.L
    if len(c) < 60:
        return None
    e21 = c.ewm(span=21).mean(); s50full = c.rolling(50).mean(); s200 = c.rolling(200).mean()
    ma = {"10E": c.ewm(span=10).mean(), "20E": c.ewm(span=20).mean(), "50S": s50full}.get(ma_choice, c.ewm(span=20).mean())
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    px = c.iloc[-1]; man = ma.iloc[-1]; map_ = ma.iloc[-2]; pxp = c.iloc[-2]
    dev21 = (px - e21.iloc[-1]) / atr.iloc[-1]
    uptrend = (px > s50full.iloc[-1]) and (pd.isna(s200.iloc[-1]) or px > s200.iloc[-1])
    in_trend = px > man * 0.99
    buy = (pxp <= map_) and (px > man) and uptrend
    exit_ = (pxp >= map_ * 0.99) and (px < man * 0.99)
    reduce = in_trend and dev21 >= REDUCE_DEV
    sig = "买入" if buy else ("退出" if exit_ else ("减仓" if reduce else ("持有" if in_trend else "观望")))
    return dict(px=px, ma=man, ma_choice=ma_choice, dev21=dev21, stop=man * 0.99, sig=sig,
                in_trend=in_trend, bar_date=str(c.index[-1].date()))


def main():
    status = "--status" in sys.argv
    if not SWING_POOL:
        print("⚠️ 未找到 output/swing_ranked.csv, 先跑 python3 swing_backtest.py"); return
    ctx = QuoteContext(Config.from_env())
    print(f"📊 波段择时信号 (日线·仅提醒不下单) | 池 {len(SWING_POOL)} 只(各用最优跟踪线)")
    state = load_state()
    fired = 0
    # 按 Calmar 排序展示
    items = sorted(SWING_POOL.items(), key=lambda kv: -kv[1]["calmar"])
    for sym, cfg in items:
        df = pull(ctx, sym)
        if df is None:
            continue
        a = analyze(df, cfg["ma"])
        if a is None:
            continue
        emoji = {"买入": "🟢🆕", "退出": "🔴", "减仓": "🟠", "持有": "🔵", "观望": "⚪"}[a["sig"]]
        if status:
            print(f"  {emoji} {sym:9}{cfg['name'][:9]:10} {a['sig']:4} 价{a['px']:.2f} "
                  f"{cfg['ma']}止损{a['stop']:.2f} 距21E{a['dev21']:+.1f}ATR (波段{cfg['ret']:+.0f}%/Calmar{cfg['calmar']})")
        if a["sig"] in ("买入", "退出", "减仓"):
            stamp = f"{a['bar_date']}:{a['sig']}"
            if state.get(sym) == stamp:
                continue   # 同一bar同一信号已推过, 防重复(每日自动跑)
            state[sym] = stamp
            fired += 1
            risk = (a["px"] - a["stop"]) / a["px"] * 100
            title = f"📈 波段{a['sig']}: {sym} {cfg['name']}"
            body = (f"信号: 波段{a['sig']} ({cfg['verdict']}, 历史波段{cfg['ret']:+.0f}%/Calmar{cfg['calmar']})\n"
                    f"现价 {a['px']:.2f} · {cfg['ma']}跟踪止损 {a['stop']:.2f} ({-risk:.1f}%)\n"
                    f"距21EMA {a['dev21']:+.1f} ATR" + (" (涨过头, 减1/3~1/2)" if a['sig'] == '减仓' else "") + "\n"
                    f"⚠️ 波段·请手动下单(本系统不自动执行)")
            send_telegram(title, body)
    save_state(state)
    if not status and fired == 0:
        print("   今日无波段转档信号 (用 --status 看所有票当前状态)")


if __name__ == "__main__":
    main()
