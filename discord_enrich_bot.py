#!/usr/bin/env python3
"""
discord_enrich_bot.py — 监听 Discord #期权-波段-enrich 信号 → 解析 → LongPort 模拟盘买期权 + 仓位管理。

链路: 站长转发(bot id锁死) 发英文原文 → enrich_parser 严格五要素解析
      → BUY: 限价买入 N 张 → 成交后自动挂止盈 → EXIT信号跟随平仓 → 到期日强平

🔒 安全护栏 (缺一不跑):
  1. 启动时 LongPort 模拟盘三重校验 (JWT.ac / JWT.ik / API channel 全=lb_papertrading), 不过即退出
  2. 只认 频道ID+作者ID 白名单 (昵称可仿冒, ID不可)
  3. 默认 DRY_RUN; ENRICH_LIVE=true 才真下(仍是模拟盘)
  4. 去重: 同一期权同一天只开一次; 消息ID去重
  5. 限价单 only, 单张权利金>MAX_PREMIUM 拒绝; 已到期信号跳过

仓位管理 (无期权行情权限 → 全部不依赖盯价):
  · 止盈: 入场成交 → 挂 GTC 限价卖 (TP_MULT×成本, 默认2.0=+100%), 卖出 半仓(2张卖1张; 1张全卖)
  · 出场跟随: 站长 EXIT 信号 (scaling out/all out/stopped...) → 撤止盈单 + 剩仓市价全平
  · 到期强平: 到期日 15:40 ET 剩仓市价全平, 防归零/行权
  · 天然风控: 买期权最大亏=权利金 (每笔 ≤ MAX_PREMIUM×100×张数)

用法:
  python3 discord_enrich_bot.py                                  # DRY_RUN
  ENRICH_LIVE=true OPTION_CONTRACTS=2 python3 discord_enrich_bot.py   # 模拟盘真下单
环境: DISCORD_BOT_TOKEN(必须) / OPTION_CONTRACTS(默认1) / MAX_PREMIUM(默认5.0) / TP_MULT(默认2.0)
      / DISCORD_WEBHOOK_URL(可选回报推送)
"""
import os, re, sys, json, base64, time
from datetime import datetime, date, time as dtime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")

import discord
from discord.ext import tasks
from enrich_parser import parse_signal, to_longport_symbol
from notify import push_discord
from llm_classifier import classify as llm_classify

LLM_ON = os.environ.get("LLM_CLASSIFY", "on").lower() != "off"   # LLM语义层(claude CLI); off退纯规则
import threading
_handle_lock = threading.Lock()   # LLM调用期间串行处理消息, 防持仓并发写

# ── 白名单 (2026-07-14 实测抓取, 锁ID) ──
CHANNEL_ID = 1392361900217602108          # #期权-波段-enrich
AUTHOR_ID  = 1392020997393088542          # 站长转发 (bot)
ANDY_CHANNEL_ID = 1523725658935656448     # #andy-option — 仅观察记录, 绝不下单

LIVE = os.environ.get("ENRICH_LIVE", "").lower() == "true"
CONTRACTS = int(os.environ.get("OPTION_CONTRACTS", "1"))
MAX_PREMIUM = float(os.environ.get("MAX_PREMIUM", "5.0"))
TP_MULT = float(os.environ.get("TP_MULT", "2.0"))     # 止盈倍数 (2.0 = +100%)
STOP_MULT = float(os.environ.get("STOP_MULT", "0.7")) # lotto止损 (0.7=-30%; lotto会归零必须止损)
SWING_STOP_MULT = float(os.environ.get("SWING_STOP_MULT", "0.5"))  # 波段单兜底 (-50%; 抗回撤跟他出场, 网格+30.3%)
LOTTO_CONTRACTS = int(os.environ.get("LOTTO_CONTRACTS", "1"))  # 歧义/lotto单张数(POSITION_USD=0时用)
POSITION_USD = float(os.environ.get("POSITION_USD", "0"))   # >0: 固定金额模式(旧)
LOTTO_USD = float(os.environ.get("LOTTO_USD", "0")) or (POSITION_USD / 5)
POSITION_FRAC = float(os.environ.get("POSITION_FRAC", "0"))  # >0: 按账户净值比例, 常规单=净值×此值
LOTTO_FRAC = float(os.environ.get("LOTTO_FRAC", "0.3333"))   # lotto/歧义=净值×此值
ZERO_DTE_FRAC = float(os.environ.get("ZERO_DTE_FRAC", "0.10"))  # 0DTE更小: 净值×1/10 (归零常态,限损)
OI_CAP_PCT = 0.10   # 流动性帽: 张数≤未平仓量10% (防模拟盘假成交失真)
# ── QA加固常量 (2026-07-20 对抗性QA: 8猎手+逐条核实) ──
OI_UNKNOWN_CAP = int(os.environ.get("OI_UNKNOWN_CAP", "20"))     # OI拿不到=流动性未知 → 保守帽(原为不封顶, $0.05票可下6666张)
# 全账户在险权利金上限(原无上限: 一天3条常规信号=1.5×净值, 第3条起被券商购买力不足随机拒单)。
# ⚠️ 这是策略参数不是纯安全参数: 设太低会系统性漏单, 而回测的+16%/笔是在"无上限"下跑出来的。
# 默认1.0 = 最多压满净值(允许2笔常规单并发), 只挡掉>100%杠杆和随机拒单; 设0则完全关闭该闸。
MAX_GROSS_FRAC = float(os.environ.get("MAX_GROSS_FRAC", "1.0"))
MIN_TICK = 0.01     # 期权最小报价档: 止损价低于此值则永不触发(低价期权-60%止损失效)
OPT_FAIL_ALERT = int(os.environ.get("OPT_FAIL_ALERT", "5"))      # 期权报价连续失败N轮 → 告警(模拟盘止损唯一通道)

# ── 出场模式 (2026-07-19 最终定稿, 用户拍板; 复盘+网格+对抗审查三轮定案) ──
# mechanical = 机械出场(无AI): +30%卖⅓ → +60%再卖⅓ → 剩⅓runner(-60%初始止损全程, 无保本)
#              → runner摸到+60%(武装)后启动【标的】15分9ema连破2根拖尾 → 到期强平
#              回测(2-7月n=127): 加权+16%/笔 真实K+30% 胜率52% 中位+10% 最差月-6% (bt_mechanical_v2)
# mirror     = 跟站长出场(LLM语义层) — 旧模式, 保留可回切
EXIT_MODE = os.environ.get("EXIT_MODE", "mirror").lower()
MECH_TP_MULT = float(os.environ.get("MECH_TP_MULT", "1.3"))     # 一档止盈 +30% 卖⅓
MECH_TP2_MULT = float(os.environ.get("MECH_TP2_MULT", "1.6"))   # 二档止盈 +60% 卖⅓; 触及即武装9ema
MECH_STOP_MULT = float(os.environ.get("MECH_STOP_MULT", "0.4")) # 初始止损 -60% 全程有效(无保本移动)
MECH_EMA_MIN = int(os.environ.get("MECH_EMA_MIN", "15"))        # 9ema时间框架(分钟)
MECH_EMA_N = int(os.environ.get("MECH_EMA_N", "2"))             # 连破N根(已完成bar)出runner
ENTRY_TTL_SEC = int(os.environ.get("ENTRY_TTL_SEC", "1200"))    # 在途入场单TTL: 20分钟未成交撤单
# (审计发现+MSFT复盘: 挂一天的限价单只会在期权崩盘穿价时成交=专门接刀; "不追高"的另一半是"不接刀")
if EXIT_MODE == "mechanical":
    LLM_ON = False   # 机械模式全程无LLM: 进场纯规则解析, 出场纯技术, 站长出场消息仅提醒
ET = ZoneInfo("America/New_York")

OUT = Path(__file__).parent / "output"
SEEN_JSON = OUT / "enrich_seen.json"
POS_JSON = OUT / "enrich_positions.json"
LOG = OUT / "enrich_bot.log"
JOURNAL = OUT / "enrich_journal.jsonl"   # 结构化交易日志 (回测原料, 入库)


def journal(**kv):
    """追加一行结构化事件 (JSONL)。永不抛错。"""
    kv["ts"] = datetime.now(ZoneInfo("Asia/Singapore")).isoformat(timespec="seconds")
    try:
        OUT.mkdir(exist_ok=True)
        with open(JOURNAL, "a") as f:
            f.write(json.dumps(kv, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass

_trade_ctx = None
_quote_ctx = None    # OPRA期权行情 (止损轮询用); 拿不到就自动关止损


_rl_until = 0.0   # 429退避: 限流风暴期间暂停报价/查单30秒 (7/15八连429教训)


def _rl_hit(e) -> bool:
    global _rl_until
    if "429" in str(e) or "limited" in str(e).lower():
        _rl_until = time.time() + 30
        log("⏳ API限流 → 退避30秒")
        return True
    return False


def _option_last(osi: str):
    """期权最新价 (需OPRA权限)。失败返回 None。"""
    global _quote_ctx
    if time.time() < _rl_until:
        return None
    try:
        if _quote_ctx is None:
            from longport.openapi import Config, QuoteContext
            _quote_ctx = QuoteContext(Config.from_env())
        o = _quote_ctx.option_quote([osi])
        return float(o[0].last_done) if o else None
    except Exception as e:
        _rl_hit(e)
        return None


def _osi(ticker, expiry_iso_or_date, right, strike):
    d = expiry_iso_or_date
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return f"{ticker}{d:%y%m%d}{right}{int(round(strike * 1000)):06d}.US"


def _tp_params(remain: int, avg: float):
    """(止盈张数, 止盈价) — 按出场模式。mechanical: 一档卖⅓ @+30%; mirror: 卖半 @TP_MULT。"""
    if EXIT_MODE == "mechanical":
        return max(1, round(remain / 3)), round(avg * MECH_TP_MULT, 2)
    return (max(1, remain // 2) if remain >= 2 else remain), round(avg * TP_MULT, 2)


def _mech_split(filled: int):
    """机械模式两档张数: (q1@+30%, q2@+60%)。q=1→(1,0); q=2→(1,1,无runner); q≥3→各⅓留runner。"""
    q1 = max(1, round(filled / 3))
    q2 = min(max(1, round(filled / 3)), filled - q1) if filled - q1 >= 1 else 0
    return q1, q2


_ema_cache = {}   # ticker -> (checked_at_epoch, break_count)
_optfail = {}     # osi -> 期权报价连续失败轮数 (止损通道健康度; 模拟盘唯一止损通道)
_closing = set()  # 正在平仓的osi — 只存内存, 绝不落盘(落盘会导致崩溃重启后永久无法平仓)
_TERMINAL = frozenset(("Filled", "Canceled", "Expired", "Rejected", "PartialWithdrawal"))
_CANCELLED = frozenset(("Canceled", "Expired", "Rejected", "PartialWithdrawal"))
_alert_noexit_warned = [False]


def _alert(msg: str):
    """关键告警: 记日志 + 推Discord。未配webhook时显式标注"告警无出口",
    否则整个人工兜底层是空的却看不出来(实测本机从未配置过 DISCORD_WEBHOOK_URL)。"""
    log(msg)
    try:
        if not push_discord(msg) and not _alert_noexit_warned[0]:
            _alert_noexit_warned[0] = True
            log("   ⚠️ 告警无出口: 未配置 DISCORD_WEBHOOK_URL/URLS, 所有告警只会留在本日志")
    except Exception as e:
        log(f"   ⚠️ 告警推送异常: {e}")


def _ema15_break_count(ticker: str):
    """标的15分9ema连续破位根数(只算已完成bar)。5分钟节流/票。失败返回None(不出场)。"""
    global _quote_ctx
    now = time.time()
    hit = _ema_cache.get(ticker)
    if hit and now - hit[0] < 300:
        return hit[1]
    try:
        if _quote_ctx is None:
            from longport.openapi import Config, QuoteContext
            _quote_ctx = QuoteContext(Config.from_env())
        from longport.openapi import Period, AdjustType
        bars = _quote_ctx.candlesticks(f"{ticker}.US", Period.Min_15, 80, AdjustType.ForwardAdjust)
        from datetime import timezone as _tz, timedelta as _td
        now_utc = datetime.now(_tz.utc)
        closes = [float(b.close) for b in bars
                  if b.timestamp.astimezone(_tz.utc) + _td(minutes=MECH_EMA_MIN) <= now_utc]  # 剔进行中bar
        if len(closes) < 15:
            return None
        k = 2 / 10; e = closes[0]; emas = []
        for c in closes:
            e = c * k + e * (1 - k); emas.append(e)
        cnt = 0
        for c, em in zip(reversed(closes), reversed(emas)):
            if c < em:
                cnt += 1
            else:
                break
        _ema_cache[ticker] = (now, cnt)
        return cnt
    except Exception as e:
        log(f"   15m9ema获取失败 {ticker}: {e}")
        return None


def resolve_direction(s):
    """缺方向的信号: 拉同行权价 call/put 实时报价, 谁的价跟信号权利金匹配(0.4~2.2x窗口)
       且恰好只有一边匹配 → 那边就是方向。返回 ('C'/'P', 说明) 或 (None, 原因)。"""
    global _quote_ctx
    try:
        if _quote_ctx is None:
            from longport.openapi import Config, QuoteContext
            _quote_ctx = QuoteContext(Config.from_env())
        syms = [_osi(s.ticker, s.expiry, r, s.strike) for r in ("C", "P")]
        px = {}
        for o in _quote_ctx.option_quote(syms):
            px[o.symbol] = float(o.last_done or 0)   # OptionQuote只有last_done, 无bid/ask(实测)
        pc, pp = px.get(syms[0], 0), px.get(syms[1], 0)
        if pc <= 0 and pp <= 0:
            return None, "两边都无报价"
        inwin = [r for r, p in (("C", pc), ("P", pp))
                 if p > 0 and 0.4 <= p / s.limit_price <= 2.2]
        if len(inwin) != 1:
            return None, f"消歧失败(C={pc} P={pp} 信号${s.limit_price}, 匹配{len(inwin)}边)"
        return inwin[0], f"C={pc} P={pp} vs 信号${s.limit_price} → {inwin[0]}"
    except Exception as e:
        return None, f"报价异常: {e}"


_last_equity = [None]

def account_equity_usd():
    """账户净值(USD)。HKD净值按7.8折算。失败用上次值; 从未成功→None。"""
    try:
        for b in _trade_ctx.account_balance():
            cur, na = str(b.currency), float(b.net_assets)
            if na > 0:
                usd = na if cur == "USD" else na / 7.8
                _last_equity[0] = usd
                return usd
    except Exception:
        pass
    return _last_equity[0]


def size_qty(premium: float, budget: float, osi: str, fallback: int) -> tuple:
    """按预算算张数(带OI流动性帽)。返回 (张数, 说明)。budget<=0 → 固定张数fallback。"""
    if budget <= 0:
        return max(1, int(fallback)), "固定张数"
    qty = max(1, int(budget // (premium * 100)))
    note = f"${budget:.0f}预算"
    oi = 0
    try:
        global _quote_ctx
        if _quote_ctx is None:
            from longport.openapi import Config, QuoteContext
            _quote_ctx = QuoteContext(Config.from_env())
        o = _quote_ctx.option_quote([osi])
        oi = int(o[0].open_interest or 0) if o else 0
    except Exception:
        oi = 0
    if oi > 0:
        cap = max(1, int(oi * OI_CAP_PCT))
        if qty > cap:
            note += f", OI帽{oi}×{OI_CAP_PCT:.0%}={cap}张(原{qty})"
            qty = cap
    elif qty > OI_UNKNOWN_CAP:
        # QA: 原实现 OI=0/None 时静默不封顶($0.05票能下6666张, 日志还看不出帽子没生效)
        note += f", ⚠️OI未知→保守帽{OI_UNKNOWN_CAP}张(原{qty})"
        qty = OI_UNKNOWN_CAP
    else:
        note += ", OI未知"
    return max(1, int(qty)), note


def us_rth_now() -> bool:
    """美股常规时段(期权只在RTH交易): 周一-五 9:15-16:20 ET (留边)。"""
    now = datetime.now(ET)
    return now.weekday() < 5 and dtime(9, 15) <= now.time() <= dtime(16, 20)


def log(msg: str):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        OUT.mkdir(exist_ok=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _load(p: Path, strict: bool = False) -> dict:
    """读状态。不存在→{}; 损坏→先试.bak; 仍不行: strict=True 抛错拒启, 否则告警+返回{}。
    QA-CRITICAL: 原实现裸except返回{}, 半截JSON=bot静默忘记全部持仓(止损没了/runner裸奔/挂单变孤儿)。
    仓位/去重表宁可响亮地起不来, 也绝不带着空表上线。
    但 strict 只能用于 POS_JSON/SEEN_JSON —— 消息锚点(LAST_MSG_JSON)每条消息都写, 且 bump_last
    在 on_message 第一行【裸调】, 抛错会让所有信号静默丢失, 必须走宽松模式(复审发现)。"""
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log(f"🚨 状态文件损坏 {p.name}: {e}")
        bak = p.with_suffix(p.suffix + ".bak")
        if bak.exists():
            try:
                d = json.loads(bak.read_text())
                log(f"🚨 已从 {bak.name} 恢复 {len(d)} 条 — 请人工核对券商侧实际持仓!")
                try:
                    push_discord(f"🚨 enrich状态文件 {p.name} 损坏, 已从备份恢复{len(d)}条 — 请核对券商实际持仓")
                except Exception:
                    pass
                return d
            except Exception:
                pass
        if not strict:
            log(f"⚠️ {p.name} 损坏且无备份, 非关键状态 → 按空表继续(最坏是重放一次追赶)")
            return {}
        try:
            (OUT / "REFUSED_TO_START").write_text(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} 状态文件 {p.name} 损坏且无可用备份, bot拒绝启动。\n"
                f"请人工核对券商侧实际持仓后, 修复或删除该文件再启动。\n")
        except Exception:
            pass
        banner = f"🚨🚨 状态文件 {p.name} 损坏且无可用备份 — bot拒绝启动, 请人工核对券商持仓 🚨🚨"
        print("\n" + "=" * 78 + f"\n{banner}\n" + "=" * 78 + "\n", flush=True)
        log(banner)
        try:
            push_discord(banner)
        except Exception:
            pass
        raise RuntimeError(f"状态文件 {p} 损坏且无可用备份 — 拒绝以空状态启动(会导致在手持仓永久裸奔)")


def _save(p: Path, d: dict):
    """原子落盘: 先备份上一版 → 写tmp → fsync → os.replace(原子替换)。
    QA-CRITICAL: 原实现 write_text 是"先截断再写", 写一半被kill=文件损坏。"""
    try:
        OUT.mkdir(exist_ok=True)
        if p.exists():
            try:                      # .bak 同样原子化: 直写会被kill打断产生半截备份
                _bt = p.with_suffix(p.suffix + ".bak.tmp")
                _bt.write_bytes(p.read_bytes())
                os.replace(_bt, p.with_suffix(p.suffix + ".bak"))
            except Exception:
                pass
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps(d, ensure_ascii=False, indent=1))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)          # 原子: 要么旧版要么新版, 不存在半截
    except Exception as e:
        log(f"⚠️ 状态落盘失败 {p.name}: {e}")


def verify_paper_trading() -> bool:
    """铁律: 三处独立验证全是 lb_papertrading 才放行。"""
    try:
        tok = os.environ["LONGPORT_ACCESS_TOKEN"]
        part = tok.split(".")[1]
        p = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        ac, ik = p.get("ac"), p.get("ik", "")
        from longport.openapi import Config, TradeContext
        global _trade_ctx
        _trade_ctx = TradeContext(Config.from_env())
        chans = {c.account_channel for c in _trade_ctx.stock_positions().channels}
        ok = ac == "lb_papertrading" and ik.startswith("lb_papertrading_") and chans == {"lb_papertrading"}
        log(f"模拟盘校验: ac={ac} ik前缀={ik[:16]} channels={chans} → {'✅通过' if ok else '❌不通过'}")
        return ok
    except Exception as e:
        log(f"模拟盘校验异常: {e}")
        return False


# ── 下单原语 (全部模拟盘) ──

def _submit(osi: str, side_buy: bool, qty: int, price: float | None, tif_gtc=False,
            remark="enrich", trigger: float | None = None):
    """trigger→MIT触价市价 / price→LO限价 / 都无→MO市价。返回 (ok, order_id或错误)。"""
    from longport.openapi import OrderType, OrderSide, TimeInForceType
    qty = int(qty)
    if qty <= 0:      # QA: 原无防线, budget=0且OPTION_CONTRACTS=0时会真提交qty=0的单并占住osi槽位
        log(f"   🚫 拒绝提交 {osi} qty={qty} (非正数)")
        return False, f"qty<=0 ({qty})"
    try:
        kw = dict(symbol=osi,
                  side=OrderSide.Buy if side_buy else OrderSide.Sell,
                  submitted_quantity=Decimal(str(qty)),
                  time_in_force=TimeInForceType.GoodTilCanceled if tif_gtc else TimeInForceType.Day,
                  remark=remark)
        if trigger is not None:
            kw.update(order_type=OrderType.MIT, trigger_price=Decimal(f"{trigger:.2f}"))
        elif price is not None:
            kw.update(order_type=OrderType.LO, submitted_price=Decimal(f"{price:.2f}"))
        else:
            kw.update(order_type=OrderType.MO)
        resp = _trade_ctx.submit_order(**kw)
        return True, resp.order_id
    except Exception as e:
        return False, str(e)


_MIT_OK = None   # None=未探测 / True=真实账户支持触价单 / False=模拟盘604050不支持


def ensure_protection(positions: dict, osi: str, p: dict):
    """给剩仓配保护腿(自适应):
       ① 优先挂券商侧MIT止损(bot死了也在) — 真实账户支持
       ② 模拟盘不支持MIT → 止盈挂券商侧限价(抓尖峰) + 止损靠轮询兜底"""
    global _MIT_OK
    remain = p.get("filled", 0) - p.get("sold", 0)
    if remain <= 0 or p.get("avg", 0) <= 0:
        return
    _sm = p.get("stop_mult", STOP_MULT)
    if _sm > 0 and not p.get("stop_order_id") and _MIT_OK is not False:
        trig = round(p["avg"] * _sm, 2)
        ok, r = _submit(osi, side_buy=False, qty=remain, price=None, tif_gtc=True,
                        remark="stop", trigger=trig)
        if ok:
            _MIT_OK = True
            p["stop_order_id"], p["stop_qty"] = r, remain
            log(f"🛡️ {osi} 券商侧止损已挂: {remain}张 触发${trig} (-{(1-_sm)*100:.0f}%)")
            journal(ev="stop_place", osi=osi, trigger=trig, qty=remain, order_id=r)
        elif "604050" in str(r) or "not supported" in str(r).lower():
            _MIT_OK = False
            log("ℹ️ 模拟盘不支持触价单 → 止损轮询兜底, 止盈挂券商侧限价 (真实账户会自动切回止损常驻)")
        else:
            log(f"⚠️ {osi} 止损挂单失败: {r}")
    # 回退模式: 无券商侧止损 → 止盈限价挂券商侧(抓尖峰)
    if EXIT_MODE == "mechanical" and not p.get("stop_order_id"):
        # 两档GTC同时挂: ⅓@+30%, ⅓@+60% (二档成交=runner武装9ema)
        q1, q2 = _mech_split(p.get("filled", 0))
        if q1 >= 1 and not p.get("tp_order_id") and not p.get("tp1_done"):
            px1 = round(p["avg"] * MECH_TP_MULT, 2)
            ok, r = _submit(osi, side_buy=False, qty=q1, price=px1, tif_gtc=True, remark="tp1")
            if ok:
                p["tp_order_id"], p["tp_qty"] = r, q1
                log(f"🎯 {osi} 挂一档止盈: 卖{q1}张 @ ${px1} (+30%)")
                journal(ev="tp_place", osi=osi, px=px1, qty=q1, order_id=r)
            else:
                log(f"⚠️ {osi} 一档止盈挂单失败: {r}")
        if q2 >= 1 and not p.get("tp2_order_id") and not p.get("tp2_done"):
            px2 = round(p["avg"] * MECH_TP2_MULT, 2)
            ok, r = _submit(osi, side_buy=False, qty=q2, price=px2, tif_gtc=True, remark="tp2")
            if ok:
                p["tp2_order_id"], p["tp2_qty"] = r, q2
                log(f"🎯 {osi} 挂二档止盈: 卖{q2}张 @ ${px2} (+60%, 成交即武装9ema)")
                journal(ev="tp2_place", osi=osi, px=px2, qty=q2, order_id=r)
            else:
                log(f"⚠️ {osi} 二档止盈挂单失败: {r}")
    elif not p.get("stop_order_id") and not p.get("tp_order_id") and not p.get("reduced"):
        tp_qty, tp_px = _tp_params(remain, p["avg"])
        ok, r = _submit(osi, side_buy=False, qty=tp_qty, price=tp_px, tif_gtc=True, remark="tp")
        if ok:
            p["tp_order_id"], p["tp_qty"] = r, tp_qty
            log(f"🎯 {osi} 挂止盈: 卖{tp_qty}张 @ ${tp_px} (+{(tp_px/p['avg']-1)*100:.0f}%)")
            journal(ev="tp_place", osi=osi, px=tp_px, qty=tp_qty, order_id=r)
        else:
            log(f"⚠️ {osi} 止盈挂单失败: {r} (靠轮询/出场跟随/到期强平)")
    _save(POS_JSON, positions)


def cancel_stop(p: dict):
    if p.get("stop_order_id"):
        _cancel(p["stop_order_id"])
        p["stop_order_id"] = None


def _cancel(order_id: str):
    try:
        _trade_ctx.cancel_order(order_id)
        return True
    except Exception as e:
        log(f"   撤单失败 {order_id}: {e}")
        return False


def _order_state(order_id: str):
    """(状态名, 已成交张数, 成交均价) — 只读轮询, 不需要行情权限。"""
    if time.time() < _rl_until:
        return None, 0, 0.0
    try:
        od = _trade_ctx.order_detail(order_id)
        st = str(od.status).split(".")[-1]        # Filled/New/PartialFilled/Canceled/Expired/Rejected
        exq = int(od.executed_quantity or 0)
        exp = float(od.executed_price) if od.executed_price else 0.0
        return st, exq, exp
    except Exception as e:
        if not _rl_hit(e):
            log(f"   查单失败 {order_id}: {e}")
        return None, 0, 0.0


# ── 仓位管理 ──

def close_position(positions: dict, osi: str, reason: str):
    """撤所有挂单 + 剩仓市价全平。

    QA-CRITICAL 三修 (原实现可致裸空/超卖/弃仓):
      ①撤单前后必须与券商对账: _cancel返回值原本被丢弃, 撤失败时挂单还活着, 再市价卖=双卖裸空。
      ②止盈腿"部分成交后被撤"的成交量原本不计入sold, remain虚高 → 超卖。
      ③市价卖失败时原本照样 status=closed, 抛弃券商侧仍持有的多头, -60%止损从此空转。
    方向性原则: 宁可少卖(仍持多头, 亏损有界=权利金), 绝不多卖(裸空期权=无限风险)。
    """
    p = positions.get(osi)
    if not p or p["status"] == "closed" or osi in _closing:
        return
    # 防重入守卫必须是【内存态】: 若写进 p 再被 _save 落盘, 崩溃重启后该标志永久为真,
    # close_position 将永远早退 → 该仓位再也平不掉, 止损空转(自查发现, 与本次要修的bug同类)
    _closing.add(osi)
    try:
        log(f"🔻 平仓 {osi} ({reason})")
        # 撤三条保护腿, 每条都"撤完复查"把真实成交量记进 sold 再算 remain
        unresolved = []
        for id_key in ("tp_order_id", "tp2_order_id", "stop_order_id"):
            oid = p.get(id_key)
            if not oid:
                continue
            ok_c = _cancel(oid)
            st, exq, _ = _order_state(oid)          # 撤单后复查真实成交
            if st is None and not ok_c:
                # 撤失败且查不到真实成交 → 【绝不猜】。保留腿ID, 本轮放弃平仓, 下轮重试。
                # (曾经写成"假定该腿全成交", 会把sold顶到>=filled → remain<=0 → 直接标closed,
                #  结果一张没卖却认为已平, 券商侧多头彻底失管 — 比超卖更隐蔽的灾难; 自查+复审发现)
                unresolved.append(id_key)
                continue
            if exq > 0:
                p["sold"] = p.get("sold", 0) + exq
                log(f"   ℹ️ {osi} {id_key} 撤单前已成交{exq}张, 已记账(防超卖)")
            p[id_key] = None
        # 入场买腿: 只要还在途就必须撤。不能再用 status=='pending' 判断 ——
        # 部分成交时 status 已被改成 open 而 entry_order_id 仍在, 否则剩余买腿会在
        # 刚触发止损的暴跌里继续挂着接刀, 且仓位标closed后再没人管(复审发现)
        if p.get("entry_order_id"):
            if _cancel(p["entry_order_id"]):
                p["entry_order_id"] = None
            else:
                unresolved.append("entry_order_id")
        if unresolved:
            log(f"   ⚠️ {osi} 挂单未确认撤销 {unresolved} → 本轮不平仓, 保持{p['status']}, 下轮重试")
            journal(ev="close_deferred", osi=osi, unresolved=unresolved, reason=reason)
            _alert(f"⚠️ enrich {osi} 平仓推迟({reason}): {unresolved} 撤单+查单均失败, 下轮重试")
            _save(POS_JSON, positions)
            return
        remain = p.get("filled", 0) - p.get("sold", 0)
        if remain > 0:
            ok, r = _submit(osi, side_buy=False, qty=remain, price=None, remark=f"exit-{reason[:12]}")
            log(f"   市价卖 {remain} 张: {'✅' + str(r) if ok else '❌' + str(r)}")
            journal(ev="close_sell", osi=osi, qty=remain, reason=reason, order_id=(r if ok else None), ok=ok)
            push_discord(f"🔻 enrich平仓 {osi} ×{remain}张 ({reason}) {'✅' if ok else '❌' + str(r)}")
            if ok:
                p["sold"] = p.get("sold", 0) + remain
                p["status"] = "closed"
            else:
                # ③不弃仓: 保持open, 下一轮轮询会再次触发止损/强平重试
                log(f"   ⚠️ {osi} 市价平仓失败, 保持open等下轮重试(不标closed以免止损空转)")
                journal(ev="close_sell_failed", osi=osi, qty=remain, reason=reason, err=str(r))
                push_discord(f"🚨 enrich平仓失败 {osi} ×{remain}张 ({reason}) — 仓位仍在, 下轮重试: {r}")
        else:
            p["status"] = "closed"
        _save(POS_JSON, positions)
    finally:
        _closing.discard(osi)
        if positions.get(osi, {}).get("status") == "closed":
            _optfail.pop(osi, None)


def mirror_reduce(positions: dict, osi: str, level: str):
    """镜像站长减仓 (2张粒度): 首次部分减→卖1张留跑; 已减过→partial忽略/vague全平。"""
    p = positions.get(osi)
    if not p or p["status"] == "closed":
        return
    if p["status"] == "pending":            # 还没成交他就开始出 → 撤单/全清, 别再进
        close_position(positions, osi, "站长已出(未完全入场)")
        return
    remain = p.get("filled", 0) - p.get("sold", 0)
    if remain <= 0:
        p["status"] = "closed"; _save(POS_JSON, positions); return
    if not p.get("reduced"):
        if remain >= 2:
            half = max(1, remain // 2)
            cancel_stop(p)                 # 先撤止损防双卖, 卖完给剩仓重挂
            ok, r = _submit(osi, side_buy=False, qty=half, price=None, remark="mirror-scale")
            if ok:
                p["sold"] = p.get("sold", 0) + half
                p["reduced"] = True
                log(f"🪞 {osi} 镜像减仓: 市价卖{half}张 (单{r}), 剩{remain-half}张继续跑")
                journal(ev="mirror_sell", osi=osi, qty=half, order_id=r)
                push_discord(f"🪞 enrich镜像减仓 {osi} 卖{half}张, 剩{remain-half}张跑趋势")
            else:
                log(f"⚠️ {osi} 镜像减仓下单失败: {r}")
            ensure_protection(positions, osi, p)  # 剩仓重配保护腿
            _save(POS_JSON, positions)
        else:                               # 只剩1张, 部分减也=全出
            close_position(positions, osi, "站长减仓(仅剩1张)")
    else:
        if level == "vague":                # 已减过+模糊催促 → 保守全平
            close_position(positions, osi, "站长模糊出场(已减过)")
        else:                               # 已减过+再次partial → 他在连续撤退, 清runner
            close_position(positions, osi, "站长二次减仓")   # 回测: IBM305 -26%→+56%


def manage_positions(positions: dict):
    """轮询: 入场单成交→挂止盈; 止盈成交→记账; 到期日强平。"""
    now_et = datetime.now(ET)
    for osi, p in list(positions.items()):
        if p["status"] == "closed":
            continue
        time.sleep(0.4)   # API限速保护 (429防护, 昨夜网络抖动后重试风暴教训)
        # ① 入场单状态 (只要入场单还在途就持续对账, 不限于pending)
        # QA修①: 原 `if st is None: continue` — 查单失败(429退避/断网)会把本轮后面整段
        #   ④d入场TTL一并跳过 → 限价单挂一整天专接刀。TTL只依赖本地submitted_ts, 不该被API阻断。
        # QA修②: 原只在 Filled/终态 才转open, PartialFilled 两个分支都不进 → 永久卡pending →
        #   ①b不挂保护腿 + ④止损要求open被跳过 + ④d因filled>0不撤单 = 已成交部分整日裸奔。
        if p.get("entry_order_id"):
            st, exq, exp = _order_state(p["entry_order_id"])
            if st is not None:
                if exq > p.get("filled", 0):
                    p["filled"] = exq
                    # 成交均价拿不到时【保留旧avg, 绝不清零】: _order_state 对在途单可能返回
                    # executed_price 为空→exp=0.0, 写进去会让 avg=0 → ensure_protection 早退、
                    # ④轮询止损的 avg>0 前置失败 = 保护腿+止损全灭且落盘不可逆(复审发现)
                    if exp > 0:
                        p["avg"] = exp
                    log(f"📥 {osi} 入场成交 {exq}张 @ ${exp or p.get('avg')}")
                    journal(ev="entry_fill", osi=osi, qty=exq, avg=p.get("avg"))
                    push_discord(f"📥 enrich成交 {osi} ×{exq}张 @ ${p.get('avg')} (成本${p.get('avg',0)*100*exq:.0f})")
                if st in _TERMINAL:
                    p["entry_order_id"] = None          # 终态: 停止轮询该单
                    if exq > 0:
                        if p["status"] == "pending":
                            p["status"] = "open"
                            ensure_protection(positions, osi, p)   # 🛡️ MIT止损(真账户)或止盈限价(模拟盘)
                    elif p["status"] == "pending":
                        log(f"🗑️ {osi} 入场未成交已失效 ({st})")
                        p["status"] = "closed"
                elif exq > 0 and p["status"] == "pending":
                    # 部分成交且单仍在途: 立刻保护已成交部分, 剩余买腿交给 ④d TTL 撤
                    p["status"] = "open"
                    log(f"⚠️ {osi} 入场部分成交 {exq}张(单仍在途) → 先保护已成交部分, 余量待TTL撤")
                    journal(ev="entry_partial", osi=osi, qty=exq, avg=exp)
                    ensure_protection(positions, osi, p)
                _save(POS_JSON, positions)
        # ①b 保护腿对账 (审计bug#1: 纯事件驱动下ID丢失/挂单失效=永久裸奔; ensure_protection幂等, 每轮补齐缺失的腿)
        if p["status"] == "open" and p.get("filled", 0) - p.get("sold", 0) > 0 and p.get("avg", 0) > 0:
            ensure_protection(positions, osi, p)
        # ② 券商侧止损单状态 (真实账户模式)
        if p["status"] == "open" and p.get("stop_order_id"):
            st, exq, exp = _order_state(p["stop_order_id"])
            if st == "Filled":
                p["sold"] = p.get("sold", 0) + p.get("stop_qty", 0)
                p["stop_order_id"] = None
                p["status"] = "closed"
                log(f"🛑 {osi} 券商侧止损成交 {p.get('stop_qty')}张 @ ${exp}")
                journal(ev="stop_fill", osi=osi, px=exp, qty=p.get("stop_qty"))
                push_discord(f"🛑 enrich止损成交 {osi} ×{p.get('stop_qty')}张 @ ${exp}")
                _save(POS_JSON, positions)
                continue
            elif st in _CANCELLED:
                # 审计bug#2: 止损被外部撤销若不清ID, 轮询止损被stop_order_id条件跳过=止损通道全灭
                if exq > 0:
                    p["sold"] = p.get("sold", 0) + exq
                p["stop_order_id"] = None
                log(f"⚠️ {osi} 券商侧止损单失效({st}) → 清ID, 轮询止损接管+重挂")
                journal(ev="stop_order_lost", osi=osi, state=st)
                ensure_protection(positions, osi, p)
                _save(POS_JSON, positions)
        # ③ 券商侧一档止盈状态
        if p["status"] == "open" and p.get("tp_order_id"):
            st, exq, exp = _order_state(p["tp_order_id"])
            if st == "Filled":
                p["sold"] = p.get("sold", 0) + p.get("tp_qty", 0)
                p["tp_order_id"] = None
                p["reduced"] = True          # 止盈成交=完成首次减仓
                p["tp1_done"] = True
                log(f"💰 {osi} 一档止盈成交 {p.get('tp_qty')}张 @ ${exp}")
                journal(ev="tp_fill", osi=osi, px=exp, qty=p.get("tp_qty"))
                push_discord(f"💰 enrich止盈成交 {osi} ×{p.get('tp_qty')}张 @ ${exp} — 剩{p['filled']-p['sold']}张")
                if p["filled"] - p["sold"] <= 0:
                    p["status"] = "closed"
                _save(POS_JSON, positions)   # 机械模式: 不移保本, -60%初始止损全程 (最终定稿)
            elif st in _CANCELLED:
                if exq > 0:                  # 部分成交后被撤: 先记账防超卖 (审计bug#4)
                    p["sold"] = p.get("sold", 0) + exq
                p["tp_order_id"] = None      # 挂单失效 → 重配保护
                ensure_protection(positions, osi, p)
        # ③c 券商侧二档止盈状态 (机械模式; 成交=runner武装9ema)
        if p["status"] == "open" and p.get("tp2_order_id"):
            st, exq, exp = _order_state(p["tp2_order_id"])
            if st == "Filled":
                p["sold"] = p.get("sold", 0) + p.get("tp2_qty", 0)
                p["tp2_order_id"] = None
                p["tp2_done"] = True
                p["reduced"] = True
                p["armed"] = True            # 摸到+60% → runner进入9ema拖尾管理
                log(f"💰 {osi} 二档止盈成交 {p.get('tp2_qty')}张 @ ${exp} — runner已武装15m9ema")
                journal(ev="tp2_fill", osi=osi, px=exp, qty=p.get("tp2_qty"))
                push_discord(f"💰 enrich二档止盈 {osi} ×{p.get('tp2_qty')}张 @ ${exp} — 剩{p['filled']-p['sold']}张runner(9ema拖尾)")
                if p["filled"] - p["sold"] <= 0:
                    p["status"] = "closed"
                _save(POS_JSON, positions)
            elif st in _CANCELLED:
                if exq > 0:
                    p["sold"] = p.get("sold", 0) + exq
                p["tp2_order_id"] = None
                ensure_protection(positions, osi, p)
        # ③b 轮询止盈 (真实账户模式: 无止盈挂单时; 只管一档)
        remain = p.get("filled", 0) - p.get("sold", 0)
        if p["status"] == "open" and not p.get("reduced") and not p.get("tp_order_id") \
                and not p.get("tp2_order_id") and remain > 0 and p.get("avg", 0) > 0:
            last = _option_last(osi)
            tp_qty, tp_px = _tp_params(remain, p["avg"])
            if last is not None and last >= tp_px:
                cancel_stop(p)
                ok, r = _submit(osi, side_buy=False, qty=tp_qty, price=None, remark="tp-poll")
                if ok:
                    p["sold"] = p.get("sold", 0) + tp_qty
                    p["reduced"] = True
                    p["tp1_done"] = True
                    log(f"💰 {osi} 轮询止盈: 最新${last}≥${tp_px}, 市价卖{tp_qty}张")
                    journal(ev="tp_poll_sell", osi=osi, last=last, qty=tp_qty, order_id=r)
                    push_discord(f"💰 enrich止盈 {osi} ×{tp_qty}张 @≈${last} — 剩{remain-tp_qty}张跑趋势")
                if remain - tp_qty <= 0:
                    p["status"] = "closed"
                else:
                    ensure_protection(positions, osi, p)   # 机械: 补挂二档; 不移保本(-60%全程)
                _save(POS_JSON, positions)
        # ④ 轮询止损 (模拟盘唯一止损通道; 真实账户仅当MIT挂失败时兜底)
        _sm = p.get("stop_mult", STOP_MULT)
        if _sm > 0 and p["status"] == "open" and not p.get("stop_order_id") \
                and p.get("avg", 0) > 0 and p.get("filled", 0) - p.get("sold", 0) > 0:
            last = _option_last(osi)
            if last is None:
                # QA: 模拟盘不支持MIT触价单, 这是【唯一】止损通道。报价拿不到(OPRA权限/429/合约不识别)
                #     原本整段静默跳过, 启动横幅还在宣传"止损-60%全程", 用户看不出止损已经不存在。
                _n = _optfail.get(osi, 0) + 1
                _optfail[osi] = _n
                # 周期性重报(原为 ==N 精确相等, 失效5小时也只告警一次)
                if _n >= OPT_FAIL_ALERT and _n % OPT_FAIL_ALERT == 0:
                    journal(ev="stop_channel_down", osi=osi, fails=_n)
                    _alert(f"🚨 enrich {osi} 期权报价连续{_n}轮拿不到 → -{(1-_sm)*100:.0f}%止损通道失效, "
                           f"该仓位当前仅剩到期强平保护, 请人工盯")
            else:
                _optfail.pop(osi, None)
                # 报价顺带武装检查: 摸过+60%即武装9ema (TP2成交也会武装, 这里是兜底)
                if EXIT_MODE == "mechanical" and not p.get("armed") and last >= p["avg"] * MECH_TP2_MULT:
                    p["armed"] = True
                    log(f"🎯 {osi} 报价${last}≥成本×{MECH_TP2_MULT} → runner武装15m9ema")
                    journal(ev="runner_armed", osi=osi, last=last)
                    _save(POS_JSON, positions)
                # QA: 低价期权 avg×_sm 可能落在最小报价档以下($0.02×0.4=$0.008), last永远够不到 → 止损失效
                stop_px = max(MIN_TICK, p["avg"] * _sm)
                if last <= stop_px:
                    lab = "保本止损" if _sm >= 0.999 else f"止损-{(1-_sm)*100:.0f}%"
                    log(f"🛑 {osi} 轮询止损: 最新${last} ≤ ${stop_px:.2f} (成本${p['avg']}×{_sm})")
                    journal(ev="stop_trigger", osi=osi, last=last, avg=p["avg"], mult=_sm, stop_px=stop_px)
                    close_position(positions, osi, lab)
                    continue
        # ④c 机械模式: runner【武装后】守标的15分9ema, 连破N根(已完成bar)→全平
        #    (最终定稿: 摸到+60%才武装; 未武装的runner只受-60%止损+到期强平管, 防早盘回踩误洗肥尾)
        if EXIT_MODE == "mechanical" and p["status"] == "open" and p.get("armed") \
                and p.get("filled", 0) - p.get("sold", 0) > 0 and us_rth_now():
            cnt = _ema15_break_count(p["ticker"])
            if cnt is not None and cnt >= MECH_EMA_N:
                log(f"📉 {osi} 标的{p['ticker']} 15分9ema连破{cnt}根 → 趋势破位出场(runner已武装)")
                journal(ev="ema_exit", osi=osi, ticker=p["ticker"], break_count=cnt)
                close_position(positions, osi, f"15m9ema连破{cnt}")
                continue
        # ④d 在途入场单TTL: 挂20分钟未成交→撤(不接刀; MSFT接盘教训+审计bug)
        # QA: 原条件含 filled==0 → 部分成交的入场单永不撤, 剩余买腿整日挂着专接崩盘穿价
        if p.get("entry_order_id") and p.get("submitted_ts") \
                and time.time() - p["submitted_ts"] > ENTRY_TTL_SEC:
            _f = p.get("filled", 0)
            _d = f"部分成交{_f}张" if _f else "未成交"
            if not _cancel(p["entry_order_id"]):
                # 撤单失败仍清ID的话, ①就不再轮询该单, filled 永远冻结在此刻的值, 而券商侧
                # 可能继续成交 → 多出来的张数裸多无人管、无止损(复审实测: 账面1张 vs 实际3张)
                log(f"   ⚠️ {osi} 入场单TTL撤单失败, 保留ID下轮重试")
                journal(ev="entry_ttl_cancel_failed", osi=osi, order_id=p["entry_order_id"])
                _save(POS_JSON, positions)
                continue
            log(f"⏳ {osi} 入场单挂{ENTRY_TTL_SEC//60}分钟{_d} → 已撤剩余(不接刀)")
            journal(ev="entry_ttl_cancel", osi=osi, order_id=p["entry_order_id"], ttl=ENTRY_TTL_SEC, filled=_f)
            push_discord(f"⏳ enrich入场单超时撤单 {osi} (挂{ENTRY_TTL_SEC//60}分钟, {_d}, 撤剩余不接刀)")
            p["entry_order_id"] = None
            if _f > 0:
                if p["status"] == "pending":
                    p["status"] = "open"
                ensure_protection(positions, osi, p)   # 已成交部分转正常保护
            else:
                p["status"] = "closed"
            _save(POS_JSON, positions)
            continue
        # ④ 到期日强平 (15:40 ET 后)
        if p["status"] in ("pending", "open"):
            try:
                exp_d = date.fromisoformat(p["expiry"])
                # QA: 原 `date>=exp_d and (h,m)>=(15,40)` 元组比较陷阱 — 若到期日15:40那刻没跑到,
                #     次日09:30时 (9,30)>=(15,40) 为False, 要拖到次日15:40才平, 已到期合约多挂近一天
                if now_et.date() > exp_d or (now_et.date() == exp_d
                                             and (now_et.hour, now_et.minute) >= (15, 40)):
                    close_position(positions, osi, "到期强平")
            except Exception:
                pass


# ── andy 前向观察 (只记录不下单; 子集=波段+明确止损, 回测PF1.28待前向验证) ──
ANDY_TRACK = OUT / "andy_tracked.json"
_andy_last_tk, _andy_last_ts = None, None


def handle_andy(text: str, msg_ts, msg_id: int):
    global _andy_last_tk, _andy_last_ts, _quote_ctx
    from backtest_andy import parse_entry, osi as andy_osi, EXIT_FULL_RE, EXIT_PART_RE, BE_RE, TICKER_RE
    tracked = _load(ANDY_TRACK)
    up = " ".join(text.split())
    e = parse_entry(text, msg_ts.date()) if "RedAlert" in text else None
    if e:
        sym = andy_osi(e)
        subset = (not e["lotto"]) and e["expiry"] > msg_ts.date() and bool(e["stop"])
        snap = {}
        try:
            if _quote_ctx is None:
                from longport.openapi import Config, QuoteContext
                _quote_ctx = QuoteContext(Config.from_env())
            o = _quote_ctx.option_quote([sym])
            if o:
                snap = dict(last=float(o[0].last_done or 0), oi=int(o[0].open_interest or 0),
                            iv=float(getattr(o[0], "implied_volatility", 0) or 0))
        except Exception:
            pass
        journal(ev="andy_entry", osi=sym, ticker=e["ticker"], prem=e["prem"], stop=e["stop"],
                expiry=str(e["expiry"]), lotto=e["lotto"], subset=subset, quote=snap, sig=up[:130])
        _andy_last_tk, _andy_last_ts = e["ticker"], msg_ts
        if subset:
            tracked[e["ticker"]] = dict(osi=sym, ts=str(msg_ts), prem=e["prem"], stop=e["stop"])
            _save(ANDY_TRACK, tracked)
            log(f"📒 andy观察(不下单): {sym} @${e['prem']} SL${e['stop']} | 实时{snap}")
            push_discord(f"📒 andy观察单 {e['ticker']} {e['expiry']} ${e['strike']}{'C' if e['right']=='C' else 'P'} "
                         f"@${e['prem']} SL${e['stop']} (仅记录)")
        else:
            log(f"📒 andy跳过(lotto/0DTE/无止损): {up[:70]}")
        return
    # 出场/BE: 票名∩已跟踪票; 无票名→30分钟内最近提及票 (与回测同规则)
    lv = "full" if EXIT_FULL_RE.search(up) else ("partial" if EXIT_PART_RE.search(up) else None)
    be = bool(BE_RE.search(up))
    mention = [x for x in TICKER_RE.findall(up) if x in tracked]
    uniq = list(dict.fromkeys(mention))
    if len(uniq) == 1:
        _andy_last_tk, _andy_last_ts = uniq[0], msg_ts
    if not lv and not be:
        return
    tk = uniq[0] if len(uniq) == 1 else (
        _andy_last_tk if not uniq and _andy_last_tk and _andy_last_ts
        and (msg_ts - _andy_last_ts).total_seconds() <= 1800 else None)
    if tk:
        journal(ev="andy_exit", ticker=tk, level=("be" if be else lv), sig=up[:130])
        log(f"📒 andy出场[{'be' if be else lv}] {tk}: {up[:60]}")


LAST_MSG_JSON = OUT / "enrich_last_msg.json"   # 每频道最后处理的消息id (停机追赶用)


async def catch_up(client, seen, positions):
    """重连后回看停机期间错过的消息:
       enrich BUY→仅提醒(旧价不补单); enrich EXIT→照常处理(持仓晚跟好过不跟); andy→照常记录。"""
    state = _load(LAST_MSG_JSON)
    for ch_id in (CHANNEL_ID, ANDY_CHANNEL_ID):
        ch = client.get_channel(ch_id)
        if ch is None:
            continue
        key = str(ch_id)
        last = state.get(key)
        try:
            if not last:                      # 首次运行: 锚定到最新, 不回放历史
                async for m in ch.history(limit=1):
                    state[key] = str(m.id)
                continue
            missed = [m async for m in ch.history(limit=100, after=discord.Object(id=int(last)),
                                                  oldest_first=True)]
        except Exception as e:
            log(f"追赶失败 ch={ch_id}: {e}")
            continue
        for m in missed:
            state[key] = str(m.id)
            if m.author.id != AUTHOR_ID or not m.content:
                continue
            one = " ".join(m.content.split())[:90]
            try:
                if ch_id == ANDY_CHANNEL_ID:
                    handle_andy(m.content, m.created_at, m.id)   # 纯记录, 无下单
                    continue
                s = parse_signal(m.content, m.created_at.date())
                if s.kind in ("BUY", "BUY_AMBIG"):
                    note = f"⏰ 停机期间错过的enrich买入信号 (仅提醒, 不按旧价补单): {one}"
                    log(note); push_discord(note)
                    journal(ev="missed_during_downtime", sig=one, ts_signal=str(m.created_at))
                elif s.kind == "EXIT":
                    # 必须 to_thread: handle 要抢 _handle_lock, 而 _managed_tick 现在会长时间
                    # 持有该锁; 在事件循环线程里同步抢锁会挂住 discord 心跳 → 断线重连正反馈(复审发现)
                    import asyncio as _aio
                    await _aio.to_thread(handle, m.content, m.created_at.date(),
                                         m.id, seen, positions)   # 出场晚跟好过不跟
            except Exception as e:
                log(f"追赶处理异常: {e}")
        if missed:
            log(f"⏰ 追赶完成 ch={ch_id}: 回看了 {len(missed)} 条停机期间消息")
    _save(LAST_MSG_JSON, state)


def bump_last(ch_id, msg_id):
    state = _load(LAST_MSG_JSON)
    state[str(ch_id)] = str(msg_id)
    _save(LAST_MSG_JSON, state)


# ── 信号处理 ──

STALE_BUY_SEC = int(os.environ.get("STALE_BUY_SEC", "180"))   # 买入信号迟到>此秒数→只提醒(scalp几分钟就死)
_recent_exits = {}   # 出场去重: {规范化文本: 时刻} — 站长13秒重发同文本曾致减仓执行两次


def handle(text: str, msg_date: date, msg_id: int, seen: dict, positions: dict, msg_dt=None):
    with _handle_lock:
        return _handle(text, msg_date, msg_id, seen, positions, msg_dt)


def _llm_exit_targets(positions, v):
    """LLM出场判定 → 目标持仓列表 (tickers数组/scope/except数组)。"""
    out = []
    excepts = set(v.get("except") or [])
    tks = set(v.get("tickers") or [])
    for osi, p in positions.items():
        if p["status"] not in ("pending", "open"):
            continue
        if p.get("ticker") in excepts:
            continue
        if v.get("scope") == "ticker" and tks and p.get("ticker") not in tks:
            continue
        out.append(osi)
    return out


def _handle(text: str, msg_date: date, msg_id: int, seen: dict, positions: dict, msg_dt=None):
    s = parse_signal(text, msg_date)
    one = " ".join(text.split())[:100]
    if s.kind == "NOISE":
        held_any = any(p["status"] in ("pending", "open") for p in positions.values())
        if LLM_ON and LIVE and held_any and len(one) > 15 and "http" not in text:
            held_tks = [p["ticker"] for p in positions.values() if p["status"] in ("pending", "open")]
            v = llm_classify(text, held_tks)
            if v:
                journal(ev="llm_classify", rule="NOISE", verdict=v, sig=one)
            if v and v["action"] in ("exit_full", "exit_partial") and v["confidence"] >= 0.75:
                targets = _llm_exit_targets(positions, v)
                if targets:
                    log(f"🤖 LLM捞漏出场[{v['action']}] conf={v['confidence']} ({v.get('why','')}): {one}")
                    push_discord(f"🤖 LLM识别出场(规则未识别) [{v['action']}]: {one}")
                    for osi in targets:
                        if v["action"] == "exit_full":
                            close_position(positions, osi, "LLM出场判定")
                        else:
                            mirror_reduce(positions, osi, "partial")
        return

    if s.kind == "EXIT":
        # 出场文本去重: 同文本10分钟内只执行一次 (站长爱重发)
        _norm = one.lower().strip()
        _now = time.time()
        for k in [k for k, t0 in _recent_exits.items() if _now - t0 > 600]:
            _recent_exits.pop(k, None)
        if _norm in _recent_exits:
            log(f"↩️ 重复出场消息跳过(10分钟内同文本): {one[:60]}")
            return
        _recent_exits[_norm] = _now
        if EXIT_MODE == "mechanical":
            # 机械模式: 出场全靠 +30%卖⅓/保本/15m9ema/-50止损, 站长出场只提醒不执行
            note = f"🟠 站长出场[{s.exit_level}] [{s.ticker}] — 机械出场模式不跟(仅提醒): {one}"
            log(note); push_discord(note)
            journal(ev="exit_signal_mech_ignored", ticker=s.ticker, level=s.exit_level, sig=one)
            return
        held = [osi for osi, p in positions.items()
                if p["status"] in ("pending", "open")
                and (s.ticker == "*" or p.get("ticker") == s.ticker)]
        if s.exit_level == "alert" and LIVE:
            # 多票/豁免词 → LLM仲裁: 能解析出明确tickers就按票执行 (审计漏洞#1: 多票出场瘫痪)
            held_all = [p["ticker"] for p in positions.values() if p["status"] in ("pending", "open")]
            v = llm_classify(text, held_all) if (LLM_ON and held_all) else None
            if v:
                journal(ev="llm_classify", rule="EXIT/alert", verdict=v, sig=one)
            if v and v["action"] in ("exit_full", "exit_partial") and v["confidence"] >= 0.75                     and (v.get("tickers") or v.get("scope") == "all"):
                targets = _llm_exit_targets(positions, v)
                if targets:
                    lvl = "full" if v["action"] == "exit_full" else "partial"
                    log(f"🤖 LLM仲裁多票出场[{lvl}] {v.get('tickers') or 'ALL'} 豁免{v.get('except')}: {one}")
                    push_discord(f"🤖 多票出场(LLM仲裁)[{lvl}]: {one}")
                    for osi in targets:
                        if lvl == "full":
                            close_position(positions, osi, "多票出场(LLM仲裁)")
                        else:
                            mirror_reduce(positions, osi, "partial")
                    return
            note = f"⚠️ enrich出场信号含多票/豁免词, LLM未能明确, 仅提醒 [{s.ticker}]: {one}"
            log(note); push_discord(note)
            return
        if not held or not LIVE:
            note = f"🟠 enrich出场提醒 [{s.ticker}·{s.exit_level}] (无持仓/DRY_RUN): {one}"
            log(note); push_discord(note)
            return
        log(f"🟠 站长出场[{s.exit_level}] [{s.ticker}] → 处理 {len(held)} 个持仓: {one}")
        journal(ev="exit_signal", ticker=s.ticker, level=s.exit_level, held=len(held), sig=one)
        push_discord(f"🟠 enrich出场[{s.exit_level}] [{s.ticker}]: {one}")
        for osi in held:
            if s.exit_level == "full":
                close_position(positions, osi, "站长清仓")
            else:                           # partial / vague → 镜像
                mirror_reduce(positions, osi, s.exit_level)
        return

    # 🛡️ Hedge单: 不跳过, 按lotto小仓跟 (2026-07-19 用户改)
    #   旧: 一刀切跳过(XOM -30%教训) → 但复盘发现该XOM hedge实为+265%波段, 跳过=错失大赢家。
    #   机械出场的-50%止损已兜住"赌方向"下行, 小仓(净值×⅓)限损即可。标记→走lotto仓位档 + 免LLM否决。
    is_hedge = bool(re.search(r"\bhedge\b", one, re.IGNORECASE))
    if is_hedge:
        log(f"🛡️ enrich对冲单 → 按lotto小仓跟 (机械止损兜底下行): {one}")
        journal(ev="hedge_as_lotto", sig=one)

    # 🚫 迟到闸门: 信号已过时效的买入绝不进场 (用户铁律; MSFT -42%教训)
    if msg_dt is not None:
        from datetime import timezone as _tz
        age = (datetime.now(_tz.utc) - msg_dt).total_seconds()
        if age > STALE_BUY_SEC:
            note = f"🚫 信号已过时效 {age/60:.0f}分钟, 按规则不进场(仅提醒): {one}"
            log(note); push_discord(note)
            journal(ev="stale_buy_skipped", age_sec=int(age), sig=one)
            return

    # BUY_NOEXPIRY: 四要素齐缺到期(已推断周五/0DTE) → LLM buy≥0.85佐证才下, lotto档 (审计漏洞#2)
    qty = CONTRACTS
    is_ambig = False
    if s.kind == "BUY_NOEXPIRY":
        if LLM_ON and LIVE:
            held_tks = [p["ticker"] for p in positions.values() if p["status"] in ("pending", "open")]
            v = llm_classify(text, held_tks)
            if v:
                journal(ev="llm_classify", rule="BUY_NOEXPIRY", verdict=v, sig=one)
            if not v or v["action"] != "buy" or v["confidence"] < 0.85:
                note = f"❓ 缺到期信号LLM佐证不足({v['action'] if v else 'fail'}), 仅提醒: {one}"
                log(note); push_discord(note)
                return
            log(f"🔍 缺到期信号LLM佐证buy({v['confidence']}), 推断到期{s.expiry}, lotto档跟进")
        else:
            # 无LLM(机械模式): 直接按lotto档跟 (2026-07-19复盘: 历史11/11条BUY_NOEXPIRY全为
            # 真实lotto/scalp入场, 零误报; ARM 7/17漏单教训。兜底: 权利金≤$5 + TTL20分 + -60%止损 + 小仓)
            log(f"🎯 缺到期信号→推断到期{s.expiry}" + ("(0DTE档)" if s.expiry == msg_date else "")
                + ", lotto档直接跟 (历史11/11真信号)")
            journal(ev="noexpiry_accept", ticker=s.ticker, strike=s.strike,
                    expiry=str(s.expiry), limit=s.limit_price, sig=one)
        s.kind = "BUY"
        is_ambig = True     # 走lotto仓位档 (到期是推断的, 小仓; 0DTE自动再降到⅒档)
    if s.kind == "BUY_AMBIG":
        is_ambig = True
        side, info = resolve_direction(s)
        if side is None:
            note = f"❓ enrich歧义单无法消歧, 仅提醒 [{s.ticker} ${s.strike} {s.expiry}]: {info}\n原文: {one}"
            log(note); push_discord(note)
            return
        s.right, s.kind = side, "BUY"
        qty = LOTTO_CONTRACTS               # lotto/歧义单: 小仓 (他自己都喊small)
        log(f"🔍 报价消歧: {info}")
        journal(ev="disambig", ticker=s.ticker, strike=s.strike, expiry=str(s.expiry), info=info, sig=one)

    # 🤖 LLM否决闸: 规则说买, LLM说不是买(对冲/评论/出场) → 保守不下单
    if LLM_ON and LIVE:
        held_tks = [p["ticker"] for p in positions.values() if p["status"] in ("pending", "open")]
        v = llm_classify(text, held_tks)
        if v:
            journal(ev="llm_classify", rule=s.kind, verdict=v, sig=one)
        if v and v["action"] != "buy" and v["confidence"] >= 0.7 and not is_hedge:
            note = (f"🤖 LLM否决买入: 规则判BUY但LLM判[{v['action']}] conf={v['confidence']} "
                    f"({v.get('why','')}), 保守不下单: {one}")
            log(note); push_discord(note)
            journal(ev="llm_veto", verdict=v, sig=one)
            return

    # BUY
    osi = to_longport_symbol(s)
    key = f"{osi}:{msg_date}"
    if str(msg_id) in seen or key in seen:
        log(f"↩️ 重复信号跳过: {osi}")
        return
    if s.expiry < msg_date:
        log(f"⏭️ 已到期跳过: {osi}")
        return
    if s.limit_price > MAX_PREMIUM:
        log(f"🚫 权利金${s.limit_price}>上限${MAX_PREMIUM}, 拒绝: {one}")
        return
    # QA-CRITICAL: 去重key含日期, 跨日复述同一合约会走到 positions[osi]=dict(...) 把未平仓记录
    #   整条覆盖 → filled/avg清零(止损前置条件失效) + 旧GTC止盈单变孤儿 + 到期强平只卖新批量
    old = positions.get(osi)
    if old and old.get("status") in ("pending", "open"):
        _r = old.get("filled", 0) - old.get("sold", 0)
        log(f"↩️ {osi} 已有未平仓持仓(status={old['status']}, 剩{_r}张) → 跳过重复建仓(防状态覆盖丢账)")
        journal(ev="dup_open_skip", osi=osi, old_status=old.get("status"), remain=_r)
        push_discord(f"↩️ enrich跳过重复建仓 {osi} — 已持未平仓位(剩{_r}张), 防状态覆盖")
        return
    # QA: BUY_NOEXPIRY推断到期 + 站长随后澄清(如0DTE) → 同一交易意图开出两个不同合约双份敞口。
    #   按 (票/方向/行权价/当天) 加一道软去重。
    soft_key = f"soft:{s.ticker}{s.right}{s.strike}:{msg_date}"
    if soft_key in seen:
        log(f"↩️ 软去重跳过(同票同向同行权价当天已建仓): {osi}")
        journal(ev="soft_dup_skip", osi=osi, soft_key=soft_key)
        return

    # 按金额定张数 (POSITION_USD>0时; lotto/歧义单用LOTTO_USD)
    is_lotto = is_hedge or is_ambig or "lotto" in (s.size_tag or "").lower() or "scalp" in one.lower() or s.expiry == msg_date
    is_0dte = s.expiry == msg_date
    if POSITION_FRAC > 0:
        eq = account_equity_usd()
        if eq:
            frac = ZERO_DTE_FRAC if is_0dte else (LOTTO_FRAC if is_lotto else POSITION_FRAC)
            budget = eq * frac
            size_src = f"净值${eq:,.0f}×{frac:.2f}" + ("(0DTE档)" if is_0dte else "")
        else:
            budget = LOTTO_USD if is_lotto else POSITION_USD
            size_src = "净值获取失败,退回固定额"
            log("⚠️ 账户净值获取失败, 用固定金额兜底")
    else:
        budget = LOTTO_USD if is_lotto else POSITION_USD
        size_src = "固定额"
    qty, size_note = size_qty(s.limit_price, budget, osi, fallback=qty)
    # QA: 原来每条信号各自按"当前净值×frac"定张, 没有任何组合层上限 —— 站长一天发3-5条是常态,
    #   3条常规单即 1.5×净值(实测), 第三条起被券商以购买力不足随机拒单, 且全账户压在两三个合约上。
    _eq_now = _last_equity[0] or account_equity_usd()
    if _eq_now and MAX_GROSS_FRAC > 0:
        # 必须扣掉已卖出部分: 机械模式 +30%/+60% 会卖掉⅔, 若按 filled 全额计,
        # 一笔成交就把额度占死(实测高估200%), 当天后续信号会被系统性拒掉(复审发现)
        gross = sum(max(0, (q.get("filled") or q.get("qty", 0)) - q.get("sold", 0))
                    * q.get("limit", 0) * 100
                    for q in positions.values() if q.get("status") in ("pending", "open"))
        room = _eq_now * MAX_GROSS_FRAC - gross
        max_q = int(room // (s.limit_price * 100)) if room > 0 else 0
        if max_q < 1:
            log(f"🚫 敞口闸: 在险${gross:,.0f} 已达净值${_eq_now:,.0f}×{MAX_GROSS_FRAC:.0%}上限 → 拒绝新仓 {osi}")
            journal(ev="gross_cap_skip", osi=osi, gross=gross, eq=_eq_now, cap=MAX_GROSS_FRAC)
            push_discord(f"🚫 enrich敞口已满 (在险${gross:,.0f} / 上限{MAX_GROSS_FRAC:.0%}×净值) — 跳过 {osi}: {one}")
            return
        if qty > max_q:
            log(f"⚠️ 敞口闸: {qty}张→{max_q}张 (在险${gross:,.0f}, 剩余额度${room:,.0f})")
            size_note += f"; 敞口闸{max_q}张(原{qty})"
            qty = max_q
    size_note = f"{size_src}; {size_note}"
    log(f"📐 仓位: {qty}张 ({size_note})")

    plan = (f"{'🚀模拟盘' if LIVE else '🧪DRY-RUN'} enrich买入\n"
            f"  {s.ticker} {s.expiry} ${s.strike} {'CALL' if s.right=='C' else 'PUT'}  ({osi})\n"
            f"  限价 ${s.limit_price} × {qty}张 (≈${s.limit_price*100*qty:.0f})"
            + (f"  [{s.size_tag}]" if s.size_tag else "") + f"\n  原文: {one}")
    log(plan)

    if LIVE:
        ok, r = _submit(osi, side_buy=True, qty=qty, price=s.limit_price, remark="enrich-entry")
        if ok:
            _sm0 = MECH_STOP_MULT if EXIT_MODE == "mechanical" else (STOP_MULT if is_lotto else SWING_STOP_MULT)
            positions[osi] = dict(ticker=s.ticker, entry_order_id=r, qty=qty,
                                  limit=s.limit_price, expiry=s.expiry.isoformat(),
                                  filled=0, sold=0, avg=0.0, tp_order_id=None, tp_qty=0,
                                  tp2_order_id=None, tp2_qty=0, tp1_done=False, tp2_done=False,
                                  armed=False, submitted_ts=time.time(),
                                  stop_mult=_sm0,
                                  status="pending", opened=str(msg_date))
            _save(POS_JSON, positions)
            _tpm = MECH_TP_MULT if EXIT_MODE == "mechanical" else TP_MULT
            plan += (f"\n  ✅已提交 order_id={r} (成交后自动挂+{(_tpm-1)*100:.0f}%止盈"
                     + (", 首档后移保本+15m9ema管runner" if EXIT_MODE == "mechanical" else "") + ")")
            log(f"  ✅已提交 {r}")
            journal(ev="entry_submit", osi=osi, ticker=s.ticker, right=s.right, strike=s.strike,
                    expiry=str(s.expiry), limit=s.limit_price, qty=qty, order_id=r, sig=one)
        else:
            plan += f"\n  ❌下单失败: {r}"
            log(f"  ❌下单失败: {r}")
    else:
        plan += "\n  (DRY_RUN 未下单)"
    push_discord(plan)

    seen[str(msg_id)] = key
    seen[key] = str(msg_id)
    # 软去重只在【真的建了仓】时登记: 下单失败还登记的话, 当天站长澄清到期日的正确信号会被永久拒掉
    if osi in positions:
        seen[soft_key] = osi
    _save(SEEN_JSON, seen)


_lockf = None   # 单实例锁句柄 (须常驻引用)


def main():
    global _lockf
    # 单实例锁 (审计bug#5: dev/mock实例与launchd并发写同一状态曾致真实止盈单变孤儿)
    import fcntl
    _lockf = open(OUT / "enrich_bot.lock", "w")
    try:
        fcntl.flock(_lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lockf.write(str(os.getpid())); _lockf.flush()
    except OSError:
        print("另一实例已持锁运行, 本实例退出 (output/enrich_bot.lock)"); sys.exit(1)
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("缺 DISCORD_BOT_TOKEN"); sys.exit(1)

    if LIVE:
        if not verify_paper_trading():
            print("❌ 模拟盘三重校验不通过, 拒绝启动 LIVE"); sys.exit(1)
        if POSITION_FRAC > 0:
            size_s = (f"动态仓位: 常规=净值×{POSITION_FRAC:.2f} / lotto=×{LOTTO_FRAC:.2f} "
                      f"/ 0DTE=×{ZERO_DTE_FRAC:.2f} (OI帽{OI_CAP_PCT:.0%})")
        elif POSITION_USD > 0:
            size_s = f"每信号${POSITION_USD:,.0f}/lotto${LOTTO_USD:,.0f} (OI帽{OI_CAP_PCT:.0%})"
        else:
            size_s = f"每信号{CONTRACTS}张"
        if EXIT_MODE == "mechanical":
            log(f"🚀 LIVE(模拟盘)·机械出场(最终定稿): {size_s} | 权利金上限${MAX_PREMIUM} | "
                f"+{(MECH_TP_MULT-1)*100:.0f}%卖⅓ → +{(MECH_TP2_MULT-1)*100:.0f}%卖⅓(武装) | "
                f"runner武装后守{MECH_EMA_MIN}分9ema连破{MECH_EMA_N}根 | 止损-{(1-MECH_STOP_MULT)*100:.0f}%全程无保本 | "
                f"入场TTL{ENTRY_TTL_SEC//60}分 | 到期强平 | 无LLM")
        else:
            log(f"🚀 LIVE(模拟盘): {size_s} | 权利金上限${MAX_PREMIUM} | 止盈+{(TP_MULT-1)*100:.0f}%卖半仓 | 镜像出场 | 止损: lotto-{(1-STOP_MULT)*100:.0f}%/波段-{(1-SWING_STOP_MULT)*100:.0f}%兜底 | 到期强平")
    else:
        log("🧪 DRY_RUN: 只解析播报, 不下单")

    # 首次启动先给关键状态文件播一份.bak, 否则"第一次损坏"时无备份可回落=直接拒启且救不回来
    for _p in (SEEN_JSON, POS_JSON):
        _b = _p.with_suffix(_p.suffix + ".bak")
        if _p.exists() and not _b.exists():
            try:
                _b.write_bytes(_p.read_bytes())
                log(f"🗂️ 已为 {_p.name} 建立初始备份")
            except Exception:
                pass
    seen = _load(SEEN_JSON, strict=True)      # 仓位/去重表损坏 → 宁可拒启不带空表上线
    positions = _load(POS_JSON, strict=True)
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    def _managed_tick():
        # QA-CRITICAL: manage_positions 原本在事件循环里裸调且不持 _handle_lock, 与
        #   on_message→to_thread(handle) 并发改写同一个 positions dict; close_position 的
        #   "已closed就返回"检查与置位之间存在TOCTOU, 重入即双份市价卖(mock复现: 3张持仓卖出6张)。
        #   另: 它内部有 time.sleep(0.4)+同步API调用, 裸调会阻塞整个 discord 事件循环。
        with _handle_lock:
            manage_positions(positions)

    @tasks.loop(seconds=60)
    async def manager():
        if LIVE and us_rth_now():   # 期权只在美股RTH交易, 盘外不轮询(省API防429)
            try:
                import asyncio
                await asyncio.to_thread(_managed_tick)
            except Exception as e:
                log(f"仓位管理异常: {e}")

    @client.event
    async def on_ready():
        log(f"✅ 已连接 Discord: {client.user} | 监听频道 {CHANNEL_ID} 作者 {AUTHOR_ID}")
        try:
            await catch_up(client, seen, positions)
        except Exception as e:
            log(f"追赶异常: {e}")
        if not manager.is_running():
            manager.start()

    @client.event
    async def on_message(msg):
        if msg.channel.id in (CHANNEL_ID, ANDY_CHANNEL_ID):
            bump_last(msg.channel.id, msg.id)
        if msg.author.id != AUTHOR_ID or not msg.content:
            return
        if msg.channel.id == ANDY_CHANNEL_ID:      # andy: 只观察记录, 永不下单
            try:
                handle_andy(msg.content, msg.created_at, msg.id)
            except Exception as e:
                log(f"andy处理异常: {e}")
            return
        if msg.channel.id != CHANNEL_ID:
            return
        try:
            import asyncio
            await asyncio.to_thread(handle, msg.content, msg.created_at.date(), msg.id,
                                    seen, positions, msg.created_at)
        except Exception as e:
            log(f"处理消息异常: {e}")

    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
