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
from datetime import datetime, date, time as dtime, timezone as _tzone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).parent))
import warnings; warnings.filterwarnings("ignore")

import discord
from discord.ext import tasks
from enrich_parser import parse_signal, to_longport_symbol
from notify import push_discord
import threading
# 2026-07-20 用户令: 实盘bot移除全部LLM。理由 ——
#   ① 进场95%+可纯规则解析, 出场按机械规则(+30%卖⅓/+60%卖⅓/15m9ema/-60%止损)不需要语义层
#   ② claude CLI 冷启动数十秒且会握着 _handle_lock, 期间止损轮询/到期强平/EXIT全部排队(审查项#13)
#   ③ 少一个不可靠的外部依赖 = 少一类静默失败
# llm_classifier.py 保留在仓库里, 仍被 build_interp_audit.py / backtest_agent_ab.py 等研究脚本使用。
_handle_lock = threading.Lock()   # 消息处理与仓位轮询串行, 防持仓并发写

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
# 报价陈旧阈值。900秒对低流动性期权过严: $0.05 的OTM周权盘中15分钟无成交是常态, 会导致
# "唯一止损通道"整日不可用。放宽到1小时 —— -60%是很宽的止损, 1小时前的价格跌破它已是强信号。
QUOTE_MAX_AGE_SEC = int(os.environ.get("QUOTE_MAX_AGE_SEC", "3600"))
EXIT_STUCK_SEC = int(os.environ.get("EXIT_STUCK_SEC", "300"))    # 卖单在途超过此秒数仍未终态 → 告警

# ── 出场模式 (2026-07-19 最终定稿, 用户拍板; 复盘+网格+对抗审查三轮定案) ──
# mechanical = 机械出场(无AI): +30%卖⅓ → +60%再卖⅓ → 剩⅓runner(-60%初始止损全程, 无保本)
#              → runner摸到+60%(武装)后启动【标的】15分9ema连破2根拖尾 → 到期强平
#              回测(2-7月n=127): 加权+16%/笔 真实K+30% 胜率52% 中位+10% 最差月-6% (bt_mechanical_v2)
# mirror     = 跟站长出场(纯规则解析) — 旧模式, 保留可回切
EXIT_MODE = os.environ.get("EXIT_MODE", "mirror").lower()
MECH_TP_MULT = float(os.environ.get("MECH_TP_MULT", "1.3"))     # 一档止盈 +30% 卖⅓
MECH_TP2_MULT = float(os.environ.get("MECH_TP2_MULT", "1.6"))   # 二档止盈 +60% 卖⅓; 触及即武装9ema
MECH_STOP_MULT = float(os.environ.get("MECH_STOP_MULT", "0.4")) # 初始止损 -60% 全程有效(无保本移动)
MECH_EMA_MIN = int(os.environ.get("MECH_EMA_MIN", "15"))        # 9ema时间框架(分钟)
MECH_EMA_N = int(os.environ.get("MECH_EMA_N", "2"))             # 连破N根(已完成bar)出runner
ENTRY_TTL_SEC = int(os.environ.get("ENTRY_TTL_SEC", "1200"))    # 在途入场单TTL: 20分钟未成交撤单
# (审计发现+MSFT复盘: 挂一天的限价单只会在期权崩盘穿价时成交=专门接刀; "不追高"的另一半是"不接刀")
ET = ZoneInfo("America/New_York")

OUT = Path(__file__).parent / "output"
# DRY_RUN 用独立去重表: 否则 DRY_RUN 期间"看过"的信号, 切回 LIVE 后会被当成重复信号永久跳过
SEEN_JSON = OUT / ("enrich_seen.json" if os.environ.get("ENRICH_LIVE", "").lower() == "true"
                   else "enrich_seen_dry.json")
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
    """期权最新价 (需OPRA权限)。返回 None = 【价格不可用】, 调用方必须跳过止损判定。

    校验三项(原实现直接返回 last_done, 会打出假止损):
      ① last_done>0 —— 无成交时返回0, 而 0 <= 任何止损价 → 立刻假止损全平
      ② 报价新鲜度 —— 低流动性期权的"最新成交"可能是几天前的旧价, 拿它判止损等于看历史
      ③ trade_status==Normal —— 停牌/熔断/退市期间的价格不可据以交易
    """
    global _quote_ctx
    if time.time() < _rl_until:
        return None
    try:
        if _quote_ctx is None:
            from longport.openapi import Config, QuoteContext
            _quote_ctx = QuoteContext(Config.from_env())
        o = _quote_ctx.option_quote([osi])
        if not o:
            return None
        q = o[0]
        last = float(q.last_done or 0)
        if last <= 0:
            return None
        ts = getattr(q, "timestamp", None)
        if ts is not None:
            try:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=_tzone.utc)   # naive 一律按UTC解释, 不能让
                    # astimezone 用本机时区(SGT+8)去猜 → 会把所有报价算成8小时前=止损全灭
                age = (datetime.now(_tzone.utc) - ts.astimezone(_tzone.utc)).total_seconds()
                if age > QUOTE_MAX_AGE_SEC:
                    log(f"   ⏱️ {osi} 报价陈旧({age/60:.0f}分钟前), 本轮不据此判止损")
                    return None
            except Exception:
                pass
        tst = str(getattr(q, "trade_status", "Normal")).split(".")[-1]
        if tst != "Normal":
            log(f"   ⛔ {osi} 交易状态={tst}, 本轮不据此判止损")
            return None
        return last
    except Exception as e:
        _rl_hit(e)
        return None


def _await_terminal(oid: str, tries: int = 6, delay: float = 0.5):
    """轮询订单直到终态。返回 (是否终态, 状态名, 累计成交量)。
    非终态或查询失败 → 第一项 False, 此时调用方【禁止】再发新的卖单(旧单可能仍会成交)。"""
    st, exq = None, 0
    for i in range(tries):
        st, exq, _ = _order_state(oid)
        if st in _TERMINAL:
            return True, st, exq
        if i < tries - 1:
            time.sleep(delay)
    return False, st, exq


def cancel_and_reconcile(oid: str):
    """撤单 + 确认终态 + 取回最终成交量。返回 (已确认, 累计成交量, 状态名)。

    撤单接口只是"请求撤销", 返回成功不代表订单已停止 —— 可能正在撮合、可能已部分成交。
    必须查到终态才能确定它不会再吃单; 查不到就当它还活着, 绝不能在它之上再发卖单(=超卖裸空)。"""
    _cancel(oid)                      # 撤单请求本身失败也继续确认(可能已成交/已在撤销中)
    ok, st, exq = _await_terminal(oid)
    return ok, exq, st


def _save_critical(positions: dict, osi: str, p: dict, what: str) -> bool:
    """落盘"券商侧订单ID"这类关键状态。失败意味着: 券商已收单而本地不知道 ——
    崩溃重启后会重复下单甚至裸空。因此失败即把该仓位置 fail_stop, 停止对它的一切自动交易,
    只告警等人工处理。(入场路径已有专门的撤单兜底; 卖腿/保护腿无法安全撤销时只能 fail-stop)"""
    if _save(POS_JSON, positions):
        return True
    p["fail_stop"] = True
    journal(ev="persist_failed", osi=osi, what=what)
    _alert(f"🚨 enrich {osi} {what} 已提交但状态落盘失败 → 该仓位进入 fail-stop"
           f"(停止自动交易), 请人工核对券商挂单与持仓")
    return False


def _outstanding_sells(p: dict) -> int:
    """券商侧尚未成交的卖单总张数(已成交部分已进 sold, 这里只算未成交余量)。"""
    n = 0
    for id_key, qty_key in (("tp_order_id", "tp_qty"), ("tp2_order_id", "tp2_qty"),
                            ("stop_order_id", "stop_qty"), ("exit_order_id", "exit_qty")):
        if p.get(id_key):
            n += max(0, int(p.get(qty_key, 0)) - int(p.get(id_key + "_counted", 0)))
    return n


def _sell_budget(p: dict) -> int:
    """I3 硬闸: 还能再挂多少张卖单而不超卖 = 剩仓 − 已挂未成交卖单。

    任何挂卖单的路径都必须先过这道闸。没有它就只能靠"每处都算对张数"这种脆弱约定 ——
    实际上 ensure_protection 曾按 filled 而非 remain 定张, 部分平仓后 sold>0 时挂出
    多于持仓的卖单 = 裸空期权(无限风险)。"""
    return (p.get("filled", 0) - p.get("sold", 0)) - _outstanding_sells(p)


def _credit_leg(p: dict, id_key: str, exq: int) -> int:
    """把某条卖腿的【累计】成交量记入 sold, 只补差额。
    同一条腿可能被多处对账(②③③c/close_position), 直接 sold+=exq 会重复计数 → 虚增已卖 → 少卖真仓。"""
    k = id_key + "_counted"
    delta = max(0, int(exq) - int(p.get(k, 0)))
    if delta:
        p["sold"] = p.get("sold", 0) + delta
        p[k] = int(exq)
    return delta


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
# 订单终态【白名单】。长桥 OrderStatus 共18种, 其余(New/PartialFilled/PendingCancel/WaitToCancel/
# NotReported/Replaced/Unknown/...)全部视为在途未决 —— 未决状态下【绝不允许】再发新的卖单。
# (原实现是"列举几个已撤销状态", 漏一个在途态就会误判成已结束 → 旧单还活着又发新单 = 超卖裸空)
# closing = 卖单已发出但未确认成交 —— 仓位【尚未平掉】, 一切"是否还持有"的判断都必须算上它:
# 重复建仓守卫/敞口计算/EXIT目标查找 漏掉它会导致 覆盖在途仓位 / 低估敞口 / 忽略"all out"。
ACTIVE_STATUSES = ("pending", "open", "closing")
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
    if premium is None or premium <= 0:
        return 0, "权利金<=0, 拒绝定张"      # 原会 budget//(0*100) 直接 ZeroDivisionError
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
    """美股期权可交易窗口: 周一-五 9:31-15:58 ET (两端留安全边)。
    原为 9:15-16:20, 会在正式开盘前和收盘后继续轮询/触发止损/提交市价单。
    到期强平(15:40)在窗口内, 不受影响。"""
    now = datetime.now(ET)
    return now.weekday() < 5 and dtime(9, 31) <= now.time() <= dtime(15, 58)


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
        return True
    except Exception as e:
        log(f"⚠️ 状态落盘失败 {p.name}: {e}")
        return False


def reconcile_with_broker(positions: dict) -> dict:
    """启动时与券商【实际】状态对账 (只读, 不自动改仓)。

    本地JSON格式完好 ≠ 与券商一致。崩溃/漏落盘/人工干预都会造成偏差, 而 .bak 和格式校验
    都救不了这一层。这里把三方摆出来比对并告警, 由人决定怎么处理:
      ① 券商侧期权持仓  vs 本地 filled-sold
      ② 券商侧当日未终态挂单 vs 本地记录的各腿 order_id
    返回 {"mismatch": [...], "orphan_orders": [...], "orphan_positions": [...]}
    """
    out = {"mismatch": [], "orphan_orders": [], "orphan_positions": []}
    try:
        broker_pos = {}
        for ch in _trade_ctx.stock_positions().channels:
            for sp in getattr(ch, "positions", []) or []:
                sym = str(getattr(sp, "symbol", ""))
                qty = int(float(getattr(sp, "quantity", 0) or 0))
                if qty:
                    broker_pos[sym] = broker_pos.get(sym, 0) + qty
        live_oids = {}
        try:
            for od in _trade_ctx.today_orders() or []:
                st = str(od.status).split(".")[-1]
                if st not in _TERMINAL:
                    live_oids[str(od.order_id)] = (str(od.symbol), st)
        except Exception as e:
            log(f"   对账: 拉取当日订单失败 {e}")

        local_active = {o: p for o, p in positions.items() if p.get("status") in ACTIVE_STATUSES}
        # ① 持仓数量比对
        for osi, p in local_active.items():
            local_q = p.get("filled", 0) - p.get("sold", 0)
            bq = broker_pos.get(osi, 0)
            if local_q != bq:
                out["mismatch"].append(f"{osi}: 本地{local_q}张 vs 券商{bq}张")
        for sym, bq in broker_pos.items():
            if bq > 0 and sym not in local_active and sym.endswith(".US") and len(sym) > 15:
                out["orphan_positions"].append(f"{sym}: 券商持有{bq}张但本地无活跃记录")
        # ② 挂单比对: 本地记着的腿是否还活着 / 券商侧有没有本地不认识的活单
        known = set()
        for osi, p in local_active.items():
            for k in ("entry_order_id", "tp_order_id", "tp2_order_id", "stop_order_id", "exit_order_id"):
                if p.get(k):
                    known.add(str(p[k]))
        for oid, (sym, st) in live_oids.items():
            if oid not in known and len(sym) > 15:
                out["orphan_orders"].append(f"{oid} {sym} ({st})")
    except Exception as e:
        log(f"⚠️ 券商对账失败(不阻断启动): {e}")
        return out

    total = len(out["mismatch"]) + len(out["orphan_orders"]) + len(out["orphan_positions"])
    if total == 0:
        log("🔍 券商对账: 本地持仓/挂单与券商一致 ✅")
    else:
        lines = ["🚨 券商对账发现不一致 (bot不会自动处理, 请人工确认):"]
        for t, items in (("数量不符", out["mismatch"]), ("券商有仓本地无记录", out["orphan_positions"]),
                         ("券商有活单本地不认识", out["orphan_orders"])):
            for it in items:
                lines.append(f"   · [{t}] {it}")
        msg = "\n".join(lines)
        log(msg)
        journal(ev="broker_reconcile_mismatch", **out)
        _alert(msg[:1800])
    return out


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
    if _sm > 0 and not p.get("stop_order_id") and _MIT_OK is not False and _sell_budget(p) >= 1:
        _sq = min(remain, _sell_budget(p))
        trig = round(p["avg"] * _sm, 2)
        ok, r = _submit(osi, side_buy=False, qty=_sq, price=None, tif_gtc=True,
                        remark="stop", trigger=trig)
        if ok:
            _MIT_OK = True
            p["stop_order_id"], p["stop_qty"], p["stop_order_id_counted"] = r, _sq, 0
            log(f"🛡️ {osi} 券商侧止损已挂: {_sq}张 触发${trig} (-{(1-_sm)*100:.0f}%)")
            journal(ev="stop_place", osi=osi, trigger=trig, qty=_sq, order_id=r)
        elif "604050" in str(r) or "not supported" in str(r).lower():
            _MIT_OK = False
            log("ℹ️ 模拟盘不支持触价单 → 止损轮询兜底, 止盈挂券商侧限价 (真实账户会自动切回止损常驻)")
        else:
            log(f"⚠️ {osi} 止损挂单失败: {r}")
    # 回退模式: 无券商侧止损 → 止盈限价挂券商侧(抓尖峰)
    if EXIT_MODE == "mechanical" and not p.get("stop_order_id"):
        # 两档GTC同时挂: ⅓@+30%, ⅓@+60% (二档成交=runner武装9ema)
        # 【按 remain 而非 filled 定张】: 部分平仓后 sold>0, 用 filled 会挂出多于持仓的卖单。
        # 正常路径(新建仓 sold=0)下 remain==filled, 档位与回测口径完全一致, 不改变策略。
        q1, q2 = _mech_split(remain)
        _bud = _sell_budget(p)
        if q1 >= 1 and not p.get("tp_order_id") and not p.get("tp1_done") and _bud >= 1:
            q1 = min(q1, _bud)
            px1 = round(p["avg"] * MECH_TP_MULT, 2)
            ok, r = _submit(osi, side_buy=False, qty=q1, price=px1, tif_gtc=True, remark="tp1")
            if ok:
                p["tp_order_id"], p["tp_qty"], p["tp_order_id_counted"] = r, q1, 0
                log(f"🎯 {osi} 挂一档止盈: 卖{q1}张 @ ${px1} (+30%)")
                journal(ev="tp_place", osi=osi, px=px1, qty=q1, order_id=r)
            else:
                log(f"⚠️ {osi} 一档止盈挂单失败: {r}")
            _bud = _sell_budget(p)
        if q2 >= 1 and not p.get("tp2_order_id") and not p.get("tp2_done") and _bud >= 1:
            q2 = min(q2, _bud)
            px2 = round(p["avg"] * MECH_TP2_MULT, 2)
            ok, r = _submit(osi, side_buy=False, qty=q2, price=px2, tif_gtc=True, remark="tp2")
            if ok:
                p["tp2_order_id"], p["tp2_qty"], p["tp2_order_id_counted"] = r, q2, 0
                log(f"🎯 {osi} 挂二档止盈: 卖{q2}张 @ ${px2} (+60%, 成交即武装9ema)")
                journal(ev="tp2_place", osi=osi, px=px2, qty=q2, order_id=r)
            else:
                log(f"⚠️ {osi} 二档止盈挂单失败: {r}")
    elif not p.get("stop_order_id") and not p.get("tp_order_id") and not p.get("reduced"):
        tp_qty, tp_px = _tp_params(remain, p["avg"])
        tp_qty = min(tp_qty, _sell_budget(p))
        if tp_qty < 1:
            _save(POS_JSON, positions); return
        ok, r = _submit(osi, side_buy=False, qty=tp_qty, price=tp_px, tif_gtc=True, remark="tp")
        if ok:
            p["tp_order_id"], p["tp_qty"], p["tp_order_id_counted"] = r, tp_qty, 0
            log(f"🎯 {osi} 挂止盈: 卖{tp_qty}张 @ ${tp_px} (+{(tp_px/p['avg']-1)*100:.0f}%)")
            journal(ev="tp_place", osi=osi, px=tp_px, qty=tp_qty, order_id=r)
        else:
            log(f"⚠️ {osi} 止盈挂单失败: {r} (靠轮询/出场跟随/到期强平)")
    if p.get("tp_order_id") or p.get("tp2_order_id") or p.get("stop_order_id"):
        _save_critical(positions, osi, p, "保护腿订单ID")   # 挂了单就必须落盘, 否则重启后变孤儿
    else:
        _save(POS_JSON, positions)


def cancel_stop(p: dict) -> bool:
    """撤券商侧止损腿【并确认终态】。返回是否撤净 —— False 时旧单可能仍会成交, 不可再发卖单。
    (原实现丢弃 _cancel 返回值就清ID: 撤单失败也当撤掉了 → 旧止损单+新卖单 = 超卖裸空)"""
    oid = p.get("stop_order_id")
    if not oid:
        return True
    done, exq, st = cancel_and_reconcile(oid)
    if exq > 0:
        _credit_leg(p, "stop_order_id", exq)
    if done:
        p["stop_order_id"] = None
    else:
        log(f"   ⚠️ 止损腿 {oid} 未确认终态({st}), 暂不发新卖单")
    return done


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

def _live_sell_order(osi: str):
    """券商侧该合约当前是否已有【未终态的卖单】。返回 (order_id, 状态) 或 (None, None)。

    应用层幂等: longport SDK 的 submit_order 没有 client_request_id(3.0.23 与最新 4.3.3 都没有,
    幂等键只存在于 REST 层), 所以网络超时"券商已收单但客户端认为失败"这类重复提交只能靠提交前
    查重来挡。查不到不代表没有(查询本身可能失败), 因此这只是减少重复, 不是保证。"""
    try:
        for od in _trade_ctx.today_orders(symbol=osi) or []:
            if str(getattr(od, "side", "")).split(".")[-1] != "Sell":
                continue
            st = str(od.status).split(".")[-1]
            if st not in _TERMINAL:
                return str(od.order_id), st
    except Exception as e:
        if not _rl_hit(e):
            log(f"   查在途卖单失败 {osi}: {e}")
    return None, None


def _start_exit(positions: dict, osi: str, p: dict, qty: int, reason: str,
                intent: str = "full") -> bool:
    """提交市价卖 → 进入 closing 态等待【真实成交】。

    关键: 这里【绝不】更新 sold, 也【绝不】标 closed。
    submit_order 成功只代表拿到委托号, 不代表成交 —— 订单随后可能被拒绝、部分成交或长期挂起。
    sold 只在 manage_positions ⓪ 依据 order_detail 的 executed_quantity 更新。"""
    # 提交前查重: 上一轮可能"券商已收单但我们以为失败"。认领它而不是再发一张。
    dup_oid, dup_st = _live_sell_order(osi)
    if dup_oid and dup_oid not in (p.get("tp_order_id"), p.get("tp2_order_id"),
                                   p.get("stop_order_id")):
        log(f"   ♻️ {osi} 券商侧已有在途卖单 {dup_oid}({dup_st}) → 认领它, 不重复提交")
        journal(ev="exit_reclaim", osi=osi, order_id=dup_oid, state=dup_st, reason=reason)
        p["exit_order_id"], p["exit_qty"], p["exit_reason"] = dup_oid, int(qty), reason
        p["exit_intent"], p["exit_ts"] = intent, time.time()
        p["status"] = "closing"
        _save_critical(positions, osi, p, f"认领在途卖单{dup_oid}")
        return True
    ok, r = _submit(osi, side_buy=False, qty=qty, price=None, remark=f"exit-{reason[:12]}")
    if not ok:
        log(f"   ⚠️ {osi} 卖单提交失败({reason}), 保持{p['status']}等下轮重试: {r}")
        journal(ev="exit_submit_failed", osi=osi, qty=qty, reason=reason, err=str(r))
        _alert(f"🚨 enrich {osi} 卖单提交失败 ×{qty}张 ({reason}) — 仓位仍在, 下轮重试: {r}")
        _save(POS_JSON, positions)
        return False
    p["exit_order_id"], p["exit_qty"], p["exit_reason"] = r, int(qty), reason
    p["exit_intent"] = intent          # full=全平 / tp1=一档止盈(成交后才置reduced+tp1_done)
    p["exit_ts"] = time.time()
    p["status"] = "closing"
    log(f"   📤 {osi} 市价卖 {qty}张 已受理 order={r} ({reason}) → 等待成交确认")
    journal(ev="exit_submit", osi=osi, qty=qty, reason=reason, order_id=r)
    push_discord(f"📤 enrich卖出已受理 {osi} ×{qty}张 ({reason}) — 待成交确认")
    _save_critical(positions, osi, p, f"卖单{r}")
    return True


def close_position(positions: dict, osi: str, reason: str):
    """撤净所有挂单(确认终态) → 提交剩仓市价卖 → 进入 closing 等成交确认。

    生命周期原则(三轮返工后定稿):
      · 撤单 = 只是"请求撤销"。必须查到终态才算撤净; 查不到就当它还活着, 绝不在其上再发卖单。
      · 卖单提交成功 ≠ 成交。sold/closed 只能由 ⓪ 对账依据 executed_quantity 更新。
      · 方向性: 宁可少卖(仍持多头, 亏损上限=权利金), 绝不多卖(裸空期权=无限风险)。
    """
    p = positions.get(osi)
    if not p or p["status"] == "closed" or osi in _closing:
        return
    if p["status"] == "closing":
        # 已有卖单在途(可能只是部分减仓)。不能重复卖, 但也【绝不能静默丢弃】这次出场意图 ——
        # 止损/到期强平因条件持续存在会自然重来, 而站长"all out"/9ema出场是【一次性】的。
        # 记成待办, 等 ⓪ 把仓位收敛回 open 后由 ⓪b 立即执行(动作升级: 部分减仓 → 全平)。
        p["pending_action"] = "full_exit"
        p["pending_action_reason"] = reason
        p["pending_action_ts"] = time.time()
        log(f"   ⏸️ {osi} 卖单在途, 记为待办全平({reason}), 待其终态后执行")
        journal(ev="exit_deferred_closing", osi=osi, reason=reason)
        _save(POS_JSON, positions)
        return
    # 防重入守卫必须是【内存态】: 若写进 p 再被 _save 落盘, 崩溃重启后该标志永久为真,
    # close_position 将永远早退 → 该仓位再也平不掉, 止损空转(自查发现, 与本次要修的bug同类)
    _closing.add(osi)
    try:
        log(f"🔻 平仓 {osi} ({reason})")
        unresolved = []
        # 四条腿全部"撤单+确认终态+回填成交量"。入场腿也必须撤 —— 部分成交时 status 已是 open
        # 而 entry_order_id 仍在, 漏撤会让剩余买腿在暴跌里继续接刀且仓位标closed后无人管。
        for id_key in ("tp_order_id", "tp2_order_id", "stop_order_id", "entry_order_id"):
            oid = p.get(id_key)
            if not oid:
                continue
            done, exq, st = cancel_and_reconcile(oid)
            if not done:
                unresolved.append(f"{id_key}({st})")
                continue
            if id_key == "entry_order_id":
                # 撤单竞态期: 上次查询到撤单生效之间可能又成交了几张, 必须回填(绝对量)
                if exq > p.get("filled", 0):
                    log(f"   ℹ️ {osi} 撤单竞态期入场又成交至{exq}张, 已回填")
                    p["filled"] = exq
            else:
                d = _credit_leg(p, id_key, exq)
                if d:
                    log(f"   ℹ️ {osi} {id_key} 撤单前已成交{exq}张(补记{d}), 防超卖")
            p[id_key] = None
        if unresolved:
            # 必须持久化重试意图: 否则一次性出场信号(站长all out/9ema出场)就永久丢了
            p["pending_action"] = "full_exit"
            p["pending_action_reason"] = reason
            p["pending_action_ts"] = time.time()
            log(f"   ⚠️ {osi} 挂单未确认终态 {unresolved} → 本轮不发卖单, 已记待办下轮重试")
            journal(ev="close_deferred", osi=osi, unresolved=unresolved, reason=reason)
            _alert(f"⚠️ enrich {osi} 平仓推迟({reason}): {unresolved} 未确认终态, 下轮重试")
            _save(POS_JSON, positions)
            return
        remain = p.get("filled", 0) - p.get("sold", 0)
        if remain > 0:
            _start_exit(positions, osi, p, remain, reason)
        else:
            p["status"] = "closed"
            p.pop("pending_action", None); p.pop("pending_action_reason", None)
            log(f"✅ {osi} 无剩仓, 直接标记已平 ({reason})")
            _save(POS_JSON, positions)
    finally:
        _closing.discard(osi)
        if positions.get(osi, {}).get("status") == "closed":
            _optfail.pop(osi, None)


def mirror_reduce(positions: dict, osi: str, level: str):
    """镜像站长减仓 (2张粒度): 首次部分减→卖1张留跑; 已减过→partial忽略/vague全平。"""
    p = positions.get(osi)
    if not p or p["status"] in ("closed", "closing"):
        return                              # closing = 已有卖单在途, 等它终态再说, 否则重复卖
    if p["status"] == "pending":            # 还没成交他就开始出 → 撤单/全清, 别再进
        close_position(positions, osi, "站长已出(未完全入场)")
        return
    remain = p.get("filled", 0) - p.get("sold", 0)
    if remain <= 0:
        if not p.get("entry_order_id"):
            p["status"] = "closed"
        _save(POS_JSON, positions); return
    if not p.get("reduced"):
        if remain >= 2:
            half = max(1, remain // 2)
            if not cancel_stop(p):         # 必须确认止损腿撤净, 否则旧止损单+新卖单=双卖
                log(f"   ⏸️ {osi} 镜像减仓暂缓: 止损腿未确认撤净")
                _save(POS_JSON, positions); return
            log(f"🪞 {osi} 镜像减仓: 市价卖{half}张, 剩{remain-half}张继续跑")
            # sold 由 ⓪ 在确认成交后更新(原实现提交成功即 sold+=half, 订单被拒也算已卖)
            _start_exit(positions, osi, p, half, "站长减仓", intent="reduce")
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
        if p.get("fail_stop"):
            continue      # 关键状态落盘失败 → 停止对该仓位的自动交易, 等人工介入(见 _save_critical)
        time.sleep(0.4)   # API限速保护 (429防护, 昨夜网络抖动后重试风暴教训)
        # ⓪ 在途卖单对账 —— 【sold 只在这里更新】
        #   提交成功≠成交: submit_order 只返回委托号, 订单随后可能被拒绝/部分成交/长期挂起。
        #   原实现在提交成功时就 sold+=qty 并标 closed, 一旦订单没真成交, bot 就再也不管这个仓位了。
        if p["status"] == "closing" and p.get("exit_order_id"):
            st, exq, exp = _order_state(p["exit_order_id"])
            if st in _TERMINAL:
                want = p.get("exit_qty", 0)
                if exq > 0:
                    p["sold"] = p.get("sold", 0) + exq
                p["exit_order_id"] = None
                journal(ev="exit_fill", osi=osi, qty=exq, want=want, state=st, px=exp,
                        reason=p.get("exit_reason"))
                # 只有【全部成交】才落标志: 部分成交也置位会让 ensure_protection 的
                # not tp1_done / ③b 的 not reduced 守卫永久关闭该止盈通道, 剩余张数再无止盈出口
                if exq >= want > 0 and p.get("exit_intent") in ("tp1", "reduce"):
                    p["reduced"] = True
                    if p.get("exit_intent") == "tp1":
                        p["tp1_done"] = True
                if exq >= want and want > 0:
                    log(f"✅ {osi} 卖单全部成交 {exq}张 @ ${exp} ({p.get('exit_reason')})")
                    push_discord(f"🔻 enrich已成交 {osi} ×{exq}张 @ ${exp} ({p.get('exit_reason')})")
                else:
                    _alert(f"⚠️ enrich {osi} 卖单只成交 {exq}/{want}张 ({st}) — 剩余仓位交回正常管理")
                rem = p.get("filled", 0) - p.get("sold", 0)
                if rem <= 0 and not p.get("entry_order_id"):
                    p["status"] = "closed"
                    p.pop("pending_action", None); p.pop("pending_action_reason", None)
                    _optfail.pop(osi, None)
                    _save(POS_JSON, positions)
                    continue
                # 还有剩仓(或入场单仍在途) → 回到 open 由后续分支重新配保护腿, 绝不提前 closed
                p["status"] = "open"
                _save(POS_JSON, positions)
            elif time.time() - p.get("exit_ts", 0) > EXIT_STUCK_SEC:
                # 逃生舱: 没有它, 一张永远查不到终态的卖单(如券商归档旧单致 order_detail 持续报错)
                # 会让仓位永久卡在 closing —— ⓪ 的 continue 会跳过止损/9ema, 而到期强平的门是
                # status in (pending,open) 也进不来 → runner 无止损裸奔至归零。
                done_x, exq_x, st_x = cancel_and_reconcile(p["exit_order_id"])
                if done_x:
                    if exq_x > 0:
                        p["sold"] = p.get("sold", 0) + exq_x
                    p["exit_order_id"] = None
                    # 必须与正常终态分支同一口径: 入场单仍在途时绝不标closed(否则后续成交=孤儿仓)
                    _rem_x = p.get("filled", 0) - p.get("sold", 0)
                    p["status"] = "closed" if (_rem_x <= 0 and not p.get("entry_order_id")) else "open"
                    _alert(f"⚠️ enrich {osi} 卖单久挂未成({p.get('exit_reason')}) → 已撤并对账"
                           f"(成交{exq_x}张), 仓位交回正常管理")
                    journal(ev="exit_stuck_recovered", osi=osi, exq=exq_x, state=st_x)
                    _save(POS_JSON, positions)
                    if p["status"] == "closed":
                        continue
                else:
                    p["exit_stuck_alerts"] = p.get("exit_stuck_alerts", 0) + 1
                    if p["exit_stuck_alerts"] % 10 == 1:      # 节流, 否则每60秒刷一条
                        _alert(f"🚨 enrich {osi} 卖单卡死无法撤销({st_x}, {p.get('exit_reason')})"
                               f" — 仓位当前无任何自动保护, 请人工处理券商挂单")
                    _save(POS_JSON, positions)
                    continue
            else:
                continue      # 卖单在途期间不做任何其它动作, 防重复卖
        # ⓪b 延迟动作重试 —— P0: 上轮因挂单未确认终态而推迟的平仓必须真的重来。
        #     止损/到期强平会因条件持续而自然重触发, 但站长"all out"/9ema出场是
        #     【一次性】信号, 没有这个待办队列就永久丢失。
        if p.get("pending_action") == "full_exit" and p["status"] in ("pending", "open"):
            _rsn = p.get("pending_action_reason") or "延迟平仓重试"
            log(f"🔁 {osi} 重试待办平仓 ({_rsn})")
            close_position(positions, osi, _rsn)
            if p.get("status") in ("closing", "closed") or p.get("pending_action"):
                continue        # 已发出卖单 或 仍未成功 → 本轮到此为止
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
                elif exq > 0:
                    # 部分成交 → 【立刻撤掉未成交余量】, 不等TTL。
                    #   保护腿按 filled 定张且有 not tp_order_id 守卫, 不会随后续成交扩容 →
                    #   账面3张券商侧只保护1张; 且TP成交后 filled-sold==0 会被标closed, 而入场
                    #   单还在 → 后续成交变孤儿。先撤净让 filled 定型, 状态机才可对账。
                    #   策略上也一致: "不追高"的另一半就是不摊。
                    done2, exq2, st2 = cancel_and_reconcile(p["entry_order_id"])
                    if exq2 > p.get("filled", 0):
                        p["filled"] = exq2
                    if done2:
                        p["entry_order_id"] = None
                        log(f"⚠️ {osi} 入场部分成交{p['filled']}张 → 已撤未成交余量(不摊)")
                        journal(ev="entry_partial_cancel", osi=osi, filled=p["filled"])
                    else:
                        log(f"   ⚠️ {osi} 入场部分成交但撤单未确认({st2}), 下轮重试")
                    # 无论撤单是否确认, 已成交部分立即转open并配保护腿:
                    # 保护腿是卖单且张数≤filled, 最坏只是欠保护(有界), 绝不会超卖。
                    if p["status"] == "pending":
                        p["status"] = "open"
                    ensure_protection(positions, osi, p)
                _save(POS_JSON, positions)
        # ①b 保护腿对账 (审计bug#1: 纯事件驱动下ID丢失/挂单失效=永久裸奔; ensure_protection幂等, 每轮补齐缺失的腿)
        if p["status"] == "open" and p.get("filled", 0) - p.get("sold", 0) > 0 and p.get("avg", 0) > 0:
            ensure_protection(positions, osi, p)
        # ② 券商侧止损单状态 (真实账户模式)
        if p["status"] == "open" and p.get("stop_order_id"):
            st, exq, exp = _order_state(p["stop_order_id"])
            if st == "Filled":
                _credit_leg(p, "stop_order_id", exq or p.get("stop_qty", 0))
                p["stop_order_id"] = None
                if p["filled"] - p["sold"] <= 0 and not p.get("entry_order_id"):
                    p["status"] = "closed"
                log(f"🛑 {osi} 券商侧止损成交 {p.get('stop_qty')}张 @ ${exp}")
                journal(ev="stop_fill", osi=osi, px=exp, qty=p.get("stop_qty"))
                push_discord(f"🛑 enrich止损成交 {osi} ×{p.get('stop_qty')}张 @ ${exp}")
                _save(POS_JSON, positions)
                continue
            elif st in _CANCELLED:
                # 审计bug#2: 止损被外部撤销若不清ID, 轮询止损被stop_order_id条件跳过=止损通道全灭
                _credit_leg(p, "stop_order_id", exq)
                p["stop_order_id"] = None
                log(f"⚠️ {osi} 券商侧止损单失效({st}) → 清ID, 轮询止损接管+重挂")
                journal(ev="stop_order_lost", osi=osi, state=st)
                ensure_protection(positions, osi, p)
                _save(POS_JSON, positions)
        # ③ 券商侧一档止盈状态
        if p["status"] == "open" and p.get("tp_order_id"):
            st, exq, exp = _order_state(p["tp_order_id"])
            if st == "Filled":
                _credit_leg(p, "tp_order_id", exq or p.get("tp_qty", 0))
                p["tp_order_id"] = None
                p["reduced"] = True          # 止盈成交=完成首次减仓
                p["tp1_done"] = True
                log(f"💰 {osi} 一档止盈成交 {p.get('tp_qty')}张 @ ${exp}")
                journal(ev="tp_fill", osi=osi, px=exp, qty=p.get("tp_qty"))
                push_discord(f"💰 enrich止盈成交 {osi} ×{p.get('tp_qty')}张 @ ${exp} — 剩{p['filled']-p['sold']}张")
                # 入场单仍在途时绝不标closed: 否则后续成交的张数会变成无人管理的孤儿仓
                if p["filled"] - p["sold"] <= 0 and not p.get("entry_order_id"):
                    p["status"] = "closed"
                _save(POS_JSON, positions)   # 机械模式: 不移保本, -60%初始止损全程 (最终定稿)
            elif st in _CANCELLED:
                _credit_leg(p, "tp_order_id", exq)   # 部分成交后被撤: 先记账防超卖(审计bug#4)
                p["tp_order_id"] = None      # 挂单失效 → 重配保护
                ensure_protection(positions, osi, p)
        # ③c 券商侧二档止盈状态 (机械模式; 成交=runner武装9ema)
        if p["status"] == "open" and p.get("tp2_order_id"):
            st, exq, exp = _order_state(p["tp2_order_id"])
            if st == "Filled":
                _credit_leg(p, "tp2_order_id", exq or p.get("tp2_qty", 0))
                p["tp2_order_id"] = None
                p["tp2_done"] = True
                p["reduced"] = True
                p["armed"] = True            # 摸到+60% → runner进入9ema拖尾管理
                log(f"💰 {osi} 二档止盈成交 {p.get('tp2_qty')}张 @ ${exp} — runner已武装15m9ema")
                journal(ev="tp2_fill", osi=osi, px=exp, qty=p.get("tp2_qty"))
                push_discord(f"💰 enrich二档止盈 {osi} ×{p.get('tp2_qty')}张 @ ${exp} — 剩{p['filled']-p['sold']}张runner(9ema拖尾)")
                if p["filled"] - p["sold"] <= 0 and not p.get("entry_order_id"):
                    p["status"] = "closed"
                _save(POS_JSON, positions)
            elif st in _CANCELLED:
                _credit_leg(p, "tp2_order_id", exq)
                p["tp2_order_id"] = None
                ensure_protection(positions, osi, p)
        # ③b 轮询止盈 (真实账户模式: 无止盈挂单时; 只管一档)
        remain = p.get("filled", 0) - p.get("sold", 0)
        if p["status"] == "open" and not p.get("reduced") and not p.get("tp_order_id") \
                and not p.get("tp2_order_id") and remain > 0 and p.get("avg", 0) > 0:
            last = _option_last(osi)
            tp_qty, tp_px = _tp_params(remain, p["avg"])
            if last is not None and last >= tp_px:
                # 必须先确认止损腿撤净, 否则"旧止损单 + 新止盈市价单"会双卖
                if not cancel_stop(p):
                    log(f"   ⏸️ {osi} 轮询止盈暂缓: 止损腿未确认撤净")
                    _save(POS_JSON, positions)
                    continue
                log(f"💰 {osi} 轮询止盈触发: 最新${last}≥${tp_px}, 市价卖{tp_qty}张")
                # 提交后进入 closing, sold/reduced/tp1_done 由 ⓪ 在确认成交后才落
                # (原实现提交成功即记账, 且 "remain-tp_qty<=0 → closed" 还在 if ok 之外:
                #  只剩1张时下单失败照样标closed, 真实仓位从此无人管理)
                _start_exit(positions, osi, p, tp_qty, "轮询止盈", intent="tp1")
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
            # 必须走 cancel_and_reconcile: 撤单只是"请求", 第20分钟这一刻订单可能正好成交,
            # 撤单请求照样返回成功。原写法据此清ID并在 filled==0 时标closed →
            # 券商侧真实多头永久失联(无止损/无强平/日志无痕), 是本次重构要根除的那类bug的残留。
            done_t, exq_t, st_t = cancel_and_reconcile(p["entry_order_id"])
            if exq_t > p.get("filled", 0):
                p["filled"] = exq_t
                _f = exq_t
                _d = f"部分成交{_f}张"
            if not done_t:
                log(f"   ⚠️ {osi} 入场单TTL撤单未确认终态({st_t}), 保留ID下轮重试")
                journal(ev="entry_ttl_cancel_unconfirmed", osi=osi, state=st_t)
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
            # 锚点【处理成功后才前移】。原本在循环首行无条件前移, 处理某条停机期间的 EXIT 抛异常时
            # 锚点已越过它并最终落盘 → 该出场信号永久丢失(与 on_message 同一类缺陷, 上轮只修了实时路径)
            if m.author.id != AUTHOR_ID or not m.content:
                state[key] = str(m.id)        # 明确无需处理 → 可安全前移
                continue
            one = " ".join(m.content.split())[:90]
            try:
                if ch_id == ANDY_CHANNEL_ID:
                    handle_andy(m.content, m.created_at, m.id)   # 纯记录, 无下单
                    continue
                s = parse_signal(m.content, m.created_at.date())
                if s.kind in ("BUY", "BUY_AMBIG", "BUY_NOEXPIRY"):
                    note = f"⏰ 停机期间错过的enrich买入信号 (仅提醒, 不按旧价补单): {one}"
                    log(note); push_discord(note)
                    journal(ev="missed_during_downtime", sig=one, ts_signal=str(m.created_at))
                elif s.kind == "EXIT":
                    # 必须 to_thread: handle 要抢 _handle_lock, 而 _managed_tick 现在会长时间
                    # 持有该锁; 在事件循环线程里同步抢锁会挂住 discord 心跳 → 断线重连正反馈(复审发现)
                    import asyncio as _aio
                    await _aio.to_thread(handle, m.content, m.created_at.date(),
                                         m.id, seen, positions)   # 出场晚跟好过不跟
                state[key] = str(m.id)        # 处理成功才前移锚点
            except Exception as e:
                # 保留锚点: 下次重连会重新回看这条(BUY有seen[msg_id]兜底, EXIT重复执行也安全 ——
                # close_position 按 remain=filled-sold 计算, 已平仓则直接返回)
                log(f"追赶处理异常(保留锚点待下次重试): {e}")
                break                          # 不跳过它继续推进后面的消息, 否则锚点仍会越过它
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


def _handle(text: str, msg_date: date, msg_id: int, seen: dict, positions: dict, msg_dt=None):
    s = parse_signal(text, msg_date)
    one = " ".join(text.split())[:100]
    if s.kind == "NOISE":
        return          # 规则未识别的消息一律忽略(原有LLM"捞漏出场"分支已随LLM移除)

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
                if p["status"] in ACTIVE_STATUSES
                and (s.ticker == "*" or p.get("ticker") == s.ticker)]
        if s.exit_level == "alert" and LIVE:
            # 多票/豁免词的出场信号规则层无法可靠切分到具体标的 → 仅提醒, 由人判断。
            # (原走LLM仲裁, 已随LLM移除; 宁可漏跟也不要按错误的标的集合平仓)
            note = f"⚠️ enrich出场信号含多票/豁免词, 规则无法明确标的, 仅提醒 [{s.ticker}]: {one}"
            log(note); push_discord(note)
            journal(ev="exit_alert_ambiguous", ticker=s.ticker, sig=one)
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
    #   机械出场的-50%止损已兜住"赌方向"下行, 小仓(净值×⅓)限损即可。标记→走lotto仓位档(小仓)。
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

    # BUY_NOEXPIRY: 四要素齐、仅缺到期(已推断周五/0DTE) → 直接按lotto档跟
    # (2026-07-19复盘: 历史11/11条BUY_NOEXPIRY全为真实lotto/scalp入场, 零误报; ARM 7/17漏单教训。
    #  兜底: 权利金≤$5 + TTL20分 + -60%止损 + 小仓。原有LLM佐证闸已随LLM移除)
    qty = CONTRACTS
    is_ambig = False
    if s.kind == "BUY_NOEXPIRY":
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
    if old and old.get("status") in ACTIVE_STATUSES:
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
        # 潜在最大敞口: ①含 closing(卖单未确认成交=仓位还在) ②入场单仍在途时按 max(下单量, 已成交)
        #   计 —— 计划20张只成交1张时 filled=1, 但另19张买单还活着, 按 filled 算会严重低估
        def _exp_qty(q):
            base = max(q.get("qty", 0), q.get("filled", 0)) if q.get("entry_order_id") \
                else q.get("filled", 0) or q.get("qty", 0)
            return max(0, base - q.get("sold", 0))
        gross = sum(_exp_qty(q) * q.get("limit", 0) * 100
                    for q in positions.values() if q.get("status") in ACTIVE_STATUSES)
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
            if not _save(POS_JSON, positions):
                # 券商已收单但本地存不下 → 崩溃/重启后完全不知道有这个仓位。撤单是唯一安全解。
                _alert(f"🚨 enrich {osi} 入场单已提交但仓位落盘失败 → 立即撤单, 勿留无记录持仓")
                journal(ev="entry_persist_failed", osi=osi, order_id=r)
                done_c, exq_c, st_c = cancel_and_reconcile(r)
                if done_c and exq_c == 0:
                    positions.pop(osi, None)
                    log(f"   ✅ {osi} 已撤净(未成交), 本次进场作废")
                else:
                    _alert(f"🚨 enrich {osi} 撤单未确认或已成交{exq_c}张({st_c}) — 请立即人工核对券商持仓!")
                push_discord(plan)
                return
            _tpm = MECH_TP_MULT if EXIT_MODE == "mechanical" else TP_MULT
            plan += (f"\n  ✅已提交 order_id={r} (成交后自动挂+{(_tpm-1)*100:.0f}%止盈"
                     + (f", 二档后武装15m9ema管runner, -{(1-MECH_STOP_MULT)*100:.0f}%止损全程不移保本"
                        if EXIT_MODE == "mechanical" else "") + ")")
            log(f"  ✅已提交 {r}")
            journal(ev="entry_submit", osi=osi, ticker=s.ticker, right=s.right, strike=s.strike,
                    expiry=str(s.expiry), limit=s.limit_price, qty=qty, order_id=r, sig=one)
        else:
            plan += f"\n  ❌下单失败: {r}"
            log(f"  ❌下单失败: {r}")
    else:
        plan += "\n  (DRY_RUN 未下单)"
    push_discord(plan)

    # 去重分两层(原来混在一起, 导致下单API临时失败后同一合约当天再也进不去):
    #   · 消息级(msg_id): 同一条Discord消息不重复处理 —— 总是登记
    #   · 交易级(osi:date / soft_key): 只有券商真的接受了入场单才登记
    seen[str(msg_id)] = key
    # 判据必须是"本次是否真的新建了仓", 不能用 `osi in positions` ——
    # positions 从不删除已平仓记录, 同合约当日二次信号下单失败时会被误判成建仓成功
    if positions.get(osi, {}).get("entry_order_id") and positions[osi].get("opened") == str(msg_date):
        seen[key] = str(msg_id)
        seen[soft_key] = osi
    else:
        log(f"   ↩️ {osi} 未建仓(未下单/下单失败), 不写交易级去重键, 允许后续同合约信号重试")
    _save(SEEN_JSON, seen)


_lockf = None   # 单实例锁句柄 (须常驻引用)


def main():
    global _lockf
    # 单实例锁 (审计bug#5: dev/mock实例与launchd并发写同一状态曾致真实止盈单变孤儿)
    import fcntl
    OUT.mkdir(parents=True, exist_ok=True)   # 全新部署时 output/ 不存在, 否则开锁文件即 FileNotFoundError
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
    if LIVE:
        reconcile_with_broker(positions)      # 只读对账: 本地JSON格式完好≠与券商一致
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
        # 锚点只在【处理成功或明确无需处理】之后才前移。原实现在函数第一行就 bump_last,
        # 一旦后面处理抛异常, 锚点已越过该消息 → 重启后 catch_up 不会再补 → 信号永久丢失。
        # (bump_last 内部还会 _load 状态文件, 裸调若抛错会让整个 on_message 在第一行就断)
        if msg.channel.id not in (CHANNEL_ID, ANDY_CHANNEL_ID):
            return
        def _anchor():
            try:
                bump_last(msg.channel.id, msg.id)
            except Exception as e:
                log(f"锚点更新异常(不影响本条处理): {e}")
        if msg.author.id != AUTHOR_ID or not msg.content:
            _anchor()                              # 明确无需处理 → 可以安全前移
            return
        if msg.channel.id == ANDY_CHANNEL_ID:      # andy: 只观察记录, 永不下单
            try:
                handle_andy(msg.content, msg.created_at, msg.id)
                _anchor()
            except Exception as e:
                log(f"andy处理异常(保留锚点待重试): {e}")
            return
        try:
            import asyncio
            await asyncio.to_thread(handle, msg.content, msg.created_at.date(), msg.id,
                                    seen, positions, msg.created_at)
            _anchor()                              # 处理成功才前移
        except Exception as e:
            _alert(f"🚨 enrich 消息处理异常, 已保留锚点待重启补处理: {e}")

    client.run(token, log_handler=None)


if __name__ == "__main__":
    main()
