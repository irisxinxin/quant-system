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

仓位管理 (机械出场, 无AI; 站长的出场消息只播报不执行 —— 2026-07-22 F不对称过夜定案):
  · 止盈: 入场成交 → 挂单档 GTC 限价卖 +30%卖½ (剩半跑runner; 不再挂二档卖单)
  · runner: 不保本(MECH_BE=0, 让它扛回撤接大runner如GOOGL) → 摸+60%武装 → 守【标的】15分9ema连破2根市价平
  · 止损: -50% 全程 (MECH_STOP_MULT=0.5; 模拟盘唯一通道是轮询, 真实盘走券商侧MIT)
  · 过夜: 【未落袋满仓】(not reduced)非到期日15:40 ET收盘前平, 不裸扛过夜避IBM式-94%gap; 已落袋runner留过夜
  · 到期强平: 到期日 15:40 ET 剩仓市价全平, 防归零/行权
  · 天然风控: 买期权最大亏=权利金 (每笔 ≤ MAX_PREMIUM×100×张数)

用法:
  python3 discord_enrich_bot.py                                  # DRY_RUN
  ENRICH_LIVE=true OPTION_CONTRACTS=2 python3 discord_enrich_bot.py   # 模拟盘真下单
环境: DISCORD_BOT_TOKEN(必须) / OPTION_CONTRACTS(默认1) / MAX_PREMIUM(默认5.0)
      / MECH_TP_MULT(1.3) / MECH_TP_FRAC(0.5) / MECH_TP2_MULT(1.6) / MECH_STOP_MULT(0.5)
      / MECH_BE(F策略=0) / F_EOD_CLOSE_UNREDUCED(1) / DISCORD_WEBHOOK_URL(可选回报推送)
"""
import os, re, sys, json, base64, time, math
from datetime import datetime, date, time as dtime, timezone as _tzone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).parent))
# 仓库根也要进 path: 本 bot 复用根目录的 backtest_andy 的出场正则(parse_entry/EXIT_*_RE)。
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(1, str(REPO_ROOT))
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
LOTTO_CONTRACTS = int(os.environ.get("LOTTO_CONTRACTS", "1"))  # 歧义/lotto单张数(POSITION_USD=0时用)
# (TP_MULT / STOP_MULT / SWING_STOP_MULT 随 mirror 出场模式一并删除 —— 2026-07-20)
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

# 歧义单(缺 call/put)消歧阈值, 单位是 |ln(报价/信号价)| 偏离度。
#   MAX_DEV   最贴合的一边也偏离超过它 → 两边都不像, 不猜 (0.79 ≈ ln2.2, 与原窗口上界等价)
#   MIN_MARGIN 两边偏离度之差小于它 → 模棱两可, 不猜
# 阈值按真实数据定, 不是拍脑袋: 历史两次真实消歧的差距是 0.90 / 0.88, 基线 1.16;
# 而应当拒绝的误判场景差距是 0.01 / 0.12 —— 中间一大片真空, 取 0.5 两侧都不贴边。
AMBIG_MAX_DEV = float(os.environ.get("AMBIG_MAX_DEV", "0.79"))
AMBIG_MIN_MARGIN = float(os.environ.get("AMBIG_MIN_MARGIN", "0.5"))

# 停机追赶: 单频道最多回看多少条(翻页累计)。超出会告警而不是静默截断。
CATCHUP_MAX = int(os.environ.get("CATCHUP_MAX", "500"))
# 停机期间的出场信号超过此秒数则只提醒不执行。买入侧是 STALE_BUY_SEC=180(scalp几分钟就死),
# 出场侧放宽到 6 小时: 持仓晚跟好过不跟, 但隔夜/隔日的清仓令绝不能拿来平今天新建的仓。
STALE_EXIT_SEC = int(os.environ.get("STALE_EXIT_SEC", "21600"))

# ── 出场规则 (2026-07-22 引擎修bug后重测定案, 用户拍板 F不对称过夜; bt_be_grid + overnight_asym) ──
# 机械出场(无AI): +30%卖½ → runner不保本(MECH_BE=0, 维持-50%不移入场价) → 剩仓摸+60%武装
#                → 守【标的】15分9ema连破2根拖尾 → 到期强平。初始止损-50%。
#                → 未落袋满仓(not reduced)非到期日15:40收盘前平(F_EOD_CLOSE_UNREDUCED), 不裸扛过夜。
# 回测教训: 旧引擎缺【武装门控】(runner一进场就守9ema)→ 结论"保本>不保本"是bug造成的假象;
#   修复后重测(n=86, 加TTL/premium$5/到期15:40对齐bot): 保本vs不保本是【权衡】——保本胜率79%但砍死
#   回撤后爆发的runner(GOOGL被砍+15%次日却+1270%), 不保本加权+42%接得住。F不对称=runner不保本扛过夜
#   接GOOGL + 未落袋满仓收盘平避IBM式-94%裸扛gap: 加权+39% 胜率56% 最差-50% 灾难9(A现状是-94%/16)。
# 2026-07-19 更早旧定稿"两档+30卖⅓/+60卖⅓ -60全程无保本"亦已被替换。只信相对排序不信幅度(真实K仅10单)。
# 站长的出场消息【只播报不执行】。
#
# 2026-07-20 删除了 EXIT_MODE 开关与 mirror(跟站长出场)模式。理由:
#   · 策略已定稿用机械出场, mirror 不再是候选
#   · 留着一条不跑、没测透的路径本身就是风险源 —— 它的默认值曾经是 fail-open
#     (环境变量没到位就静默切成另一套策略), 且仿真在它上面挖出过裸空和三处信号丢弃
# 要回切请从 git 恢复(commit e6a0041 及之前), 别在这里加开关。
# 2026-07-21 回测定案(bt_be_grid, n=127 2-7月): 单档止盈 + 首档后保本 + 15m9ema。
#   2026-07-22 引擎修bug(武装门控缺失+sticky)后重测: 保本vs不保本是【权衡】不是碾压——
#   保本胜率高(79%)但会把回撤到入场价的runner砍死(GOOGL被砍在+15%, 次日却+1270%);
#   不保本加权高(+42%)因接得住这类。IBM-94%则是【没落袋满仓裸扛过夜】被逆gap打穿。
#   → 采用【F不对称过夜策略】: runner不保本(MECH_BE=0)让它扛过夜接大runner;
#     未落袋满仓(not reduced)非到期日收盘前平, 不裸扛过夜。详见 bt_be_grid.py + scratchpad/overnight_asym.py。
MECH_TP_MULT = float(os.environ.get("MECH_TP_MULT", "1.3"))     # 首档止盈 +30%
MECH_TP_FRAC = float(os.environ.get("MECH_TP_FRAC", "0.5"))     # 首档卖多少(0.5=卖半, 剩半跑runner)
MECH_TP2_MULT = float(os.environ.get("MECH_TP2_MULT", "1.6"))   # +60% = runner武装9ema的阈值(不再挂二档卖单)
MECH_BE = os.environ.get("MECH_BE", "1") == "1"                 # 首档止盈后止损移到入场价(保本); F策略=0(runner不保本, 让它扛回撤接大runner)
MECH_STOP_MULT = float(os.environ.get("MECH_STOP_MULT", "0.5")) # 初始止损 -50% (首档止盈前有效; 之后若MECH_BE则移保本, 否则维持-50%)
MECH_EMA_MIN = int(os.environ.get("MECH_EMA_MIN", "15"))        # 9ema时间框架(分钟)
MECH_EMA_N = int(os.environ.get("MECH_EMA_N", "2"))             # 连破N根(已完成bar)出runner
ENTRY_TTL_SEC = int(os.environ.get("ENTRY_TTL_SEC", "1200"))    # 在途入场单TTL: 20分钟未成交撤单
# (审计发现+MSFT复盘: 挂一天的限价单只会在期权崩盘穿价时成交=专门接刀; "不追高"的另一半是"不接刀")
F_EOD_CLOSE_UNREDUCED = os.environ.get("F_EOD_CLOSE_UNREDUCED", "1") == "1"  # F策略: 非到期日15:40 ET平【未落袋满仓】(不裸扛过夜); 已落袋runner留过夜
ET = ZoneInfo("America/New_York")

# ⚠ 状态目录固定在【仓库根】的 output/, 不跟着本文件走。
# bot 从根目录搬进 bots/enrich/ 时, 原来的 Path(__file__).parent/"output" 会指向
# bots/enrich/output/ → 持仓/去重表/追赶锚点全部读不到 = 状态静默清零。
# 归档 job(enrich_archive.py) 和一堆回测脚本也都按仓库根的 output/ 找文件。
OUT = REPO_ROOT / "output"
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


def _quote_usable(q, osi: str):
    """把一条 OptionQuote 校验成可交易价格。返回价格, 或 None=不可用。

    三项校验(原实现直接返回 last_done, 会打出假止损):
      ① last_done>0 —— 无成交时返回0, 而 0 <= 任何止损价 → 立刻假止损全平
      ② 报价新鲜度 —— 低流动性期权的"最新成交"可能是几天前的旧价, 拿它判止损等于看历史
      ③ trade_status==Normal —— 停牌/熔断/退市期间的价格不可据以交易
    抽出来给 _option_last 和 resolve_direction 共用 —— 判方向和判止损必须同一套标准,
    而且【只能有一份实现】(测试里另写一份 _osi 的教训: 两份实现只会一起错)。"""
    last = float(getattr(q, "last_done", 0) or 0)
    if last <= 0:
        return None
    ts = getattr(q, "timestamp", None)
    if ts is not None:
        try:
            if ts.tzinfo is None:
                # SDK 返回的是 naive datetime, 实测其值 = 【本机时区】的墙上时间
                # (2026-07-20 用真实API核对: SDK ts 23:29:27 / 本机SGT 23:29:28 / UTC 15:29:28
                #  → 按SGT解释年龄1秒✅; 按UTC解释 -28799秒❌)。astimezone() 对 naive 值
                # 正是"按本机时区解释", 所以直接用它。
                #
                # ⚠ 这里曾经写 replace(tzinfo=utc), 注释还断言"按本机时区解释会算成8小时前
                # =止损全灭" —— 诊断反了。真实后果是年龄恒等于【真实年龄 − 28800】, 于是
                # `age > QUOTE_MAX_AGE_SEC` 的实际拒绝线从 1 小时变成 9 小时, 覆盖整个交易日
                # 加隔夜: 流动性差的合约今天没成交时, 会拿昨天的 last_done 判今天的止损
                # → 明明已经跌穿却判"没跌穿"(漏止损)。
                # 仿真抓不到这类问题 —— FakeQuotes 的 quote_age_sec 是直接赋值的, 根本不过时区。
                ts = ts.astimezone()
            age = (datetime.now(_tzone.utc) - ts.astimezone(_tzone.utc)).total_seconds()
            if age > QUOTE_MAX_AGE_SEC:
                log(f"   ⏱️ {osi} 报价陈旧({age/60:.0f}分钟前), 不可用")
                return None
        except Exception:
            pass
    tst = str(getattr(q, "trade_status", "Normal")).split(".")[-1]
    if tst != "Normal":
        log(f"   ⛔ {osi} 交易状态={tst}, 不可用")
        return None
    return last


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
        return _quote_usable(o[0], osi)
    except Exception as e:
        _rl_hit(e)
        return None


def _await_terminal(oid: str, tries: int = 6, delay: float = 0.5):
    """轮询订单直到终态。返回 (是否终态, 状态名, 累计成交量)。
    非终态或查询失败 → 第一项 False, 此时调用方【禁止】再发新的卖单(旧单可能仍会成交)。"""
    st, exq, exp = None, 0, 0.0
    for i in range(tries):
        st, exq, exp = _order_state(oid)
        if st in _TERMINAL:
            return True, st, exq, exp
        if i < tries - 1:
            time.sleep(delay)
    return False, st, exq, exp


def cancel_and_reconcile(oid: str):
    """撤单 + 确认终态 + 取回最终成交量。返回 (已确认, 累计成交量, 状态名)。

    撤单接口只是"请求撤销", 返回成功不代表订单已停止 —— 可能正在撮合、可能已部分成交。
    必须查到终态才能确定它不会再吃单; 查不到就当它还活着, 绝不能在它之上再发卖单(=超卖裸空)。"""
    _cancel(oid)                      # 撤单请求本身失败也继续确认(可能已成交/已在撤销中)
    ok, st, exq, exp = _await_terminal(oid)
    return ok, exq, st, exp


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
    """(止盈张数, 止盈价) — 首档卖 MECH_TP_FRAC @+MECH_TP_MULT。"""
    return max(1, round(remain * MECH_TP_FRAC)), round(avg * MECH_TP_MULT, 2)


def _stop_price(p: dict) -> float:
    """当前止损价 —— 止损判定/挂单/心跳显示【唯一】来源, 三处都调它, 不各写一份。

    保本(MECH_BE 开 且已首档止盈): 止损=入场价 avg。回测(bt_be_grid)结论: 保本在每个结构
    上胜率+5~10pp、最差月更好, 且不牺牲上行(只在首档止盈后启用, 那时已落袋一档利润)。
    否则: avg × stop_mult。都不低于 MIN_TICK —— 低价期权 avg×0.5 可能落在最小报价档以下,
    last 永远够不到 → 止损失效(旧bug)。

    保本是【全局】行为, 对所有仓位一视同仁(2026-07-21 用户拍板: 模拟单不做新旧隔离)。"""
    avg = p.get("avg", 0) or 0.0
    if MECH_BE and (p.get("reduced") or p.get("tp1_done")):
        return max(MIN_TICK, avg)
    return max(MIN_TICK, avg * p.get("stop_mult", MECH_STOP_MULT))


_ema_cache = {}   # ticker -> (checked_at_epoch, break_count)
_optfail = {}     # osi -> 期权报价连续失败轮数 (止损通道健康度; 模拟盘唯一止损通道)
_closing = set()  # 正在平仓的osi — 只存内存, 绝不落盘(落盘会导致崩溃重启后永久无法平仓)
_anchor_mem = {}  # 消息锚点的内存镜像 — 锚点文件损坏且无.bak时用它兜底, 免得一个频道的
                  # 损坏把另一个频道的游标也带走(那会让它退化成"首次运行", 静默丢弃停机窗口)
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
        # 走 _option_last 而不是裸读 last_done: 自动继承 >0 / 新鲜度 / trade_status 三项校验。
        # 原实现裸读 —— 判方向这个不可逆得多的决策, 反而比判止损更不设防:
        # 停牌合约的隔夜旧价照样参与拍板(仿真复现: 停牌CALL+24小时前旧价 → 照买)。
        # 一次查两边(而不是分别调 _option_last): 必须能区分两种情形 ——
        #   · 查询【失败】(网络/API异常) → 那边价格未知, 无从比较 → 整体抛异常 → 拒绝
        #   · 查到了但【不可交易】(无成交/停牌/陈旧) → 那边确认没价, 可用另一边判
        # 分别调 _option_last 会把两者都变成 None, 分不开: 要么误拒(单边无成交也不敢判),
        # 要么误判(单边查询失败却拿另一边拍板 —— 仿真抓到过: CALL查询失败 → 反手下 PUT)。
        # 站长报了权利金说明【他买的那边有成交】, 所以"无成交的那边"大概率不是他买的。
        qs = {q.symbol: q for q in _quote_ctx.option_quote(syms)}
        # 缺条目 = 那边【确认没数据】(无成交), 不是查询失败 —— 查询失败会抛异常, 由外层 except
        # 兜住并整体拒绝。这两者只能靠"抛没抛异常"区分, 拿"返回是否完整"当失败判据会误拒
        # 正常的单边无成交。
        pc = _quote_usable(qs[syms[0]], syms[0]) if syms[0] in qs else None
        pp = _quote_usable(qs[syms[1]], syms[1]) if syms[1] in qs else None
        if not pc and not pp:
            return None, "两边都无可用报价(无成交/停牌/报价陈旧)"

        # 判据是"哪一边【贴合信号价】", 用 |ln(报价/信号价)| 度量偏离(对涨跌对称)。
        # 不能用"哪边便宜"排除: 同行权价的 C/P 结构上必然一虚一实, 站长发的是虚值(便宜),
        # 另一边天然是实值(贵) —— "另一边过高"是常态而非异常。历史两次真实消歧
        # (C=0.8/P=1.975, C=1.17/P=2.81)都是这个形态。
        def _dev(px):
            return None if (not px or px <= 0) else abs(math.log(px / s.limit_price))

        dc, dp = _dev(pc), _dev(pp)
        detail = f"C={pc} P={pp} vs 信号${s.limit_price}"
        cand = [(d, r, px) for d, r, px in ((dc, "C", pc), (dp, "P", pp)) if d is not None]
        cand.sort()
        best_d, best_r, _best_px = cand[0]
        if best_d > AMBIG_MAX_DEV:
            # 最贴合的一边也偏离信号价太远 → 两边都不像, 不猜
            return None, f"消歧失败({detail}) — 最接近的一边偏离{best_d:.2f}已超阈值"
        if len(cand) == 2 and (cand[1][0] - best_d) < AMBIG_MIN_MARGIN:
            # 两边贴合度接近 → 模棱两可。原实现在这里最危险: 用固定窗口[0.4,2.2]做"命中/落选",
            # 于是正确方向涨出上界被判落选、反方向跌进窗口成为唯一命中 → 反手买错方向。
            # 且是悬崖式的: 2.20x 两边命中→安全拒绝, 2.21x→直接下反方向, 一分钱翻转结论。
            return None, (f"消歧失败({detail}) — 两边贴合度接近"
                          f"(C偏离{dc if dc is None else round(dc,2)} "
                          f"P偏离{dp if dp is None else round(dp,2)}), 不猜方向")
        return best_r, f"{detail} → {best_r}(偏离{best_d:.2f})"
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
        # 丢响应: submit_order 抛异常, 但券商【可能已收单】(网络超时/客户端断连)。SDK 无
        # client_request_id 幂等键 → 只能反查 today_orders 看是否真下单了。查到匹配的未终态单
        # → 认领它(返回其id, 视为成功), 否则这张单 bot 不追踪 = 孤儿(对抗测试挖出: 孤儿保护
        # 腿会在+30%意外成交摧毁runner, 或崩盘后残留成裸空空头)。查不到/查失败 → 按失败返回。
        oid = _reclaim_lost_submit(osi, side_buy, remark)
        if oid:
            log(f"   🔁 {osi} 提交丢响应但券商已收单 → 认领 {oid} (remark={remark})")
            journal(ev="submit_lost_reclaimed", osi=osi, order_id=oid, remark=remark)
            return True, oid
        return False, str(e)


def _reclaim_lost_submit(osi: str, side_buy: bool, remark: str):
    """提交丢响应后反查券商是否真收单。返回匹配的【未终态】订单id, 或 None。
    按 symbol+side+remark 精确匹配 —— remark 编码了腿类型(enrich-entry/tp1/stop/exit-*),
    正常路径下同类腿券商侧至多一张在途(挂腿前都有 not xxx_order_id 守卫)。"""
    want_side = "Buy" if side_buy else "Sell"
    try:
        for od in _trade_ctx.today_orders(symbol=osi) or []:
            if str(getattr(od, "symbol", "")) != osi:
                continue
            if str(getattr(od, "side", "")).split(".")[-1] != want_side:
                continue
            if str(getattr(od, "remark", "")) != remark:
                continue
            if str(od.status).split(".")[-1] in _TERMINAL:
                continue
            return str(od.order_id)
    except Exception as e:
        _rl_hit(e)
    return None


_MIT_OK = None   # None=未探测 / True=真实账户支持触价单 / False=模拟盘604050不支持


def ensure_protection(positions: dict, osi: str, p: dict):
    """给剩仓配保护腿(自适应):
       ① 优先挂券商侧MIT止损(bot死了也在) — 真实账户支持
       ② 模拟盘不支持MIT → 止盈挂券商侧限价(抓尖峰) + 止损靠轮询兜底"""
    global _MIT_OK
    remain = p.get("filled", 0) - p.get("sold", 0)
    if remain <= 0 or p.get("avg", 0) <= 0:
        return
    _sm = p.get("stop_mult", MECH_STOP_MULT)
    if _sm > 0 and not p.get("stop_order_id") and _MIT_OK is not False and _sell_budget(p) >= 1:
        _sq = min(remain, _sell_budget(p))
        trig = round(_stop_price(p), 2)      # 保本时=入场价, 否则 avg×stop_mult
        ok, r = _submit(osi, side_buy=False, qty=_sq, price=None, tif_gtc=True,
                        remark="stop", trigger=trig)
        if ok:
            _MIT_OK = True
            p["stop_order_id"], p["stop_qty"], p["stop_order_id_counted"] = r, _sq, 0
            _be = MECH_BE and (p.get("reduced") or p.get("tp1_done"))
            log(f"🛡️ {osi} 券商侧止损已挂: {_sq}张 触发${trig} " +
                ("(保本)" if _be else f"(-{(1-_sm)*100:.0f}%)"))
            journal(ev="stop_place", osi=osi, trigger=trig, qty=_sq, order_id=r)
        elif "604050" in str(r) or "not supported" in str(r).lower():
            _MIT_OK = False
            log("ℹ️ 模拟盘不支持触价单 → 止损轮询兜底, 止盈挂券商侧限价 (真实账户会自动切回止损常驻)")
        else:
            log(f"⚠️ {osi} 止损挂单失败: {r}")
    # 回退模式: 无券商侧止损 → 止盈限价挂券商侧(抓尖峰)
    if not p.get("stop_order_id"):
        # 单档GTC挂: 卖 MECH_TP_FRAC @+MECH_TP_MULT (2026-07-21回测定案, 不再挂二档止盈)。
        # runner 靠"价格摸+60%武装 → 15分9ema拖尾"管, 不依赖二档卖单成交。
        # 【按 remain 而非 filled 定张】: 部分平仓后 sold>0, 用 filled 会挂出多于持仓的卖单。
        # 正常路径(新建仓 sold=0)下 remain==filled, 档位与回测口径完全一致。
        _bud = _sell_budget(p)
        q1, px1 = _tp_params(remain, p["avg"])
        if q1 >= 1 and not p.get("tp_order_id") and not p.get("tp1_done") and _bud >= 1:
            q1 = min(q1, _bud)
            ok, r = _submit(osi, side_buy=False, qty=q1, price=px1, tif_gtc=True, remark="tp1")
            if ok:
                p["tp_order_id"], p["tp_qty"], p["tp_order_id_counted"] = r, q1, 0
                log(f"🎯 {osi} 挂止盈: 卖{q1}张 @ ${px1} (+{(MECH_TP_MULT-1)*100:.0f}%, 剩仓跑runner)")
                journal(ev="tp_place", osi=osi, px=px1, qty=q1, order_id=r)
            else:
                log(f"⚠️ {osi} 止盈挂单失败: {r}")
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
    done, exq, st, _exp = cancel_and_reconcile(oid)
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

def _broker_qty(osi: str):
    """券商侧该合约【实际】持仓张数。拿不到返回 None(调用方必须保守处理)。

    这是 I3 的终极保障: 本地记账再怎么错, 只要卖出量以券商真值封顶, 就不可能裸空。
    (仿真实测: 券商已收单而客户端超时 → 本地以为没卖 → 下一轮重复卖 → 净持仓 -6 张期权空头)"""
    try:
        for ch in _trade_ctx.stock_positions(symbols=[osi]).channels:
            for sp in getattr(ch, "positions", []) or []:
                if str(getattr(sp, "symbol", "")) == osi:
                    return int(float(getattr(sp, "quantity", 0) or 0))
        return 0
    except Exception as e:
        if not _rl_hit(e):
            log(f"   查券商持仓失败 {osi}: {e}")
        return None


def _live_sell_order(osi: str):
    """券商侧该合约当前是否已有【未终态的出场卖单】(remark=exit-*)。返回 (id, 状态) 或 (None,None)。

    应用层幂等: longport SDK 的 submit_order 没有 client_request_id(3.0.23 与最新 4.3.3 都没有,
    幂等键只存在于 REST 层), 所以网络超时"券商已收单但客户端认为失败"这类重复提交只能靠提交前
    查重来挡。查不到不代表没有(查询本身可能失败), 因此这只是减少重复, 不是保证。

    ⚠ 只认【出场单】(remark 以 exit 开头), 跳过 tp/stop 保护腿: _start_exit 靠它防重复【出场】,
    若把 tp1 限价保护腿也当出场单认领 → 假 closing、④轮询止损(要求status==open)被禁用、
    那张 tp 限价单价没到永不成交 → 仓位卡死无止损(对抗测试挖出, 尤其孤儿 tp 不在排除列表时)。"""
    try:
        for od in _trade_ctx.today_orders(symbol=osi) or []:
            if str(getattr(od, "side", "")).split(".")[-1] != "Sell":
                continue
            if not str(getattr(od, "remark", "")).startswith("exit"):
                continue
            st = str(od.status).split(".")[-1]
            if st not in _TERMINAL:
                return str(od.order_id), st
    except Exception as e:
        if not _rl_hit(e):
            log(f"   查在途卖单失败 {osi}: {e}")
    return None, None


def _live_orphan_sells(osi: str, known_ids: set) -> list:
    """券商侧该合约【未终态、bot 不追踪(不在 known_ids)】的卖单 id = 孤儿。
    丢响应且反查(_reclaim_lost_submit)也失败时会留下这种孤儿。平仓时必须撤净它们,
    否则标 closed 后残留, 价格回弹成交 → 未授权空头裸空(对抗测试挖出)。"""
    out = []
    try:
        for od in _trade_ctx.today_orders(symbol=osi) or []:
            if str(getattr(od, "side", "")).split(".")[-1] != "Sell":
                continue
            if str(od.status).split(".")[-1] in _TERMINAL:
                continue
            oid = str(od.order_id)
            if oid not in known_ids:
                out.append(oid)
    except Exception as e:
        _rl_hit(e)
    return out


def _start_exit(positions: dict, osi: str, p: dict, qty: int, reason: str,
                intent: str = "full") -> bool:
    """提交市价卖 → 进入 closing 态等待【真实成交】。

    关键: 这里【绝不】更新 sold, 也【绝不】标 closed。
    submit_order 成功只代表拿到委托号, 不代表成交 —— 订单随后可能被拒绝、部分成交或长期挂起。
    sold 只在 manage_positions ⓪ 依据 order_detail 的 executed_quantity 更新。"""
    # ① 以券商实际持仓封顶 —— 本地账本可能因"提交超时但券商已收单"而偏高
    bq = _broker_qty(osi)
    if bq is None:
        # 查不到券商真值 → 【不卖】。原实现是 `if bq is not None:` 包住封顶逻辑, 查询失败就
        # 跳过封顶按本地账本照常卖 —— 而"账本偏高"与"查询失败"往往是同一次网络故障的两面,
        # 不是两个独立巧合: 卖单券商已收单但客户端超时(sold未更新), 同期 _broker_qty 与
        # _live_sell_order 两道防线一起失灵 → 再卖一次。仿真复现: 券商净持仓 -6 张裸空。
        # 取舍依 I3: 少卖亏损上限=权利金(有限), 多卖=裸空期权(无限风险)。
        n = p["broker_qty_fail"] = p.get("broker_qty_fail", 0) + 1
        p["pending_action"] = "full_exit"
        p["pending_action_reason"] = reason
        p["pending_action_ts"] = time.time()
        log(f"   ⏸️ {osi} 查不到券商持仓(第{n}次) → 本轮不卖, 记待办下轮重试 ({reason})")
        journal(ev="exit_blocked_no_broker_qty", osi=osi, want=qty, reason=reason, fails=n)
        if n >= 3:
            _alert(f"🚨 enrich {osi} 连续{n}轮查不到券商持仓, 出场被阻塞({reason}) — "
                   f"为避免裸空已拒绝卖出, 请人工核对券商侧持仓")
        _save(POS_JSON, positions)
        return False
    p.pop("broker_qty_fail", None)          # 查通了 → 清失败计数
    if bq <= 0:
        # 券商侧已无持仓 = 之前那张卖单其实成交了(客户端误判为失败)。绝不能再卖。
        log(f"   🔄 {osi} 券商侧已无持仓 → 上一笔卖出实际已成交, 对账收口(不重复卖)")
        journal(ev="exit_reconciled_flat", osi=osi, want=qty, reason=reason)
        _alert(f"🔄 enrich {osi} 与券商对账: 实际已无持仓(上笔卖出已成交), 本地补记并收口")
        p["sold"] = p.get("filled", 0)
        if not p.get("entry_order_id"):
            p["status"] = "closed"
            p.pop("pending_action", None); p.pop("pending_action_reason", None)
        _save(POS_JSON, positions)
        return False
    if qty > bq:
        log(f"   ✂️ {osi} 卖出量按券商实际持仓封顶: {qty}→{bq}张")
        journal(ev="exit_qty_clamped", osi=osi, want=qty, broker=bq)
        qty = bq
    # ①b I3 硬闸: 剩仓 − 已挂未成交卖单。券商净持仓封顶挡不住"在挂卖单+新卖单"叠加超卖
    #     (mirror_reduce 不撤止盈腿时: 持仓6 挂3 再卖3 → 都成交就卖了7张 = 裸空)。
    #     _sell_budget 的 docstring 写明"任何挂卖单的路径都必须先过这道闸", ensure_protection
    #     三处都过了, _start_exit 是唯一漏网的 —— 补在这里, 以后新增调用方自动受保护。
    _bud = _sell_budget(p)
    if _bud <= 0:
        log(f"   ⛔ {osi} 卖出预算为0(剩仓{p.get('filled',0)-p.get('sold',0)}张, "
            f"已挂{_outstanding_sells(p)}张) → 不再叠加卖单 ({reason})")
        journal(ev="exit_blocked_no_budget", osi=osi, want=qty, reason=reason)
        _save(POS_JSON, positions)
        return False
    if qty > _bud:
        log(f"   ✂️ {osi} 卖出量按剩余预算封顶: {qty}→{_bud}张(已挂{_outstanding_sells(p)}张)")
        journal(ev="exit_qty_budget_clamped", osi=osi, want=qty, budget=_bud)
        qty = _bud
    # ② 提交前查重: 上一轮可能"券商已收单但我们以为失败"。认领它而不是再发一张。
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
            done, exq, st, _exp = cancel_and_reconcile(oid)
            if not done:
                # 【卖】腿未确认 → 绝不能再发卖单(会双卖裸空), 必须推迟。
                # 【买】腿未确认 → 不阻断: 卖出量 ≤ 已成交量, 不可能裸空; 阻断反而会让一张
                #   撤不掉的买腿把到期强平永久卡死(仿真实测: 持仓被留到到期归零)。
                if id_key == "entry_order_id":
                    log(f"   ⚠️ {osi} 入场买腿未确认({st}), 不阻断平仓(卖已成交量不会裸空)")
                else:
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
        # 孤儿清扫(纵深防御): 撤完 bot 追踪的四条腿后, 券商侧若还有该合约未终态卖单, 就是
        # bot 不认识的孤儿(丢响应且反查也失败留下的)。平仓=清空该合约, 这些必须撤 —— 否则标
        # closed 后残留, 价格回弹成交 = 未授权空头裸空(对抗测试全真实链路复现: 净持仓-3)。
        known = {str(p.get(k)) for k in ("tp_order_id", "tp2_order_id", "stop_order_id",
                                         "exit_order_id") if p.get(k)}
        for oid in _live_orphan_sells(osi, known):
            done_o, exq_o, st_o, _ = cancel_and_reconcile(oid)
            if exq_o > 0:
                p["sold"] = p.get("sold", 0) + exq_o     # 孤儿撤单前已成交也要记账
            if done_o:
                log(f"   🧹 {osi} 撤掉券商侧孤儿卖单 {oid}(成交{exq_o})")
                journal(ev="orphan_sell_cancelled", osi=osi, order_id=oid, exq=exq_o)
                _alert(f"🧹 enrich {osi} 平仓时发现并撤掉券商侧孤儿卖单 {oid}(成交{exq_o}张) — "
                       f"疑似丢响应残留, 请留意是否有更多")
            else:
                unresolved.append(f"orphan({st_o})")     # 撤不掉 → 推迟, 不标 closed
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
            if not _start_exit(positions, osi, p, remain, reason):
                # 提交失败 → 必须留待办。止损/到期强平会因条件持续自然重来, 但站长"all out"
                # 是一次性的, 不记就永久丢失(与"挂单未确认终态"那条路径同一纪律)
                p["pending_action"] = "full_exit"
                p["pending_action_reason"] = reason
                p["pending_action_ts"] = time.time()
                _save(POS_JSON, positions)
        elif p.get("entry_order_id"):
            # 无剩仓但【买腿仍在途】: 不能标 closed —— manage_positions 首行会 continue 掉
            # closed 仓位, 那张买单后续成交就成了无人管理的孤儿多头(仿真抓到)
            log(f"   ⏸️ {osi} 已无剩仓, 但入场买腿仍在途 → 保持 open 直到它终态")
            journal(ev="close_wait_entry_leg", osi=osi, entry=p["entry_order_id"])
            p["status"] = "open"
            _save(POS_JSON, positions)
        else:
            p["status"] = "closed"
            p.pop("pending_action", None); p.pop("pending_action_reason", None)
            log(f"✅ {osi} 无剩仓, 直接标记已平 ({reason})")
            _save(POS_JSON, positions)
    finally:
        _closing.discard(osi)
        if positions.get(osi, {}).get("status") == "closed":
            _optfail.pop(osi, None)




HEARTBEAT_SEC = int(os.environ.get("HEARTBEAT_SEC", "600"))   # 心跳间隔(秒)
_last_hb = [0.0]


def _heartbeat(snap: list, positions: dict):
    """低频心跳 —— 证明【这一轮轮询真的执行了】。

    模拟盘不支持 MIT 触价单, 止损只有轮询这一条通道, 而它平时完全静默:
    tasks.loop 若因 discord 重连/事件循环异常停摆, 外部没有任何信号, 直到某天
    需要止损时才发现没止损。心跳把"还活着"变成可观测的。
    价格取自 ④ 止损判定时已经查过的那次, 不额外打 API。
    """
    now = time.time()
    if now - _last_hb[0] < HEARTBEAT_SEC:
        return
    _last_hb[0] = now
    act = [(o, p) for o, p in positions.items() if p.get("status") in ACTIVE_STATUSES]
    if not act:
        log("💓 轮询正常 | 无持仓")
        return
    px = dict(snap)
    parts = []
    for osi, p in act:
        remain = p.get("filled", 0) - p.get("sold", 0)
        avg = p.get("avg", 0) or 0
        last = px.get(osi)
        legs = sum(1 for k in ("tp_order_id", "tp2_order_id", "stop_order_id") if p.get(k))
        if last and avg > 0:
            stop_px = _stop_price(p)         # 与真正止损判定同一来源(保本时=入场价)
            _be = MECH_BE and (p.get("reduced") or p.get("tp1_done"))
            room = (1 - stop_px / last) * 100 if last > 0 else 0
            parts.append(f"{osi} {remain}张 ${last:.2f}({(last/avg-1)*100:+.0f}%) "
                         f"距止损${stop_px:.2f}({'保本' if _be else '-'+str(round((1-p.get('stop_mult',MECH_STOP_MULT))*100))+'%'})还有{room:.0f}% 保护腿{legs}条")
        else:
            # 报价拿不到 = 止损通道对这个仓位是瞎的, 心跳必须显式说出来
            parts.append(f"{osi} {remain}张 ⚠️报价不可用(止损通道失效) 保护腿{legs}条 "
                         f"status={p.get('status')}")
    log("💓 轮询正常 | " + " | ".join(parts))


def manage_positions(positions: dict):
    """轮询: 入场单成交→挂止盈; 止盈成交→记账; 到期日强平。"""
    now_et = datetime.now(ET)
    _hb_snap = []          # [(osi, last)] — 供心跳复用, 不额外查价
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
                # 只有【全部成交】才落 reduced/tp1_done: 部分成交也置位会让 ensure_protection 的
                # not tp1_done / ③b 的 not reduced 守卫永久关闭该止盈通道, 剩余张数再无止盈出口
                if exq >= want > 0 and p.get("exit_intent") == "tp1":
                    p["reduced"] = True
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
                done_x, exq_x, st_x, _ = cancel_and_reconcile(p["exit_order_id"])
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
                # 卖单在途期间不发【新卖单】(防重复卖), 但【入场单仍在途】必须继续对账+续撤:
                # 保本止损触发 → close_position 会撤入场单, 若那次撤单失败(网络/券商), 入场单
                # 继续成交的张数在此被 continue 跳过 → 不纳入 filled → 变孤儿仓(仿真抓到, 保本策略
                # 引爆的潜伏缺陷: 旧-60%止损在价格回入场价时不触发, 从没走到"边平边买"这一步)。
                # cancel_and_reconcile 只读对账成交量 + 撤未成交余量, 不发任何卖单, 不会重复卖。
                if p.get("entry_order_id"):
                    done_e, exq_e, st_e, exp_e = cancel_and_reconcile(p["entry_order_id"])
                    if exq_e > p.get("filled", 0):
                        p["filled"] = exq_e
                        if exp_e > 0:
                            p["avg"] = exp_e
                        journal(ev="entry_fill_while_closing", osi=osi, qty=exq_e)
                    if done_e:
                        p["entry_order_id"] = None
                    _save(POS_JSON, positions)
                continue      # 防重复卖
        # ⓪b 延迟动作重试 —— P0: 上轮因挂单未确认终态而推迟的平仓必须真的重来。
        #     止损/到期强平会因条件持续而自然重触发, 但 9ema 出场是【一次性】判定,
        #     没有这个待办队列就永久丢失。
        #     ("reduce" 意图随 mirror 出场模式一并删除; 旧状态文件里若残留也按全平处理 ——
        #      宁可多平也不能让一个仓位卡在无人认领的待办上)
        if p.get("pending_action") and p["status"] in ("pending", "open"):
            _rsn = p.get("pending_action_reason") or "延迟出场重试"
            log(f"🔁 {osi} 重试待办平仓 ({_rsn})")
            p.pop("pending_action", None)      # 先清再执行: 失败路径会自行重新记, 否则永久自触发
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
                    done2, exq2, st2, exp2 = cancel_and_reconcile(p["entry_order_id"])
                    if exq2 > p.get("filled", 0):
                        p["filled"] = exq2
                        if exp2 > 0:
                            p["avg"] = exp2
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
                _save(POS_JSON, positions)
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
        _sm = p.get("stop_mult", MECH_STOP_MULT)
        if _sm > 0 and p["status"] == "open" and not p.get("stop_order_id") \
                and p.get("avg", 0) > 0 and p.get("filled", 0) - p.get("sold", 0) > 0:
            last = _option_last(osi)
            _hb_snap.append((osi, last))      # 给心跳复用这次查价的结果
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
                if not p.get("armed") and last >= p["avg"] * MECH_TP2_MULT:
                    p["armed"] = True
                    log(f"🎯 {osi} 报价${last}≥成本×{MECH_TP2_MULT} → runner武装15m9ema")
                    journal(ev="runner_armed", osi=osi, last=last)
                    _save(POS_JSON, positions)
                # stop_px 走 _stop_price 单一来源(保本时=入场价, 否则 avg×stop_mult; 均≥MIN_TICK)
                stop_px = _stop_price(p)
                _be = MECH_BE and (p.get("reduced") or p.get("tp1_done"))
                if last <= stop_px:
                    lab = "保本止损" if _be else f"止损-{(1-_sm)*100:.0f}%"
                    log(f"🛑 {osi} 轮询止损: 最新${last} ≤ ${stop_px:.2f} ({lab})")
                    journal(ev="stop_trigger", osi=osi, last=last, avg=p["avg"], stop_px=stop_px, be=bool(_be))
                    close_position(positions, osi, lab)
                    continue
        # ④c 机械模式: runner【武装后】守标的15分9ema, 连破N根(已完成bar)→全平
        #    (最终定稿: 摸到+60%才武装; 未武装的runner只受-60%止损+到期强平管, 防早盘回踩误洗肥尾)
        if p["status"] == "open" and p.get("armed") \
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
            done_t, exq_t, st_t, exp_t = cancel_and_reconcile(p["entry_order_id"])
            if exq_t > p.get("filled", 0):
                p["filled"] = exq_t
                # 必须同时回填成交均价! avg=0 会让 ensure_protection 早退(一条腿都不挂)+
                # ④轮询止损的 avg>0 前置失败, 而此时 entry_order_id 已清空, ① 再也不跑 →
                # 该仓位止盈腿与-60%止损【全部永久失效】(仿真实测跌到-85%仍无人管)
                if exp_t > 0:
                    p["avg"] = exp_t
                _f = exq_t
                _d = f"部分成交{_f}张"
            if not done_t:
                # 撤单未确认: 【保留 entry_order_id】继续重试, 但【不 continue】——
                # 到期强平就写在下面, continue 会让一张撤不掉的买腿无限期阻断已成交多头的强平
                # (仿真实测: 持仓被留到期权到期归零)。卖"已成交量"不可能裸空, 无需等买腿终态。
                p["ttl_cancel_fails"] = p.get("ttl_cancel_fails", 0) + 1
                log(f"   ⚠️ {osi} 入场单TTL撤单未确认({st_t}), 保留ID重试(第{p['ttl_cancel_fails']}次)")
                journal(ev="entry_ttl_cancel_unconfirmed", osi=osi, state=st_t,
                        fails=p["ttl_cancel_fails"])
                if p["ttl_cancel_fails"] % 10 == 1:
                    _alert(f"⚠️ enrich {osi} 入场单撤不掉({st_t}, 第{p['ttl_cancel_fails']}次) — "
                           f"已成交{p.get('filled',0)}张仍按正常管理, 请人工查券商挂单")
                if p.get("filled", 0) > 0 and p["status"] == "pending":
                    p["status"] = "open"
                    ensure_protection(positions, osi, p)
                _save(POS_JSON, positions)
            else:
                # 只有【确认撤净】才允许清 ID / 标 closed。
                # (曾经把 continue 一删了事, 结果未确认时也落进这段 → 清了ID还报"已撤剩余",
                #  券商侧买单变孤儿 —— 仿真抓到, 是修 BUG3 时自己引入的)
                log(f"⏳ {osi} 入场单挂{ENTRY_TTL_SEC//60}分钟{_d} → 已撤剩余(不接刀)")
                journal(ev="entry_ttl_cancel", osi=osi, order_id=p["entry_order_id"],
                        ttl=ENTRY_TTL_SEC, filled=_f)
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
        # ④ 到期日强平 (15:40 ET 后) + F不对称过夜: 非到期日收盘平【未落袋满仓】
        if p["status"] in ("pending", "open"):
            try:
                exp_d = date.fromisoformat(p["expiry"])
                hm = (now_et.hour, now_et.minute)
                # QA: 原 `date>=exp_d and (h,m)>=(15,40)` 元组比较陷阱 — 若到期日15:40那刻没跑到,
                #     次日09:30时 (9,30)>=(15,40) 为False, 要拖到次日15:40才平, 已到期合约多挂近一天
                if now_et.date() > exp_d or (now_et.date() == exp_d and hm >= (15, 40)):
                    close_position(positions, osi, "到期强平")   # 到期无上界: 必须避行权/归零, 收盘后也要平
                elif F_EOD_CLOSE_UNREDUCED and (15, 40) <= hm < (16, 0) \
                        and p["status"] == "open" and not p.get("reduced"):
                    # F: 从没首档止盈的满仓=裸扛, 一个逆gap就-94%(如IBM 07-08), 收盘前平掉不过夜。
                    # 已落袋+30%的runner(reduced)=赢来的钱, 留过夜接GOOGL这类回撤后爆发(见bt_be_grid回测)。
                    # 窗口封在 15:40-16:00: 只在市场还开着能市价卖时平; 16:00后收盘无法成交, 不 fire(等次日)。
                    close_position(positions, osi, "收盘平未落袋(F:不裸扛过夜)")
            except Exception:
                pass
    # 心跳放在【循环外、函数最末】: 只有整轮真正跑完才打, 中途 continue 不影响。
    # 单个仓位抛异常不该让心跳失真, 所以它自己不吞异常 —— 真出问题宁可看到 traceback。
    _heartbeat(_hb_snap, positions)


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
    _anchor_mem.update(state)                 # 内存镜像: 锚点文件损坏时用它兜底(见 _merge_save_anchor)
    for ch_id in (CHANNEL_ID, ANDY_CHANNEL_ID):
        ch = client.get_channel(ch_id)
        if ch is None:
            continue
        key = str(ch_id)
        last = state.get(key)
        try:
            if not last:                      # 首次运行: 锚定到最新, 不回放历史
                async for m in ch.history(limit=1):
                    _bump(state, key, m.id)
                continue
            # 翻页取完整个停机窗口。原来只 history(limit=100) 一把: 停机漏 130 条时只吃到前100,
            # 锚点停在第100; 随后第一条实时消息到达, on_message 把锚点直接推到最新 id
            # → 101..130 永不回看(锚点只进不退, 那段窗口就此消失)。
            missed, cur = [], last
            while len(missed) < CATCHUP_MAX:
                page = [m async for m in ch.history(limit=100,
                                                    after=discord.Object(id=int(cur)),
                                                    oldest_first=True)]
                if not page:
                    break
                missed += page
                cur = str(page[-1].id)
                if len(page) < 100:
                    break
            if len(missed) >= CATCHUP_MAX:
                # 不静默截断: 超出上限说明停机窗口过大, 必须让人知道有多少没回看
                _alert(f"⚠️ enrich 追赶 ch={ch_id} 达到上限 {CATCHUP_MAX} 条, "
                       f"更早的消息不再回看 — 请人工核对停机窗口")
        except Exception as e:
            log(f"追赶失败 ch={ch_id}: {e}")
            continue
        for m in missed:
            # 锚点【处理成功后才前移】。原本在循环首行无条件前移, 处理某条停机期间的 EXIT 抛异常时
            # 锚点已越过它并最终落盘 → 该出场信号永久丢失(与 on_message 同一类缺陷, 上轮只修了实时路径)
            if m.author.id != AUTHOR_ID or not m.content:
                _bump(state, key, m.id)       # 明确无需处理 → 可安全前移
                continue
            one = " ".join(m.content.split())[:90]
            try:
                if ch_id == ANDY_CHANNEL_ID:
                    handle_andy(m.content, m.created_at, m.id)   # 纯记录, 无下单
                    _bump(state, key, m.id)   # ← 原来这里直接 continue, 跳过了锚点前移:
                    continue                  #   andy 锚点永不推进 → 每次重连重复记账, 窗口不收敛
                s = parse_signal(m.content, m.created_at.date())
                if s.kind in ("BUY", "BUY_AMBIG", "BUY_NOEXPIRY"):
                    note = f"⏰ 停机期间错过的enrich买入信号 (仅提醒, 不按旧价补单): {one}"
                    log(note); push_discord(note)
                    journal(ev="missed_during_downtime", sig=one, ts_signal=str(m.created_at))
                elif s.kind == "EXIT":
                    # 时效闸: BUY 有 STALE_BUY_SEC 且这里直接降级为提醒, EXIT 原本一道闸都没有 ——
                    # 锚点一旦因任何原因退回(并发覆盖 / 锚点文件损坏回落到陈旧 .bak),
                    # 三天前的"全部清仓"会当场平掉今天刚建的仓。
                    age = (datetime.now(_tzone.utc)
                           - m.created_at.replace(tzinfo=m.created_at.tzinfo or _tzone.utc)
                           ).total_seconds()
                    if age > STALE_EXIT_SEC:
                        note = (f"⏰ 停机期间的enrich出场信号已过期{age/3600:.1f}小时 "
                                f"(>{STALE_EXIT_SEC/3600:.0f}h, 仅提醒不执行): {one}")
                        log(note); push_discord(note)
                        journal(ev="stale_exit_skipped", sig=one, age_sec=int(age),
                                ts_signal=str(m.created_at))
                    else:
                        # 必须 to_thread: handle 要抢 _handle_lock, 而 _managed_tick 现在会长时间
                        # 持有该锁; 在事件循环线程里同步抢锁会挂住 discord 心跳 → 断线重连正反馈(复审发现)
                        import asyncio as _aio
                        await _aio.to_thread(handle, m.content, m.created_at.date(),
                                             m.id, seen, positions)   # 出场晚跟好过不跟
                _bump(state, key, m.id)       # 处理成功才前移锚点
            except Exception as e:
                # 保留锚点: 下次重连会重新回看这条(BUY有seen[msg_id]兜底, EXIT重复执行也安全 ——
                # close_position 按 remain=filled-sold 计算, 已平仓则直接返回)
                log(f"追赶处理异常(保留锚点待下次重试): {e}")
                break                          # 不跳过它继续推进后面的消息, 否则锚点仍会越过它
        if missed:
            log(f"⏰ 追赶完成 ch={ch_id}: 回看了 {len(missed)} 条停机期间消息")
    _merge_save_anchor(state)


def _bump(state: dict, key: str, msg_id) -> None:
    """锚点只进不退。Discord 消息 id 是雪花号(单调递增), 直接取 max 即可。

    原来是盲写 state[key] = msg_id: 配合 catch_up 的分页/重试, 任何一次写入较小的 id
    都会让锚点倒退, 那段窗口的消息下次重连被整段重放(出场信号重复执行)。"""
    old = state.get(key)
    try:
        if old is not None and int(old) >= int(msg_id):
            return
    except (TypeError, ValueError):
        pass                                  # 旧值不是数字(脏数据) → 直接用新值覆盖
    state[key] = str(msg_id)


def _merge_save_anchor(state: dict) -> bool:
    """把锚点合并进磁盘上的最新值再落盘(每键取 max), 而不是整份覆盖。

    ① 防 lost update: catch_up 开头 _load 拿的是快照, 中途 await 会交还事件循环, 期间
       on_message → bump_last 可能已落了更新的锚点; 结尾整份覆盖就把它冲掉 → 消息被重放。
    ② 防文件损坏连累别的频道: _load 遇损坏且无 .bak 时返回 {}, 直接写回去就只剩当前频道,
       另一频道退化成"首次运行"、整个停机窗口静默丢弃。本进程内存里其实还记得那个值
       (catch_up 启动时读过), 用 _anchor_mem 补回来。"""
    cur = _load(LAST_MSG_JSON)
    if not cur and (_anchor_mem or LAST_MSG_JSON.exists()):
        # 文件存在却读出空 = 损坏且 .bak 也不可用。内存里还记得的能救回来(bot 跑起来后
        # catch_up/bump_last 都会填充), 内存也没有的就是真丢了 —— 那种情况下别的频道会
        # 退化成"首次运行"(锚定最新、不回放), 停机窗口静默消失, 只能靠告警让人知道。
        _alert(f"🚨 enrich 锚点文件损坏且无可用备份 — "
               f"内存中保有 {sorted(_anchor_mem)} 的游标可恢复; "
               f"不在其中的频道将退化为'首次运行'并丢失停机窗口, 请人工核对")
    for k, v in _anchor_mem.items():          # 先用内存补全, 再让本次的新值覆盖
        _bump(cur, k, v)
    for k, v in state.items():
        _bump(cur, k, v)
    _anchor_mem.update(cur)
    return _save(LAST_MSG_JSON, cur)


def bump_last(ch_id, msg_id):
    # 走合并落盘: 原实现是 _load → 改一个键 → _save 整份覆盖, 而 _load 遇到损坏文件会走宽松
    # 分支返回 {} → 写回去只剩当前频道, 【另一个频道的游标被抹掉】→ 它退化成"首次运行",
    # 整个停机窗口被静默丢弃。
    _merge_save_anchor({str(ch_id): str(msg_id)})


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
        # 站长出场【一律只提醒不执行】。出场全部由机械规则接管:
        #   +30%卖½ → 首档后止损移保本 → 剩仓摸+60%武装守15分9ema → 到期强平(见文件头)
        # (2026-07-20 删除 mirror 跟单出场模式; 历史实现见 git: e6a0041 之前)
        note = f"🟠 站长出场[{s.exit_level}] [{s.ticker}] — 机械出场, 不跟(仅提醒): {one}"
        log(note); push_discord(note)
        journal(ev="exit_signal_mech_ignored", ticker=s.ticker, level=s.exit_level, sig=one)
        # 去重表在【播报成功之后】才写: 上面若抛异常就不留痕, catch_up 重连回看 / 站长重发
        # 时才能真的重来。原实现在函数开头就无条件写死, 于是重试撞上"重复出场消息跳过"被
        # 静默吞掉, 而这次不抛异常 → 调用方认定成功 → 锚点前移 → 信号永久丢失。
        _recent_exits[_norm] = _now
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
            _sm0 = MECH_STOP_MULT
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
                done_c, exq_c, st_c, _ = cancel_and_reconcile(r)
                if done_c and exq_c == 0:
                    positions.pop(osi, None)
                    log(f"   ✅ {osi} 已撤净(未成交), 本次进场作废")
                else:
                    _alert(f"🚨 enrich {osi} 撤单未确认或已成交{exq_c}张({st_c}) — 请立即人工核对券商持仓!")
                push_discord(plan)
                return
            plan += (f"\n  ✅已提交 order_id={r} (成交后自动挂+{(MECH_TP_MULT-1)*100:.0f}%止盈, "
                     f"二档后武装15m9ema管runner, -{(1-MECH_STOP_MULT)*100:.0f}%止损全程不移保本)")
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
        log(f"🚀 LIVE(模拟盘)·机械出场(2026-07-22 F不对称过夜定案): {size_s} | 权利金上限${MAX_PREMIUM} | "
            f"+{(MECH_TP_MULT-1)*100:.0f}%卖{MECH_TP_FRAC:.0%} → "
            f"{'首档后止损移保本' if MECH_BE else 'runner不保本(维持止损, 扛回撤接大runner)'} | "
            f"剩仓摸+{(MECH_TP2_MULT-1)*100:.0f}%武装 → 守{MECH_EMA_MIN}分9ema连破{MECH_EMA_N}根 | "
            f"初始止损-{(1-MECH_STOP_MULT)*100:.0f}% | 入场TTL{ENTRY_TTL_SEC//60}分 | "
            f"{'未落袋满仓15:40收盘平不过夜' if F_EOD_CLOSE_UNREDUCED else ''} | 到期强平 | 无LLM | 站长出场只播报")
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
