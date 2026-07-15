#!/usr/bin/env python3
"""
enrich_parser.py — 解析 enrich 期权信号 (Discord #期权-波段-enrich 频道英文原文)。

严格文法, 宁可漏不可错 (漏了=没跟到一笔; 错了=乱下单):
  入场 BUY:  $TICKER + 到期(weekly|0DTE|M/D) + $行权价 + calls/puts + $权利金
             五要素缺一 → 不算入场 (降级为 EXIT/NOISE)。
  出场 EXIT: $TICKER + scale out/all out/closing/trim/sell/secure profits 等关键词 → 仅提醒。
  其余 NOISE: watchlist/recap/levels/评论 → 忽略。

到期解析:
  0DTE   → 信号当天
  weekly → 信号所在周的周五 (信号日已过周五则下周五)
  M/D    → 显式日期 (年份取信号年, 若已过则+1年)

行权价 vs 权利金消歧: 行权价 = 紧邻 calls/puts 前的 $数; 权利金 = 其余 $数里最小的
(或 for/@ 后面的数)。权利金区间 ($2.40-$2.70) 取上界做限价 (区间内都愿意买)。
"""
import re
from datetime import date, timedelta
from dataclasses import dataclass, field


# ── 结果 ──
@dataclass
class Signal:
    kind: str                 # BUY / EXIT / NOISE
    ticker: str = ""
    right: str = ""           # C / P
    strike: float = 0.0
    expiry: date | None = None
    limit_price: float = 0.0  # 权利金限价 (区间取上界)
    size_tag: str = ""        # lotto / 1% / small / "" (口头仓位提示)
    exit_level: str = ""      # EXIT分级: full全平/partial部分减/vague模糊/alert仅提醒(多票或有豁免词)
    reason: str = ""          # 判定依据 (调试/审计)
    raw: str = ""


TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")
# $数字 (价格): $362.50 / $.80 / $1300
PRICE_RE = re.compile(r"\$(\d*\.?\d+)")
DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")
CALLPUT_RE = re.compile(r"\b(calls?|puts?)\b", re.I)
# 出场关键词 (自然语言, 只提醒)
EXIT_RE = re.compile(
    r"\b(scal(?:e|ing)\s+(?:out|down)|all\s+out|closing|close\s+out|trim|"
    r"sell(?:ing)?|sold|secure\s+profits?|lock(?:ing)?\s+(?:in\s+)?(?:those\s+)?gains|"
    r"take\s+(?:the\s+)?(?:profits?|gains)|stopped\s+out|cutting)\b", re.I)
# 噪音标志 (整条丢弃, 优先级最高)
NOISE_RE = re.compile(r"\b(watchlist|recap|levels\s+for\s+the\s+day|mentorship)\b", re.I)
SIZE_RE = re.compile(r"\b(lotto|small|starter|light)\b|(\d{1,2})%\s*(?:position|starter)?", re.I)

# $TICKER 误匹配保护: 常见非票词
NOT_TICKER = {"DTE", "EMA", "ITM", "OTM", "ATM", "PDH", "PMH", "PWH", "EOD", "HTF", "VIP", "AND"}

# 出场分级 (镜像跟单用)
EXIT_FULL_RE = re.compile(
    r"\ball\s+out\b|\bclos(?:e|ing|ed)\b|\bcutting\b|\bstopped(?:\s+out)?\b|\bflat\b|"
    r"\bsold\s+(?:the\s+)?rest\b|\bselling\s+everything\b", re.I)
EXIT_PARTIAL_RE = re.compile(
    r"scal(?:e|ing)\s+(?:out|down)|\btrim\w*\b|\b1/[234]\b|\bhalf\b|"
    r"selling\s+\d{1,2}%|\bdown\s+to\b|taking\s+(?:1/[234]|some|partial)", re.I)
EXIT_EXEMPT_RE = re.compile(r"\boutside\s+of\b|\bexcept\b|\bother\s+than\b", re.I)


def _exit_level(t: str, tickers: list) -> str:
    """full=清仓跟随 / partial=减仓跟随 / vague=模糊 / alert=仅提醒(多票复盘或有豁免词)。"""
    if len(set(tickers)) > 1 or EXIT_EXEMPT_RE.search(t):
        return "alert"
    if EXIT_FULL_RE.search(t):
        return "full"
    if EXIT_PARTIAL_RE.search(t):
        return "partial"
    return "vague"


def _next_friday(d: date) -> date:
    """本周五; 若 d 已过周五(周六/日), 下周五。周五当天算当周。"""
    off = (4 - d.weekday()) % 7
    return d + timedelta(days=off)


def parse_signal(text: str, msg_date: date) -> Signal:
    raw = text.strip()
    t = raw.replace("@everyone", " ").replace("$alert", " ")

    if NOISE_RE.search(t):
        return Signal(kind="NOISE", reason="watchlist/recap/levels", raw=raw)

    tickers = [x for x in TICKER_RE.findall(t) if x not in NOT_TICKER]
    if not tickers:
        # 无票名的全局出场令: "All cash now" / "Down to runners on all" / "Closing everything"
        if not EXIT_EXEMPT_RE.search(t):
            if re.search(r"\ball\s+cash\b|\ball\s+out\b|\bclos(?:e|ing)\s+everything\b|\bflat\s+now\b", t, re.I):
                return Signal(kind="EXIT", ticker="*", exit_level="full", reason="全局清仓令(无票名)", raw=raw)
            if re.search(r"down\s+to\s+runners\s+on\s+all|trimm?(?:ing)?\s+(?:all|everything)", t, re.I):
                return Signal(kind="EXIT", ticker="*", exit_level="partial", reason="全局减仓令(无票名)", raw=raw)
        return Signal(kind="NOISE", reason="无ticker", raw=raw)

    cp = CALLPUT_RE.search(t)
    # ── 尝试入场五要素 ──
    if cp:
        right = "C" if cp.group(1).lower().startswith("call") else "P"
        # 到期
        expiry = None; exp_src = ""
        if re.search(r"\b0\s*DTE\b", t, re.I):
            expiry, exp_src = msg_date, "0DTE"
        else:
            md = DATE_RE.search(t)
            if md:
                m_, d_ = int(md.group(1)), int(md.group(2))
                if 1 <= m_ <= 12 and 1 <= d_ <= 31:
                    y = msg_date.year
                    exp = date(y, m_, d_)
                    if exp < msg_date:          # 已过 → 明年 (跨年信号)
                        exp = date(y + 1, m_, d_)
                    expiry, exp_src = exp, f"{m_}/{d_}"
            if expiry is None and re.search(r"\bweekl(?:y|ies)\b", t, re.I):
                expiry, exp_src = _next_friday(msg_date), "weekly"

        # 行权价 = 紧邻 calls/puts 前最近的 $数
        strike = 0.0
        before = t[:cp.start()]
        b_prices = PRICE_RE.findall(before)
        if b_prices:
            strike = float(b_prices[-1])
        # 权利金 = 其余 $数里 (≠strike) 的候选; 优先 for/@ 之后; 区间取上界
        prem = 0.0
        after = t[cp.end():]
        m_for = re.search(r"(?:for|@|\bin\b)\s*\$(\d*\.?\d+)(?:\s*-\s*\$(\d*\.?\d+))?", t, re.I)
        if m_for:
            prem = float(m_for.group(2) or m_for.group(1))
        else:
            a_prices = [float(x) for x in PRICE_RE.findall(after)]
            # 区间: 取上界 (最后一个); 单值: 取第一个
            rng = re.search(r"\$(\d*\.?\d+)\s*-\s*\$(\d*\.?\d+)", after)
            if rng:
                prem = float(rng.group(2))
            elif a_prices:
                prem = a_prices[0]
            else:
                # calls 前面同时有 strike 和权利金的排布 (如 "0DTE $.70-$.80 $207.50 …calls")
                b_all = [float(x) for x in b_prices[:-1]]
                if b_all:
                    prem = min(b_all)
        # 合理性: 权利金应远小于行权价; 权利金>0; strike>0
        five_ok = (strike > 0 and prem > 0 and expiry is not None
                   and prem < strike * 0.5 and len(set(tickers)) == 1)
        if five_ok:
            sz = SIZE_RE.search(t)
            size_tag = (sz.group(0).strip() if sz else "")
            return Signal(kind="BUY", ticker=tickers[0], right=right, strike=strike,
                          expiry=expiry, limit_price=prem, size_tag=size_tag,
                          reason=f"五要素齐: 到期={exp_src}", raw=raw)
        # 要素不齐 → 落到出场/噪音判定
        missing = []
        if strike <= 0: missing.append("行权价")
        if prem <= 0: missing.append("权利金")
        if expiry is None: missing.append("到期")
        if len(set(tickers)) != 1: missing.append("多ticker")
        if EXIT_RE.search(t):
            return Signal(kind="EXIT", ticker=tickers[0], exit_level=_exit_level(t, tickers),
                          reason=f"出场关键词(且入场要素缺{'/'.join(missing)})", raw=raw)
        return Signal(kind="NOISE", ticker=tickers[0],
                      reason=f"有calls/puts但缺{'/'.join(missing)}=评论", raw=raw)

    # ── 无 calls/puts: 先试歧义买入(缺方向, 靠实时报价消歧), 再出场/噪音 ──
    #    要素: 单ticker + 到期token + ≥2个$价(最大=行权价, 需≥5×权利金分离) → BUY_AMBIG
    if len(set(tickers)) == 1:
        expiry = None; exp_src = ""
        if re.search(r"\b0\s*DTE\b", t, re.I):
            expiry, exp_src = msg_date, "0DTE"
        else:
            md = DATE_RE.search(t)
            if md and 1 <= int(md.group(1)) <= 12 and 1 <= int(md.group(2)) <= 31:
                exp = date(msg_date.year, int(md.group(1)), int(md.group(2)))
                if exp < msg_date:
                    exp = date(msg_date.year + 1, int(md.group(1)), int(md.group(2)))
                expiry, exp_src = exp, "M/D"
            elif re.search(r"\bweekl(?:y|ies)\b", t, re.I):
                expiry, exp_src = _next_friday(msg_date), "weekly"
        prices = [float(x) for x in PRICE_RE.findall(t)]
        if expiry is not None and len(prices) >= 2:
            strike = max(prices)
            m_for = re.search(r"(?:for|@)\s*\$(\d*\.?\d+)(?:\s*-\s*\$(\d*\.?\d+))?", t, re.I)
            if m_for:
                prem = float(m_for.group(2) or m_for.group(1))
            else:
                rng = re.search(r"\$(\d*\.?\d+)\s*-\s*\$(\d*\.?\d+)", t)
                prem = float(rng.group(2)) if rng else min(prices)
            if strike > 0 and 0 < prem < strike / 5 and prem != strike:
                sz = SIZE_RE.search(t)
                return Signal(kind="BUY_AMBIG", ticker=tickers[0], right="?", strike=strike,
                              expiry=expiry, limit_price=prem,
                              size_tag=(sz.group(0).strip() if sz else ""),
                              reason=f"缺方向待报价消歧: 到期={exp_src}", raw=raw)
    if EXIT_RE.search(t):
        return Signal(kind="EXIT", ticker=tickers[0], exit_level=_exit_level(t, tickers),
                      reason="出场关键词", raw=raw)
    return Signal(kind="NOISE", ticker=tickers[0], reason="无calls/puts无出场词", raw=raw)


def to_longport_symbol(s: Signal) -> str:
    """OSI 期权代码: TSLA260708C250000.US (strike×1000, 至少6位)。"""
    assert s.kind == "BUY" and s.expiry
    return f"{s.ticker}{s.expiry:%y%m%d}{s.right}{int(round(s.strike * 1000)):06d}.US"


if __name__ == "__main__":
    # 快速自测 (真实历史样本)
    cases = [
        ("$GOOGL 7/15 $362.50 calls - scaling in $.80", date(2026, 7, 14)),
        ("$HOOD weekly $120 calls $.83", date(2026, 7, 9)),
        ("Lotto $IREN 0DTE $60 calls $.68", date(2026, 6, 12)),
        ("$IBM 7/10 $270 calls - I will scale in $2.40-$2.70", date(2026, 6, 22)),
        ("Weekly $HOOD $90 calls $1.20", date(2026, 6, 8)),
        ("$IBM $325 weeklies are not a bad lotto here with profits.", date(2026, 7, 7)),
        ("$DELL - Scaling out 1/2 on this pump.", date(2026, 7, 9)),
        ("Updated watchlist:\n$GOOGL\n$PLTR", date(2026, 7, 14)),
    ]
    for txt, d in cases:
        s = parse_signal(txt, d)
        head = f"[{s.kind}]"
        if s.kind == "BUY":
            head += f" {to_longport_symbol(s)} 限价${s.limit_price} ({s.size_tag or '常规'})"
        print(f"{head}  ← {txt[:60]}  |{s.reason}")
