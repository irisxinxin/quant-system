#!/usr/bin/env python3
"""sim/fakes.py — 高保真假券商 / 假行情 / 假时钟。

目的: 让 discord_enrich_bot 的【真实逻辑】跑在可控环境里, 端到端验证行为是否符合预期。

设计铁律(否则测试会变成自我循环, 拿我自己的错误假设验证我自己的代码):
  1. 假券商维护【独立的持仓真值】self.position[symbol]。bot 的 positions.json 只是它的"看法"。
     所有不变式都用券商真值断言 —— 尤其是 position < 0 (裸空) 必须永远为假。
  2. 撤单是【请求】不是结果: cancel_order 把订单置 PendingCancel, 下一 tick 才可能变 Canceled;
     若该 tick 恰好触价, 它会【先成交】。这正是真实竞态, bot 必须自己处理。
  3. 订单不会瞬间成交: 限价单要价格真的触及; 市价单按下一 tick 开盘价。
  4. 故障可注入: 抛异常/429/查单失败/拒单/部分成交/订单卡在 PendingCancel 不动。

只实现 bot 真正用到的接口面(已 grep 确认):
  TradeContext: submit_order / cancel_order / order_detail / today_orders / stock_positions / account_balance
  QuoteContext: option_quote / candlesticks
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import itertools

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

# 长桥订单状态(与 SDK 一致)。终态白名单必须与 bot 里的 _TERMINAL 相同。
TERMINAL = {"Filled", "Canceled", "Expired", "Rejected", "PartialWithdrawal"}


class SimError(Exception):
    """模拟的 API 异常(网络/限流/权限)。"""


class FakeClock:
    """可控时钟。bot 用 time.time() 和 datetime.now(ET), 两者都由它驱动。"""

    def __init__(self, start_et: datetime):
        if start_et.tzinfo is None:
            start_et = start_et.replace(tzinfo=ET)
        self._now = start_et

    def now(self, tz=None):
        return self._now.astimezone(tz) if tz else self._now

    def time(self):
        return self._now.timestamp()

    def advance(self, seconds):
        self._now += timedelta(seconds=seconds)

    def set_et(self, y, mo, d, h, mi):
        self._now = datetime(y, mo, d, h, mi, tzinfo=ET)


class _Order:
    _ids = itertools.count(1000)

    def __init__(self, symbol, side, qty, otype, price, trigger, tif, remark, ts):
        self.order_id = f"SIM{next(_Order._ids)}"
        self.symbol = symbol
        self.side = side                  # "Buy" / "Sell"
        self.submitted_quantity = int(qty)
        self.order_type = otype           # "LO" / "MO" / "MIT"
        self.price = price
        self.trigger = trigger
        self.tif = tif                    # "Day" / "GTC"
        self.remark = remark
        self.status = "New"
        self.executed_quantity = 0
        self.executed_price = 0.0
        self.submitted_at = ts
        self.triggered = False            # MIT 是否已触发
        self.cancel_requested = False

    # bot 读的是 str(od.status).split(".")[-1], 这里直接给状态名
    def __repr__(self):
        return (f"<{self.order_id} {self.side} {self.symbol} {self.executed_quantity}"
                f"/{self.submitted_quantity} {self.status}>")


class FakeQuotes:
    """假行情。价格路径由场景喂入: {symbol: [(last, high, low), ...]}, 每 tick 前进一格。"""

    def __init__(self, clock: FakeClock):
        self.clock = clock
        self.paths = {}          # symbol -> list of (last, high, low)
        self.idx = {}            # symbol -> 当前下标
        self.open_interest = {}  # symbol -> OI
        self.trade_status = {}   # symbol -> "Normal"/"Halted"/...
        self.quote_age_sec = {}  # symbol -> 报价滞后秒数(模拟低流动性无成交)
        self.fail = set()        # 这些 symbol 的报价查询直接抛错
        self.candles = {}        # ticker -> 15分K收盘价序列(给9ema用)

    # ── 场景配置 ──
    def set_path(self, symbol, path):
        self.paths[symbol] = [(p, p, p) if not isinstance(p, tuple) else p for p in path]
        self.idx[symbol] = 0

    def cur(self, symbol):
        """当前 tick 的 (last, high, low)。路径走完就停在最后一格。"""
        pth = self.paths.get(symbol)
        if not pth:
            return None
        i = min(self.idx.get(symbol, 0), len(pth) - 1)
        return pth[i]

    def step(self):
        for sym in self.paths:
            self.idx[sym] = min(self.idx.get(sym, 0) + 1, len(self.paths[sym]) - 1)

    # ── QuoteContext 接口 ──
    def option_quote(self, symbols):
        out = []
        for sym in symbols:
            if sym in self.fail:
                raise SimError(f"no quote permission for {sym}")
            c = self.cur(sym)
            if c is None:
                continue
            q = type("Q", (), {})()
            q.symbol = sym
            q.last_done = c[0]
            q.open_interest = self.open_interest.get(sym, 5000)
            q.trade_status = self.trade_status.get(sym, "Normal")
            q.timestamp = self.clock.now(UTC) - timedelta(seconds=self.quote_age_sec.get(sym, 0))
            out.append(q)
        return out

    def candlesticks(self, symbol, period, count, adjust):
        tk = symbol.replace(".US", "")
        closes = self.candles.get(tk)
        if closes is None:
            raise SimError(f"no candles for {symbol}")
        bars = []
        base = self.clock.now(UTC) - timedelta(minutes=15 * len(closes))
        for i, c in enumerate(closes[-count:]):
            b = type("B", (), {})()
            b.close = c
            b.timestamp = base + timedelta(minutes=15 * i)
            bars.append(b)
        return bars


class FakeBroker:
    """假券商。持有【持仓真值】, 是所有不变式的裁判。"""

    def __init__(self, clock: FakeClock, quotes: FakeQuotes, equity=100_000.0):
        self.clock = clock
        self.quotes = quotes
        self.equity = equity
        self.orders = {}                 # oid -> _Order
        self.position = {}               # symbol -> 净持仓(负数=裸空, 永远不该出现)
        self.fills = []                  # (ts, oid, symbol, side, qty, price)
        self.rejected_naked = []         # 券商侧本应拒绝的裸卖(见 allow_naked)
        # ── 故障注入开关 ──
        self.fail_submit = 0             # >0: 接下来N次 submit 抛错
        self.fail_cancel = 0             # >0: 接下来N次 cancel 抛错
        self.fail_detail = 0             # >0: 接下来N次 order_detail 抛错
        self.fail_today_orders = False
        self.fail_balance = False
        self.reject_next_submit = 0      # >0: 接下来N次 submit 返回成功但订单立刻 Rejected
        self.stuck_cancel = set()        # 这些 oid 撤单请求永远停在 PendingCancel
        self.liquidity = {}              # symbol -> 每 tick 最多成交张数(默认全量)
        self.submit_but_lose_response = 0  # >0: 券商收单成功但抛异常给客户端(幂等场景)
        self.allow_naked = True          # True=允许卖超(用于暴露bug); False=券商侧拒单

    # ── TradeContext 接口 ──
    def submit_order(self, symbol, order_type, side, submitted_quantity,
                     time_in_force, submitted_price=None, trigger_price=None,
                     remark="", **kw):
        if self.fail_submit > 0:
            self.fail_submit -= 1
            raise SimError("network timeout on submit")
        ot = str(order_type).split(".")[-1]
        sd = str(side).split(".")[-1]
        tif = str(time_in_force).split(".")[-1]
        if ot == "MIT":
            raise SimError("604050 condition order not supported")   # 模拟盘不支持触价单
        o = _Order(symbol, sd, int(submitted_quantity), ot,
                   float(submitted_price) if submitted_price is not None else None,
                   float(trigger_price) if trigger_price is not None else None,
                   tif, remark, self.clock.time())
        self.orders[o.order_id] = o
        if self.reject_next_submit > 0:
            self.reject_next_submit -= 1
            o.status = "Rejected"
        if self.submit_but_lose_response > 0:
            self.submit_but_lose_response -= 1
            raise SimError("network timeout after broker accepted")  # 单已进券商, 客户端不知道
        return type("R", (), {"order_id": o.order_id})()

    def cancel_order(self, order_id):
        if self.fail_cancel > 0:
            self.fail_cancel -= 1
            raise SimError("network error on cancel")
        o = self.orders.get(order_id)
        if not o:
            raise SimError(f"unknown order {order_id}")
        if o.status in TERMINAL:
            return
        o.cancel_requested = True
        o.status = "PendingCancel"      # 只是"请求撤销", 真正终止要等下一 tick

    def order_detail(self, order_id):
        if self.fail_detail > 0:
            self.fail_detail -= 1
            raise SimError("429 rate limited")
        o = self.orders.get(order_id)
        if not o:
            raise SimError(f"unknown order {order_id}")
        d = type("D", (), {})()
        d.order_id = o.order_id
        d.symbol = o.symbol
        d.side = o.side
        d.status = o.status
        d.executed_quantity = o.executed_quantity
        d.executed_price = o.executed_price
        return d

    def today_orders(self, symbol=None, **kw):
        if self.fail_today_orders:
            raise SimError("today_orders unavailable")
        out = []
        for o in self.orders.values():
            if symbol and o.symbol != symbol:
                continue
            d = type("D", (), {})()
            d.order_id, d.symbol, d.side = o.order_id, o.symbol, o.side
            d.status, d.executed_quantity = o.status, o.executed_quantity
            d.executed_price = o.executed_price
            # 真实 LongPort Order 带 remark(2026-07-21实测 today_orders 返回'tp2'), fake 原漏了
            # → bot 的丢响应认领/出场认领按 remark 甄别腿类型时全拿到空串, 匹配失效。补齐:
            d.remark = o.remark
            d.submitted_quantity = o.submitted_quantity
            out.append(d)
        return out

    def stock_positions(self, symbols=None):
        ps = []
        for sym, q in self.position.items():
            if q == 0:
                continue
            sp = type("P", (), {})()
            sp.symbol, sp.quantity = sym, q
            ps.append(sp)
        ch = type("C", (), {})()
        ch.account_channel = "lb_papertrading"
        ch.positions = ps
        return type("R", (), {"channels": [ch]})()

    def account_balance(self):
        if self.fail_balance:
            raise SimError("balance unavailable")
        b = type("B", (), {})()
        b.currency, b.net_assets = "USD", self.equity
        return [b]

    # ── 撮合: 每 tick 调一次 ──
    def tick(self, step_price=True):
        """推进一步。step_price=False 用于"时间流逝但行情不动"(如 bot 在轮询等撤单终态)。"""
        for o in list(self.orders.values()):
            if o.status in TERMINAL:
                continue
            c = self.quotes.cur(o.symbol)
            # ① 先撮合(�leave 单在被撤销前仍可能成交 —— 这是真实竞态)
            if c is not None:
                last, high, low = c
                fillable = 0
                if o.order_type == "LO":
                    if o.side == "Buy" and low <= o.price:
                        fillable = o.submitted_quantity - o.executed_quantity
                        px = min(o.price, last)
                    elif o.side == "Sell" and high >= o.price:
                        fillable = o.submitted_quantity - o.executed_quantity
                        px = max(o.price, last)
                    else:
                        px = None
                elif o.order_type == "MO":
                    fillable = o.submitted_quantity - o.executed_quantity
                    px = last
                else:
                    px = None
                if fillable > 0 and px is not None:
                    cap = self.liquidity.get(o.symbol)
                    if cap is not None:
                        fillable = min(fillable, cap)
                    if fillable > 0:
                        self._apply_fill(o, fillable, px)
            # ② 再处理撤单请求(本 tick 若已成交, 成交优先)
            if o.status == "PendingCancel" and o.order_id not in self.stuck_cancel:
                if o.executed_quantity == 0:
                    o.status = "Canceled"
                elif o.executed_quantity >= o.submitted_quantity:
                    o.status = "Filled"
                else:
                    o.status = "PartialWithdrawal"
            # ③ Day 单收盘作废
            if o.tif == "Day" and o.status not in TERMINAL:
                if self.clock.now(ET).time() > datetime(2000, 1, 1, 16, 0).time():
                    o.status = "Expired" if o.executed_quantity == 0 else "PartialWithdrawal"
        self.quotes.step()

    def _apply_fill(self, o, qty, px):
        prev_val = o.executed_price * o.executed_quantity
        o.executed_quantity += qty
        o.executed_price = round((prev_val + px * qty) / o.executed_quantity, 4)
        if o.executed_quantity >= o.submitted_quantity:
            o.status = "Filled"
        else:
            o.status = "PartialFilled"
        delta = qty if o.side == "Buy" else -qty
        newpos = self.position.get(o.symbol, 0) + delta
        if newpos < 0 and not self.allow_naked:
            # 真实券商会拒绝卖超; 这里回滚并记录
            o.executed_quantity -= qty
            o.status = "Rejected"
            self.rejected_naked.append((o.order_id, o.symbol, qty))
            return
        self.position[o.symbol] = newpos
        self.fills.append((self.clock.time(), o.order_id, o.symbol, o.side, qty, px))

    # ── 给不变式用的真值查询 ──
    def net_position(self, symbol):
        return self.position.get(symbol, 0)

    def live_orders(self, symbol=None):
        return [o for o in self.orders.values()
                if o.status not in TERMINAL and (symbol is None or o.symbol == symbol)]
