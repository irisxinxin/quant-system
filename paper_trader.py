"""
paper_trader.py — LongPort 模拟账户自动下单 (ORB 策略专用)

功能:
  - 信号触发时自动下 BUY 限价单
  - 持仓期间监测 5m K 线, 触发止损/止盈/EOD 时自动平仓
  - 全程通过 TradeContext, 但只用模拟账户 (LBPT...)

环境变量需求:
  - LONGPORT_APP_KEY/SECRET/ACCESS_TOKEN  (复用 quote 用的)
  - LONGPORT_TRADE_PASSWORD               (新增, 6 位数字交易密码)

风险控制:
  - 单笔最大风险: 模拟账户的 1% (env: PAPER_RISK_USD, 默认 100)
  - 仓位上限: 单股不超过总资产 30%
"""
import os
import time
from decimal import Decimal
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from typing import Optional
from dataclasses import dataclass, field

from longport.openapi import (
    Config, TradeContext,
    OrderType, OrderSide, TimeInForceType, OutsideRTH,
)


ET = ZoneInfo("US/Eastern")
EOD_CUTOFF = dtime(15, 50)   # 15:50 ET 强制平仓 (留 10 分钟收盘前缓冲)

# 单笔风险 (USD), 决定下多少股
RISK_PER_TRADE_USD = float(os.environ.get("PAPER_RISK_USD", "100"))


@dataclass
class PaperPosition:
    """单只股票的持仓状态"""
    symbol: str
    entry_order_id: Optional[str] = None
    entry_filled: bool = False
    entry_price: float = 0.0
    quantity: int = 0
    stop_price: float = 0.0
    tp_price: float = 0.0
    direction: int = 1   # 1=多, -1=空
    entered_at: Optional[datetime] = None
    closed: bool = False


class PaperTrader:
    """LongPort 模拟账户自动下单器"""

    def __init__(self):
        self.trade_ctx: Optional[TradeContext] = None
        self.unlocked: bool = False
        self.positions: dict = {}   # symbol → PaperPosition

    def init(self) -> bool:
        """初始化 TradeContext + 解锁交易. 失败返回 False (脚本仍能跑, 只是不自动下单)"""
        password = os.environ.get("LONGPORT_TRADE_PASSWORD")
        if not password:
            print("⚠️ LONGPORT_TRADE_PASSWORD 未设置 → 自动 paper trade 关闭, 仅推送信号")
            return False

        try:
            config = Config.from_env()
            self.trade_ctx = TradeContext(config)
            print("🔌 TradeContext 连接...")

            # 解锁交易
            self.trade_ctx.unlock_trade(password)
            self.unlocked = True
            print("✅ Paper trade 解锁成功 (模拟账户)")

            # 注册订单状态变化回调
            self.trade_ctx.set_on_order_changed(self._on_order_changed)

            return True
        except Exception as e:
            print(f"❌ TradeContext 初始化失败: {e}")
            print("   常见原因:")
            print("     - 交易密码错误")
            print("     - 模拟账户未开通模拟交易功能")
            print("     - API 凭证只读没有交易权限")
            return False

    def calculate_quantity(self, entry: float, stop: float) -> int:
        """根据风险预算算下多少股. risk_per_trade / risk_per_share"""
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0
        shares = int(RISK_PER_TRADE_USD / risk_per_share)
        return max(1, shares)

    def submit_buy(self, symbol: str, entry: float, stop: float, tp: float,
                   direction: int = 1) -> bool:
        """
        信号触发时调用: 下买入限价单
        direction: 1=多 (BUY), -1=空 (SELL Short)
        """
        if not self.unlocked:
            return False
        if symbol in self.positions and not self.positions[symbol].closed:
            print(f"   ⚠️ {symbol} 已有未平仓位, 跳过新下单")
            return False

        qty = self.calculate_quantity(entry, stop)
        side = OrderSide.Buy if direction == 1 else OrderSide.Sell

        try:
            resp = self.trade_ctx.submit_order(
                symbol=symbol,
                order_type=OrderType.LO,
                side=side,
                submitted_quantity=Decimal(str(qty)),
                submitted_price=Decimal(str(round(entry, 3))),
                time_in_force=TimeInForceType.Day,
                outside_rth=OutsideRTH.RTHOnly,
                remark=f"ORB-{datetime.now(ET).strftime('%m%d')}",
            )
            order_id = resp.order_id
            print(f"   📤 PAPER 下单成功: {symbol} {qty}股 @ {entry} (order_id={order_id})")

            # 记录持仓 (尚未成交)
            pos = PaperPosition(
                symbol=symbol,
                entry_order_id=order_id,
                entry_price=entry,
                quantity=qty,
                stop_price=stop,
                tp_price=tp,
                direction=direction,
                entered_at=datetime.now(ET),
            )
            self.positions[symbol] = pos
            return True
        except Exception as e:
            print(f"   ❌ PAPER 下单失败 ({symbol}): {e}")
            return False

    def submit_close(self, symbol: str, reason: str = "EOD") -> bool:
        """平仓 (市价单)"""
        if not self.unlocked:
            return False
        if symbol not in self.positions or self.positions[symbol].closed:
            return False

        pos = self.positions[symbol]
        if not pos.entry_filled:
            # 还没成交, 直接撤单
            try:
                self.trade_ctx.cancel_order(pos.entry_order_id)
                pos.closed = True
                print(f"   🗑️ PAPER 取消未成交订单: {symbol}")
            except Exception as e:
                print(f"   ⚠️ 取消订单失败 ({symbol}): {e}")
            return True

        # 已成交, 反向市价平仓
        side = OrderSide.Sell if pos.direction == 1 else OrderSide.Buy
        try:
            resp = self.trade_ctx.submit_order(
                symbol=symbol,
                order_type=OrderType.MO,
                side=side,
                submitted_quantity=Decimal(str(pos.quantity)),
                time_in_force=TimeInForceType.Day,
                outside_rth=OutsideRTH.RTHOnly,
                remark=f"ORB-CLOSE-{reason}",
            )
            print(f"   📤 PAPER 平仓 ({reason}): {symbol} {pos.quantity}股 (order_id={resp.order_id})")
            pos.closed = True
            return True
        except Exception as e:
            print(f"   ❌ PAPER 平仓失败 ({symbol}): {e}")
            return False

    def check_exits_on_bar(self, symbol: str, bar_high: float, bar_low: float,
                            bar_close: float, bar_time_et: datetime):
        """每根新 bar 调用, 检查是否触发止损/止盈/EOD"""
        if not self.unlocked:
            return
        pos = self.positions.get(symbol)
        if not pos or pos.closed or not pos.entry_filled:
            return

        # 1. EOD 强制平仓 (15:50 ET 之后)
        if bar_time_et.time() >= EOD_CUTOFF:
            self.submit_close(symbol, reason="EOD")
            return

        # 2. 止损 (多单: low <= stop; 空单: high >= stop)
        if pos.direction == 1 and bar_low <= pos.stop_price:
            self.submit_close(symbol, reason="STOP")
            return
        if pos.direction == -1 and bar_high >= pos.stop_price:
            self.submit_close(symbol, reason="STOP")
            return

        # 3. 止盈
        if pos.direction == 1 and bar_high >= pos.tp_price:
            self.submit_close(symbol, reason="TP")
            return
        if pos.direction == -1 and bar_low <= pos.tp_price:
            self.submit_close(symbol, reason="TP")
            return

    def _on_order_changed(self, event):
        """订单状态变化回调 (成交/拒单/撤单等)"""
        try:
            symbol = event.symbol
            status = str(event.status)
            if symbol not in self.positions:
                return
            pos = self.positions[symbol]

            # 成交时更新 entry_filled
            if "Filled" in status and not pos.entry_filled:
                pos.entry_filled = True
                if hasattr(event, "executed_price") and event.executed_price:
                    pos.entry_price = float(event.executed_price)
                print(f"   ✅ PAPER 成交: {symbol} @ {pos.entry_price} ({status})")
            elif "Rejected" in status or "Canceled" in status:
                print(f"   ⚠️ PAPER 订单 {status}: {symbol}")
                pos.closed = True
        except Exception as e:
            print(f"   ⚠️ on_order_changed 异常: {e}")


# 全局单例 (供 signal_live_longport.py 调用)
_paper_trader: Optional[PaperTrader] = None


def get_paper_trader() -> PaperTrader:
    """获取全局 PaperTrader 实例 (lazy init)"""
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader()
        _paper_trader.init()
    return _paper_trader
