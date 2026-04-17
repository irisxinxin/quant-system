"""
config.py — 全局配置
所有模块从这里读取参数，改一处全局生效
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CACHE_DIR  = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"

for d in [CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────
DATA_START     = "2015-01-01"    # 回测起始日期
DATA_END       = None             # None = 今天
CACHE_EXPIRE_H = 8                # 日线缓存过期时间（小时）

# ─────────────────────────────────────────────
# 板块 ETF 映射
# ─────────────────────────────────────────────
SECTOR_ETFS = {
    "半导体":   "SMH",
    "软件/科技": "IGV",
    "AI概念":   "AIQ",
    "能源/油":  "XLE",
    "重金属":   "GDX",
    "可选消费":  "XLY",
    "必选消费":  "XLP",
    "金融":     "XLF",
    "医疗":     "XLV",
    "工业":     "XLI",
    "材料":     "XLB",
    "公用事业":  "XLU",
}

# 大盘基准
BENCHMARK = "SPY"
BENCHMARK_BOND = "TLT"

# CTA 复制模型监控的核心资产
CTA_UNIVERSE = {
    "SPX":   "SPY",
    "NDX":   "QQQ",
    "Bonds": "TLT",
    "Gold":  "GLD",
    "Oil":   "USO",
    "USD":   "UUP",
}

# ─────────────────────────────────────────────
# 策略参数
# ─────────────────────────────────────────────

# 趋势跟踪
TREND_MA_FAST  = 50
TREND_MA_SLOW  = 200
DUAL_MOM_LK    = 252   # Dual Momentum 回望期（交易日）

# 均值回归
MR_RSI_PERIOD  = 10
MR_RSI_OB      = 70    # 超买
MR_RSI_OS      = 30    # 超卖
MR_BB_PERIOD   = 20
MR_BB_STD      = 2.0
MR_TREND_MA    = 200   # 趋势过滤均线

# CTA 趋势复制
CTA_LOOKBACKS  = [80, 160, 260]   # 短/中/长期
CTA_VOL_WIN    = 90
CTA_COT_PCT_HI = 0.85             # 极度多头阈值
CTA_COT_PCT_LO = 0.15             # 极度空头阈值

# 因子选股
FACTOR_UNIV_SIZE   = 500    # 从多大股票池选（市值前500）
FACTOR_TOP_N       = 20     # 每月选出 top N 只股
FACTOR_REBAL_FREQ  = "ME"   # 月末再平衡

# Smart Money (SMC)
SMC_SWING_LK   = 10    # 摆动高低点回望期
SMC_FVG_MIN    = 0.003 # 最小 FVG 比例（0.3%）

# 裸K
PA_BODY_RATIO  = 0.6   # 实体/总长度比（判断强势K线）
PA_SHADOW_MULT = 2.0   # 影线/实体比（判断锤子/流星）

# 突破系统（Donchian）
DONCHIAN_PERIOD    = 20    # 入场通道周期（日）
DONCHIAN_EXIT      = 10    # 出场通道周期（日，更短）
VIX_HIGH           = 25    # VIX 超过此值 = 高波动，仓位减半
VIX_EXTREME        = 40    # VIX 超过此值 = 极端行情，禁止新开仓
BREAKOUT_ATR_STOP  = 2.0   # 止损 = 入场 - N×ATR
BREAKOUT_ATR_TGT   = 4.0   # 止盈参考 = 入场 + N×ATR

# ─────────────────────────────────────────────
# 回测参数
# ─────────────────────────────────────────────
BACKTEST_INIT_CASH   = 100_000   # 初始资金
BACKTEST_COMMISSION  = 0.001     # 手续费 0.1%
BACKTEST_SLIPPAGE    = 0.0005    # 滑点 0.05%

# Walk-Forward
WF_TRAIN_YEARS = 3
WF_TEST_YEARS  = 1

# ─────────────────────────────────────────────
# 风控参数
# ─────────────────────────────────────────────
RISK_MAX_DD       = 0.15   # 最大允许回撤（触发降仓）
RISK_PER_TRADE    = 0.02   # 单笔最大亏损占比
RISK_MAX_SECTOR   = 0.30   # 单板块最大仓位
RISK_MAX_POSITION = 0.10   # 单只股票最大仓位
