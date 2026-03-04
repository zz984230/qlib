"""动态策略选择器 - 根据市场状态选择合适策略

强趋势/弱趋势 → 海龟趋势跟踪策略
震荡 → 均值回归策略
异常波动 → 空仓观望
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

MarketState = Literal["strong_trend", "weak_trend", "ranging", "volatile"]
StrategyType = Literal["turtle", "mean_reversion", None]


class StrategySelector:
    """根据市场状态动态选择策略"""

    def __init__(self):
        self.current_strategy = None
        self.current_market_state = None

    def select_strategy(self, market_state: MarketState) -> StrategyType:
        """根据市场状态选择策略类型

        Args:
            market_state: 市场状态

        Returns:
            策略类型：turtle, mean_reversion, 或 None（空仓）
        """
        self.current_market_state = market_state

        if market_state == "volatile":
            # 异常波动市：空仓观望
            self.current_strategy = None
            logger.info("异常波动市场：空仓观望")
            return None

        elif market_state in ["strong_trend", "weak_trend"]:
            # 强/弱趋势市：海龟趋势跟踪
            self.current_strategy = "turtle"
            logger.info(f"{market_state}市场：使用海龟趋势跟踪策略")
            return "turtle"

        elif market_state == "ranging":
            # 震荡市：均值回归策略
            self.current_strategy = "mean_reversion"
            logger.info("震荡市场：使用均值回归策略")
            return "mean_reversion"

        else:
            # 未知状态：默认使用海龟
            self.current_strategy = "turtle"
            logger.warning(f"未知市场状态{market_state}，默认使用海龟策略")
            return "turtle"

    def get_strategy_name(self, strategy_type: StrategyType) -> str:
        """获取策略名称

        Args:
            strategy_type: 策略类型

        Returns:
            策略名称
        """
        if strategy_type == "turtle":
            return "海龟趋势跟踪策略"
        elif strategy_type == "mean_reversion":
            return "均值回归策略"
        else:
            return "空仓观望"

    def get_strategy_description(self, strategy_type: StrategyType) -> str:
        """获取策略描述

        Args:
            strategy_type: 策略类型

        Returns:
            策略描述
        """
        descriptions = {
            "turtle": (
                "海龟趋势跟踪策略：追逐趋势，让利润奔跑。"
                "入场：突破20日高点。止损：入场价 - 2*ATR。"
                "加仓：每0.5*ATR加仓，最多4单位。"
            ),
            "mean_reversion": (
                "均值回归策略：低买高卖，区间交易。"
                "入场：RSI超买超卖或触及布林带。止损：1.5*ATR。"
                "出场：价格回归均值或RSI回归中性区域。"
            ),
            None: "空仓观望：避免在异常波动市场交易。"
        }

        return descriptions.get(strategy_type, "未知策略")

    def get_position_multiplier(self, market_state: MarketState | None = None) -> float:
        """获取仓位系数

        Args:
            market_state: 市场状态

        Returns:
            仓位系数 (0-1)
        """
        if market_state is None:
            market_state = self.current_market_state

        multipliers = {
            "strong_trend": 1.0,    # 100%仓位
            "weak_trend": 0.6,      # 60%仓位
            "ranging": 0.25,        # 25%仓位
            "volatile": 0.0,        # 空仓
        }

        return multipliers.get(market_state, 0.5)  # 默认50%

    def get_stop_loss_atr(self, market_state: MarketState | None = None) -> tuple[float, float]:
        """获取止损ATR倍数范围

        Args:
            market_state: 市场状态

        Returns:
            (最小倍数, 最大倍数)
        """
        if market_state is None:
            market_state = self.current_market_state

        stop_ranges = {
            "strong_trend": (2.0, 2.5),   # 海龟标准：2ATR
            "weak_trend": (1.8, 2.2),     # 略紧
            "ranging": (1.0, 1.5),       # 均值回归：窄止损
            "volatile": (0.5, 1.0),      # 超紧止损
        }

        return stop_ranges.get(market_state, (1.5, 2.0))

    def should_trade(self, market_state: MarketState | None = None) -> bool:
        """判断是否应该交易

        Args:
            market_state: 市场状态

        Returns:
            是否可以交易
        """
        if market_state is None:
            market_state = self.current_market_state

        # 异常波动市不交易
        if market_state == "volatile":
            return False

        return True


# 便捷函数
def get_strategy_for_market(market_state: MarketState) -> str | None:
    """根据市场状态获取策略类型

    Args:
        market_state: 市场状态

    Returns:
        策略类型或None
    """
    selector = StrategySelector()
    return selector.select_strategy(market_state)
