"""参数适配器 - 根据市场状态提供参数配置

为四种市场状态定义不同的参数配置，实现动态风险调整。
"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# 市场状态类型
MarketState = Literal["strong_trend", "weak_trend", "ranging", "volatile"]


@dataclass
class MarketParameters:
    """市场状态对应的参数配置"""
    # 信号阈值
    signal_threshold: tuple[float, float]  # (min, max)
    exit_threshold: tuple[float, float]    # (min, max)

    # 止损参数
    stop_loss_atr: tuple[float, float]     # ATR倍数范围
    trailing_stop_trigger: tuple[float, float]

    # 仓位控制
    position_multiplier: float             # 仓位系数 (0-1)
    max_units: int                         # 最大加仓单位数

    # 遗传算法
    factor_count_range: tuple[int, int]    # 因子数量范围


# 四种市场状态的参数配置
MARKET_PARAM_CONFIG: dict[MarketState, MarketParameters] = {
    "strong_trend": MarketParameters(
        # 强趋势：放宽入场门槛，宽止损，满仓运行
        signal_threshold=(0.05, 0.15),
        exit_threshold=(0.05, 0.15),
        stop_loss_atr=(2.5, 3.5),
        trailing_stop_trigger=(1.5, 2.5),
        position_multiplier=1.0,    # 100%仓位
        max_units=4,
        factor_count_range=(2, 5),
    ),
    "weak_trend": MarketParameters(
        # 弱趋势：中等门槛，中等止损，60%仓位
        signal_threshold=(0.10, 0.20),
        exit_threshold=(0.08, 0.18),
        stop_loss_atr=(2.0, 2.5),
        trailing_stop_trigger=(1.0, 1.5),
        position_multiplier=0.6,    # 60%仓位
        max_units=3,
        factor_count_range=(2, 4),
    ),
    "ranging": MarketParameters(
        # 震荡：收紧入场，窄止损，轻仓降低磨损
        signal_threshold=(0.15, 0.30),
        exit_threshold=(0.10, 0.20),
        stop_loss_atr=(1.0, 1.5),
        trailing_stop_trigger=(0.5, 1.0),
        position_multiplier=0.25,   # 25%仓位
        max_units=2,
        factor_count_range=(1, 3),
    ),
    "volatile": MarketParameters(
        # 异常波动：严格入场，超紧止损，空仓观望
        signal_threshold=(0.25, 0.40),
        exit_threshold=(0.15, 0.25),
        stop_loss_atr=(0.5, 1.0),
        trailing_stop_trigger=(0.3, 0.7),
        position_multiplier=0.0,    # 0%仓位（空仓）
        max_units=1,
        factor_count_range=(1, 2),
    ),
}


class ParameterAdapter:
    """参数适配器

    根据市场状态提供对应的参数配置，实现动态风险调整。
    """

    def __init__(self, default_state: MarketState = "ranging"):
        """初始化适配器

        Args:
            default_state: 默认市场状态（当无法识别时使用）
        """
        self.default_state = default_state

    def get_parameters(self, market_state: MarketState | None = None) -> MarketParameters:
        """获取市场状态对应的参数配置

        Args:
            market_state: 市场状态，如果为 None 则使用默认状态

        Returns:
            参数配置对象
        """
        if market_state is None:
            market_state = self.default_state

        if market_state not in MARKET_PARAM_CONFIG:
            logger.warning(f"未知市场状态: {market_state}，使用默认状态 {self.default_state}")
            market_state = self.default_state

        return MARKET_PARAM_CONFIG[market_state]

    def adapt_genetic_config(
        self,
        market_state: MarketState | None = None
    ) -> dict:
        """获取遗传算法参数配置

        返回适用于指定市场状态的遗传算法参数范围。

        Args:
            market_state: 市场状态

        Returns:
            参数范围字典
        """
        params = self.get_parameters(market_state)

        return {
            "min_signal_threshold": params.signal_threshold[0],
            "max_signal_threshold": params.signal_threshold[1],
            "min_exit_threshold": params.exit_threshold[0],
            "max_exit_threshold": params.exit_threshold[1],
            "min_stop_loss_atr": params.stop_loss_atr[0],
            "max_stop_loss_atr": params.stop_loss_atr[1],
            "min_trailing_stop_trigger": params.trailing_stop_trigger[0],
            "max_trailing_stop_trigger": params.trailing_stop_trigger[1],
            "min_factor_count": params.factor_count_range[0],
            "max_factor_count": params.factor_count_range[1],
        }

    def get_position_multiplier(self, market_state: MarketState | None = None) -> float:
        """获取仓位系数

        Args:
            market_state: 市场状态

        Returns:
            仓位系数 (0-1)
        """
        params = self.get_parameters(market_state)
        return params.position_multiplier

    def get_max_units(self, market_state: MarketState | None = None) -> int:
        """获取最大加仓单位数

        Args:
            market_state: 市场状态

        Returns:
            最大单位数
        """
        params = self.get_parameters(market_state)
        return params.max_units

    def get_stop_loss_range(
        self,
        market_state: MarketState | None = None
    ) -> tuple[float, float]:
        """获取止损ATR倍数范围

        Args:
            market_state: 市场状态

        Returns:
            (最小倍数, 最大倍数)
        """
        params = self.get_parameters(market_state)
        return params.stop_loss_atr

    def get_risk_description(self, market_state: MarketState | None = None) -> str:
        """获取市场状态的风险描述

        Args:
            market_state: 市场状态

        Returns:
            风险描述字符串
        """
        params = self.get_parameters(market_state)

        descriptions = {
            "strong_trend": (
                f"强趋势市场：满仓({params.position_multiplier*100:.0f}%)运行，"
                f"宽止损({params.stop_loss_atr[0]}-{params.stop_loss_atr[1]}ATR)，"
                f"最多{params.max_units}个加仓单位"
            ),
            "weak_trend": (
                f"弱趋势市场：{params.position_multiplier*100:.0f}%仓位，"
                f"中等止损({params.stop_loss_atr[0]}-{params.stop_loss_atr[1]}ATR)，"
                f"最多{params.max_units}个加仓单位"
            ),
            "ranging": (
                f"震荡市场：轻仓({params.position_multiplier*100:.0f}%)降低磨损，"
                f"窄止损({params.stop_loss_atr[0]}-{params.stop_loss_atr[1]}ATR)，"
                f"最多{params.max_units}个加仓单位"
            ),
            "volatile": (
                f"异常波动：空仓({params.position_multiplier*100:.0f}%)观望，"
                f"超紧止损({params.stop_loss_atr[0]}-{params.stop_loss_atr[1]}ATR)，"
                f"避免交易风险"
            ),
        }

        if market_state and market_state in descriptions:
            return descriptions[market_state]

        return descriptions[self.default_state]


# 便捷函数
def get_market_parameters(market_state: MarketState) -> MarketParameters:
    """便捷函数：获取市场参数配置

    Args:
        market_state: 市场状态

    Returns:
        参数配置对象
    """
    adapter = ParameterAdapter()
    return adapter.get_parameters(market_state)


def get_position_multiplier(market_state: MarketState) -> float:
    """便捷函数：获取仓位系数

    Args:
        market_state: 市场状态

    Returns:
        仓位系数 (0-1)
    """
    adapter = ParameterAdapter()
    return adapter.get_position_multiplier(market_state)
