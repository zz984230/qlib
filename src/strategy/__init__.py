"""策略模块"""

from src.strategy.base import BaseStrategy, SimpleFactorStrategy, MomentumStrategy
from src.strategy.advanced import (
    MultiFactorStrategy,
    DualMAStrategy,
    MeanReversionStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    BreakoutStrategy,
    get_strategy,
    list_strategies,
)

__all__ = [
    # 基础策略
    "BaseStrategy",
    "SimpleFactorStrategy",
    "MomentumStrategy",
    # 高级策略
    "MultiFactorStrategy",
    "DualMAStrategy",
    "MeanReversionStrategy",
    "RSIStrategy",
    "BollingerBandsStrategy",
    "BreakoutStrategy",
    # 工具函数
    "get_strategy",
    "list_strategies",
]
