"""策略模块"""

from src.strategy.base import BaseStrategy, SimpleFactorStrategy, MomentumStrategy
from src.strategy.advanced import (
    MultiFactorStrategy,
    DualMAStrategy,
    MeanReversionStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    BreakoutStrategy,
    OscillationStrategy,
    get_strategy,
    list_strategies,
)
from src.strategy.regime import RegimeDetector, MarketRegime, MarketState, get_recommended_strategy
from src.strategy.adaptive import AdaptiveStrategy, DynamicAllocationStrategy
from src.strategy.factor_driven import FactorDrivenStrategy
from src.strategy.turtle_position import (
    TurtlePositionManager,
    TurtleRiskManager,
    Position,
    PortfolioState,
)
from src.strategy.turtle_signals import TurtleSignalGenerator

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
    "OscillationStrategy",
    # 因子驱动策略
    "FactorDrivenStrategy",
    # 自适应策略
    "AdaptiveStrategy",
    "DynamicAllocationStrategy",
    # 市场状态
    "RegimeDetector",
    "MarketRegime",
    "MarketState",
    "get_recommended_strategy",
    # 工具函数
    "get_strategy",
    "list_strategies",
    # 海龟交易法则模块
    "TurtlePositionManager",
    "TurtleRiskManager",
    "Position",
    "PortfolioState",
    "TurtleSignalGenerator",
]
