"""
自适应策略

根据市场状态自动切换策略
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.strategy.base import BaseStrategy
from src.strategy.regime import MarketState, RegimeDetector, get_recommended_strategy
from src.strategy.advanced import (
    DualMAStrategy,
    MeanReversionStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    BreakoutStrategy,
    MultiFactorStrategy,
)

logger = logging.getLogger(__name__)


class AdaptiveStrategy(BaseStrategy):
    """
    自适应策略

    根据市场状态自动选择最合适的子策略
    """

    # 市场状态 -> 策略映射
    DEFAULT_STRATEGY_MAP = {
        MarketState.TRENDING: "breakout",  # 趋势市场用突破
        MarketState.OSCILLATING: "mean_reversion",  # 震荡市场用均值回归
        MarketState.VOLATILE: "bollinger",  # 高波动用布林带
        MarketState.QUIET: "multi_factor",  # 平稳市场用多因子
    }

    def __init__(
        self,
        strategy_map: Optional[dict[MarketState, str]] = None,
        min_confidence: float = 0.3,
        **kwargs,
    ):
        """
        初始化自适应策略

        Args:
            strategy_map: 自定义状态-策略映射
            min_confidence: 最小置信度阈值，低于此值使用多因子策略
        """
        super().__init__(**kwargs)

        self.strategy_map = strategy_map or self.DEFAULT_STRATEGY_MAP
        self.min_confidence = min_confidence
        self.detector = RegimeDetector()

        # 初始化子策略
        self._sub_strategies = {
            "dual_ma": DualMAStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "rsi": RSIStrategy(),
            "bollinger": BollingerBandsStrategy(),
            "breakout": BreakoutStrategy(),
            "multi_factor": MultiFactorStrategy(),
        }

        # 最近的市场状态
        self._last_regime = None

        logger.info(
            f"自适应策略初始化: 映射={self.strategy_map}, "
            f"最小置信度={self.min_confidence}"
        )

    def get_factors(self) -> list[str]:
        """返回所有子策略的因子"""
        all_factors = []
        for strategy in self._sub_strategies.values():
            all_factors.extend(strategy.get_factors())
        return list(set(all_factors))

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        生成信号

        根据市场状态选择最合适的子策略
        """
        # 检测市场状态
        regime = self.detector.detect(data)
        self._last_regime = regime

        logger.debug(
            f"市场状态: {regime.state.value}, 置信度: {regime.confidence:.2f}, "
            f"ADX: {regime.adx:.1f}, ATR比率: {regime.atr_ratio:.2%}"
        )

        # 置信度过低时使用多因子策略
        if regime.confidence < self.min_confidence:
            strategy_name = "multi_factor"
        else:
            strategy_name = self.strategy_map.get(regime.state, "multi_factor")

        strategy = self._sub_strategies.get(strategy_name)
        if strategy is None:
            logger.warning(f"未知策略: {strategy_name}, 使用多因子策略")
            strategy = self._sub_strategies["multi_factor"]

        # 生成基础信号
        signals = strategy.generate_signals(data)

        # 根据置信度调整信号强度
        signals = signals * (0.5 + 0.5 * regime.confidence)

        # 高波动市场减仓
        if regime.state == MarketState.VOLATILE:
            signals = signals * 0.6
            logger.debug("高波动市场，信号强度降低 40%")

        return signals

    def get_strategy_config(self) -> dict[str, Any]:
        """返回策略配置"""
        config = super().get_strategy_config()
        config["strategy_map"] = {k.value: v for k, v in self.strategy_map.items()}
        config["min_confidence"] = self.min_confidence
        if self._last_regime:
            config["last_regime"] = {
                "state": self._last_regime.state.value,
                "confidence": self._last_regime.confidence,
                "adx": self._last_regime.adx,
            }
        return config

    def get_current_strategy(self) -> str:
        """获取当前使用的子策略名称"""
        if self._last_regime is None:
            return "multi_factor"

        if self._last_regime.confidence < self.min_confidence:
            return "multi_factor"

        return self.strategy_map.get(self._last_regime.state, "multi_factor")


class DynamicAllocationStrategy(BaseStrategy):
    """
    动态配置策略

    同时运行多个策略，根据历史表现动态分配权重
    """

    def __init__(
        self,
        strategies: Optional[list[str]] = None,
        lookback: int = 60,
        rebalance_freq: int = 5,
        **kwargs,
    ):
        """
        初始化动态配置策略

        Args:
            strategies: 策略名称列表
            lookback: 绩效回看期
            rebalance_freq: 权重调整频率（天数）
        """
        super().__init__(**kwargs)

        self.strategy_names = strategies or [
            "dual_ma",
            "mean_reversion",
            "rsi",
            "bollinger",
            "breakout",
        ]
        self.lookback = lookback
        self.rebalance_freq = rebalance_freq

        # 初始化子策略
        self._strategies = {}
        for name in self.strategy_names:
            self._strategies[name] = self._create_strategy(name)

        # 策略权重
        self._weights = {name: 1.0 / len(self.strategy_names) for name in self.strategy_names}

        # 历史信号记录
        self._signal_history = {name: [] for name in self.strategy_names}
        self._days_since_rebalance = 0

    def _create_strategy(self, name: str) -> BaseStrategy:
        """创建策略实例"""
        strategy_classes = {
            "dual_ma": DualMAStrategy,
            "mean_reversion": MeanReversionStrategy,
            "rsi": RSIStrategy,
            "bollinger": BollingerBandsStrategy,
            "breakout": BreakoutStrategy,
            "multi_factor": MultiFactorStrategy,
        }

        cls = strategy_classes.get(name)
        if cls:
            return cls()
        raise ValueError(f"未知策略: {name}")

    def get_factors(self) -> list[str]:
        """返回所有策略的因子"""
        all_factors = []
        for strategy in self._strategies.values():
            all_factors.extend(strategy.get_factors())
        return list(set(all_factors))

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """生成加权信号"""
        signals = np.zeros(len(data))

        # 获取每个策略的信号
        strategy_signals = {}
        for name, strategy in self._strategies.items():
            sig = strategy.generate_signals(data)
            strategy_signals[name] = sig

        # 加权组合
        for name, sig in strategy_signals.items():
            weight = self._weights.get(name, 0)
            signals += weight * sig

        # 定期调整权重（简化版：基于波动率倒数）
        self._days_since_rebalance += 1
        if self._days_since_rebalance >= self.rebalance_freq:
            self._rebalance_weights(data)
            self._days_since_rebalance = 0

        return signals

    def _rebalance_weights(self, data: pd.DataFrame) -> None:
        """调整策略权重"""
        # 简单实现：基于最近波动率分配权重
        # 波动率低的策略获得更高权重
        volatilities = {}

        for name, strategy in self._strategies.items():
            sig = strategy.generate_signals(data)
            if len(sig) > self.lookback:
                recent_sig = sig[-self.lookback:]
                vol = np.std(recent_sig)
                volatilities[name] = vol if vol > 0 else 0.01

        # 波动率倒数加权
        inv_vols = {name: 1.0 / vol for name, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol > 0:
            self._weights = {name: v / total_inv_vol for name, v in inv_vols.items()}
            logger.debug(f"权重调整: {self._weights}")

    def get_strategy_config(self) -> dict[str, Any]:
        """返回策略配置"""
        config = super().get_strategy_config()
        config["strategies"] = self.strategy_names
        config["weights"] = self._weights
        config["lookback"] = self.lookback
        return config
