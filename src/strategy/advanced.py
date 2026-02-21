"""高级策略模板"""

import numpy as np
import pandas as pd
from typing import Any

from src.strategy.base import BaseStrategy
from src.strategy.factors import (
    MomentumFactor,
    RSIFactor,
    VolumeRatioFactor,
    MAFactor,
    BollingerBandsFactor,
)


class MultiFactorStrategy(BaseStrategy):
    """
    多因子策略

    组合多个因子生成综合信号
    """

    def __init__(
        self,
        factors: list | None = None,
        weights: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 默认因子组合
        self.factors = factors or [
            MomentumFactor(window=20),
            RSIFactor(window=14),
            VolumeRatioFactor(window=20),
            MAFactor(window=20),
        ]

        # 默认等权
        self.weights = weights or [1.0 / len(self.factors)] * len(self.factors)

    def get_factors(self) -> list[str]:
        """返回因子表达式"""
        expressions = []
        for factor in self.factors:
            try:
                expr = factor.to_qlib_expression()
                expressions.append(expr)
            except NotImplementedError:
                expressions.append(f"custom:{factor.name}")
        return expressions

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """生成综合信号"""
        signals = np.zeros(len(data))

        for factor, weight in zip(self.factors, self.weights):
            factor_values = factor.calculate(data)
            # 标准化
            if not np.all(np.isnan(factor_values)):
                valid_mask = ~np.isnan(factor_values)
                mean = np.nanmean(factor_values)
                std = np.nanstd(factor_values)
                if std > 0:
                    factor_values = (factor_values - mean) / std
                signals += weight * np.nan_to_num(factor_values, nan=0)

        return signals


class DualMAStrategy(BaseStrategy):
    """
    双均线策略

    短期均线上穿长期均线时买入，下穿时卖出
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def get_factors(self) -> list[str]:
        return [
            f"Mean($close, {self.short_window})",
            f"Mean($close, {self.long_window})",
        ]

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        signals = np.zeros(len(close))

        if len(close) < self.long_window:
            return signals

        # 计算短期和长期均线
        short_ma = np.convolve(close, np.ones(self.short_window) / self.short_window, mode="valid")
        long_ma = np.convolve(close, np.ones(self.long_window) / self.long_window, mode="valid")

        # 对齐数组
        offset = self.long_window - self.short_window
        short_ma_aligned = short_ma[offset:]

        # 生成信号
        for i in range(1, len(long_ma)):
            idx = i + self.long_window - 1
            # 金叉买入
            if short_ma_aligned[i - 1] <= long_ma[i - 1] and short_ma_aligned[i] > long_ma[i]:
                signals[idx] = 1
            # 死叉卖出
            elif short_ma_aligned[i - 1] >= long_ma[i - 1] and short_ma_aligned[i] < long_ma[i]:
                signals[idx] = -1

        return signals

    def get_strategy_config(self) -> dict[str, Any]:
        config = super().get_strategy_config()
        config["short_window"] = self.short_window
        config["long_window"] = self.long_window
        return config


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略

    价格偏离均值过大时反向操作
    """

    def __init__(
        self,
        window: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def get_factors(self) -> list[str]:
        return [
            f"($close - Mean($close, {self.window})) / Std($close, {self.window})",
        ]

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        signals = np.zeros(len(close))

        if len(close) < self.window:
            return signals

        # 计算 Z-score
        for i in range(self.window - 1, len(close)):
            window_data = close[i - self.window + 1 : i + 1]
            mean = np.mean(window_data)
            std = np.std(window_data)

            if std > 0:
                z_score = (close[i] - mean) / std

                # 超跌买入
                if z_score < -self.entry_z:
                    signals[i] = 1
                # 超涨卖出
                elif z_score > self.entry_z:
                    signals[i] = -1
                # 回归中性
                elif abs(z_score) < self.exit_z:
                    signals[i] = 0

        return signals


class RSIStrategy(BaseStrategy):
    """
    RSI 策略

    RSI < 30 超卖买入，RSI > 70 超买卖出
    """

    def __init__(
        self,
        window: int = 14,
        oversold: float = 30,
        overbought: float = 70,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window = window
        self.oversold = oversold
        self.overbought = overbought
        self._rsi_factor = RSIFactor(window=window)

    def get_factors(self) -> list[str]:
        return [f"RSI($close, {self.window})"]

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        rsi = self._rsi_factor.calculate(data)
        signals = np.zeros(len(rsi))

        for i in range(len(rsi)):
            if np.isnan(rsi[i]):
                continue

            if rsi[i] < self.oversold:
                signals[i] = 1  # 超卖买入
            elif rsi[i] > self.overbought:
                signals[i] = -1  # 超买卖出

        return signals


class BollingerBandsStrategy(BaseStrategy):
    """
    布林带策略

    价格触及下轨买入，触及上轨卖出
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window = window
        self.num_std = num_std
        self._bb_factor = BollingerBandsFactor(window=window, num_std=num_std)

    def get_factors(self) -> list[str]:
        return [
            f"($close - Mean($close, {self.window}) + {self.num_std} * Std($close, {self.window})) / (2 * {self.num_std} * Std($close, {self.window}))",
        ]

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        bb = self._bb_factor.calculate(data)
        signals = np.zeros(len(bb))

        for i in range(len(bb)):
            if np.isnan(bb[i]):
                continue

            if bb[i] < 0.1:  # 接近下轨
                signals[i] = 1
            elif bb[i] > 0.9:  # 接近上轨
                signals[i] = -1

        return signals


class BreakoutStrategy(BaseStrategy):
    """
    突破策略

    价格突破 N 日高点买入，跌破 N 日低点卖出
    """

    def __init__(
        self,
        window: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.window = window

    def get_factors(self) -> list[str]:
        return [
            f"$close / Max($close, {self.window})",
            f"$close / Min($close, {self.window})",
        ]

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        signals = np.zeros(len(close))

        if len(close) < self.window:
            return signals

        for i in range(self.window, len(close)):
            window_high = np.max(close[i - self.window : i])
            window_low = np.min(close[i - self.window : i])

            # 突破高点
            if close[i] > window_high:
                signals[i] = 1
            # 跌破低点
            elif close[i] < window_low:
                signals[i] = -1

        return signals


# 策略注册表
STRATEGY_REGISTRY = {
    "simple": "SimpleFactorStrategy",
    "momentum": "MomentumStrategy",
    "multi_factor": MultiFactorStrategy,
    "dual_ma": DualMAStrategy,
    "mean_reversion": MeanReversionStrategy,
    "rsi": RSIStrategy,
    "bollinger": BollingerBandsStrategy,
    "breakout": BreakoutStrategy,
}


def get_strategy(name: str, **params) -> BaseStrategy:
    """获取策略实例"""
    from src.strategy.base import SimpleFactorStrategy, MomentumStrategy

    strategy_map = {
        "simple": SimpleFactorStrategy,
        "momentum": MomentumStrategy,
        "multi_factor": MultiFactorStrategy,
        "dual_ma": DualMAStrategy,
        "mean_reversion": MeanReversionStrategy,
        "rsi": RSIStrategy,
        "bollinger": BollingerBandsStrategy,
        "breakout": BreakoutStrategy,
    }

    if name not in strategy_map:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategy_map.keys())}")

    return strategy_map[name](**params)


def list_strategies() -> list[str]:
    """列出所有可用策略"""
    return list(STRATEGY_REGISTRY.keys())
