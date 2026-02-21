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


class OscillationStrategy(BaseStrategy):
    """
    震荡市场优化策略

    在震荡市场中表现更佳，使用多个指标确认:
    - RSI 超卖/超买
    - 布林带上下轨
    - 随机指标
    - ADX 过滤（避免在趋势中做均值回归）
    """

    def __init__(
        self,
        rsi_window: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        bb_window: int = 20,
        bb_std: float = 2.0,
        stoch_window: int = 14,
        adx_threshold: float = 25,
        min_confirmations: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.stoch_window = stoch_window
        self.adx_threshold = adx_threshold
        self.min_confirmations = min_confirmations

        # 初始化因子
        self._rsi_factor = RSIFactor(window=rsi_window)
        self._bb_factor = BollingerBandsFactor(window=bb_window, num_std=bb_std)

    def get_factors(self) -> list[str]:
        return [
            f"RSI($close, {self.rsi_window})",
            f"($close - Mean($close, {self.bb_window})) / ({self.bb_std} * Std($close, {self.bb_window}))",
        ]

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        high = data["high"].values if "high" in data.columns else close
        low = data["low"].values if "low" in data.columns else close

        signals = np.zeros(len(close))

        if len(close) < max(self.rsi_window, self.bb_window, self.stoch_window) + 1:
            return signals

        # 计算 RSI
        rsi = self._rsi_factor.calculate(data)

        # 计算布林带位置
        bb_position = self._bb_factor.calculate(data)

        # 计算随机指标
        stoch_k = self._calculate_stochastic(close, high, low)

        # 计算 ADX
        adx = self._calculate_adx(data)

        for i in range(len(close)):
            if np.isnan(rsi[i]) or np.isnan(bb_position[i]) or np.isnan(stoch_k[i]):
                continue

            # ADX 过滤：趋势市场中不做均值回归
            if adx[i] > self.adx_threshold:
                continue

            # 统计确认信号数量
            buy_signals = 0
            sell_signals = 0

            # RSI 信号
            if rsi[i] < self.rsi_oversold:
                buy_signals += 1
            elif rsi[i] > self.rsi_overbought:
                sell_signals += 1

            # 布林带信号
            if bb_position[i] < 0.1:  # 接近下轨
                buy_signals += 1
            elif bb_position[i] > 0.9:  # 接近上轨
                sell_signals += 1

            # 随机指标信号
            if stoch_k[i] < 20:  # 超卖
                buy_signals += 1
            elif stoch_k[i] > 80:  # 超买
                sell_signals += 1

            # 需要足够多的确认
            if buy_signals >= self.min_confirmations:
                signals[i] = buy_signals / 3.0  # 信号强度
            elif sell_signals >= self.min_confirmations:
                signals[i] = -sell_signals / 3.0

        return signals

    def _calculate_stochastic(
        self, close: np.ndarray, high: np.ndarray, low: np.ndarray
    ) -> np.ndarray:
        """计算随机指标 K 值"""
        stoch_k = np.full(len(close), np.nan)

        for i in range(self.stoch_window - 1, len(close)):
            window_low = np.min(low[i - self.stoch_window + 1 : i + 1])
            window_high = np.max(high[i - self.stoch_window + 1 : i + 1])

            if window_high - window_low > 0:
                stoch_k[i] = 100 * (close[i] - window_low) / (window_high - window_low)

        return stoch_k

    def _calculate_adx(self, data: pd.DataFrame) -> np.ndarray:
        """简化版 ADX 计算"""
        close = data["close"].values
        adx = np.full(len(close), np.nan)

        if "high" not in data.columns or "low" not in data.columns:
            return adx

        high = data["high"].values
        low = data["low"].values
        period = 14

        if len(close) < period * 2:
            return adx

        # 计算 TR
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # 计算 +DM 和 -DM
        plus_dm = np.zeros(len(close))
        minus_dm = np.zeros(len(close))
        for i in range(1, len(close)):
            up = high[i] - high[i - 1]
            down = low[i - 1] - low[i]
            plus_dm[i] = up if up > down and up > 0 else 0
            minus_dm[i] = down if down > up and down > 0 else 0

        # 平滑
        atr = self._wilder_smooth(tr, period)
        plus_dm_smooth = self._wilder_smooth(plus_dm, period)
        minus_dm_smooth = self._wilder_smooth(minus_dm, period)

        # DI
        plus_di = np.where(atr > 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr > 0, 100 * minus_dm_smooth / atr, 0)

        # DX
        dx = np.where(
            (plus_di + minus_di) > 0,
            100 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
            0,
        )

        # ADX
        adx_smooth = self._wilder_smooth(dx, period)
        adx[period * 2 - 1 :] = adx_smooth[period - 1 :]

        return adx

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder 平滑"""
        result = np.zeros(len(data))
        if len(data) < period:
            return result

        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period
        return result

    def get_strategy_config(self) -> dict[str, Any]:
        config = super().get_strategy_config()
        config["rsi_window"] = self.rsi_window
        config["rsi_oversold"] = self.rsi_oversold
        config["rsi_overbought"] = self.rsi_overbought
        config["bb_window"] = self.bb_window
        config["min_confirmations"] = self.min_confirmations
        return config


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
    "oscillation": OscillationStrategy,
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
        "oscillation": OscillationStrategy,
    }

    if name not in strategy_map:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategy_map.keys())}")

    return strategy_map[name](**params)


def list_strategies() -> list[str]:
    """列出所有可用策略"""
    return list(STRATEGY_REGISTRY.keys())
