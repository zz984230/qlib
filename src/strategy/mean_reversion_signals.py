"""均值回归策略信号生成器

用于震荡市场的均值回归交易策略。
当价格偏离均值时入场，当价格回归均值时出场。

特点：
- 使用 RSI 超买超卖信号
- 使用布林带位置信号
- 使用均值偏离度信号
- 目标是低买高卖，而非追逐趋势
"""

import logging
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from src.optimizer.individual import Individual

if TYPE_CHECKING:
    from src.strategy.turtle_signals import IndicatorCache

logger = logging.getLogger(__name__)


class MeanReversionSignalGenerator:
    """均值回归策略信号生成器

    用于震荡市场（ADX <= 20）的交易信号生成。

    核心理念：
    - 价格偏离均值时入场（超买时做空，超卖时做多）
    - 价格回归均值时出场
    - 使用较小的止损和目标位

    入场信号：
    - RSI < 30（超卖）或 RSI > 70（超买）
    - 价格触及布林带下轨或上轨
    - 价格显著偏离MA均值

    出场信号：
    - RSI 回归到 40-60 中性区域
    - 价格回归到 MA 均值附近
    - 达到预设目标位
    """

    def __init__(
        self,
        individual: Individual,
        indicator_cache: "IndicatorCache | None" = None
    ):
        """初始化均值回归信号生成器

        Args:
            individual: 包含因子权重和阈值的个体
            indicator_cache: 预计算的指标缓存（可选）
        """
        self.individual = individual
        self.factor_weights = individual.factor_weights
        self.signal_threshold = individual.signal_threshold
        self.exit_threshold = individual.exit_threshold
        self.indicator_cache = indicator_cache

        # 均值回归策略特有参数
        self.rsi_oversold = 30.0   # RSI超卖阈值
        self.rsi_overbought = 70.0  # RSI超买阈值
        self.bb_threshold = 0.2    # 布林带位置阈值（接近下轨0.2或上轨0.8）

    def set_cache(self, cache: "IndicatorCache"):
        """设置指标缓存"""
        self.indicator_cache = cache

    def generate_entry_signal(self, data: pd.DataFrame | None = None, idx: int = -1) -> float:
        """生成入场信号强度

        均值回归策略的入场信号：
        - 超卖信号（买入）：RSI低、价格在下轨附近
        - 超买信号（卖出）：RSI高、价格在上轨附近

        Args:
            data: 包含 OHLCV 数据的 DataFrame（当没有缓存时使用）
            idx: 数据索引位置（使用缓存时）

        Returns:
            入场信号强度 [0, 1]，值越高表示信号越强
        """
        if self.indicator_cache is not None:
            return self._generate_signal_cached(idx)
        else:
            return self._generate_signal_no_cache(data, idx)

    def _generate_signal_cached(self, idx: int) -> float:
        """使用缓存生成入场信号"""
        cache = self.indicator_cache
        if idx < 20:
            return 0.0

        # 获取关键指标
        rsi = cache.get("rsi14")
        bb_upper = cache.get("bb_upper")
        bb_lower = cache.get("bb_lower")
        ma20 = cache.get("ma20")
        close = cache.close

        if rsi is None or bb_upper is None or bb_lower is None or ma20 is None:
            return 0.0

        signal_strength = 0.0

        # 1. RSI 超买超卖信号
        current_rsi = float(rsi[idx])
        if not np.isnan(current_rsi):
            # 超卖信号（买入）
            if current_rsi < self.rsi_oversold:
                rsi_signal = (self.rsi_oversold - current_rsi) / self.rsi_oversold
                signal_strength += rsi_signal * 0.4
            # 超买信号（卖出）
            elif current_rsi > self.rsi_overbought:
                rsi_signal = (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
                signal_strength += rsi_signal * 0.4

        # 2. 布林带位置信号
        current_close = float(close[idx])
        current_upper = float(bb_upper[idx])
        current_lower = float(bb_lower[idx])

        if not np.isnan(current_upper) and not np.isnan(current_lower):
            bb_range = current_upper - current_lower
            if bb_range > 1e-10:
                bb_position = (current_close - current_lower) / bb_range

                # 接近下轨（买入）
                if bb_position < self.bb_threshold:
                    bb_signal = (self.bb_threshold - bb_position) / self.bb_threshold
                    signal_strength += bb_signal * 0.3
                # 接近上轨（卖出）
                elif bb_position > (1 - self.bb_threshold):
                    bb_signal = (bb_position - (1 - self.bb_threshold)) / self.bb_threshold
                    signal_strength += bb_signal * 0.3

        # 3. 均值偏离度信号
        current_ma = float(ma20[idx])
        if not np.isnan(current_ma) and current_ma > 0:
            deviation = (current_close - current_ma) / current_ma

            # 价格低于均值（买入）
            if deviation < -0.02:  # 低于均值2%
                deviation_signal = abs(deviation) / 0.05  # 假设5%偏离为强信号
                signal_strength += min(deviation_signal, 1.0) * 0.2
            # 价格高于均值（卖出）
            elif deviation > 0.02:
                deviation_signal = deviation / 0.05
                signal_strength += min(deviation_signal, 1.0) * 0.2

        # 4. 因子权重加权
        factor_contribution = 0.0
        total_weight = 0.0

        for factor_name, weight in self.factor_weights.items():
            factor_value = self._calculate_factor_value_cached(factor_name, idx)

            if factor_value is None or np.isnan(factor_value):
                continue

            # 均值回归策略使用反向因子：极值信号
            normalized_value = self._normalize_factor_for_reversion(factor_name, factor_value)
            factor_contribution += normalized_value * weight
            total_weight += weight

        if total_weight > 0:
            factor_contribution /= total_weight
            signal_strength += factor_contribution * 0.1

        return float(np.clip(signal_strength, 0, 1))

    def _generate_signal_no_cache(self, data: pd.DataFrame | None, idx: int) -> float:
        """不使用缓存生成入场信号"""
        if data is None or len(data) < 20:
            return 0.0

        close = data["close"].values
        high = data["high"].values
        low = data["low"].values

        # 计算 RSI
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        gain_series = pd.Series(gain)
        loss_series = pd.Series(loss)
        avg_gain = gain_series.ewm(span=14, adjust=False).mean().values
        avg_loss = loss_series.ewm(span=14, adjust=False).mean().values

        rs = np.divide(avg_gain, avg_loss, out=np.ones(len(avg_gain)), where=avg_loss > 1e-10)
        rsi_values = 100 - (100 / (1 + rs))

        if len(rsi_values) == 0:
            return 0.0

        current_rsi = rsi_values[-1]
        signal_strength = 0.0

        # RSI 信号
        if current_rsi < self.rsi_oversold:
            signal_strength += ((self.rsi_oversold - current_rsi) / self.rsi_oversold) * 0.4
        elif current_rsi > self.rsi_overbought:
            signal_strength += ((current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)) * 0.4

        # MA 偏离度信号
        ma20 = pd.Series(close).rolling(20).mean().values
        if len(ma20) > 0 and not np.isnan(ma20[-1]):
            deviation = (close[-1] - ma20[-1]) / ma20[-1]
            if abs(deviation) > 0.02:
                signal_strength += min(abs(deviation) / 0.05, 1.0) * 0.3

        return float(np.clip(signal_strength, 0, 1))

    def generate_exit_signal(self, data: pd.DataFrame | None = None, idx: int = -1) -> float:
        """生成出场信号强度

        均值回归策略的出场信号：
        - 价格回归到均值附近
        - RSI 回归到中性区域（40-60）
        - 达到预设目标位

        Args:
            data: 包含 OHLCV 数据的 DataFrame
            idx: 数据索引位置（使用缓存时）

        Returns:
            出场信号强度 [0, 1]，值越高表示出场信号越强
        """
        if self.indicator_cache is not None:
            cache = self.indicator_cache
            if idx < 20:
                return 0.0

            rsi = cache.get("rsi14")
            ma20 = cache.get("ma20")
            close = cache.close

            if rsi is None or ma20 is None:
                return 0.0

            exit_strength = 0.0

            # 1. RSI 回归中性区域
            current_rsi = float(rsi[idx])
            if not np.isnan(current_rsi):
                if 40 <= current_rsi <= 60:
                    exit_strength += 0.5
                elif 35 <= current_rsi <= 65:
                    exit_strength += 0.3

            # 2. 价格回归均值
            current_close = float(close[idx])
            current_ma = float(ma20[idx])

            if not np.isnan(current_ma) and current_ma > 0:
                deviation = abs(current_close - current_ma) / current_ma
                if deviation < 0.01:  # 接近均值1%以内
                    exit_strength += 0.5
                elif deviation < 0.02:
                    exit_strength += 0.3

            return float(np.clip(exit_strength, 0, 1))
        else:
            if data is None or len(data) < 20:
                return 0.0

            close = data["close"].values
            ma20 = pd.Series(close).rolling(20).mean().values

            if len(ma20) == 0 or np.isnan(ma20[-1]):
                return 0.0

            deviation = abs(close[-1] - ma20[-1]) / ma20[-1]

            if deviation < 0.01:
                return 1.0
            elif deviation < 0.02:
                return 0.5
            else:
                return 0.0

    def should_enter(self, data: pd.DataFrame | None = None, idx: int = -1) -> bool:
        """判断是否应该入场

        均值回归策略入场条件：
        1. 信号强度超过阈值
        2. 市场处于震荡状态（由外部判断）
        """
        entry_signal = self.generate_entry_signal(data, idx)
        return entry_signal >= self.signal_threshold

    def should_exit(self, data: pd.DataFrame | None = None, idx: int = -1) -> bool:
        """判断是否应该出场

        均值回归策略出场条件：
        1. 出场信号强度超过阈值
        2. 价格回归均值
        3. RSI 回归中性区域
        """
        exit_signal = self.generate_exit_signal(data, idx)
        return exit_signal >= self.exit_threshold

    def _calculate_factor_value_cached(self, factor_name: str, idx: int) -> float | None:
        """使用缓存计算因子值（用于均值回归策略）"""
        cache = self.indicator_cache
        close = cache.close
        high = cache.high
        low = cache.low
        volume = cache.volume

        try:
            if factor_name == "rsi":
                rsi = cache.get("rsi14")
                if rsi is not None:
                    val = float(rsi[idx])
                    # 均值回归：关注极值（超买超卖）
                    if val < 30:
                        return (30 - val) / 30  # 超卖程度
                    elif val > 70:
                        return (val - 70) / 30  # 超买程度
                    return 0.0

            elif factor_name == "bb_ratio":
                ma20 = cache.get("ma20")
                std20 = cache.get("std20")
                if ma20 is not None and std20 is not None:
                    upper = ma20[idx] + 2 * std20[idx]
                    lower = ma20[idx] - 2 * std20[idx]
                    position = (close[idx] - lower) / (upper - lower + 1e-10)
                    # 均值回归：关注极值位置
                    if position < 0.2:
                        return (0.2 - position) / 0.2
                    elif position > 0.8:
                        return (position - 0.8) / 0.2
                    return 0.0

            elif factor_name == "ma_ratio":
                ma5 = cache.get("ma5")
                ma20 = cache.get("ma20")
                if ma5 is not None and ma20 is not None:
                    ratio = ma5[idx] / ma20[idx] - 1
                    # 均值回归：关注过度偏离
                    return abs(ratio) * 10

            elif factor_name == "volatility":
                std20 = cache.get("std20")
                if std20 is not None:
                    vol = std20[idx] / close[idx]
                    # 均值回归偏好适度波动
                    if 0.01 < vol < 0.03:
                        return 1.0
                    return 0.5

            elif factor_name == "cci":
                cci = cache.get("cci14")
                if cci is not None:
                    val = float(cci[idx])
                    # CCI < -100 超卖，> 100 超买
                    if val < -100:
                        return abs(val + 100) / 100
                    elif val > 100:
                        return (val - 100) / 100
                    return 0.0

            elif factor_name == "williams_r":
                if idx >= 14:
                    highest = np.max(high[idx-13:idx+1])
                    lowest = np.min(low[idx-13:idx+1])
                    if highest - lowest > 1e-10:
                        wr = -100 * (highest - close[idx]) / (highest - lowest)
                        # Williams %R < -80 超卖，> -20 超买
                        if wr < -80:
                            return abs(wr + 80) / 20
                        elif wr > -20:
                            return (wr + 20) / 80
                    return 0.0

            # 其他因子使用默认计算
            return 0.0

        except Exception as e:
            logger.warning(f"Factor {factor_name} calculation failed: {e}")
            return 0.0

    def _normalize_factor_for_reversion(self, factor_name: str, value: float) -> float:
        """为均值回归策略归一化因子值

        均值回归策略关注极值，因此归一化方式与趋势策略不同。
        """
        # 已经在 _calculate_factor_value_cached 中处理过了
        return np.clip(value, 0, 1)

    def get_signal_info(self, data: pd.DataFrame | None = None, idx: int = -1) -> dict:
        """获取信号详细信息"""
        entry_signal = self.generate_entry_signal(data, idx)
        exit_signal = self.generate_exit_signal(data, idx)

        return {
            "strategy_type": "mean_reversion",
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
            "signal_threshold": self.signal_threshold,
            "exit_threshold": self.exit_threshold,
            "should_enter": entry_signal >= self.signal_threshold,
            "should_exit": exit_signal >= self.exit_threshold,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "bb_threshold": self.bb_threshold,
        }
