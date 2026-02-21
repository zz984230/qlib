"""
多因子驱动策略

支持自定义因子表达式和权重，通过因子组合生成交易信号
"""

from typing import Any

import numpy as np
import pandas as pd

from src.strategy.base import BaseStrategy


class FactorDrivenStrategy(BaseStrategy):
    """
    多因子驱动策略

    通过组合多个因子生成综合信号，支持自定义因子表达式和权重
    """

    def __init__(
        self,
        factor_expressions: list[str] | None = None,
        factor_weights: list[float] | None = None,
        signal_threshold_buy: float = 0.3,
        signal_threshold_sell: float = -0.2,
        **kwargs,
    ):
        """
        初始化多因子策略

        Args:
            factor_expressions: 因子表达式列表
            factor_weights: 因子权重列表 (和为1)
            signal_threshold_buy: 买入信号阈值
            signal_threshold_sell: 卖出信号阈值
        """
        super().__init__(**kwargs)

        self.factor_expressions = factor_expressions or [
            "($close - Ref($close, 5)) / Ref($close, 5)",  # 5日动量
            "($close - Mean($close, 20)) / Mean($close, 20)",  # 均线偏离
        ]

        # 默认等权
        if factor_weights is None:
            self.factor_weights = [1.0 / len(self.factor_expressions)] * len(self.factor_expressions)
        else:
            # 归一化权重
            total = sum(factor_weights)
            self.factor_weights = [w / total for w in factor_weights]

        self.signal_threshold_buy = signal_threshold_buy
        self.signal_threshold_sell = signal_threshold_sell

        # 缓存因子名称
        self._factor_names = [f"factor_{i}" for i in range(len(self.factor_expressions))]

    def get_factors(self) -> list[str]:
        """返回因子表达式列表"""
        return self.factor_expressions

    def get_factor_names(self) -> list[str]:
        """获取因子名称列表"""
        return self._factor_names

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        生成交易信号

        1. 计算各因子值
        2. 标准化到 [-1, 1] 范围
        3. 加权组合
        4. 根据阈值生成买卖信号

        Args:
            data: 包含价格数据的 DataFrame (需有 close, high, low, volume 列)

        Returns:
            信号数组，正值表示买入信号强度，负值表示卖出信号强度
        """
        close = data["close"].values
        signals = np.zeros(len(close))

        if len(close) < 20:  # 至少需要20天数据
            return signals

        # 计算所有因子值
        factor_values = self._calculate_factors(data)

        # 标准化每个因子到 [-1, 1]
        normalized_factors = []
        for fv in factor_values:
            nf = self._normalize_factor(fv)
            normalized_factors.append(nf)

        # 加权组合
        combined_signal = np.zeros(len(close))
        for nf, weight in zip(normalized_factors, self.factor_weights):
            combined_signal += weight * np.nan_to_num(nf, nan=0)

        # 生成离散信号
        for i in range(len(combined_signal)):
            if combined_signal[i] >= self.signal_threshold_buy:
                # 买入信号强度
                signals[i] = min(combined_signal[i], 1.0)
            elif combined_signal[i] <= self.signal_threshold_sell:
                # 卖出信号强度
                signals[i] = max(combined_signal[i], -1.0)

        return signals

    def _calculate_factors(self, data: pd.DataFrame) -> list[np.ndarray]:
        """
        计算所有因子值

        Args:
            data: 价格数据 DataFrame

        Returns:
            因子值数组列表
        """
        factor_values = []

        for expr in self.factor_expressions:
            fv = self._evaluate_expression(expr, data)
            factor_values.append(fv)

        return factor_values

    def _evaluate_expression(self, expr: str, data: pd.DataFrame) -> np.ndarray:
        """
        计算因子表达式

        支持 Qlib 风格的表达式:
        - $close, $open, $high, $low, $volume
        - Ref($var, n): n日前值
        - Mean($var, n): n日均值
        - Std($var, n): n日标准差
        - Max($var, n): n日最大值
        - Min($var, n): n日最小值
        - RSI($close, n): RSI指标

        Args:
            expr: 因子表达式
            data: 价格数据

        Returns:
            因子值数组
        """
        close = data["close"].values
        high = data["high"].values if "high" in data.columns else close
        low = data["low"].values if "low" in data.columns else close
        volume = data["volume"].values if "volume" in data.columns else np.ones(len(close))

        n = len(close)
        result = np.full(n, np.nan)

        # 简单解析表达式
        try:
            # 价格动量
            if "Ref($close" in expr:
                # 提取天数
                import re
                match = re.search(r'Ref\(\$close,\s*(\d+)\)', expr)
                if match:
                    days = int(match.group(1))
                    if "(Ref" in expr:
                        # 动量率: ($close - Ref($close, n)) / Ref($close, n)
                        result[days:] = (close[days:] - close[:-days]) / close[:-days]
                    else:
                        result[days:] = close[:-days]

            # 均线偏离
            elif "Mean($close" in expr and "($close - Mean" in expr:
                import re
                match = re.search(r'Mean\(\$close,\s*(\d+)\)', expr)
                if match:
                    window = int(match.group(1))
                    ma = self._sma(close, window)
                    result = (close - ma) / ma

            # 简单均线
            elif "Mean($close" in expr:
                import re
                match = re.search(r'Mean\(\$close,\s*(\d+)\)', expr)
                if match:
                    window = int(match.group(1))
                    result = self._sma(close, window)

            # 均线比率
            elif "Mean($close, 5) / Mean($close, 20)" in expr:
                ma5 = self._sma(close, 5)
                ma20 = self._sma(close, 20)
                result = ma5 / ma20

            # 成交量比率
            elif "Mean($volume" in expr:
                import re
                matches = re.findall(r'Mean\(\$volume,\s*(\d+)\)', expr)
                if len(matches) >= 2:
                    w1, w2 = int(matches[0]), int(matches[1])
                    vol_ma1 = self._sma(volume, w1)
                    vol_ma2 = self._sma(volume, w2)
                    result = vol_ma1 / vol_ma2

            # 波动率
            elif "Std($close" in expr:
                import re
                match = re.search(r'Std\(\$close,\s*(\d+)\)', expr)
                if match:
                    window = int(match.group(1))
                    std = self._rolling_std(close, window)
                    ma = self._sma(close, window)
                    result = std / ma

            # RSI
            elif "RSI($close" in expr:
                import re
                match = re.search(r'RSI\(\$close,\s*(\d+)\)', expr)
                if match:
                    period = int(match.group(1))
                    rsi = self._calculate_rsi(close, period)
                    if "/ 100" in expr:
                        result = rsi / 100
                    else:
                        result = rsi

            # 布林带位置
            elif "bb_position" in expr or "($close - Mean" in expr and "2 * Std" in expr:
                window = 20
                ma = self._sma(close, window)
                std = self._rolling_std(close, window)
                result = (close - ma) / (2 * std)

            # K线位置
            elif "($close - $low) / ($high - $low)" in expr:
                range_hl = high - low
                range_hl[range_hl == 0] = np.nan
                result = (close - low) / range_hl

            # 振幅
            elif "($high - $low) / $close" in expr:
                result = (high - low) / close

            # 成交量变化
            elif "($volume - Ref($volume" in expr:
                vol_change = np.zeros(n)
                vol_change[1:] = (volume[1:] - volume[:-1]) / volume[:-1]
                result = vol_change

            else:
                # 默认：尝试作为动量因子
                result[5:] = (close[5:] - close[:-5]) / close[:-5]

        except Exception:
            result = np.full(n, np.nan)

        return result

    def _normalize_factor(self, factor: np.ndarray) -> np.ndarray:
        """
        标准化因子到 [-1, 1] 范围

        使用滚动 z-score 然后通过 tanh 映射到 [-1, 1]

        Args:
            factor: 因子值数组

        Returns:
            标准化后的因子值
        """
        result = np.full(len(factor), np.nan)

        # 使用全局统计进行标准化
        valid_mask = ~np.isnan(factor)
        if not np.any(valid_mask):
            return result

        valid_values = factor[valid_mask]
        mean = np.mean(valid_values)
        std = np.std(valid_values)

        if std > 0:
            z_score = (factor - mean) / std
            # 使用 tanh 将 z-score 映射到 [-1, 1]
            result = np.tanh(z_score)

        return result

    def _sma(self, data: np.ndarray, window: int) -> np.ndarray:
        """计算简单移动平均"""
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        return result

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """计算滚动标准差"""
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1])
        return result

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> np.ndarray:
        """计算 RSI"""
        rsi = np.full(len(close), 50.0)
        if len(close) < period + 1:
            return rsi

        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(close) - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))

        return rsi

    def get_strategy_config(self) -> dict[str, Any]:
        """获取策略配置"""
        config = super().get_strategy_config()
        config["factor_count"] = len(self.factor_expressions)
        config["factor_weights"] = self.factor_weights
        config["signal_threshold_buy"] = self.signal_threshold_buy
        config["signal_threshold_sell"] = self.signal_threshold_sell
        return config

    def __str__(self) -> str:
        return f"FactorDrivenStrategy(factors={len(self.factor_expressions)}, buy_threshold={self.signal_threshold_buy})"


# 注册到策略注册表
from src.strategy.advanced import STRATEGY_REGISTRY

STRATEGY_REGISTRY["factor_driven"] = FactorDrivenStrategy
