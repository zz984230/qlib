"""因子基类"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseFactor(ABC):
    """因子基类"""

    def __init__(self, name: str = None, **params):
        self.name = name or self.__class__.__name__
        self.params = params

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """
        计算因子值

        Args:
            data: 包含 OHLCV 数据的 DataFrame

        Returns:
            因子值数组
        """
        raise NotImplementedError

    def __call__(self, data: pd.DataFrame) -> np.ndarray:
        """调用计算"""
        return self.calculate(data)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"<{self.name}({params_str})>"

    def to_qlib_expression(self) -> str:
        """
        转换为 Qlib 因子表达式

        Returns:
            Qlib 表达式字符串
        """
        # 默认实现，子类可以覆盖
        raise NotImplementedError(f"{self.name} does not support Qlib expression yet")


def rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """滚动均值"""
    result = np.full_like(data, np.nan, dtype=np.float64)
    if len(data) < window:
        return result
    cumsum = np.cumsum(data)
    cumsum = np.insert(cumsum, 0, 0)
    result[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return result


def rolling_std(data: np.ndarray, window: int) -> np.ndarray:
    """滚动标准差"""
    result = np.full_like(data, np.nan, dtype=np.float64)
    if len(data) < window:
        return result

    for i in range(window - 1, len(data)):
        result[i] = np.std(data[i - window + 1 : i + 1], ddof=1)
    return result


def rolling_max(data: np.ndarray, window: int) -> np.ndarray:
    """滚动最大值"""
    result = np.full_like(data, np.nan, dtype=np.float64)
    if len(data) < window:
        return result

    from scipy.ndimage import maximum_filter1d

    result[window - 1 :] = maximum_filter1d(data, size=window)[window - 1 :]
    return result


def rolling_min(data: np.ndarray, window: int) -> np.ndarray:
    """滚动最小值"""
    result = np.full_like(data, np.nan, dtype=np.float64)
    if len(data) < window:
        return result

    from scipy.ndimage import minimum_filter1d

    result[window - 1 :] = minimum_filter1d(data, size=window)[window - 1 :]
    return result


def exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均"""
    alpha = 2 / (span + 1)
    result = np.zeros_like(data, dtype=np.float64)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result
