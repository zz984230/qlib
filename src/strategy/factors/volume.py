"""成交量因子"""

import numpy as np
import pandas as pd

from src.strategy.factors.base import BaseFactor, rolling_mean


class VolumeRatioFactor(BaseFactor):
    """
    成交量比率因子

    当前成交量与均量的比值
    """

    def __init__(self, window: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.params["window"] = window

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        volume = data["volume"].values
        result = np.full(len(volume), np.nan, dtype=np.float64)

        if len(volume) < self.window:
            return result

        # 计算滚动均量
        vol_mean = rolling_mean(volume, self.window)

        # 计算比率
        result[self.window - 1 :] = volume[self.window - 1 :] / vol_mean[self.window - 1 :]

        return result

    def to_qlib_expression(self) -> str:
        return f"$volume / Mean($volume, {self.window})"


class OBVFactor(BaseFactor):
    """
    能量潮因子 (On-Balance Volume)

    累积成交量指标
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        volume = data["volume"].values

        result = np.zeros(len(close), dtype=np.float64)

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:
                result[i] = result[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                result[i] = result[i - 1] - volume[i]
            else:
                result[i] = result[i - 1]

        return result


class VolumeMomentumFactor(BaseFactor):
    """
    成交量动量因子

    成交量变化率
    """

    def __init__(self, window: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.params["window"] = window

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        volume = data["volume"].values
        result = np.full(len(volume), np.nan, dtype=np.float64)

        if len(volume) < self.window:
            return result

        result[self.window :] = (
            volume[self.window :] - volume[: -self.window]
        ) / volume[: -self.window]

        return result


class VolumePriceTrendFactor(BaseFactor):
    """
    量价趋势因子

    成交量与价格变化的综合指标
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        volume = data["volume"].values

        result = np.zeros(len(close), dtype=np.float64)

        for i in range(1, len(close)):
            price_change = (close[i] - close[i - 1]) / close[i - 1]
            result[i] = result[i - 1] + volume[i] * price_change

        return result
