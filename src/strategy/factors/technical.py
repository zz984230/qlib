"""技术因子"""

import numpy as np
import pandas as pd

from src.strategy.factors.base import (
    BaseFactor,
    exponential_moving_average,
    rolling_mean,
    rolling_std,
)


class MAFactor(BaseFactor):
    """
    移动平均因子

    价格与移动平均的偏离度
    """

    def __init__(self, window: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.params["window"] = window

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        result = np.full(len(close), np.nan, dtype=np.float64)

        if len(close) < self.window:
            return result

        ma = rolling_mean(close, self.window)
        result[self.window - 1 :] = (close[self.window - 1 :] - ma[self.window - 1 :]) / ma[self.window - 1 :]

        return result

    def to_qlib_expression(self) -> str:
        return f"($close - Mean($close, {self.window})) / Mean($close, {self.window})"


class RSIFactor(BaseFactor):
    """
    相对强弱指数因子 (Relative Strength Index)

    衡量价格变动的速度和变化
    """

    def __init__(self, window: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.params["window"] = window

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        result = np.full(len(close), np.nan, dtype=np.float64)

        if len(close) < self.window + 1:
            return result

        # 计算价格变化
        delta = np.diff(close)

        # 上涨和下跌
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # 使用 EMA 计算平均涨跌
        avg_gain = exponential_moving_average(gains, self.window)
        avg_loss = exponential_moving_average(losses, self.window)

        # 计算 RSI
        for i in range(self.window, len(delta)):
            if avg_loss[i] == 0:
                result[i + 1] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                result[i + 1] = 100 - (100 / (1 + rs))

        return result

    def to_qlib_expression(self) -> str:
        # Qlib 内置 RSI
        return f"RSI($close, {self.window})"


class BollingerBandsFactor(BaseFactor):
    """
    布林带因子

    价格相对于布林带的位置
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.num_std = num_std
        self.params.update({"window": window, "num_std": num_std})

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        result = np.full(len(close), np.nan, dtype=np.float64)

        if len(close) < self.window:
            return result

        # 计算中轨 (MA)
        middle = rolling_mean(close, self.window)

        # 计算标准差
        std = rolling_std(close, self.window)

        # 计算上下轨
        upper = middle + self.num_std * std
        lower = middle - self.num_std * std

        # 计算 %B (价格在布林带中的位置)
        for i in range(self.window - 1, len(close)):
            if upper[i] != lower[i]:
                result[i] = (close[i] - lower[i]) / (upper[i] - lower[i])
            else:
                result[i] = 0.5

        return result


class StochasticFactor(BaseFactor):
    """
    随机指标因子 (Stochastic Oscillator)

    %K 线：当前价格相对于最近 N 天价格范围的位置
    """

    def __init__(self, k_window: int = 14, d_window: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.k_window = k_window
        self.d_window = d_window
        self.params.update({"k_window": k_window, "d_window": d_window})

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        result = np.full(len(close), np.nan, dtype=np.float64)

        if len(close) < self.k_window:
            return result

        # 计算 %K
        for i in range(self.k_window - 1, len(close)):
            highest = np.max(high[i - self.k_window + 1 : i + 1])
            lowest = np.min(low[i - self.k_window + 1 : i + 1])

            if highest != lowest:
                result[i] = (close[i] - lowest) / (highest - lowest) * 100
            else:
                result[i] = 50

        return result


class WilliamsRFactor(BaseFactor):
    """
    威廉指标因子 (Williams %R)

    衡量超买超卖水平
    """

    def __init__(self, window: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.params["window"] = window

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        result = np.full(len(close), np.nan, dtype=np.float64)

        if len(close) < self.window:
            return result

        for i in range(self.window - 1, len(close)):
            highest = np.max(high[i - self.window + 1 : i + 1])
            lowest = np.min(low[i - self.window + 1 : i + 1])

            if highest != lowest:
                result[i] = (highest - close[i]) / (highest - lowest) * -100
            else:
                result[i] = -50

        return result
