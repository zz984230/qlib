"""动量因子"""

import numpy as np
import pandas as pd

from src.strategy.factors.base import BaseFactor, exponential_moving_average


class MomentumFactor(BaseFactor):
    """
    动量因子

    计算过去 N 天的收益率
    """

    def __init__(self, window: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.params["window"] = window

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        result = np.full(len(close), np.nan, dtype=np.float64)

        if len(close) > self.window:
            result[self.window :] = (
                close[self.window :] - close[: -self.window]
            ) / close[: -self.window]

        return result

    def to_qlib_expression(self) -> str:
        return f"Ref($close, -{self.window}) / $close - 1"


class ROCFactor(BaseFactor):
    """
    变化率因子 (Rate of Change)

    计算价格相对于 N 天前的变化百分比
    """

    def __init__(self, window: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.params["window"] = window

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        result = np.full(len(close), np.nan, dtype=np.float64)

        if len(close) > self.window:
            result[self.window :] = (
                (close[self.window :] - close[: -self.window])
                / close[: -self.window]
                * 100
            )

        return result

    def to_qlib_expression(self) -> str:
        return f"($close / Ref($close, {self.window}) - 1) * 100"


class MACDFactor(BaseFactor):
    """
    MACD 因子 (Moving Average Convergence Divergence)

    计算 MACD 线 (快线 - 慢线)
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.params.update({"fast": fast, "slow": slow, "signal": signal})

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values

        # 计算 EMA
        ema_fast = exponential_moving_average(close, self.fast)
        ema_slow = exponential_moving_average(close, self.slow)

        # MACD 线
        macd_line = ema_fast - ema_slow

        # 信号线
        signal_line = exponential_moving_average(macd_line, self.signal)

        # MACD 柱
        histogram = macd_line - signal_line

        return histogram

    def to_qlib_expression(self) -> str:
        # Qlib 中需要使用内置的 MACD 函数
        return f"EMA($close, {self.fast}) - EMA($close, {self.slow})"


class RSIFFactor(BaseFactor):
    """
    相对强弱因子 (基于 RSI 思想)

    计算上涨幅度占比
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

        # 滚动平均
        for i in range(self.window, len(delta)):
            avg_gain = np.mean(gains[i - self.window + 1 : i + 1])
            avg_loss = np.mean(losses[i - self.window + 1 : i + 1])

            if avg_loss == 0:
                result[i + 1] = 100
            else:
                rs = avg_gain / avg_loss
                result[i + 1] = 100 - (100 / (1 + rs))

        return result
