"""波动率因子"""

import numpy as np
import pandas as pd

from src.strategy.factors.base import BaseFactor, rolling_mean, rolling_std


class VolatilityFactor(BaseFactor):
    """
    波动率因子

    计算收益率的标准差
    """

    def __init__(self, window: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.params["window"] = window

    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        result = np.full(len(close), np.nan, dtype=np.float64)

        if len(close) < self.window + 1:
            return result

        # 计算日收益率
        returns = np.diff(close) / close[:-1]

        # 滚动标准差
        for i in range(self.window, len(returns)):
            result[i + 1] = np.std(returns[i - self.window + 1 : i + 1], ddof=1)

        return result

    def to_qlib_expression(self) -> str:
        return f"Std($close / Ref($close, 1) - 1, {self.window})"


class ATRFactor(BaseFactor):
    """
    平均真实波幅因子 (Average True Range)

    衡量价格波动性
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

        if len(close) < self.window + 1:
            return result

        # 计算真实波幅
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]

        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # 计算 ATR (滚动平均)
        for i in range(self.window - 1, len(tr)):
            result[i] = np.mean(tr[i - self.window + 1 : i + 1])

        return result


class HighLowRatioFactor(BaseFactor):
    """
    高低比因子

    最高价与最低价的比率
    """

    def __init__(self, window: int = 20, **kwargs):
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
            window_high = np.max(high[i - self.window + 1 : i + 1])
            window_low = np.min(low[i - self.window + 1 : i + 1])
            if window_low > 0:
                result[i] = (window_high - window_low) / window_low

        return result
