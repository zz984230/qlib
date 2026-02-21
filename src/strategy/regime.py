"""
市场状态识别模块

识别当前市场的状态（趋势/震荡/高波动/平稳），用于策略选择
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketState(Enum):
    """市场状态"""

    TRENDING = "trending"  # 趋势市场
    OSCILLATING = "oscillating"  # 震荡市场
    VOLATILE = "volatile"  # 高波动市场
    QUIET = "quiet"  # 平稳市场


@dataclass
class MarketRegime:
    """市场状态信息"""

    state: MarketState
    confidence: float  # 状态判断置信度 0-1
    adx: float  # ADX 指标值
    atr_ratio: float  # ATR/价格比率
    hurst: float  # Hurst 指数
    volatility: float  # 年化波动率
    trend_direction: int  # 趋势方向 1=上涨 -1=下跌 0=无


class RegimeDetector:
    """
    市场状态识别器

    使用 ADX、ATR、Hurst 指数等指标识别市场状态
    """

    def __init__(
        self,
        adx_threshold: float = 25.0,
        adx_oscillating_threshold: float = 20.0,
        atr_high_threshold: float = 0.03,
        hurst_mean_reversion_threshold: float = 0.45,
        lookback: int = 60,
    ):
        """
        初始化市场状态识别器

        Args:
            adx_threshold: ADX 趋势阈值
            adx_oscillating_threshold: ADX 震荡阈值
            atr_high_threshold: ATR 高波动阈值
            hurst_mean_reversion_threshold: Hurst 均值回归阈值
            lookback: 回看期
        """
        self.adx_threshold = adx_threshold
        self.adx_oscillating_threshold = adx_oscillating_threshold
        self.atr_high_threshold = atr_high_threshold
        self.hurst_mean_reversion_threshold = hurst_mean_reversion_threshold
        self.lookback = lookback

    def detect(self, data: pd.DataFrame) -> MarketRegime:
        """
        识别市场状态

        Args:
            data: 包含 OHLCV 数据的 DataFrame

        Returns:
            MarketRegime 对象
        """
        if len(data) < self.lookback:
            return MarketRegime(
                state=MarketState.QUIET,
                confidence=0.0,
                adx=0.0,
                atr_ratio=0.0,
                hurst=0.5,
                volatility=0.0,
                trend_direction=0,
            )

        # 使用最近的数据
        recent_data = data.tail(self.lookback)

        # 计算各指标
        adx = self._calculate_adx(recent_data)
        atr_ratio = self._calculate_atr_ratio(recent_data)
        hurst = self._calculate_hurst(recent_data)
        volatility = self._calculate_volatility(recent_data)
        trend_direction = self._calculate_trend_direction(recent_data)

        # 状态判断
        state, confidence = self._determine_state(
            adx, atr_ratio, hurst, volatility
        )

        return MarketRegime(
            state=state,
            confidence=confidence,
            adx=adx,
            atr_ratio=atr_ratio,
            hurst=hurst,
            volatility=volatility,
            trend_direction=trend_direction,
        )

    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """
        计算 ADX (Average Directional Index)

        ADX > 25 表示强趋势
        ADX < 20 表示无趋势
        """
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        if len(close) < 14:
            return 0.0

        # 计算 +DM 和 -DM
        plus_dm = np.zeros(len(close))
        minus_dm = np.zeros(len(close))

        for i in range(1, len(close)):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # 计算 TR (True Range)
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # 平滑
        period = 14
        atr = self._smooth(tr, period)
        plus_dm_smooth = self._smooth(plus_dm, period)
        minus_dm_smooth = self._smooth(minus_dm, period)

        # 计算 +DI 和 -DI
        plus_di = np.zeros(len(close))
        minus_di = np.zeros(len(close))

        for i in range(len(close)):
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]

        # 计算 DX
        dx = np.zeros(len(close))
        for i in range(len(close)):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        # ADX 是 DX 的平滑
        adx = self._smooth(dx, period)

        return adx[-1] if len(adx) > 0 else 0.0

    def _calculate_atr_ratio(self, data: pd.DataFrame) -> float:
        """
        计算 ATR / Price 比率

        用于判断波动程度
        """
        high = data["high"].values
        low = data["low"].values
        close = data["close"].values

        if len(close) < 14:
            return 0.0

        # 计算 TR
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # ATR
        atr = np.mean(tr[-14:])
        avg_price = np.mean(close[-14:])

        return atr / avg_price if avg_price > 0 else 0.0

    def _calculate_hurst(self, data: pd.DataFrame) -> float:
        """
        计算 Hurst 指数

        H > 0.5: 趋势性
        H < 0.5: 均值回归
        H = 0.5: 随机游走
        """
        close = data["close"].values
        if len(close) < 20:
            return 0.5

        # 对数收益
        returns = np.diff(np.log(close))
        if len(returns) < 10:
            return 0.5

        # R/S 分析
        n = len(returns)
        max_k = min(n // 2, 20)

        rs_values = []
        k_values = []

        for k in range(5, max_k + 1):
            # 分割成 k 个子序列
            m = n // k
            if m < 2:
                continue

            rs_list = []
            for i in range(k):
                sub_seq = returns[i * m : (i + 1) * m]
                mean = np.mean(sub_seq)
                cum_dev = np.cumsum(sub_seq - mean)
                r = np.max(cum_dev) - np.min(cum_dev)
                s = np.std(sub_seq)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append(np.mean(rs_list))
                k_values.append(k)

        if len(rs_values) < 3:
            return 0.5

        # 线性回归求斜率
        log_k = np.log(k_values)
        log_rs = np.log(rs_values)

        slope, _ = np.polyfit(log_k, log_rs, 1)
        return float(slope)

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """计算年化波动率"""
        close = data["close"].values
        if len(close) < 2:
            return 0.0

        returns = np.diff(np.log(close))
        return np.std(returns) * np.sqrt(252)

    def _calculate_trend_direction(self, data: pd.DataFrame) -> int:
        """计算趋势方向"""
        close = data["close"].values
        if len(close) < 20:
            return 0

        # 简单使用短期和长期均线比较
        short_ma = np.mean(close[-5:])
        long_ma = np.mean(close[-20:])

        if short_ma > long_ma * 1.02:
            return 1
        elif short_ma < long_ma * 0.98:
            return -1
        return 0

    def _smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder 平滑"""
        result = np.zeros(len(data))
        if len(data) < period:
            return result

        # 初始值
        result[period - 1] = np.mean(data[:period])

        # 递归平滑
        for i in range(period, len(data)):
            result[i] = (result[i - 1] * (period - 1) + data[i]) / period

        return result

    def _determine_state(
        self,
        adx: float,
        atr_ratio: float,
        hurst: float,
        volatility: float,
    ) -> tuple[MarketState, float]:
        """
        综合判断市场状态

        Returns:
            (状态, 置信度)
        """
        # 高波动市场判断
        if atr_ratio > self.atr_high_threshold:
            return MarketState.VOLATILE, min(atr_ratio / self.atr_high_threshold, 1.0)

        # 趋势市场判断
        if adx > self.adx_threshold:
            confidence = (adx - self.adx_threshold) / self.adx_threshold
            return MarketState.TRENDING, min(confidence, 1.0)

        # 震荡市场判断
        if adx < self.adx_oscillating_threshold and hurst < self.hurst_mean_reversion_threshold:
            confidence = (self.adx_oscillating_threshold - adx) / self.adx_oscillating_threshold
            confidence *= (0.5 - hurst) / 0.5
            return MarketState.OSCILLATING, min(confidence, 1.0)

        # 默认平稳市场
        return MarketState.QUIET, 0.5


def get_recommended_strategy(regime: MarketRegime) -> str:
    """
    根据市场状态推荐策略

    Args:
        regime: 市场状态

    Returns:
        推荐的策略名称
    """
    strategy_map = {
        MarketState.TRENDING: "momentum",  # 趋势市场用动量策略
        MarketState.OSCILLATING: "mean_reversion",  # 震荡市场用均值回归
        MarketState.VOLATILE: "bollinger",  # 高波动用布林带
        MarketState.QUIET: "multi_factor",  # 平稳市场用多因子
    }

    return strategy_map.get(regime.state, "multi_factor")
