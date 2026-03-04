"""市场环境识别模块

用于识别当前市场是趋势市还是震荡市，帮助策略选择合适的交易时机。
"""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketEnvironment:
    """市场环境识别器

    通过分析价格序列特征，判断市场处于：
    - 强趋势上涨
    - 强趋势下跌
    - 震荡市
    """

    def __init__(self, lookback: int = 50):
        """初始化市场环境识别器

        Args:
            lookback: 回看周期（天数）
        """
        self.lookback = lookback

    def analyze(
        self,
        prices: np.ndarray | pd.Series,
        adx: np.ndarray | pd.Series | None = None
    ) -> dict:
        """分析市场环境

        Args:
            prices: 价格序列
            adx: ADX序列（可选）

        Returns:
            环境分析结果字典
        """
        if len(prices) < self.lookback:
            return {
                "environment": "insufficient_data",
                "is_trending": False,
                "trend_direction": 0,
                "strength": 0,
            }

        # 使用最近 lookback 天的数据
        recent_prices = prices[-self.lookback:]

        # 计算各种指标
        trend_direction = self._calculate_trend_direction(recent_prices)
        trend_strength = self._calculate_trend_strength(recent_prices)
        volatility = self._calculate_volatility(recent_prices)

        # 综合判断市场环境
        environment = self._classify_environment(
            trend_direction, trend_strength, volatility, adx
        )

        return {
            "environment": environment,
            "is_trending": environment in ["strong_uptrend", "uptrend", "downtrend", "strong_downtrend"],
            "trend_direction": trend_direction,
            "strength": trend_strength,
            "volatility": volatility,
        }

    def _calculate_trend_direction(self, prices: np.ndarray) -> float:
        """计算趋势方向

        Returns:
            趋势方向：-1(强跌) 到 +1(强涨)
        """
        # 使用线性回归斜率
        x = np.arange(len(prices))
        y = prices.values if isinstance(prices, pd.Series) else prices

        # 简单线性回归
        slope = np.cov(x, y, bias=True)[0, 1] / np.var(x)

        # 归一化到 [-1, 1]
        # 使用价格变化的标准差来归一化
        price_std = np.std(y)
        if price_std > 0:
            normalized_slope = np.clip(slope * len(x) / price_std, -1, 1)
        else:
            normalized_slope = 0

        return normalized_slope

    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """计算趋势强度

        Returns:
            趋势强度 [0, 1]，越高表示趋势越明显
        """
        # 方法1：价格变化的一致性
        price_changes = np.diff(prices)
        if len(price_changes) == 0:
            return 0

        # 计算同向变化的比例
        up_moves = np.sum(price_changes > 0)
        down_moves = np.sum(price_changes < 0)

        total_moves = len(price_changes)
        if total_moves == 0:
            return 0

        # 趋势强度 = 最大方向占比
        directional_ratio = max(up_moves, down_moves) / total_moves

        # 方法2：价格是否持续创新高/新低
        highs = np.maximum.accumulate(prices)
        lows = np.minimum.accumulate(prices)

        at_high = np.sum(prices >= highs * 0.99) / len(prices)
        at_low = np.sum(prices <= lows * 1.01) / len(prices)

        trend_indicator = max(at_high, at_low)

        # 综合两种方法
        strength = (directional_ratio + trend_indicator) / 2

        return float(strength)

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """计算波动率

        Returns:
            波动率（标准差/均值）
        """
        returns = np.diff(prices) / prices[:-1]
        if len(returns) == 0:
            return 0
        return float(np.std(returns))

    def _classify_environment(
        self,
        trend_direction: float,
        trend_strength: float,
        volatility: float,
        adx: np.ndarray | pd.Series | None = None
    ) -> str:
        """分类市场环境

        Args:
            trend_direction: 趋势方向 [-1, 1]
            trend_strength: 趋势强度 [0, 1]
            volatility: 波动率
            adx: ADX序列（可选）

        Returns:
            环境类型
        """
        # 如果有ADX，使用ADX辅助判断
        adx_value = None
        if adx is not None and len(adx) > 0:
            adx_value = float(adx[-1])

        # 强趋势判断条件（放宽阈值）
        strong_trend = (
            trend_strength > 0.50 and  # 趋势强度高（从0.65降低到0.50）
            abs(trend_direction) > 0.20 and  # 方向明显（从0.3降低到0.20）
            (adx_value is None or adx_value > 20)  # ADX确认强趋势（从30降低到20）
        )

        # 弱趋势/震荡判断条件（收紧阈值）
        ranging = (
            trend_strength < 0.40 or  # 趋势强度低（从0.55降低到0.40）
            abs(trend_direction) < 0.10 or  # 方向不明显（从0.15降低到0.10）
            (adx_value is not None and adx_value < 15)  # ADX显示无趋势（从20降低到15）
        )

        # 分类
        if strong_trend:
            if trend_direction > 0.4:
                return "strong_uptrend"
            elif trend_direction < -0.4:
                return "strong_downtrend"
            elif trend_direction > 0:
                return "uptrend"
            else:
                return "downtrend"
        elif ranging:
            return "ranging"
        else:
            # 中等趋势
            if trend_direction > 0:
                return "uptrend"
            else:
                return "weak_downtrend"

    def should_trade(
        self,
        environment: str,
        prefer_trend: bool = True
    ) -> bool:
        """判断是否应该交易

        Args:
            environment: 市场环境类型
            prefer_trend: 是否偏好趋势市（海龟策略应该设为True）

        Returns:
            是否应该交易
        """
        if prefer_trend:
            # 海龟策略只在趋势市交易
            trending_environments = [
                "uptrend", "strong_uptrend",
                "downtrend", "strong_downtrend"
            ]
            return environment in trending_environments
        else:
            # 如果不偏好趋势，在非剧烈下跌的市场都可以交易
            no_trade_environments = ["strong_downtrend", "insufficient_data"]
            return environment not in no_trade_environments


def get_market_signal(
    prices: np.ndarray | pd.Series,
    adx: np.ndarray | pd.Series | None = None,
    lookback: int = 50
) -> Tuple[bool, str]:
    """快捷函数：获取市场交易信号

    Args:
        prices: 价格序列
        adx: ADX序列（可选）
        lookback: 回看周期

    Returns:
        (是否交易, 环境类型)
    """
    analyzer = MarketEnvironment(lookback=lookback)
    analysis = analyzer.analyze(prices, adx)

    should = analyzer.should_trade(analysis["environment"], prefer_trend=True)

    return should, analysis["environment"]
