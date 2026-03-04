"""市场状态分类器 - 识别趋势/震荡/异常波动

使用ADX、ATR、波动率等指标识别当前市场状态。
"""

import logging
from typing import Literal
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 市场状态类型
MarketState = Literal["strong_trend", "weak_trend", "ranging", "volatile"]


class MarketClassifier:
    """市场状态分类器

    基于技术指标识别四种市场状态：
    - strong_trend: 强趋势（ADX > 30）
    - weak_trend: 弱趋势（20 < ADX <= 30）
    - ranging: 震荡（ADX <= 20）
    - volatile: 异常波动（波动率突增 > 2倍）
    """

    # 分类参数
    ADX_STRONG_THRESHOLD = 30.0
    ADX_WEAK_THRESHOLD = 20.0
    VOLATILITY_SPIKE_THRESHOLD = 2.0  # 波动率突增倍数

    def __init__(self, lookback_period: int = 60):
        """初始化分类器

        Args:
            lookback_period: 回看周期（天），默认60天
        """
        self.lookback_period = lookback_period

    def classify(self, data: pd.DataFrame) -> MarketState:
        """分类当前市场状态

        Args:
            data: 价格数据，必须包含 high, low, close 列

        Returns:
            市场状态字符串
        """
        if len(data) < 20:
            logger.warning("数据不足20天，默认返回ranging状态")
            return "ranging"

        # 计算技术指标
        adx = self._calculate_adx(data)
        volatility = self._calculate_volatility(data)

        if adx is None or volatility is None:
            return "ranging"

        # 获取最近的指标值
        recent_adx = adx.iloc[-1]
        recent_vol = volatility.iloc[-1] if len(volatility) > 0 else 0

        # 计算波动率突增
        vol_spike = self._calculate_volatility_spike(volatility)

        # 优先判断异常波动（风险控制优先）
        if vol_spike > self.VOLATILITY_SPIKE_THRESHOLD:
            return "volatile"

        # 趋势强度判断
        if recent_adx > self.ADX_STRONG_THRESHOLD:
            return "strong_trend"
        elif recent_adx > self.ADX_WEAK_THRESHOLD:
            return "weak_trend"
        else:
            return "ranging"

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series | None:
        """计算平均趋向指标（Average Directional Index）

        ADX 用于衡量趋势强度，范围 0-100：
        - 0-20: 弱趋势或震荡
        - 20-30: 中等趋势
        - 30+: 强趋势

        Args:
            data: 价格数据
            period: 计算周期，默认14

        Returns:
            ADX序列
        """
        if len(data) < period * 2:
            return None

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # 计算 True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # 计算 +DM 和 -DM
        plus_dm = high[1:] - high[:-1]
        minus_dm = low[:-1] - low[1:]

        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)

        # 平滑处理
        atr = pd.Series(tr).rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr

        # 计算 DX 和 ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series | None:
        """计算平均真实波幅（Average True Range）

        ATR 衡量市场波动性，值越大表示波动越剧烈。

        Args:
            data: 价格数据
            period: 计算周期，默认14

        Returns:
            ATR序列
        """
        if len(data) < 2:
            return None

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # 计算 True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # ATR = TR 的移动平均
        atr = pd.Series(tr).rolling(window=period).mean()

        return atr

    def _calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series | None:
        """计算相对波动率

        使用 ATR / close 作为相对波动率指标。

        Args:
            data: 价格数据
            period: 计算周期，默认20

        Returns:
            相对波动率序列
        """
        atr = self._calculate_atr(data, period)
        if atr is None:
            return None

        # 对齐索引
        close_aligned = data['close'].iloc[1:]

        # 相对波动率 = ATR / 收盘价
        volatility = atr / close_aligned.values

        return volatility

    def _calculate_volatility_spike(
        self,
        volatility: pd.Series,
        lookback: int = 5
    ) -> float:
        """计算波动率突增倍数

        比较当前波动率与最近N天的平均波动率。

        Args:
            volatility: 波动率序列
            lookback: 回看天数，默认5天

        Returns:
            波动率突增倍数（>1表示突增）
        """
        if len(volatility) < lookback + 1:
            return 1.0

        current_vol = volatility.iloc[-1]
        avg_vol = volatility.iloc[-lookback-1:-1].mean()

        if avg_vol == 0:
            return 1.0

        return current_vol / avg_vol

    def get_market_state_history(
        self,
        data: pd.DataFrame,
        window: int = 5
    ) -> list[dict]:
        """获取市场状态历史（带平滑）

        返回最近N天的市场状态，用于避免频繁切换。

        Args:
            data: 价格数据
            window: 平滑窗口大小

        Returns:
            市场状态历史列表
        """
        history = []

        # 滑动窗口分类
        for i in range(max(0, len(data) - window), len(data)):
            subset = data.iloc[:i+1]
            if len(subset) >= 20:
                state = self.classify(subset)
                history.append({
                    "date": data.index[i],
                    "state": state,
                })

        return history


def classify_market_state(data: pd.DataFrame) -> MarketState:
    """便捷函数：分类当前市场状态

    Args:
        data: 价格数据

    Returns:
        市场状态字符串
    """
    classifier = MarketClassifier()
    return classifier.classify(data)
