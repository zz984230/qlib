"""海龟策略因子信号生成器

使用因子组合生成入场和出场信号，与遗传算法的 Individual 基因编码对接。
"""

import logging
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from src.optimizer.individual import Individual, FACTOR_POOL

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """简单移动平均"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.empty(len(data))
    result[:period-1] = np.nan
    for i in range(period-1, len(data)):
        result[i] = np.mean(data[i-period+1:i+1])
    return result


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """指数移动平均"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    alpha = 2.0 / (period + 1)
    result = np.empty(len(data))
    result[:period-1] = np.nan
    result[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def _std(data: np.ndarray, period: int) -> np.ndarray:
    """标准差"""
    if len(data) < period:
        return np.full(len(data), np.nan)
    result = np.empty(len(data))
    result[:period-1] = np.nan
    for i in range(period-1, len(data)):
        result[i] = np.std(data[i-period+1:i+1])
    return result


def _rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI 相对强弱指标"""
    if len(data) < period + 1:
        return np.full(len(data), 50.0)
    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.empty(len(data))
    avg_loss = np.empty(len(data))
    avg_gain[:period] = np.nan
    avg_loss[:period] = np.nan

    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    for i in range(period + 1, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period

    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 1.0), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50.0
    return rsi


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR 平均真实波幅"""
    if len(close) < 2:
        return np.zeros(len(close))

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    return _ema(tr, period)


def _macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD 指标"""
    ema_fast = _ema(data, fast)
    ema_slow = _ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray,
           k_period: int = 14, d_period: int = 3) -> tuple:
    """随机指标 KDJ"""
    if len(close) < k_period:
        return np.full(len(close), 50.0), np.full(len(close), 50.0)

    lowest_low = pd.Series(low).rolling(k_period).min().values
    highest_high = pd.Series(high).rolling(k_period).max().values

    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k = np.nan_to_num(k, nan=50.0)

    d = _sma(k, d_period)
    d = np.nan_to_num(d, nan=50.0)

    return k, d


def _cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """CCI 商品通道指标"""
    if len(close) < period:
        return np.zeros(len(close))

    tp = (high + low + close) / 3
    ma_tp = _sma(tp, period)
    md = _std(tp, period) * 2  # Mean Deviation

    cci = (tp - ma_tp) / (md + 1e-10)
    cci = np.nan_to_num(cci, nan=0.0)
    return cci


def _obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """OBV 能量潮"""
    if len(close) < 2:
        return np.zeros(len(close))

    obv = np.empty(len(close))
    obv[0] = volume[0]

    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    return obv


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX 平均趋向指标"""
    if len(close) < period * 2:
        return np.zeros(len(close))

    # plus_dm 和 minus_dm 比 high 少一个元素（因为是 diff）
    plus_dm = np.where(high[1:] > high[:-1], high[1:] - high[:-1], 0)
    minus_dm = np.where(low[1:] < low[:-1], low[:-1] - low[1:], 0)

    atr = _atr(high, low, close, period)
    # atr 从索引 1 开始与 plus_dm/minus_dm 对齐
    atr_aligned = atr[1:]

    plus_di = 100 * _ema(plus_dm, period) / (atr_aligned + 1e-10)
    minus_di = 100 * _ema(minus_dm, period) / (atr_aligned + 1e-10)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

    adx = _ema(dx, period)
    adx = np.nan_to_num(adx, nan=0.0)

    # 返回与输入 close 相同长度的数组，前面填充 0
    result = np.zeros(len(close))
    result[-len(adx):] = adx

    return result


class TurtleSignalGenerator:
    """海龟策略信号生成器

    基于因子组合计算信号强度，与遗传算法的 Individual 基因编码对接。

    信号强度范围 [0, 1]：
    - 入场信号：越高表示越强烈的买入信号
    - 出场信号：越高表示越强烈的卖出信号
    """

    def __init__(self, individual: Individual):
        """初始化信号生成器

        Args:
            individual: 包含因子权重和阈值的个体
        """
        self.individual = individual
        self.factor_weights = individual.factor_weights
        self.signal_threshold = individual.signal_threshold
        self.exit_threshold = individual.exit_threshold

    def generate_entry_signal(self, data: pd.DataFrame) -> float:
        """生成入场信号强度

        综合多个因子计算加权信号强度。

        Args:
            data: 包含 OHLCV 数据的 DataFrame

        Returns:
            入场信号强度 [0, 1]，超过 signal_threshold 建议入场
        """
        if len(data) < 20:
            return 0.0

        signal_strength = 0.0
        total_weight = 0.0

        for factor_name, weight in self.factor_weights.items():
            factor_value = self._calculate_factor_value(factor_name, data)

            if factor_value is None or np.isnan(factor_value):
                continue

            # 归一化因子值到 [0, 1]
            normalized_value = self._normalize_factor(factor_name, factor_value)
            signal_strength += normalized_value * weight
            total_weight += weight

        # 归一化信号强度
        if total_weight > 0:
            signal_strength /= total_weight

        return float(np.clip(signal_strength, 0, 1))

    def generate_exit_signal(self, data: pd.DataFrame) -> float:
        """生成出场信号强度

        综合多个因子计算加权出场信号。

        Args:
            data: 包含 OHLCV 数据的 DataFrame

        Returns:
            出场信号强度 [0, 1]，超过 exit_threshold 建议出场
        """
        if len(data) < 20:
            return 0.0

        # 出场信号通常与入场信号相反
        exit_strength = 0.0
        total_weight = 0.0

        for factor_name, weight in self.factor_weights.items():
            factor_value = self._calculate_factor_value(factor_name, data)

            if factor_value is None or np.isnan(factor_value):
                continue

            # 出场信号：取反向因子值
            normalized_value = 1 - self._normalize_factor(factor_name, factor_value)
            exit_strength += normalized_value * weight
            total_weight += weight

        # 归一化出场信号强度
        if total_weight > 0:
            exit_strength /= total_weight

        return float(np.clip(exit_strength, 0, 1))

    def should_enter(self, data: pd.DataFrame) -> bool:
        """判断是否应该入场

        Args:
            data: 市场数据

        Returns:
            是否入场
        """
        entry_signal = self.generate_entry_signal(data)
        return entry_signal >= self.signal_threshold

    def should_exit(self, data: pd.DataFrame) -> bool:
        """判断是否应该出场

        Args:
            data: 市场数据

        Returns:
            是否出场
        """
        exit_signal = self.generate_exit_signal(data)
        return exit_signal >= self.exit_threshold

    def _calculate_factor_value(self, factor_name: str, data: pd.DataFrame) -> float | None:
        """计算单个因子的值

        Args:
            factor_name: 因子名称
            data: 市场数据

        Returns:
            因子值（最新值）
        """
        close = data["close"].values
        high = data["high"].values
        low = data["low"].values
        volume = data["volume"].values if "volume" in data.columns else None

        try:
            if factor_name == "ma_ratio":
                # MA5 / MA20 - 1
                ma5 = _sma(close, 5)
                ma20 = _sma(close, 20)
                if len(ma20) > 0 and not np.isnan(ma20[-1]):
                    return float(ma5[-1] / ma20[-1] - 1)
                return 0.0

            elif factor_name == "ma_cross":
                # MA5 > MA20
                ma5 = _sma(close, 5)
                ma20 = _sma(close, 20)
                if len(ma20) > 0 and not np.isnan(ma20[-1]):
                    return float(1.0 if ma5[-1] > ma20[-1] else 0.0)
                return 0.0

            elif factor_name == "momentum":
                # (close - close_n5) / close_n5
                if len(close) < 6:
                    return 0.0
                return float((close[-1] - close[-6]) / close[-6])

            elif factor_name == "price_momentum":
                # close / close_n20 - 1
                if len(close) < 21:
                    return 0.0
                return float(close[-1] / close[-21] - 1)

            elif factor_name == "rsi":
                # RSI / 100
                rsi = _rsi(close, 14)
                return float(rsi[-1] / 100) if len(rsi) > 0 else 0.5

            elif factor_name == "macd":
                # MACD / signal
                macd_line, signal_line, _ = _macd(close)
                if len(signal_line) > 0 and not np.isnan(signal_line[-1]) and signal_line[-1] != 0:
                    return float(macd_line[-1] / signal_line[-1])
                return 0.0

            elif factor_name == "kdj":
                # (K - D) / 100
                k, d = _stoch(high, low, close)
                if len(k) > 0 and not np.isnan(k[-1]) and not np.isnan(d[-1]):
                    return float((k[-1] - d[-1]) / 100)
                return 0.0

            elif factor_name == "volatility":
                # STD(close, 20) / close
                std = _std(close, 20)
                if len(std) > 0 and not np.isnan(std[-1]):
                    return float(std[-1] / close[-1])
                return 0.0

            elif factor_name == "atr_ratio":
                # ATR / close
                atr_val = _atr(high, low, close, 14)
                if len(atr_val) > 0 and not np.isnan(atr_val[-1]):
                    return float(atr_val[-1] / close[-1])
                return 0.0

            elif factor_name == "volume_ratio":
                # volume / MA(volume, 20)
                if volume is None or len(volume) < 21:
                    return 1.0
                ma_volume = _sma(volume, 20)
                if len(ma_volume) > 0 and not np.isnan(ma_volume[-1]) and ma_volume[-1] > 0:
                    return float(volume[-1] / ma_volume[-1])
                return 1.0

            elif factor_name == "volume_price":
                # 量价配合：成交量涨幅 * 价格涨幅
                if volume is None or len(volume) < 2:
                    return 0.0
                volume_change = (volume[-1] - volume[-2]) / (volume[-2] + 1e-10)
                price_change = (close[-1] - close[-2]) / (close[-2] + 1e-10)
                return float(volume_change * price_change)

            elif factor_name == "adx":
                # ADX / 100
                adx_val = _adx(high, low, close, 14)
                return float(adx_val[-1] / 100) if len(adx_val) > 0 else 0.0

            elif factor_name == "cci":
                # CCI / 200
                cci_val = _cci(high, low, close, 14)
                return float(cci_val[-1] / 200) if len(cci_val) > 0 else 0.0

            elif factor_name == "obv":
                # OBV / MA(OBV, 20) - 1
                if volume is None:
                    return 0.0
                obv_val = _obv(close, volume)
                if len(obv_val) < 21:
                    return 0.0
                ma_obv = _sma(obv_val, 20)
                if len(ma_obv) > 0 and not np.isnan(ma_obv[-1]) and ma_obv[-1] > 0:
                    return float(obv_val[-1] / ma_obv[-1] - 1)
                return 0.0

            elif factor_name == "money_flow":
                # 资金流向：典型价格与成交量的关联
                # MFI = 典型价格变化方向 * 成交量变化
                if volume is None or len(volume) < 5:
                    return 0.0
                # 计算典型价格
                tp = (high + low + close) / 3
                # 计算资金流强度 (价格涨幅 * 成交量相对强度)
                if len(tp) >= 2:
                    price_change = (tp[-1] - tp[-2]) / (tp[-2] + 1e-10)
                    volume_ma = _sma(volume, 5)
                    if len(volume_ma) > 0 and not np.isnan(volume_ma[-1]) and volume_ma[-1] > 0:
                        volume_strength = (volume[-1] - volume_ma[-1]) / volume_ma[-1]
                        return float(price_change * volume_strength)
                return 0.0

            elif factor_name == "bb_ratio":
                # 布林带位置: (close - lower) / (upper - lower)
                period = 20
                if len(close) < period:
                    return 0.5
                ma = _sma(close, period)
                std = _std(close, period)
                if len(ma) > 0 and not np.isnan(ma[-1]) and std[-1] > 0:
                    upper = ma[-1] + 2 * std[-1]
                    lower = ma[-1] - 2 * std[-1]
                    return float((close[-1] - lower) / (upper - lower + 1e-10))
                return 0.5

            elif factor_name == "roc":
                # 变动率 ROC(10)
                period = 10
                if len(close) < period + 1:
                    return 0.0
                return float((close[-1] - close[-period-1]) / (close[-period-1] + 1e-10))

            elif factor_name == "williams_r":
                # 威廉指标 %R
                period = 14
                if len(close) < period:
                    return -50.0
                highest = np.max(high[-period:])
                lowest = np.min(low[-period:])
                if highest - lowest < 1e-10:
                    return -50.0
                return float(-100 * (highest - close[-1]) / (highest - lowest))

            else:
                logger.warning(f"未知因子: {factor_name}")
                return 0.0

        except Exception as e:
            logger.warning(f"计算因子 {factor_name} 失败: {e}")
            return 0.0

    def _normalize_factor(self, factor_name: str, value: float) -> float:
        """归一化因子值到 [0, 1]

        Args:
            factor_name: 因子名称
            value: 原始因子值

        Returns:
            归一化后的值 [0, 1]
        """
        # 不同因子有不同的归一化方式
        # 使用 sigmoid 或 tanh 来平滑处理极端值

        if factor_name in ["ma_ratio", "momentum", "price_momentum", "volume_price"]:
            # 这些因子可以是负数，使用 tanh 归一化
            return (np.tanh(value * 10) + 1) / 2

        elif factor_name in ["rsi", "ma_cross", "adx", "kdj"]:
            # 这些因子已经在 [0, 1] 范围内
            return np.clip(value, 0, 1)

        elif factor_name in ["macd", "cci"]:
            # 这些因子范围约 [-2, 2]，归一化到 [0, 1]
            return np.clip((value + 2) / 4, 0, 1)

        elif factor_name in ["volatility", "atr_ratio"]:
            # 波动率类因子，使用 sigmoid
            return 1 / (1 + np.exp(-value * 20))

        elif factor_name in ["volume_ratio"]:
            # 量比，使用对数归一化
            return np.clip(np.log1p(value) / 5, 0, 1)

        elif factor_name == "obv":
            # OBV 变化率，使用 tanh
            return (np.tanh(value * 5) + 1) / 2

        elif factor_name == "bb_ratio":
            # 布林带位置已在 [0, 1]，但可能有超出
            return np.clip(value, 0, 1)

        elif factor_name == "roc":
            # ROC 变动率，使用 tanh 归一化
            return (np.tanh(value * 5) + 1) / 2

        elif factor_name == "williams_r":
            # 威廉指标范围 [-100, 0]，归一化到 [0, 1]
            return np.clip((value + 100) / 100, 0, 1)

        else:
            # 默认使用 sigmoid
            return 1 / (1 + np.exp(-value * 10))

    def get_factor_contributions(self, data: pd.DataFrame) -> dict[str, float]:
        """获取各因子对信号的贡献

        Args:
            data: 市场数据

        Returns:
            因子贡献字典
        """
        contributions = {}

        for factor_name in self.factor_weights.keys():
            factor_value = self._calculate_factor_value(factor_name, data)

            if factor_value is None or np.isnan(factor_value):
                contributions[factor_name] = 0.0
                continue

            normalized = self._normalize_factor(factor_name, factor_value)
            weight = self.factor_weights[factor_name]
            contributions[factor_name] = normalized * weight

        return contributions

    def get_signal_info(self, data: pd.DataFrame) -> dict:
        """获取信号详细信息

        Args:
            data: 市场数据

        Returns:
            信号信息字典
        """
        entry_signal = self.generate_entry_signal(data)
        exit_signal = self.generate_exit_signal(data)
        contributions = self.get_factor_contributions(data)

        return {
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
            "signal_threshold": self.signal_threshold,
            "exit_threshold": self.exit_threshold,
            "should_enter": entry_signal >= self.signal_threshold,
            "should_exit": exit_signal >= self.exit_threshold,
            "factor_contributions": contributions,
        }
