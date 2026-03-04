"""海龟策略因子信号生成器

使用因子组合生成入场和出场信号，与遗传算法的 Individual 基因编码对接。

优化版本：使用指标预计算缓存 + 市场环境识别
"""

import logging
from typing import TYPE_CHECKING, Optional
import numpy as np
import pandas as pd

from src.optimizer.individual import Individual, FACTOR_POOL
from src.strategy.market_environment import MarketEnvironment

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============ 核心计算函数（向量化优化） ============

def _sma_fast(data: np.ndarray, period: int) -> np.ndarray:
    """快速简单移动平均 (使用累加优化)"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    result = np.empty(n)
    result[:period-1] = np.nan
    # 使用累加优化 O(n) 而非 O(n*period)
    cumsum = 0.0
    for i in range(period):
        cumsum += data[i]
    result[period-1] = cumsum / period
    for i in range(period, n):
        cumsum += data[i] - data[i-period]
        result[i] = cumsum / period
    return result


def _ema_fast(data: np.ndarray, period: int) -> np.ndarray:
    """快速指数移动平均"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    alpha = 2.0 / (period + 1)
    result = np.empty(n)
    result[:period-1] = np.nan
    # 第一个有效值用 SMA
    cumsum = 0.0
    for i in range(period):
        cumsum += data[i]
    result[period-1] = cumsum / period
    # 后续用 EMA 公式
    for i in range(period, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result


def _std_fast(data: np.ndarray, period: int) -> np.ndarray:
    """快速标准差 (Pandas 向量化)"""
    return pd.Series(data).rolling(period).std().values


def _rsi_fast(data: np.ndarray, period: int = 14) -> np.ndarray:
    """快速 RSI"""
    n = len(data)
    if n < period + 1:
        result = np.empty(n)
        result.fill(50.0)
        return result

    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # 使用 Pandas 的 ewm 加速
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)

    avg_gain = gain_series.ewm(span=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, adjust=False).mean().values

    rs = np.divide(avg_gain, avg_loss, out=np.ones(n-1), where=avg_loss > 1e-10)
    rsi_values = 100 - (100 / (1 + rs))

    # 填充结果
    result = np.empty(n)
    result[:period] = 50.0
    result[period:] = rsi_values[period-1:]

    return result


def _atr_fast(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """快速 ATR (使用 Pandas 向量化)"""
    n = len(close)
    if n < 2:
        return np.zeros(n)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    # 使用 Pandas EWM
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr


def _macd_fast(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """快速 MACD"""
    ema_fast = _ema_fast(data, fast)
    ema_slow = _ema_fast(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_fast(np.nan_to_num(macd_line), signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _stoch_fast(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                k_period: int = 14, d_period: int = 3) -> tuple:
    """快速随机指标 KDJ (Pandas 向量化)"""
    n = len(close)
    if n < k_period:
        return np.full(n, 50.0), np.full(n, 50.0)

    # 使用 Pandas 的 rolling 操作
    low_series = pd.Series(low)
    high_series = pd.Series(high)

    lowest_low = low_series.rolling(k_period).min().values
    highest_high = high_series.rolling(k_period).max().values

    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k = np.nan_to_num(k, nan=50.0)

    d = _sma_fast(k, d_period)
    d = np.nan_to_num(d, nan=50.0)

    return k, d


def _adx_fast(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """快速 ADX"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)

    # 计算 +DM 和 -DM
    plus_dm = np.where(high[1:] > high[:-1], high[1:] - high[:-1], 0.0)
    minus_dm = np.where(low[1:] < low[:-1], low[:-1] - low[1:], 0.0)

    atr = _atr_fast(high, low, close, period)

    # 使用 EWM 平滑
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / (atr[1:] + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / (atr[1:] + 1e-10)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    adx = np.nan_to_num(adx, nan=0.0)

    # 返回与输入 close 相同长度的数组
    result = np.zeros(n)
    result[-len(adx):] = adx

    return result


def _cci_fast(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """快速 CCI"""
    n = len(close)
    if n < period:
        return np.zeros(n)

    tp = (high + low + close) / 3
    ma_tp = _sma_fast(tp, period)
    md = _std_fast(tp, period)

    cci = (tp - ma_tp) / (md + 1e-10)
    return np.nan_to_num(cci, nan=0.0)


def _obv_fast(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """快速 OBV (使用 cumsum)"""
    n = len(close)
    if n < 2:
        return np.zeros(n)

    direction = np.sign(np.diff(close))
    direction = np.insert(direction, 0, 1)  # 第一天设为正
    obv = np.cumsum(direction * volume)

    return obv


class IndicatorCache:
    """技术指标缓存类

    预计算所有可能需要的技术指标，避免在信号判断时重复计算。
    """

    def __init__(self, data: pd.DataFrame):
        """初始化并预计算所有指标

        Args:
            data: OHLCV 数据
        """
        self.open = data["open"].values.astype(np.float64) if "open" in data.columns else None
        self.close = data["close"].values.astype(np.float64)
        self.high = data["high"].values.astype(np.float64)
        self.low = data["low"].values.astype(np.float64)
        self.volume = data["volume"].values.astype(np.float64) if "volume" in data.columns else None

        self._cache = {}
        self._precompute()

    def _precompute(self):
        """预计算所有指标"""
        # 移动平均
        self._cache["ma5"] = _sma_fast(self.close, 5)
        self._cache["ma20"] = _sma_fast(self.close, 20)
        self._cache["ma50"] = _sma_fast(self.close, 50)

        # ATR
        self._cache["atr14"] = _atr_fast(self.high, self.low, self.close, 14)

        # RSI
        self._cache["rsi14"] = _rsi_fast(self.close, 14)

        # MACD
        macd_line, signal_line, _ = _macd_fast(self.close)
        self._cache["macd_line"] = macd_line
        self._cache["macd_signal"] = signal_line

        # KDJ
        k, d = _stoch_fast(self.high, self.low, self.close)
        self._cache["kdj_k"] = k
        self._cache["kdj_d"] = d

        # ADX
        self._cache["adx14"] = _adx_fast(self.high, self.low, self.close, 14)

        # CCI
        self._cache["cci14"] = _cci_fast(self.high, self.low, self.close, 14)

        # 波动率
        self._cache["std20"] = _std_fast(self.close, 20)

        # 布林带
        self._cache["bb_upper"] = self._cache["ma20"] + 2 * self._cache["std20"]
        self._cache["bb_lower"] = self._cache["ma20"] - 2 * self._cache["std20"]

        # 成交量指标
        if self.volume is not None:
            self._cache["vol_ma5"] = _sma_fast(self.volume, 5)
            self._cache["vol_ma20"] = _sma_fast(self.volume, 20)
            self._cache["obv"] = _obv_fast(self.close, self.volume)

    def get(self, name: str) -> np.ndarray | None:
        """获取缓存的指标"""
        return self._cache.get(name)

    def get_atr(self, idx: int = -1) -> float:
        """获取指定位置的 ATR 值"""
        atr = self._cache["atr14"]
        val = atr[idx]
        return float(val) if not np.isnan(val) else 0.0

    def _get_open(self, idx: int = -1) -> float:
        """获取指定位置的 Open 值"""
        if self.open is None:
            return 0.0
        val = self.open[idx]
        return float(val) if not np.isnan(val) else 0.0


class TurtleSignalGenerator:
    """海龟策略信号生成器

    基于因子组合计算信号强度，与遗传算法的 Individual 基因编码对接。

    信号强度范围 [0, 1]：
    - 入场信号：越高表示越强烈的买入信号
    - 出场信号：越高表示越强烈的卖出信号
    """

    def __init__(self, individual: Individual, indicator_cache: IndicatorCache | None = None):
        """初始化信号生成器

        Args:
            individual: 包含因子权重和阈值的个体
            indicator_cache: 预计算的指标缓存（可选）
        """
        self.individual = individual
        self.factor_weights = individual.factor_weights
        self.signal_threshold = individual.signal_threshold
        self.exit_threshold = individual.exit_threshold
        self.indicator_cache = indicator_cache

    def set_cache(self, cache: IndicatorCache):
        """设置指标缓存"""
        self.indicator_cache = cache

    def generate_entry_signal(self, data: pd.DataFrame | None = None, idx: int = -1) -> float:
        """生成入场信号强度

        Args:
            data: 包含 OHLCV 数据的 DataFrame（当没有缓存时使用）
            idx: 数据索引位置（使用缓存时）

        Returns:
            入场信号强度 [0, 1]
        """
        if self.indicator_cache is not None:
            # 使用缓存的指标（快速路径）
            if idx < 20:
                return 0.0

            signal_strength = 0.0
            total_weight = 0.0

            for factor_name, weight in self.factor_weights.items():
                factor_value = self._calculate_factor_value_cached(factor_name, idx)

                if factor_value is None or np.isnan(factor_value):
                    continue

                normalized_value = self._normalize_factor(factor_name, factor_value)
                signal_strength += normalized_value * weight
                total_weight += weight

            if total_weight > 0:
                signal_strength /= total_weight

            return float(np.clip(signal_strength, 0, 1))
        else:
            # 没有缓存，需要计算（慢速路径）
            if data is None or len(data) < 20:
                return 0.0

            signal_strength = 0.0
            total_weight = 0.0

            for factor_name, weight in self.factor_weights.items():
                factor_value = self._calculate_factor_value(factor_name, data)

                if factor_value is None or np.isnan(factor_value):
                    continue

                normalized_value = self._normalize_factor(factor_name, factor_value)
                signal_strength += normalized_value * weight
                total_weight += weight

            if total_weight > 0:
                signal_strength /= total_weight

            return float(np.clip(signal_strength, 0, 1))

    def generate_exit_signal(self, data: pd.DataFrame | None = None, idx: int = -1) -> float:
        """生成出场信号强度

        Args:
            data: 包含 OHLCV 数据的 DataFrame
            idx: 数据索引位置（使用缓存时）

        Returns:
            出场信号强度 [0, 1]
        """
        if self.indicator_cache is not None:
            if idx < 20:
                return 0.0

            exit_strength = 0.0
            total_weight = 0.0

            for factor_name, weight in self.factor_weights.items():
                factor_value = self._calculate_factor_value_cached(factor_name, idx)

                if factor_value is None or np.isnan(factor_value):
                    continue

                # 出场信号：取反向因子值
                normalized_value = 1 - self._normalize_factor(factor_name, factor_value)
                exit_strength += normalized_value * weight
                total_weight += weight

            if total_weight > 0:
                exit_strength /= total_weight

            return float(np.clip(exit_strength, 0, 1))
        else:
            if data is None or len(data) < 20:
                return 0.0

            exit_strength = 0.0
            total_weight = 0.0

            for factor_name, weight in self.factor_weights.items():
                factor_value = self._calculate_factor_value(factor_name, data)

                if factor_value is None or np.isnan(factor_value):
                    continue

                normalized_value = 1 - self._normalize_factor(factor_name, factor_value)
                exit_strength += normalized_value * weight
                total_weight += weight

            if total_weight > 0:
                exit_strength /= total_weight

            return float(np.clip(exit_strength, 0, 1))

    def should_enter(self, data: pd.DataFrame | None = None, idx: int = -1) -> bool:
        """判断是否应该入场

        综合考虑：
        1. 因子信号强度
        2. ADX趋势强度过滤（可选，默认禁用）
        3. 市场环境识别（可选，默认禁用）

        注意：市场环境识别和ADX过滤器默认已禁用，以增加交易机会。
        """
        entry_signal = self.generate_entry_signal(data, idx)

        # 基础信号检查
        if entry_signal < self.signal_threshold:
            return False

        # 趋势过滤器（默认禁用，可通过 use_trend_filter=True 启用）
        if self.individual.use_trend_filter:
            # ADX 趋势强度过滤器
            if self.indicator_cache is not None:
                adx_values = self.indicator_cache.get("adx14")
                if adx_values is not None and idx < len(adx_values):
                    current_adx = float(adx_values[idx])
                    if current_adx < self.individual.min_adx:
                        # ADX 趋势强度不足，不入场
                        return False

            # 市场环境识别（只在趋势市入场）
            if self.indicator_cache is not None and idx >= 50:
                # 获取最近50天的价格
                prices = self.indicator_cache.close[max(0, idx-49):idx+1]
                adx = self.indicator_cache.get("adx14")
                adx_series = adx[max(0, idx-49):idx+1] if adx is not None else None

                # 分析市场环境
                env_analyzer = MarketEnvironment(lookback=50)
                env_analysis = env_analyzer.analyze(prices, adx_series)

                # 只在趋势市入场
                if not env_analysis["is_trending"]:
                    return False

        return True

    def should_exit(self, data: pd.DataFrame | None = None, idx: int = -1) -> bool:
        """判断是否应该出场"""
        exit_signal = self.generate_exit_signal(data, idx)
        return exit_signal >= self.exit_threshold

    def _calculate_factor_value_cached(self, factor_name: str, idx: int) -> float | None:
        """使用缓存计算单个因子的值

        Args:
            factor_name: 因子名称
            idx: 数据索引位置

        Returns:
            因子值
        """
        cache = self.indicator_cache
        close = cache.close
        high = cache.high
        low = cache.low
        volume = cache.volume

        try:
            if factor_name == "ma_ratio":
                ma5 = cache.get("ma5")
                ma20 = cache.get("ma20")
                if ma5 is not None and ma20 is not None and not np.isnan(ma5[idx]) and not np.isnan(ma20[idx]):
                    return float(ma5[idx] / ma20[idx] - 1)
                return 0.0

            elif factor_name == "ma_cross":
                ma5 = cache.get("ma5")
                ma20 = cache.get("ma20")
                if ma5 is not None and ma20 is not None and not np.isnan(ma5[idx]) and not np.isnan(ma20[idx]):
                    return float(1.0 if ma5[idx] > ma20[idx] else 0.0)
                return 0.0

            elif factor_name == "momentum":
                if idx >= 5:
                    return float((close[idx] - close[idx-5]) / close[idx-5])
                return 0.0

            elif factor_name == "price_momentum":
                if idx >= 20:
                    return float(close[idx] / close[idx-20] - 1)
                return 0.0

            elif factor_name == "rsi":
                rsi = cache.get("rsi14")
                if rsi is not None:
                    return float(rsi[idx] / 100)
                return 0.5

            elif factor_name == "macd":
                macd_line = cache.get("macd_line")
                signal_line = cache.get("macd_signal")
                if macd_line is not None and signal_line is not None:
                    if not np.isnan(signal_line[idx]) and abs(signal_line[idx]) > 1e-10:
                        return float(macd_line[idx] / signal_line[idx])
                return 0.0

            elif factor_name == "kdj":
                k = cache.get("kdj_k")
                d = cache.get("kdj_d")
                if k is not None and d is not None:
                    if not np.isnan(k[idx]) and not np.isnan(d[idx]):
                        return float((k[idx] - d[idx]) / 100)
                return 0.0

            elif factor_name == "volatility":
                std20 = cache.get("std20")
                if std20 is not None and not np.isnan(std20[idx]):
                    return float(std20[idx] / close[idx])
                return 0.0

            elif factor_name == "atr_ratio":
                atr = cache.get("atr14")
                if atr is not None and not np.isnan(atr[idx]):
                    return float(atr[idx] / close[idx])
                return 0.0

            elif factor_name == "volume_ratio":
                if volume is not None:
                    vol_ma20 = cache.get("vol_ma20")
                    if vol_ma20 is not None and not np.isnan(vol_ma20[idx]) and vol_ma20[idx] > 0:
                        return float(volume[idx] / vol_ma20[idx])
                return 1.0

            elif factor_name == "volume_price":
                if volume is not None and idx >= 1:
                    volume_change = (volume[idx] - volume[idx-1]) / (volume[idx-1] + 1e-10)
                    price_change = (close[idx] - close[idx-1]) / (close[idx-1] + 1e-10)
                    return float(volume_change * price_change)
                return 0.0

            elif factor_name == "adx":
                adx = cache.get("adx14")
                if adx is not None:
                    return float(adx[idx] / 100)
                return 0.0

            elif factor_name == "cci":
                cci = cache.get("cci14")
                if cci is not None:
                    return float(cci[idx] / 200)
                return 0.0

            elif factor_name == "obv":
                if volume is not None:
                    obv = cache.get("obv")
                    if obv is not None:
                        vol_ma20 = cache.get("vol_ma20")
                        if vol_ma20 is not None and not np.isnan(vol_ma20[idx]) and vol_ma20[idx] > 0:
                            return float(obv[idx] / vol_ma20[idx] - 1)
                return 0.0

            elif factor_name == "money_flow":
                if volume is not None and idx >= 1:
                    tp = (high[idx] + low[idx] + close[idx]) / 3
                    tp_prev = (high[idx-1] + low[idx-1] + close[idx-1]) / 3
                    price_change = (tp - tp_prev) / (tp_prev + 1e-10)
                    vol_ma5 = cache.get("vol_ma5")
                    if vol_ma5 is not None and not np.isnan(vol_ma5[idx]) and vol_ma5[idx] > 0:
                        volume_strength = (volume[idx] - vol_ma5[idx]) / vol_ma5[idx]
                        return float(price_change * volume_strength)
                return 0.0

            elif factor_name == "bb_ratio":
                ma20 = cache.get("ma20")
                std20 = cache.get("std20")
                if ma20 is not None and std20 is not None and not np.isnan(ma20[idx]) and not np.isnan(std20[idx]):
                    upper = ma20[idx] + 2 * std20[idx]
                    lower = ma20[idx] - 2 * std20[idx]
                    return float((close[idx] - lower) / (upper - lower + 1e-10))
                return 0.5

            elif factor_name == "roc":
                if idx >= 10:
                    return float((close[idx] - close[idx-10]) / (close[idx-10] + 1e-10))
                return 0.0

            elif factor_name == "williams_r":
                if idx >= 14:
                    highest = np.max(high[idx-13:idx+1])
                    lowest = np.min(low[idx-13:idx+1])
                    if highest - lowest < 1e-10:
                        return -50.0
                    return float(-100 * (highest - close[idx]) / (highest - lowest))
                return -50.0

            elif factor_name == "volume_trend":
                if volume is not None:
                    vol_ma5 = cache.get("vol_ma5")
                    vol_ma20 = cache.get("vol_ma20")
                    if vol_ma5 is not None and vol_ma20 is not None:
                        if not np.isnan(vol_ma20[idx]) and vol_ma20[idx] > 0:
                            return float(vol_ma5[idx] / vol_ma20[idx])
                return 1.0

            elif factor_name == "risk_adj_momentum":
                if idx >= 10:
                    momentum_val = (close[idx] - close[idx-10]) / (close[idx-10] + 1e-10)
                    std20 = cache.get("std20")
                    if std20 is not None and not np.isnan(std20[idx]) and std20[idx] > 0:
                        return float(momentum_val / (std20[idx] / close[idx] + 1e-10))
                return 0.0

            elif factor_name == "relative_strength":
                ma50 = cache.get("ma50")
                if ma50 is not None and not np.isnan(ma50[idx]) and ma50[idx] > 0:
                    return float(close[idx] / ma50[idx] - 1)
                return 0.0

            elif factor_name == "vol_adj_return":
                if idx >= 1:
                    atr = cache.get("atr14")
                    if atr is not None and not np.isnan(atr[idx]) and atr[idx] > 0:
                        price_change = close[idx] - close[idx-1]
                        return float(price_change / atr[idx])
                return 0.0

            elif factor_name == "trend_consistency":
                if idx >= 10:
                    changes = np.diff(close[idx-10:idx+1])
                    up_days = np.sum(changes > 0)
                    return float(up_days / 10)
                return 0.5

            elif factor_name == "higher_highs":
                if idx >= 5:
                    count = 0
                    for i in range(idx-4, idx+1):
                        start = max(0, i-5)
                        if high[i] > np.max(high[start:i]):
                            count += 1
                    return float(count / 5)
                return 0.0

            # ========== 价格行为因子 (右侧交易) ==========
            elif factor_name == "new_high_count":
                # 过去10日创新高次数
                if idx >= 10:
                    count = 0
                    for i in range(idx-9, idx+1):
                        start = max(0, i-10)
                        if close[i] > np.max(close[start:i]):
                            count += 1
                    return float(count / 10)
                return 0.0

            elif factor_name == "new_low_count":
                # 过去10日创新低次数
                if idx >= 10:
                    count = 0
                    for i in range(idx-9, idx+1):
                        start = max(0, i-10)
                        if close[i] < np.min(close[start:i]):
                            count += 1
                    return float(count / 10)
                return 0.0

            elif factor_name == "consecutive_up":
                # 连续阳线天数 (归一化到0-1)
                if idx >= 1:
                    count = 0
                    for i in range(idx, max(-1, idx-10), -1):
                        if close[i] > close[i-1]:
                            count += 1
                        else:
                            break
                    return float(count / 10)
                return 0.0

            elif factor_name == "consecutive_down":
                # 连续阴线天数 (归一化到0-1)
                if idx >= 1:
                    count = 0
                    for i in range(idx, max(-1, idx-10), -1):
                        if close[i] < close[i-1]:
                            count += 1
                        else:
                            break
                    return float(count / 10)
                return 0.0

            elif factor_name == "gap_up":
                # 向上跳空: open > prev_high
                if idx >= 1:
                    return float(1.0 if cache._get_open(idx) > high[idx-1] else 0.0)
                return 0.0

            elif factor_name == "gap_down":
                # 向下跳空: open < prev_low
                if idx >= 1:
                    return float(1.0 if cache._get_open(idx) < low[idx-1] else 0.0)
                return 0.0

            else:
                logger.warning(f"Unknown factor: {factor_name}")
                return 0.0

        except Exception as e:
            logger.warning(f"Factor {factor_name} calculation failed: {e}")
            return 0.0

    def _calculate_factor_value(self, factor_name: str, data: pd.DataFrame) -> float | None:
        """计算单个因子的值（无缓存版本）

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
                ma5 = _sma_fast(close, 5)
                ma20 = _sma_fast(close, 20)
                if len(ma20) > 0 and not np.isnan(ma20[-1]):
                    return float(ma5[-1] / ma20[-1] - 1)
                return 0.0

            elif factor_name == "ma_cross":
                ma5 = _sma_fast(close, 5)
                ma20 = _sma_fast(close, 20)
                if len(ma20) > 0 and not np.isnan(ma20[-1]):
                    return float(1.0 if ma5[-1] > ma20[-1] else 0.0)
                return 0.0

            elif factor_name == "momentum":
                if len(close) < 6:
                    return 0.0
                return float((close[-1] - close[-6]) / close[-6])

            elif factor_name == "price_momentum":
                if len(close) < 21:
                    return 0.0
                return float(close[-1] / close[-21] - 1)

            elif factor_name == "rsi":
                rsi = _rsi_fast(close, 14)
                return float(rsi[-1] / 100) if len(rsi) > 0 else 0.5

            elif factor_name == "macd":
                macd_line, signal_line, _ = _macd_fast(close)
                if len(signal_line) > 0 and not np.isnan(signal_line[-1]) and signal_line[-1] != 0:
                    return float(macd_line[-1] / signal_line[-1])
                return 0.0

            elif factor_name == "kdj":
                k, d = _stoch_fast(high, low, close)
                if len(k) > 0 and not np.isnan(k[-1]) and not np.isnan(d[-1]):
                    return float((k[-1] - d[-1]) / 100)
                return 0.0

            elif factor_name == "volatility":
                std = _std_fast(close, 20)
                if len(std) > 0 and not np.isnan(std[-1]):
                    return float(std[-1] / close[-1])
                return 0.0

            elif factor_name == "atr_ratio":
                atr_val = _atr_fast(high, low, close, 14)
                if len(atr_val) > 0 and not np.isnan(atr_val[-1]):
                    return float(atr_val[-1] / close[-1])
                return 0.0

            elif factor_name == "volume_ratio":
                if volume is None or len(volume) < 21:
                    return 1.0
                ma_volume = _sma_fast(volume, 20)
                if len(ma_volume) > 0 and not np.isnan(ma_volume[-1]) and ma_volume[-1] > 0:
                    return float(volume[-1] / ma_volume[-1])
                return 1.0

            elif factor_name == "volume_price":
                if volume is None or len(volume) < 2:
                    return 0.0
                volume_change = (volume[-1] - volume[-2]) / (volume[-2] + 1e-10)
                price_change = (close[-1] - close[-2]) / (close[-2] + 1e-10)
                return float(volume_change * price_change)

            elif factor_name == "adx":
                adx_val = _adx_fast(high, low, close, 14)
                return float(adx_val[-1] / 100) if len(adx_val) > 0 else 0.0

            elif factor_name == "cci":
                cci_val = _cci_fast(high, low, close, 14)
                return float(cci_val[-1] / 200) if len(cci_val) > 0 else 0.0

            elif factor_name == "obv":
                if volume is None:
                    return 0.0
                obv_val = _obv_fast(close, volume)
                if len(obv_val) < 21:
                    return 0.0
                ma_obv = _sma_fast(obv_val, 20)
                if len(ma_obv) > 0 and not np.isnan(ma_obv[-1]) and ma_obv[-1] > 0:
                    return float(obv_val[-1] / ma_obv[-1] - 1)
                return 0.0

            elif factor_name == "money_flow":
                if volume is None or len(volume) < 5:
                    return 0.0
                tp = (high + low + close) / 3
                if len(tp) >= 2:
                    price_change = (tp[-1] - tp[-2]) / (tp[-2] + 1e-10)
                    volume_ma = _sma_fast(volume, 5)
                    if len(volume_ma) > 0 and not np.isnan(volume_ma[-1]) and volume_ma[-1] > 0:
                        volume_strength = (volume[-1] - volume_ma[-1]) / volume_ma[-1]
                        return float(price_change * volume_strength)
                return 0.0

            elif factor_name == "bb_ratio":
                period = 20
                if len(close) < period:
                    return 0.5
                ma = _sma_fast(close, period)
                std = _std_fast(close, period)
                if len(ma) > 0 and not np.isnan(ma[-1]) and std[-1] > 0:
                    upper = ma[-1] + 2 * std[-1]
                    lower = ma[-1] - 2 * std[-1]
                    return float((close[-1] - lower) / (upper - lower + 1e-10))
                return 0.5

            elif factor_name == "roc":
                period = 10
                if len(close) < period + 1:
                    return 0.0
                return float((close[-1] - close[-period-1]) / (close[-period-1] + 1e-10))

            elif factor_name == "williams_r":
                period = 14
                if len(close) < period:
                    return -50.0
                highest = np.max(high[-period:])
                lowest = np.min(low[-period:])
                if highest - lowest < 1e-10:
                    return -50.0
                return float(-100 * (highest - close[-1]) / (highest - lowest))

            elif factor_name == "volume_trend":
                if volume is None or len(volume) < 21:
                    return 1.0
                ma_vol_5 = _sma_fast(volume, 5)
                ma_vol_20 = _sma_fast(volume, 20)
                if len(ma_vol_20) > 0 and not np.isnan(ma_vol_20[-1]) and ma_vol_20[-1] > 0:
                    return float(ma_vol_5[-1] / ma_vol_20[-1])
                return 1.0

            elif factor_name == "risk_adj_momentum":
                if len(close) < 21:
                    return 0.0
                momentum_val = (close[-1] - close[-11]) / (close[-11] + 1e-10)
                std_val = _std_fast(close, 20)
                if len(std_val) > 0 and not np.isnan(std_val[-1]) and std_val[-1] > 0:
                    return float(momentum_val / (std_val[-1] / close[-1] + 1e-10))
                return 0.0

            elif factor_name == "relative_strength":
                period = 50
                if len(close) < period:
                    return 0.0
                ma50 = _sma_fast(close, period)
                if len(ma50) > 0 and not np.isnan(ma50[-1]) and ma50[-1] > 0:
                    return float(close[-1] / ma50[-1] - 1)
                return 0.0

            elif factor_name == "vol_adj_return":
                if len(close) < 2:
                    return 0.0
                atr_val = _atr_fast(high, low, close, 14)
                if len(atr_val) > 0 and not np.isnan(atr_val[-1]) and atr_val[-1] > 0:
                    price_change = close[-1] - close[-2]
                    return float(price_change / atr_val[-1])
                return 0.0

            elif factor_name == "trend_consistency":
                period = 10
                if len(close) < period + 1:
                    return 0.5
                changes = np.diff(close[-period-1:])
                up_days = np.sum(changes > 0)
                return float(up_days / period)

            elif factor_name == "higher_highs":
                period = 5
                if len(high) < period + 1:
                    return 0.0
                count = 0
                for i in range(-period, 0):
                    if high[i] > np.max(high[max(0, i-period):i]):
                        count += 1
                return float(count / period)

            elif factor_name == "new_high_count":
                # 过去10日创新高次数
                if len(close) >= 11:
                    count = 0
                    for i in range(-10, 0):
                        if close[i] > np.max(close[max(0, i-10):i]):
                            count += 1
                    return float(count / 10)
                return 0.0

            elif factor_name == "new_low_count":
                # 过去10日创新低次数
                if len(close) >= 11:
                    count = 0
                    for i in range(-10, 0):
                        if close[i] < np.min(close[max(0, i-10):i]):
                            count += 1
                    return float(count / 10)
                return 0.0

            elif factor_name == "consecutive_up":
                # 连续阳线天数(归一化到0-1)
                if len(close) >= 2:
                    count = 0
                    for i in range(-1, -len(close)-1, -1):
                        if close[i] > close[i-1]:
                            count += 1
                        else:
                            break
                    return float(count / 10)
                return 0.0
            elif factor_name == "consecutive_down":
                # 连续阴线天数(归一化到0-1)
                if len(close) >= 2:
                    count = 0
                    for i in range(-1, -len(close)-1, -1):
                        if close[i] < close[i-1]:
                            count += 1
                        else:
                            break
                    return float(count / 10)
                return 0.0
            elif factor_name == "gap_up":
                # 向上跳空: open > prev_high
                if len(close) >= 2 and "open" in data.columns:
                    open = data["open"].values
                    return float(1.0 if open[-1] > high[-2] else 0.0)
                return 0.0
            elif factor_name == "gap_down":
                # 向下跳空: open < prev_low
                if len(close) >= 2 and "open" in data.columns:
                    open = data["open"].values
                    return float(1.0 if open[-1] < low[-2] else 0.0)
                return 0.0
            else:
                logger.warning(f"Unknown factor: {factor_name}")
                return 0.0

        except Exception as e:
            logger.warning(f"Factor {factor_name} calculation failed: {e}")
            return 0.0

    def _normalize_factor(self, factor_name: str, value: float) -> float:
        """归一化因子值到 [0, 1]"""
        if factor_name in ["ma_ratio", "momentum", "price_momentum", "volume_price"]:
            return (np.tanh(value * 10) + 1) / 2
        elif factor_name in ["rsi", "ma_cross", "adx", "kdj"]:
            return np.clip(value, 0, 1)
        elif factor_name in ["macd", "cci"]:
            return np.clip((value + 2) / 4, 0, 1)
        elif factor_name in ["volatility", "atr_ratio"]:
            return 1 / (1 + np.exp(-value * 20))
        elif factor_name in ["volume_ratio"]:
            return np.clip(np.log1p(value) / 5, 0, 1)
        elif factor_name == "obv":
            return (np.tanh(value * 5) + 1) / 2
        elif factor_name == "bb_ratio":
            return np.clip(value, 0, 1)
        elif factor_name == "roc":
            return (np.tanh(value * 5) + 1) / 2
        elif factor_name == "williams_r":
            return np.clip((value + 100) / 100, 0, 1)
        elif factor_name == "volume_trend":
            return np.clip(np.log1p(value) / 3, 0, 1)
        elif factor_name == "risk_adj_momentum":
            return (np.tanh(value) + 1) / 2
        elif factor_name == "relative_strength":
            return (np.tanh(value * 10) + 1) / 2
        elif factor_name == "vol_adj_return":
            return (np.tanh(value) + 1) / 2
        elif factor_name in ["trend_consistency", "higher_highs", "new_high_count", "new_low_count", "consecutive_up", "consecutive_down", "gap_up", "gap_down"]:
            return np.clip(value, 0, 1)
        else:
            return 1 / (1 + np.exp(-value * 10))

    def get_factor_contributions(self, data: pd.DataFrame | None = None, idx: int = -1) -> dict[str, float]:
        """获取各因子对信号的贡献"""
        contributions = {}

        for factor_name in self.factor_weights.keys():
            if self.indicator_cache is not None:
                factor_value = self._calculate_factor_value_cached(factor_name, idx)
            else:
                factor_value = self._calculate_factor_value(factor_name, data) if data is not None else None

            if factor_value is None or np.isnan(factor_value):
                contributions[factor_name] = 0.0
                continue

            normalized = self._normalize_factor(factor_name, factor_value)
            weight = self.factor_weights[factor_name]
            contributions[factor_name] = normalized * weight

        return contributions

    def get_signal_info(self, data: pd.DataFrame | None = None, idx: int = -1) -> dict:
        """获取信号详细信息"""
        entry_signal = self.generate_entry_signal(data, idx)
        exit_signal = self.generate_exit_signal(data, idx)
        contributions = self.get_factor_contributions(data, idx)

        return {
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
            "signal_threshold": self.signal_threshold,
            "exit_threshold": self.exit_threshold,
            "should_enter": entry_signal >= self.signal_threshold,
            "should_exit": exit_signal >= self.exit_threshold,
            "factor_contributions": contributions,
        }
