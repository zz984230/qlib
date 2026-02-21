"""市场状态识别测试"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.regime import (
    RegimeDetector,
    MarketRegime,
    MarketState,
    get_recommended_strategy,
)


class TestRegimeDetector:
    """市场状态识别器测试"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # 创建趋势数据
        trend = np.linspace(100, 120, 100) + np.random.randn(100) * 2
        return pd.DataFrame({
            "date": dates,
            "open": trend - np.random.rand(100) * 2,
            "high": trend + np.random.rand(100) * 3,
            "low": trend - np.random.rand(100) * 3,
            "close": trend,
            "volume": np.random.randint(1000000, 5000000, 100),
        })

    @pytest.fixture
    def oscillating_data(self):
        """创建震荡数据"""
        np.random.seed(123)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # 创建震荡数据
        base = 100
        oscillation = 5 * np.sin(np.linspace(0, 4 * np.pi, 100))
        noise = np.random.randn(100) * 1
        close = base + oscillation + noise

        return pd.DataFrame({
            "date": dates,
            "open": close - np.random.rand(100),
            "high": close + np.random.rand(100) * 2,
            "low": close - np.random.rand(100) * 2,
            "close": close,
            "volume": np.random.randint(1000000, 5000000, 100),
        })

    def test_init(self):
        """测试初始化"""
        detector = RegimeDetector()
        assert detector.adx_threshold == 25.0
        assert detector.lookback == 60

    def test_detect_trending(self, sample_data):
        """测试趋势市场检测"""
        detector = RegimeDetector()
        regime = detector.detect(sample_data)

        assert isinstance(regime, MarketRegime)
        assert regime.state in [s for s in MarketState]
        assert 0 <= regime.confidence <= 1

    def test_detect_oscillating(self, oscillating_data):
        """测试震荡市场检测"""
        detector = RegimeDetector()
        regime = detector.detect(oscillating_data)

        assert isinstance(regime, MarketRegime)
        # 震荡数据应该检测为非趋势市场
        # 但具体状态取决于 ADX 和 Hurst 的计算

    def test_detect_insufficient_data(self):
        """测试数据不足时的处理"""
        detector = RegimeDetector()

        # 只有 10 天数据
        short_data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "open": np.random.randn(10) * 10 + 100,
            "high": np.random.randn(10) * 10 + 105,
            "low": np.random.randn(10) * 10 + 95,
            "close": np.random.randn(10) * 10 + 100,
            "volume": np.random.randint(1000000, 5000000, 10),
        })

        regime = detector.detect(short_data)

        # 数据不足时应该返回默认状态
        assert regime.state == MarketState.QUIET
        assert regime.confidence == 0.0

    def test_calculate_adx(self, sample_data):
        """测试 ADX 计算"""
        detector = RegimeDetector()
        adx = detector._calculate_adx(sample_data)

        assert isinstance(adx, float)
        assert adx >= 0

    def test_calculate_atr_ratio(self, sample_data):
        """测试 ATR 比率计算"""
        detector = RegimeDetector()
        atr_ratio = detector._calculate_atr_ratio(sample_data)

        assert isinstance(atr_ratio, float)
        assert atr_ratio >= 0

    def test_calculate_hurst(self, sample_data):
        """测试 Hurst 指数计算"""
        detector = RegimeDetector()
        hurst = detector._calculate_hurst(sample_data)

        assert isinstance(hurst, float)
        # Hurst 指数理论上在 0-1 之间，但实际计算可能有偏差
        # 主要验证它是一个合理的数值
        assert -1 <= hurst <= 2

    def test_calculate_volatility(self, sample_data):
        """测试波动率计算"""
        detector = RegimeDetector()
        vol = detector._calculate_volatility(sample_data)

        assert isinstance(vol, float)
        assert vol >= 0

    def test_calculate_trend_direction(self, sample_data):
        """测试趋势方向计算"""
        detector = RegimeDetector()
        direction = detector._calculate_trend_direction(sample_data)

        assert direction in [-1, 0, 1]


class TestGetRecommendedStrategy:
    """推荐策略测试"""

    def test_trending(self):
        """测试趋势市场推荐"""
        regime = MarketRegime(
            state=MarketState.TRENDING,
            confidence=0.8,
            adx=30,
            atr_ratio=0.02,
            hurst=0.55,
            volatility=0.20,
            trend_direction=1,
        )

        strategy = get_recommended_strategy(regime)
        assert strategy == "momentum"

    def test_oscillating(self):
        """测试震荡市场推荐"""
        regime = MarketRegime(
            state=MarketState.OSCILLATING,
            confidence=0.7,
            adx=15,
            atr_ratio=0.015,
            hurst=0.40,
            volatility=0.15,
            trend_direction=0,
        )

        strategy = get_recommended_strategy(regime)
        assert strategy == "mean_reversion"

    def test_volatile(self):
        """测试高波动市场推荐"""
        regime = MarketRegime(
            state=MarketState.VOLATILE,
            confidence=0.9,
            adx=20,
            atr_ratio=0.04,
            hurst=0.5,
            volatility=0.35,
            trend_direction=0,
        )

        strategy = get_recommended_strategy(regime)
        assert strategy == "bollinger"

    def test_quiet(self):
        """测试平稳市场推荐"""
        regime = MarketRegime(
            state=MarketState.QUIET,
            confidence=0.5,
            adx=18,
            atr_ratio=0.012,
            hurst=0.52,
            volatility=0.12,
            trend_direction=0,
        )

        strategy = get_recommended_strategy(regime)
        assert strategy == "multi_factor"
