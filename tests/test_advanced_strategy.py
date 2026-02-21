"""测试高级策略"""

import pytest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.advanced import (
    MultiFactorStrategy,
    DualMAStrategy,
    MeanReversionStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    BreakoutStrategy,
    get_strategy,
    list_strategies,
)


def create_test_data(n: int = 100) -> pd.DataFrame:
    """创建测试数据"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    base_price = 10
    trend = np.linspace(0, 2, n)
    noise = np.random.normal(0, 0.2, n)
    close = base_price + trend + noise

    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.uniform(-0.1, 0.1, n),
        "close": close,
        "high": close + np.random.uniform(0, 0.3, n),
        "low": close - np.random.uniform(0, 0.3, n),
        "volume": np.random.uniform(1000000, 2000000, n),
    })


class TestStrategyRegistry:
    """测试策略注册表"""

    def test_list_strategies(self):
        """测试列出策略"""
        strategies = list_strategies()
        assert len(strategies) >= 6
        assert "dual_ma" in strategies
        assert "rsi" in strategies

    def test_get_strategy(self):
        """测试获取策略"""
        strategy = get_strategy("dual_ma", short_window=5, long_window=20)
        assert strategy is not None
        assert strategy.short_window == 5


class TestDualMAStrategy:
    """测试双均线策略"""

    def test_init(self):
        """测试初始化"""
        strategy = DualMAStrategy(short_window=5, long_window=20)
        assert strategy.short_window == 5
        assert strategy.long_window == 20

    def test_generate_signals(self):
        """测试生成信号"""
        data = create_test_data()
        strategy = DualMAStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(data)

        assert len(signals) == len(data)
        # 信号应该在 -1, 0, 1 范围内
        assert np.all(np.isin(signals, [-1, 0, 1]))


class TestMeanReversionStrategy:
    """测试均值回归策略"""

    def test_generate_signals(self):
        """测试生成信号"""
        data = create_test_data()
        strategy = MeanReversionStrategy(window=20, entry_z=2.0)
        signals = strategy.generate_signals(data)

        assert len(signals) == len(data)


class TestRSIStrategy:
    """测试 RSI 策略"""

    def test_generate_signals(self):
        """测试生成信号"""
        data = create_test_data()
        strategy = RSIStrategy(window=14, oversold=30, overbought=70)
        signals = strategy.generate_signals(data)

        assert len(signals) == len(data)


class TestMultiFactorStrategy:
    """测试多因子策略"""

    def test_generate_signals(self):
        """测试生成信号"""
        data = create_test_data()
        strategy = MultiFactorStrategy()
        signals = strategy.generate_signals(data)

        assert len(signals) == len(data)


class TestBreakoutStrategy:
    """测试突破策略"""

    def test_generate_signals(self):
        """测试生成信号"""
        data = create_test_data()
        strategy = BreakoutStrategy(window=20)
        signals = strategy.generate_signals(data)

        assert len(signals) == len(data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
