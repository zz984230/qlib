"""测试策略模块"""

import pytest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.base import BaseStrategy, SimpleFactorStrategy, MomentumStrategy


class TestSimpleFactorStrategy:
    """测试 SimpleFactorStrategy"""

    def test_init(self):
        """测试初始化"""
        strategy = SimpleFactorStrategy()
        assert strategy is not None

    def test_get_factors(self):
        """测试获取因子"""
        strategy = SimpleFactorStrategy()
        factors = strategy.get_factors()

        assert isinstance(factors, list)
        assert len(factors) > 0

    def test_get_model_config(self):
        """测试获取模型配置"""
        strategy = SimpleFactorStrategy()
        config = strategy.get_model_config()

        assert "class" in config
        assert config["class"] == "LGBModel"

    def test_get_strategy_config(self):
        """测试获取策略配置"""
        strategy = SimpleFactorStrategy()
        config = strategy.get_strategy_config()

        assert "topk" in config
        assert "n_drop" in config

    def test_generate_signals(self):
        """测试生成信号"""
        strategy = SimpleFactorStrategy()

        # 创建测试数据
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        data = pd.DataFrame({
            "date": dates,
            "open": np.random.uniform(10, 11, 100),
            "close": np.random.uniform(10, 11, 100),
            "high": np.random.uniform(10.5, 11.5, 100),
            "low": np.random.uniform(9.5, 10.5, 100),
            "volume": np.random.uniform(1000000, 2000000, 100),
        })

        signals = strategy.generate_signals(data)

        assert signals is not None
        assert len(signals) == len(data)


class TestMomentumStrategy:
    """测试 MomentumStrategy"""

    def test_init(self):
        """测试初始化"""
        strategy = MomentumStrategy(lookback_period=10)
        assert strategy is not None
        assert strategy.lookback_period == 10

    def test_generate_signals(self):
        """测试生成动量信号"""
        strategy = MomentumStrategy(lookback_period=5)

        # 创建有趋势的测试数据
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        close_prices = np.linspace(10, 15, 50)  # 上升趋势
        data = pd.DataFrame({
            "date": dates,
            "close": close_prices,
            "volume": np.ones(50) * 1000000,
        })

        signals = strategy.generate_signals(data)

        assert signals is not None
        assert len(signals) == len(data)
        # 后面的信号应该为正（上升趋势）
        assert signals[-1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
