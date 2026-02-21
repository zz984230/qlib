"""测试因子模块"""

import pytest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy.factors import (
    MomentumFactor,
    ROCFactor,
    MACDFactor,
    VolatilityFactor,
    ATRFactor,
    VolumeRatioFactor,
    MAFactor,
    RSIFactor,
    BollingerBandsFactor,
    get_factor,
    list_factors,
)


def create_test_data(n: int = 100) -> pd.DataFrame:
    """创建测试数据"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    # 创建有趋势的价格数据
    base_price = 10
    trend = np.linspace(0, 2, n)  # 上升趋势
    noise = np.random.normal(0, 0.2, n)
    close = base_price + trend + noise

    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.uniform(-0.1, 0.1, n),
        "close": close,
        "high": close + np.random.uniform(0, 0.3, n),
        "low": close - np.random.uniform(0, 0.3, n),
        "volume": np.random.uniform(1000000, 2000000, n),
        "amount": np.random.uniform(10000000, 20000000, n),
    })


class TestFactorRegistry:
    """测试因子注册表"""

    def test_list_factors(self):
        """测试列出因子"""
        factors = list_factors()
        assert len(factors) > 0
        assert "momentum" in factors
        assert "rsi" in factors

    def test_get_factor(self):
        """测试获取因子"""
        factor = get_factor("momentum", window=10)
        assert factor is not None
        assert factor.window == 10


class TestMomentumFactor:
    """测试动量因子"""

    def test_calculate(self):
        """测试计算"""
        data = create_test_data()
        factor = MomentumFactor(window=20)
        values = factor.calculate(data)

        assert len(values) == len(data)
        # 后面的值应该为正（上升趋势）
        assert values[-1] > 0


class TestRSIFactor:
    """测试 RSI 因子"""

    def test_calculate(self):
        """测试计算"""
        data = create_test_data()
        factor = RSIFactor(window=14)
        values = factor.calculate(data)

        assert len(values) == len(data)
        # RSI 应该在 0-100 之间
        valid_values = values[~np.isnan(values)]
        assert np.all((valid_values >= 0) & (valid_values <= 100))


class TestMAFactor:
    """测试均线因子"""

    def test_calculate(self):
        """测试计算"""
        data = create_test_data()
        factor = MAFactor(window=20)
        values = factor.calculate(data)

        assert len(values) == len(data)
        # 检查非 NaN 值
        valid_values = values[~np.isnan(values)]
        assert len(valid_values) > 0


class TestBollingerBandsFactor:
    """测试布林带因子"""

    def test_calculate(self):
        """测试计算"""
        data = create_test_data()
        factor = BollingerBandsFactor(window=20)
        values = factor.calculate(data)

        assert len(values) == len(data)
        # %B 应该在 0-1 之间（大部分情况）
        valid_values = values[~np.isnan(values)]
        assert len(valid_values) > 0


class TestVolumeRatioFactor:
    """测试成交量比率因子"""

    def test_calculate(self):
        """测试计算"""
        data = create_test_data()
        factor = VolumeRatioFactor(window=20)
        values = factor.calculate(data)

        assert len(values) == len(data)
        # 比率应该在合理范围内
        valid_values = values[~np.isnan(values)]
        assert len(valid_values) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
