"""测试分析模块"""

import pytest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
    calculate_total_return,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_win_rate,
)
from src.analysis.visualizer import BacktestVisualizer


def create_test_portfolio(n: int = 252, seed: int = 42) -> pd.Series:
    """创建测试组合净值"""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    returns = np.random.normal(0.0005, 0.015, n)
    portfolio_value = pd.Series(1000000 * (1 + returns).cumprod(), index=dates)
    return portfolio_value


class TestPerformanceMetrics:
    """测试绩效指标"""

    def test_calculate_total_return(self):
        """测试总收益计算"""
        portfolio = create_test_portfolio()
        total_return = calculate_total_return(portfolio)
        assert -1 < total_return < 10  # 合理范围

    def test_calculate_max_drawdown(self):
        """测试最大回撤计算"""
        portfolio = create_test_portfolio()
        max_dd, duration = calculate_max_drawdown(portfolio)
        assert 0 <= max_dd < 1  # 回撤在 0-100%

    def test_calculate_sharpe_ratio(self):
        """测试夏普比率计算"""
        portfolio = create_test_portfolio()
        returns = portfolio.pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns)
        assert -10 < sharpe < 10  # 合理范围

    def test_calculate_win_rate(self):
        """测试胜率计算"""
        portfolio = create_test_portfolio()
        returns = portfolio.pct_change().dropna()
        win_rate = calculate_win_rate(returns)
        assert 0 <= win_rate <= 1

    def test_calculate_all_metrics(self):
        """测试计算所有指标"""
        portfolio = create_test_portfolio()
        metrics = calculate_all_metrics(portfolio)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_days > 0
        assert metrics.trading_days > 0
        assert metrics.total_return != 0
        assert metrics.sharpe_ratio != 0

    def test_calculate_all_metrics_with_benchmark(self):
        """测试带基准的指标计算"""
        portfolio = create_test_portfolio()
        benchmark = create_test_portfolio(seed=123)

        metrics = calculate_all_metrics(portfolio, benchmark=benchmark)

        assert metrics.benchmark_return != 0
        assert metrics.excess_return != 0
        assert metrics.beta != 0


class TestBacktestVisualizer:
    """测试可视化器"""

    def test_init(self):
        """测试初始化"""
        viz = BacktestVisualizer()
        assert viz is not None

    def test_plot_nav_curve(self):
        """测试净值曲线图"""
        portfolio = create_test_portfolio()
        viz = BacktestVisualizer()
        fig = viz.plot_nav_curve(portfolio)

        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_plot_returns_distribution(self):
        """测试收益分布图"""
        portfolio = create_test_portfolio()
        viz = BacktestVisualizer()
        fig = viz.plot_returns_distribution(portfolio)

        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_figure_to_bytes(self):
        """测试图表转字节流"""
        portfolio = create_test_portfolio()
        viz = BacktestVisualizer()
        fig = viz.plot_nav_curve(portfolio, show_drawdown=False)

        bytes_data = viz.figure_to_bytes(fig)
        assert len(bytes_data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
