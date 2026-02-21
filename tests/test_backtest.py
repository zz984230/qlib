"""测试回测模块"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.runner import BacktestRunner, BacktestResult
from src.strategy.base import SimpleFactorStrategy


class TestBacktestRunner:
    """测试 BacktestRunner"""

    def test_init(self):
        """测试初始化"""
        runner = BacktestRunner()
        assert runner is not None

    def test_run_simple_backtest(self):
        """测试简单回测"""
        runner = BacktestRunner()
        strategy = SimpleFactorStrategy()

        result = runner.run(
            strategy=strategy,
            start_date="2024-01-01",
            end_date="2024-03-31",
            cash=1000000,
        )

        assert result is not None
        assert result.start_date == "2024-01-01"
        assert result.end_date == "2024-03-31"


class TestBacktestResult:
    """测试 BacktestResult"""

    def test_properties(self):
        """测试结果属性"""
        result = BacktestResult()

        # 初始状态
        assert result.total_return == 0.0
        assert result.annual_return == 0.0
        assert result.max_drawdown == 0.0
        assert result.sharpe_ratio == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
