"""优化模块测试"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.param_optimizer import ParameterOptimizer, OptimizationResult
from src.optimization.strategy_selector import StrategySelector, StrategyRanking
from src.optimization.auto_optimizer import AutoOptimizer, suggest_param_grid


class TestParameterOptimizer:
    """参数优化器测试"""

    def test_init(self):
        """测试初始化"""
        optimizer = ParameterOptimizer(method="grid")
        assert optimizer.method == "grid"
        assert optimizer.metric == "sharpe_ratio"

    def test_invalid_method(self):
        """测试无效方法"""
        with pytest.raises(ValueError):
            ParameterOptimizer(method="invalid")

    def test_grid_search_basic(self):
        """测试网格搜索基础功能"""
        from src.strategy.advanced import DualMAStrategy

        optimizer = ParameterOptimizer(method="grid", max_evaluations=10)

        param_grid = {
            "short_window": [5, 10],
            "long_window": [20, 30],
        }

        # 使用空数据测试（会使用简单回测）
        data = pd.DataFrame()

        result = optimizer.optimize(
            DualMAStrategy,
            param_grid,
            data,
            "2023-01-01",
            "2023-03-31",
            100000,
        )

        assert isinstance(result, OptimizationResult)
        assert result.method == "grid"
        assert "short_window" in result.best_params
        assert "long_window" in result.best_params

    def test_random_search(self):
        """测试随机搜索"""
        from src.strategy.advanced import RSIStrategy

        optimizer = ParameterOptimizer(method="random", max_evaluations=5)

        param_grid = {
            "window": [7, 14, 21],
            "oversold": [25, 30],
            "overbought": [70, 75],
        }

        result = optimizer.optimize(
            RSIStrategy,
            param_grid,
            pd.DataFrame(),
            "2023-01-01",
            "2023-03-31",
        )

        assert isinstance(result, OptimizationResult)
        assert result.n_evaluations <= 5


class TestStrategySelector:
    """策略选择器测试"""

    def test_init(self):
        """测试初始化"""
        selector = StrategySelector()
        assert "sharpe_ratio" in selector.scoring_weights

    def test_calculate_score(self):
        """测试得分计算"""
        selector = StrategySelector()

        metrics = {
            "sharpe_ratio": 1.5,
            "annual_return": 0.20,
            "max_drawdown": 0.10,
            "volatility": 0.15,
        }

        score = selector._calculate_score(metrics)
        assert isinstance(score, float)

        # 高夏普、高收益、低回撤、低波动应该得高分
        assert score > 0

    def test_compare_strategies(self):
        """测试策略比较"""
        from src.strategy.advanced import DualMAStrategy, RSIStrategy

        selector = StrategySelector()

        strategies = [
            DualMAStrategy(short_window=5, long_window=20),
            RSIStrategy(window=14),
        ]

        rankings = selector.compare_strategies(
            strategies,
            "2023-01-01",
            "2023-03-31",
            100000,
        )

        assert len(rankings) <= 2
        for ranking in rankings:
            assert isinstance(ranking, StrategyRanking)
            assert ranking.rank >= 1

    def test_select_best(self):
        """测试选择最佳策略"""
        selector = StrategySelector()

        rankings = [
            StrategyRanking(
                strategy_name="strategy1",
                rank=1,
                composite_score=0.8,
                metrics={},
            ),
            StrategyRanking(
                strategy_name="strategy2",
                rank=2,
                composite_score=0.6,
                metrics={},
            ),
        ]

        best = selector.select_best(rankings, top_n=1)
        assert len(best) == 1
        assert best[0].strategy_name == "strategy1"

    def test_min_score_filter(self):
        """测试最低得分过滤"""
        selector = StrategySelector()

        rankings = [
            StrategyRanking(
                strategy_name="strategy1",
                rank=1,
                composite_score=0.8,
                metrics={},
            ),
            StrategyRanking(
                strategy_name="strategy2",
                rank=2,
                composite_score=0.3,
                metrics={},
            ),
        ]

        filtered = selector.select_best(rankings, top_n=2, min_score=0.5)
        assert len(filtered) == 1
        assert filtered[0].strategy_name == "strategy1"


class TestAutoOptimizer:
    """自动优化器测试"""

    def test_init(self):
        """测试初始化"""
        optimizer = AutoOptimizer()
        assert optimizer.optimization_method == "bayesian"
        assert optimizer.oos_ratio == 0.2

    def test_split_dates(self):
        """测试日期分割"""
        optimizer = AutoOptimizer(oos_ratio=0.2)

        is_start, is_end, oos_start, oos_end = optimizer._split_dates(
            "2023-01-01", "2023-12-31"
        )

        assert is_start == "2023-01-01"
        assert is_end < oos_start
        assert oos_end == "2023-12-31"

    def test_check_overfitting(self):
        """测试过拟合检查"""
        optimizer = AutoOptimizer(overfitting_threshold=0.5)

        # 无过拟合
        is_overfitted, degradation = optimizer._check_overfitting(1.0, 0.9)
        assert not is_overfitted
        assert degradation < 0.2

        # 有过拟合
        is_overfitted, degradation = optimizer._check_overfitting(1.0, 0.3)
        assert is_overfitted
        assert degradation > 0.5

    def test_generate_recommendation(self):
        """测试建议生成"""
        optimizer = AutoOptimizer()

        # 过拟合情况
        rec = optimizer._generate_recommendation(
            True, 0.6, {"window": 20}, {"sharpe_ratio": 0.3}
        )
        assert "过拟合" in rec

        # 良好情况
        rec = optimizer._generate_recommendation(
            False, 0.1, {"window": 20}, {"sharpe_ratio": 1.5}
        )
        assert "优秀" in rec or "良好" in rec


class TestSuggestParamGrid:
    """参数网格推荐测试"""

    def test_dual_ma(self):
        """测试双均线策略参数"""
        grid = suggest_param_grid("dual_ma")
        assert "short_window" in grid
        assert "long_window" in grid

    def test_rsi(self):
        """测试 RSI 策略参数"""
        grid = suggest_param_grid("rsi")
        assert "window" in grid
        assert "oversold" in grid
        assert "overbought" in grid

    def test_unknown_strategy(self):
        """测试未知策略"""
        grid = suggest_param_grid("unknown_strategy")
        assert grid == {}
