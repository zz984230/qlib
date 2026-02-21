"""测试 AI Agent 模块"""

import pytest
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.prompts import (
    StrategyAnalysisPrompt,
    OptimizationSuggestionPrompt,
    StrategySearchPrompt,
)
from src.agent.tools import (
    analyze_backtest_result,
    suggest_optimizations,
    search_strategies,
)


class TestPrompts:
    """测试提示模板"""

    def test_strategy_analysis_prompt(self):
        """测试策略分析提示"""
        context = {
            "strategy_name": "test_strategy",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "cash": 1000000,
            "total_return": 0.15,
            "annual_return": 0.15,
            "max_drawdown": 0.08,
            "sharpe_ratio": 1.2,
            "win_rate": 0.55,
            "metrics": {"volatility": 0.12},
        }

        prompt = StrategyAnalysisPrompt.create(context)
        assert "test_strategy" in prompt
        assert "15.00%" in prompt
        assert "1.20" in prompt

    def test_optimization_suggestion_prompt(self):
        """测试优化建议提示"""
        context = {
            "strategy_type": "dual_ma",
            "current_params": {"short_window": 5, "long_window": 20},
            "diagnosis": "夏普比率低于目标",
            "max_factors": 10,
            "max_drawdown": 0.15,
            "min_sharpe": 1.0,
        }

        prompt = OptimizationSuggestionPrompt.create(context)
        assert "dual_ma" in prompt
        assert "short_window" in prompt

    def test_strategy_search_prompt(self):
        """测试策略搜索提示"""
        context = {
            "strategy_types": ["momentum", "value"],
            "existing_strategies": ["simple", "dual_ma"],
            "target_sharpe": 1.5,
            "max_drawdown": 0.15,
        }

        prompt = StrategySearchPrompt.create(context)
        assert "momentum" in prompt
        assert "value" in prompt


class TestAnalysisTools:
    """测试分析工具"""

    def test_analyze_backtest_result_positive(self):
        """测试分析正向结果"""
        result_data = {
            "strategy_name": "test",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "total_return": 0.20,
            "annual_return": 0.20,
            "max_drawdown": 0.08,
            "sharpe_ratio": 1.5,
            "metrics": {"win_rate": 0.55, "volatility": 0.12},
        }

        analysis = analyze_backtest_result(result_data)

        assert analysis["strategy_name"] == "test"
        assert analysis["performance"]["total_return"] == 0.20
        # 表现好应该没有高危问题
        high_issues = analysis["diagnosis"]["severity_summary"]["high"]
        assert high_issues == 0

    def test_analyze_backtest_result_negative(self):
        """测试分析负向结果"""
        result_data = {
            "strategy_name": "bad_strategy",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "total_return": -0.15,
            "annual_return": -0.15,
            "max_drawdown": 0.25,
            "sharpe_ratio": -0.5,
            "metrics": {"win_rate": 0.35, "volatility": 0.20},
        }

        analysis = analyze_backtest_result(result_data)

        # 应该有高危问题
        high_issues = analysis["diagnosis"]["severity_summary"]["high"]
        assert high_issues > 0

        # 应该有建议
        assert len(analysis["recommendations"]) > 0

    def test_suggest_optimizations(self):
        """测试生成优化建议"""
        analysis = {
            "diagnosis": {
                "issues": [
                    {"type": "high_drawdown", "severity": "high"},
                    {"type": "low_sharpe", "severity": "medium"},
                ]
            }
        }

        suggestions = suggest_optimizations("dual_ma", analysis)

        assert "parameter_adjustments" in suggestions
        assert "risk_controls" in suggestions
        assert "rationale" in suggestions

    def test_search_strategies(self):
        """测试策略搜索"""
        strategies = search_strategies(max_results=3)

        assert len(strategies) <= 3
        for s in strategies:
            assert "name" in s
            assert "type" in s
            assert "description" in s
            assert "factors" in s

    def test_search_strategies_by_type(self):
        """测试按类型搜索策略"""
        strategies = search_strategies(
            strategy_types=["momentum"],
            max_results=5
        )

        for s in strategies:
            assert s["type"] == "momentum"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
