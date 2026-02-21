"""
自动优化循环

执行完整的策略优化流程，包括样本外验证和过拟合检查
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.backtest.runner import BacktestRunner
from src.optimization.param_optimizer import ParameterOptimizer, OptimizationResult
from src.optimization.strategy_selector import StrategySelector, StrategyRanking
from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class OptimizationCycleResult:
    """优化循环结果"""

    strategy_name: str
    best_params: dict[str, Any]
    in_sample_metrics: dict[str, float]
    out_of_sample_metrics: dict[str, float]
    is_overfitted: bool
    degradation: float  # OOS 相对 IS 的退化程度
    recommendation: str
    optimization_details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AutoOptimizer:
    """
    自动优化器

    执行完整的优化流程:
    1. 数据分割 (样本内/样本外)
    2. 样本内参数优化
    3. 样本外验证
    4. 过拟合检查
    5. 生成建议
    """

    def __init__(
        self,
        optimization_method: str = "bayesian",
        max_evaluations: int = 50,
        oos_ratio: float = 0.2,
        overfitting_threshold: float = 0.5,  # OOS/IS < 0.5 视为过拟合
    ):
        """
        初始化自动优化器

        Args:
            optimization_method: 优化方法
            max_evaluations: 最大评估次数
            oos_ratio: 样本外比例
            overfitting_threshold: 过拟合阈值
        """
        self.optimization_method = optimization_method
        self.max_evaluations = max_evaluations
        self.oos_ratio = oos_ratio
        self.overfitting_threshold = overfitting_threshold

    def run_cycle(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        cash: float = 1000000,
        dry_run: bool = True,
    ) -> OptimizationCycleResult:
        """
        执行优化循环

        Args:
            strategy_class: 策略类
            param_grid: 参数网格
            data: 回测数据
            start_date: 开始日期
            end_date: 结束日期
            cash: 初始资金
            dry_run: 是否为试运行（不实际应用）

        Returns:
            OptimizationCycleResult
        """
        logger.info(f"开始优化循环: {strategy_class.__name__}")
        logger.info(f"参数网格: {param_grid}")

        # 1. 分割日期
        is_start, is_end, oos_start, oos_end = self._split_dates(
            start_date, end_date
        )

        logger.info(f"样本内: {is_start} -> {is_end}")
        logger.info(f"样本外: {oos_start} -> {oos_end}")

        # 2. 样本内优化
        optimizer = ParameterOptimizer(
            method=self.optimization_method,
            max_evaluations=self.max_evaluations,
        )

        is_result = optimizer.optimize(
            strategy_class,
            param_grid,
            data,
            is_start,
            is_end,
            cash,
        )

        logger.info(
            f"样本内最佳参数: {is_result.best_params}, "
            f"夏普: {is_result.best_metrics.get('sharpe_ratio', 0):.2f}"
        )

        # 3. 样本外验证
        best_strategy = strategy_class(**is_result.best_params)
        runner = BacktestRunner()
        oos_backtest = runner.run(best_strategy, oos_start, oos_end, cash)
        oos_metrics = oos_backtest.get_all_metrics()

        logger.info(
            f"样本外验证: 夏普={oos_metrics.get('sharpe_ratio', 0):.2f}, "
            f"收益={oos_metrics.get('annual_return', 0):.1%}"
        )

        # 4. 过拟合检查
        is_sharpe = is_result.best_metrics.get("sharpe_ratio", 0)
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0)

        is_overfitted, degradation = self._check_overfitting(
            is_sharpe, oos_sharpe
        )

        # 5. 生成建议
        recommendation = self._generate_recommendation(
            is_overfitted, degradation, is_result.best_params, oos_metrics
        )

        return OptimizationCycleResult(
            strategy_name=strategy_class.__name__,
            best_params=is_result.best_params,
            in_sample_metrics=is_result.best_metrics,
            out_of_sample_metrics=oos_metrics,
            is_overfitted=is_overfitted,
            degradation=degradation,
            recommendation=recommendation,
            optimization_details={
                "optimization_method": self.optimization_method,
                "n_evaluations": is_result.n_evaluations,
                "optimization_time": is_result.optimization_time,
                "oos_ratio": self.oos_ratio,
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
            },
        )

    def run_multi_strategy_cycle(
        self,
        strategies: list[tuple[type[BaseStrategy], dict[str, list[Any]]]],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        cash: float = 1000000,
    ) -> list[OptimizationCycleResult]:
        """
        对多个策略执行优化循环

        Args:
            strategies: [(策略类, 参数网格), ...]
            data: 回测数据
            start_date: 开始日期
            end_date: 结束日期
            cash: 初始资金

        Returns:
            优化结果列表
        """
        results = []

        for strategy_class, param_grid in strategies:
            try:
                result = self.run_cycle(
                    strategy_class,
                    param_grid,
                    data,
                    start_date,
                    end_date,
                    cash,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"策略 {strategy_class.__name__} 优化失败: {e}")

        # 按样本外夏普排序
        results.sort(
            key=lambda x: x.out_of_sample_metrics.get("sharpe_ratio", 0),
            reverse=True,
        )

        return results

    def _split_dates(
        self,
        start_date: str,
        end_date: str,
    ) -> tuple[str, str, str, str]:
        """分割日期为样本内和样本外"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        total_days = (end - start).days
        oos_days = int(total_days * self.oos_ratio)

        is_end = end - pd.Timedelta(days=oos_days)
        oos_start = is_end + pd.Timedelta(days=1)

        return (
            start_date,
            str(is_end.date()),
            str(oos_start.date()),
            end_date,
        )

    def _check_overfitting(
        self,
        is_sharpe: float,
        oos_sharpe: float,
    ) -> tuple[bool, float]:
        """
        检查过拟合

        Returns:
            (是否过拟合, 退化程度)
        """
        if is_sharpe <= 0:
            return False, 0.0

        degradation = 1.0 - (oos_sharpe / is_sharpe)
        degradation = max(0, degradation)  # 负值表示 OOS 更好

        is_overfitted = (oos_sharpe / is_sharpe) < self.overfitting_threshold

        return is_overfitted, degradation

    def _generate_recommendation(
        self,
        is_overfitted: bool,
        degradation: float,
        best_params: dict[str, Any],
        oos_metrics: dict[str, float],
    ) -> str:
        """生成优化建议"""
        if is_overfitted:
            return (
                f"警告: 策略可能过拟合！样本外表现退化 {degradation:.1%}。"
                f"建议: 减少参数数量，增加正则化，或使用更简单的策略。"
            )

        if degradation > 0.3:
            return (
                f"注意: 样本外表现有一定退化 ({degradation:.1%})。"
                f"建议: 谨慎使用，建议进一步验证。"
            )

        oos_sharpe = oos_metrics.get("sharpe_ratio", 0)
        oos_return = oos_metrics.get("annual_return", 0)

        if oos_sharpe >= 1.5:
            return (
                f"优秀: 样本外夏普比率 {oos_sharpe:.2f}，年化收益 {oos_return:.1%}。"
                f"推荐参数: {best_params}。可以考虑实盘应用。"
            )
        elif oos_sharpe >= 1.0:
            return (
                f"良好: 样本外夏普比率 {oos_sharpe:.2f}。"
                f"参数 {best_params} 验证通过，可以继续观察。"
            )
        else:
            return (
                f"一般: 样本外夏普比率 {oos_sharpe:.2f}。"
                f"建议: 继续优化或尝试其他策略。"
            )


def suggest_param_grid(strategy_name: str) -> dict[str, list[Any]]:
    """
    为策略推荐参数网格

    Args:
        strategy_name: 策略名称

    Returns:
        推荐的参数网格
    """
    grids = {
        "dual_ma": {
            "short_window": [3, 5, 7, 10, 15],
            "long_window": [15, 20, 30, 40, 60],
        },
        "mean_reversion": {
            "window": [10, 15, 20, 30],
            "entry_z": [1.5, 2.0, 2.5, 3.0],
            "exit_z": [0.3, 0.5, 0.7],
        },
        "rsi": {
            "window": [7, 10, 14, 21],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
        },
        "bollinger": {
            "window": [15, 20, 25, 30],
            "num_std": [1.5, 2.0, 2.5],
        },
        "breakout": {
            "window": [10, 15, 20, 30, 40],
        },
        "oscillation": {
            "rsi_window": [10, 14, 21],
            "rsi_oversold": [25, 30],
            "rsi_overbought": [70, 75],
            "min_confirmations": [2, 3],
        },
    }

    return grids.get(strategy_name, {})
