"""
参数优化器

支持网格搜索、随机搜索和贝叶斯优化
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.runner import BacktestRunner
from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """优化结果"""

    best_params: dict[str, Any]
    best_score: float
    best_metrics: dict[str, float]
    all_results: list[dict[str, Any]] = field(default_factory=list)
    optimization_time: float = 0.0
    method: str = ""
    n_evaluations: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ParameterOptimizer:
    """
    参数优化器

    支持多种优化方法:
    - grid: 网格搜索
    - random: 随机搜索
    - bayesian: 贝叶斯优化
    """

    METHODS = ["grid", "random", "bayesian"]

    def __init__(
        self,
        method: str = "bayesian",
        metric: str = "sharpe_ratio",
        maximize: bool = True,
        max_evaluations: int = 50,
    ):
        """
        初始化参数优化器

        Args:
            method: 优化方法
            metric: 优化目标指标
            maximize: 是否最大化指标
            max_evaluations: 最大评估次数
        """
        if method not in self.METHODS:
            raise ValueError(f"未知方法: {method}. 可用方法: {self.METHODS}")

        self.method = method
        self.metric = metric
        self.maximize = maximize
        self.max_evaluations = max_evaluations

    def optimize(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        cash: float = 1000000,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> OptimizationResult:
        """
        执行参数优化

        Args:
            strategy_class: 策略类
            param_grid: 参数网格 {param_name: [possible_values]}
            data: 回测数据
            start_date: 开始日期
            end_date: 结束日期
            cash: 初始资金
            progress_callback: 进度回调函数

        Returns:
            OptimizationResult
        """
        start_time = datetime.now()
        all_results = []

        logger.info(f"开始参数优化: method={self.method}, metric={self.metric}")
        logger.info(f"参数网格: {param_grid}")

        if self.method == "grid":
            results = self._grid_search(
                strategy_class,
                param_grid,
                data,
                start_date,
                end_date,
                cash,
                progress_callback,
            )
        elif self.method == "random":
            results = self._random_search(
                strategy_class,
                param_grid,
                data,
                start_date,
                end_date,
                cash,
                progress_callback,
            )
        else:  # bayesian
            results = self._bayesian_optimization(
                strategy_class,
                param_grid,
                data,
                start_date,
                end_date,
                cash,
                progress_callback,
            )

        all_results = results

        # 找到最佳参数
        if self.maximize:
            best_result = max(all_results, key=lambda x: x["score"])
        else:
            best_result = min(all_results, key=lambda x: x["score"])

        optimization_time = (datetime.now() - start_time).total_seconds()

        return OptimizationResult(
            best_params=best_result["params"],
            best_score=best_result["score"],
            best_metrics=best_result["metrics"],
            all_results=all_results,
            optimization_time=optimization_time,
            method=self.method,
            n_evaluations=len(all_results),
        )

    def _grid_search(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        cash: float,
        progress_callback: Optional[Callable],
    ) -> list[dict]:
        """网格搜索"""
        from itertools import product

        results = []
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        total = len(combinations)
        logger.info(f"网格搜索: {total} 种参数组合")

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            try:
                score, metrics = self._evaluate_params(
                    strategy_class, params, data, start_date, end_date, cash
                )
                results.append({
                    "params": params,
                    "score": score,
                    "metrics": metrics,
                    "iteration": i + 1,
                })
            except Exception as e:
                logger.warning(f"参数评估失败: {params}, 错误: {e}")
                results.append({
                    "params": params,
                    "score": float("-inf") if self.maximize else float("inf"),
                    "metrics": {},
                    "iteration": i + 1,
                    "error": str(e),
                })

            if progress_callback:
                progress_callback(i + 1, total, results[-1]["score"])

        return results

    def _random_search(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        cash: float,
        progress_callback: Optional[Callable],
    ) -> list[dict]:
        """随机搜索"""
        results = []
        n_samples = min(self.max_evaluations, 100)

        logger.info(f"随机搜索: {n_samples} 次采样")

        for i in range(n_samples):
            # 随机采样参数
            params = {}
            for name, values in param_grid.items():
                params[name] = np.random.choice(values)

            try:
                score, metrics = self._evaluate_params(
                    strategy_class, params, data, start_date, end_date, cash
                )
                results.append({
                    "params": params,
                    "score": score,
                    "metrics": metrics,
                    "iteration": i + 1,
                })
            except Exception as e:
                logger.warning(f"参数评估失败: {params}, 错误: {e}")
                results.append({
                    "params": params,
                    "score": float("-inf") if self.maximize else float("inf"),
                    "metrics": {},
                    "iteration": i + 1,
                    "error": str(e),
                })

            if progress_callback:
                progress_callback(i + 1, n_samples, results[-1]["score"])

        return results

    def _bayesian_optimization(
        self,
        strategy_class: type[BaseStrategy],
        param_grid: dict[str, list[Any]],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        cash: float,
        progress_callback: Optional[Callable],
    ) -> list[dict]:
        """贝叶斯优化"""
        results = []

        try:
            from skopt import gp_minimize
            from skopt.space import Categorical, Integer, Real
        except ImportError:
            logger.warning("scikit-optimize 未安装，使用随机搜索替代")
            return self._random_search(
                strategy_class,
                param_grid,
                data,
                start_date,
                end_date,
                cash,
                progress_callback,
            )

        # 构建搜索空间
        space = []
        param_names = []
        param_types = []

        for name, values in param_grid.items():
            param_names.append(name)
            if isinstance(values[0], int):
                space.append(Integer(min(values), max(values), name=name))
                param_types.append("int")
            elif isinstance(values[0], float):
                space.append(Real(min(values), max(values), name=name))
                param_types.append("float")
            else:
                space.append(Categorical(values, name=name))
                param_types.append("categorical")

        def objective(params_list):
            params = {}
            for i, name in enumerate(param_names):
                if param_types[i] == "int":
                    params[name] = int(params_list[i])
                else:
                    params[name] = params_list[i]

            try:
                score, _ = self._evaluate_params(
                    strategy_class, params, data, start_date, end_date, cash
                )
                # skopt 最小化，所以如果是最大化目标，取负
                return -score if self.maximize else score
            except Exception:
                return float("inf") if self.maximize else float("-inf")

        # 运行优化
        n_initial = min(10, self.max_evaluations // 3)
        result = gp_minimize(
            objective,
            space,
            n_calls=self.max_evaluations,
            n_initial_points=n_initial,
            random_state=42,
        )

        # 提取所有评估结果
        for i, (params_list, score) in enumerate(zip(result.x_iters, result.func_vals)):
            params = {}
            for j, name in enumerate(param_names):
                if param_types[j] == "int":
                    params[name] = int(params_list[j])
                else:
                    params[name] = params_list[j]

            # 重新评估获取完整指标
            try:
                actual_score, metrics = self._evaluate_params(
                    strategy_class, params, data, start_date, end_date, cash
                )
            except Exception:
                actual_score = -score if self.maximize else score
                metrics = {}

            results.append({
                "params": params,
                "score": actual_score,
                "metrics": metrics,
                "iteration": i + 1,
            })

            if progress_callback:
                progress_callback(i + 1, self.max_evaluations, actual_score)

        return results

    def _evaluate_params(
        self,
        strategy_class: type[BaseStrategy],
        params: dict[str, Any],
        data: pd.DataFrame,
        start_date: str,
        end_date: str,
        cash: float,
    ) -> tuple[float, dict[str, float]]:
        """评估参数组合"""
        strategy = strategy_class(**params)
        runner = BacktestRunner()
        result = runner.run(strategy, start_date, end_date, cash)

        score = getattr(result, self.metric, result.sharpe_ratio)
        metrics = result.get_all_metrics()

        return score, metrics
