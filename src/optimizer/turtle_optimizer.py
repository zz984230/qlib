"""海龟遗传算法主优化器

整合所有模块，实现完整的优化流程。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.optimizer.genetic_engine import GeneticEngine, GeneticConfig
from src.optimizer.fitness import FitnessEvaluator, FitnessConfig
from src.optimizer.individual import Individual
from src.optimizer.strategy_pool import StrategyPool
from src.backtest.multi_period import MultiPeriodBacktester
from src.data.turtle_data_loader import TurtleDataLoader
from src.report.html_generator import HtmlReportGenerator
from src.optimizer.interaction import (
    confirm_continue,
    display_strategy_summary,
    display_progress,
    display_generation_summary,
    should_pause_on_valid_strategy,
    display_final_report,
    confirm_run_optimization,
)

logger = logging.getLogger(__name__)


class TurtleGeneticOptimizer:
    """海龟遗传算法主优化器

    完整的优化流程：
    1. 数据加载
    2. 遗传算法演化
    3. 多周期回测
    4. 有效策略检测
    5. 策略池保存
    6. HTML 报告生成
    """

    def __init__(
        self,
        symbol: str,
        name: str,
        initial_cash: float = 50000,
        output_dir: str = "reports/html",
        pool_dir: str = "strategies/pool",
        genetic_config: Optional[GeneticConfig] = None,
        fitness_config: Optional[FitnessConfig] = None,
        interactive: bool = True,
    ):
        """初始化优化器

        Args:
            symbol: 股票代码
            name: 股票名称
            initial_cash: 初始资金
            output_dir: 报告输出目录
            pool_dir: 策略池目录
            genetic_config: 遗传算法配置
            fitness_config: 适应度评估配置
            interactive: 是否交互式运行
        """
        self.symbol = symbol
        self.name = name
        self.cash = initial_cash
        self.output_dir = Path(output_dir)
        self.pool_dir = Path(pool_dir)
        self.interactive = interactive

        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # 初始化各模块
        self.genetic_config = genetic_config or GeneticConfig()
        self.fitness_config = fitness_config or FitnessConfig()

        self.genetic_engine = GeneticEngine(self.genetic_config)
        self.fitness_evaluator = FitnessEvaluator(self.fitness_config)
        self.strategy_pool = StrategyPool(str(self.pool_dir))
        self.multi_period_backtester = MultiPeriodBacktester(symbol, initial_cash)
        self.report_generator = HtmlReportGenerator()
        self.data_loader = TurtleDataLoader()

        # 演化历史
        self.evolution_history = []
        self.valid_strategies_found = []
        self.data = None

    def run(self, start_date: Optional[str] = None) -> dict:
        """执行优化主流程

        Args:
            start_date: 数据开始日期

        Returns:
            优化结果摘要
        """
        logger.info(f"开始海龟遗传算法优化: {self.symbol} - {self.name}")

        # 1. 确认运行
        if self.interactive and not confirm_run_optimization(self.symbol, self.cash):
            logger.info("用户取消优化")
            return {"status": "cancelled"}

        # 2. 加载数据
        logger.info("正在加载数据...")
        self.data = self.data_loader.load_stock_data(self.symbol, start_date=start_date)

        if self.data is None or len(self.data) < 400:
            logger.error(f"数据不足: {len(self.data) if self.data is not None else 0} 条")
            return {"status": "error", "message": "数据不足"}

        logger.info(f"数据加载完成: {len(self.data)} 条")

        # 3. 初始化种群
        logger.info("初始化种群...")
        population = self.genetic_engine.initialize_population()

        # 4. 开始演化
        generation = 0
        should_continue = True

        while should_continue:
            generation += 1

            # 评估适应度
            logger.info(f"评估第 {generation} 代适应度...")
            for individual in population:
                if individual.fitness == 0:
                    individual.fitness = self._evaluate_individual(individual)

            # 检查有效策略
            valid_strategies = self._find_valid_strategies(population)

            if valid_strategies:
                self.valid_strategies_found.extend(valid_strategies)

                # 保存有效策略到策略池（交互和非交互模式都保存）
                self._save_valid_strategies(valid_strategies, generation)

                # 处理有效策略（交互模式下等待用户确认）
                if self.interactive:
                    should_continue = self._handle_valid_strategies(
                        valid_strategies, generation, population
                    )

                    if should_continue == "stop":
                        break
                    elif should_continue == "adjust":
                        # 这里可以添加参数调整逻辑
                        pass

            # 演化下一代
            population = self.genetic_engine.evolve(
                population,
                generation,
                lambda ind: ind.fitness  # 使用已评估的适应度
            )

            # 显示进度
            if self.interactive:
                display_progress(
                    generation,
                    self.genetic_config.max_generations,
                    self.genetic_engine.best_fitness_history[-1] if self.genetic_engine.best_fitness_history else 0,
                    self.genetic_engine.avg_fitness_history[-1] if self.genetic_engine.avg_fitness_history else 0,
                    len(self.valid_strategies_found),
                )

            # 记录演化历史
            self.evolution_history.append({
                "generation": generation,
                "best_fitness": self.genetic_engine.best_fitness_history[-1] if self.genetic_engine.best_fitness_history else 0,
                "avg_fitness": self.genetic_engine.avg_fitness_history[-1] if self.genetic_engine.avg_fitness_history else 0,
                "valid_count": len(self.valid_strategies_found),
            })

            # 检查停止条件
            if self.genetic_engine.should_stop(generation):
                logger.info(f"达到停止条件: 代数 {generation}")
                break

        # 5. 生成最终报告
        logger.info("生成最终报告...")
        best_individual = self.genetic_engine.get_best_individual(population)

        report_path = self._generate_final_report(
            generation,
            population,
            self.valid_strategies_found,
        )

        # 6. 显示最终摘要
        if self.interactive:
            display_final_report(
                generation,
                len(self.valid_strategies_found),
                best_individual.fitness,
                report_path,
            )

        return {
            "status": "completed",
            "symbol": self.symbol,
            "generations": generation,
            "valid_strategies": len(self.valid_strategies_found),
            "best_fitness": best_individual.fitness,
            "report_path": report_path,
        }

    def _evaluate_individual(self, individual: Individual) -> float:
        """评估个体适应度

        Args:
            individual: 待评估个体

        Returns:
            适应度值
        """
        try:
            # 执行多周期回测
            backtest_results = self.multi_period_backtester.run(
                self.data,
                individual,
                periods=["1y", "3m", "1m"]
            )

            # 存储回测结果（包含详细信息和交易记录）
            individual.backtest_results = {}
            for period, result in backtest_results.items():
                # 将基准净值序列转换为 dict（日期字符串 -> 净值）
                benchmark_series = getattr(result, 'benchmark_series', None)
                if benchmark_series is not None and hasattr(benchmark_series, 'to_dict'):
                    # 将日期索引转换为字符串格式
                    benchmark_dict = {}
                    for idx, val in benchmark_series.items():
                        if hasattr(idx, 'strftime'):
                            benchmark_dict[idx.strftime('%Y-%m-%d')] = float(val)
                        else:
                            benchmark_dict[str(idx)[:10]] = float(val)
                else:
                    benchmark_dict = {}

                # 将策略净值序列转换为 dict（日期字符串 -> 净值）
                strategy_series = result.portfolio_value
                if strategy_series is not None and hasattr(strategy_series, 'to_dict'):
                    strategy_dict = {}
                    for idx, val in strategy_series.items():
                        if hasattr(idx, 'strftime'):
                            strategy_dict[idx.strftime('%Y-%m-%d')] = float(val)
                        else:
                            strategy_dict[str(idx)[:10]] = float(val)
                else:
                    strategy_dict = {}

                individual.backtest_results[period] = {
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "sharpe_ratio": result.sharpe_ratio,
                    "annual_return": result.annual_return,
                    # 周期信息
                    "period_info": getattr(result, 'period_info', {}),
                    # 交易记录
                    "trades": getattr(result, 'trades_list', []),
                    # 基准净值序列（已转换为字符串日期 -> 净值的 dict）
                    "benchmark_series": benchmark_dict,
                    # 策略净值序列（已转换为字符串日期 -> 净值的 dict）
                    "strategy_series": strategy_dict,
                }

            # 计算适应度
            fitness = self.fitness_evaluator.evaluate_backtest_based(
                individual, backtest_results
            )

            return fitness

        except Exception as e:
            logger.warning(f"评估个体失败: {e}")
            return 0.0

    def _find_valid_strategies(self, population: list[Individual]) -> list[Individual]:
        """查找有效策略

        Args:
            population: 种群

        Returns:
            有效策略列表
        """
        valid = []

        for individual in population:
            # 检查是否有效
            if self.fitness_evaluator.is_valid(
                individual,
                individual.backtest_results
            ):
                # 检查是否已存在
                if not self.strategy_pool.exists(individual, self.symbol):
                    valid.append(individual)

        return valid

    def _save_valid_strategies(
        self,
        valid_strategies: list[Individual],
        generation: int
    ) -> list[str]:
        """保存有效策略到策略池（交互和非交互模式通用）

        Args:
            valid_strategies: 有效策略列表
            generation: 当前代数

        Returns:
            保存的策略ID列表
        """
        strategy_ids = []
        for individual in valid_strategies:
            strategy_id = self.strategy_pool.add(
                individual,
                self.symbol,
                individual.backtest_results,
                {"generation": generation}
            )
            strategy_ids.append(strategy_id)
            logger.info(
                f"发现有效策略 Gen{generation}: "
                f"适应度={individual.fitness:.4f}, "
                f"ID={strategy_id}"
            )

        return strategy_ids

    def _handle_valid_strategies(
        self,
        valid_strategies: list[Individual],
        generation: int,
        population: list[Individual]
    ) -> str:
        """处理有效策略 - 交互暂停点

        Args:
            valid_strategies: 有效策略列表
            generation: 当前代数
            population: 当前种群

        Returns:
            用户选择 ('continue', 'stop', 'adjust')
        """
        # 保存到策略池
        strategy_ids = []
        for individual in valid_strategies:
            strategy_id = self.strategy_pool.add(
                individual,
                self.symbol,
                individual.backtest_results,
                {"generation": generation}
            )
            strategy_ids.append(strategy_id)

        # 生成临时报告
        temp_report_path = self.output_dir / f"temp_{self.symbol}_gen{generation}.html"

        # 获取策略池数据用于显示
        strategies_data = [self.strategy_pool.get(sid) for sid in strategy_ids]
        strategies_data = [s for s in strategies_data if s is not None]

        # 显示摘要
        if strategies_data:
            display_strategy_summary(strategies_data, top_n=3)

        # 等待用户确认
        best_fitness = max(ind.fitness for ind in valid_strategies) if valid_strategies else 0
        choice = confirm_continue(len(valid_strategies), best_fitness, generation)

        return choice

    def _generate_final_report(
        self,
        generation: int,
        population: list[Individual],
        valid_strategies: list[Individual]
    ) -> str:
        """生成最终报告

        Args:
            generation: 最终代数
            population: 最终种群
            valid_strategies: 所有发现的策略

        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"turtle_optimization_{self.symbol}_{timestamp}.html"

        self.report_generator.generate_optimization_report(
            symbol=self.symbol,
            generation=generation,
            population=population,
            valid_strategies=valid_strategies,
            evolution_history=self.evolution_history,
            output_path=str(report_path),
            metadata={
                "name": self.name,
                "initial_cash": self.cash,
                "max_drawdown_limit": self.fitness_config.max_drawdown_limit,
            }
        )

        return str(report_path)

    def get_best_strategies(self, top_n: int = 10) -> list[dict]:
        """获取最优策略

        Args:
            top_n: 返回数量

        Returns:
            策略列表
        """
        return self.strategy_pool.get_best(self.symbol, top_n)

    def get_pool_statistics(self) -> dict:
        """获取策略池统计

        Returns:
            统计信息
        """
        return self.strategy_pool.get_statistics()
