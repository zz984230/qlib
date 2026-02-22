"""遗传算法引擎

实现遗传算法的核心逻辑：选择、交叉、变异、演化。
"""

import logging
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from numpy.random import Generator, PCG64

from src.optimizer.individual import Individual, get_factor_names

logger = logging.getLogger(__name__)


@dataclass
class GeneticConfig:
    """遗传算法配置"""

    population_size: int = 50            # 种群大小
    max_generations: int = 100           # 最大演化代数
    mutation_rate: float = 0.1           # 变异率
    mutation_strength: float = 0.2       # 变异强度（标准差）
    crossover_rate: float = 0.7          # 交叉率
    elite_size: int = 5                  # 精英保留数量
    tournament_size: int = 3             # 锦标赛选择规模
    no_improvement_limit: int = 20       # 连续N代无改进则停止

    # 基因范围约束
    min_signal_threshold: float = 0.0
    max_signal_threshold: float = 1.0
    min_exit_threshold: float = 0.0
    max_exit_threshold: float = 1.0
    min_atr_period: int = 5
    max_atr_period: int = 50


class GeneticEngine:
    """遗传算法优化引擎

    通过选择、交叉、变异等遗传操作，演化出最优的因子组合策略。
    """

    def __init__(self, config: GeneticConfig | None = None, random_seed: int | None = None):
        """初始化遗传算法引擎

        Args:
            config: 遗传算法配置
            random_seed: 随机种子（用于可复现性）
        """
        self.config = config or GeneticConfig()
        self.random = Generator(PCG64(random_seed))
        self.factor_names = get_factor_names()

        # 演化历史
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history: list[float] = []
        self.generation_stats: list[dict] = []

    def initialize_population(self, generation: int = 0) -> list[Individual]:
        """初始化种群

        使用随机方式生成初始个体，确保多样性。

        Args:
            generation: 初始代数（通常为0）

        Returns:
            初始种群列表
        """
        population = []

        for i in range(self.config.population_size):
            # 随机选择 3-8 个因子
            n_factors = self.random.integers(3, min(9, len(self.factor_names) + 1))
            selected_factors = self.random.choice(
                self.factor_names, size=n_factors, replace=False
            )

            # 随机生成因子权重
            raw_weights = self.random.random(n_factors)
            weights = raw_weights / raw_weights.sum()  # 归一化

            factor_weights = dict(zip(selected_factors, weights))

            # 随机生成阈值
            signal_threshold = self.random.uniform(
                self.config.min_signal_threshold,
                self.config.max_signal_threshold
            )
            exit_threshold = self.random.uniform(
                self.config.min_exit_threshold,
                self.config.max_exit_threshold
            )

            # 随机生成 ATR 周期
            atr_period = self.random.integers(
                self.config.min_atr_period,
                self.config.max_atr_period + 1
            )

            individual = Individual(
                factor_weights=factor_weights,
                signal_threshold=signal_threshold,
                exit_threshold=exit_threshold,
                atr_period=atr_period,
                generation=generation,
                parent_ids=[],
            )

            population.append(individual)

        logger.info(f"初始化种群: {len(population)} 个个体")
        return population

    def evolve(
        self,
        population: list[Individual],
        generation: int,
        fitness_func: Callable[[Individual], float]
    ) -> list[Individual]:
        """执行一代演化

        流程：
        1. 适应度评估（如果尚未评估）
        2. 精英保留
        3. 锦标赛选择
        4. 交叉
        5. 变异

        Args:
            population: 当前种群
            generation: 当前代数
            fitness_func: 适应度计算函数

        Returns:
            新一代种群
        """
        # 确保所有个体都已评估
        for individual in population:
            if individual.fitness == 0:
                individual.fitness = fitness_func(individual)

        # 按适应度排序
        population.sort(key=lambda x: x.fitness, reverse=True)

        # 记录统计信息
        best_fitness = population[0].fitness
        avg_fitness = sum(x.fitness for x in population) / len(population)

        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        self.generation_stats.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "population_size": len(population),
        })

        logger.info(
            f"代数 {generation}: 最优适应度={best_fitness:.4f}, "
            f"平均适应度={avg_fitness:.4f}"
        )

        # 精英保留：直接复制前 N 个最优个体
        elites = population[:self.config.elite_size]

        # 生成新一代个体
        new_population = elites.copy()

        while len(new_population) < self.config.population_size:
            # 锦标赛选择父代
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            # 交叉
            if self.random.random() < self.config.crossover_rate:
                offspring1, offspring2 = self._crossover(parent1, parent2, generation)
            else:
                offspring1, offspring2 = parent1, parent2

            # 变异
            if self.random.random() < self.config.mutation_rate:
                offspring1 = self._mutate(offspring1, generation)
            if self.random.random() < self.config.mutation_rate:
                offspring2 = self._mutate(offspring2, generation)

            new_population.extend([offspring1, offspring2])

        # 截断到指定种群大小
        new_population = new_population[:self.config.population_size]

        return new_population

    def _tournament_select(self, population: list[Individual]) -> Individual:
        """锦标赛选择

        随机选择 k 个个体，返回适应度最高的个体。

        Args:
            population: 种群

        Returns:
            被选中的个体
        """
        candidates = self.random.choice(
            population,
            size=min(self.config.tournament_size, len(population)),
            replace=False
        )
        return max(candidates, key=lambda x: x.fitness)

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        generation: int
    ) -> tuple[Individual, Individual]:
        """单点交叉

        Args:
            parent1: 父代1
            parent2: 父代2
            generation: 当前代数

        Returns:
            两个子代个体
        """
        # 使用固定的因子顺序确保基因长度一致
        genes1 = parent1.to_genes(self.factor_names)
        genes2 = parent2.to_genes(self.factor_names)

        # 基因长度现在应该一致
        if len(genes1) <= 1:
            return parent1, parent2

        # 随机选择交叉点
        crossover_point = self.random.integers(1, len(genes1))

        # 交叉基因
        child1_genes = genes1.copy()
        child2_genes = genes2.copy()

        child1_genes[crossover_point:] = genes2[crossover_point:]
        child2_genes[crossover_point:] = genes1[crossover_point:]

        # 从基因创建个体
        child1 = Individual.from_genes(
            child1_genes,
            self.factor_names,
            generation=generation,
            parent_ids=[id(parent1), id(parent2)]
        )
        child2 = Individual.from_genes(
            child2_genes,
            self.factor_names,
            generation=generation,
            parent_ids=[id(parent1), id(parent2)]
        )

        return child1, child2

    def _mutate(
        self,
        individual: Individual,
        generation: int
    ) -> Individual:
        """高斯变异

        对基因添加高斯噪声，实现探索能力。

        Args:
            individual: 待变异个体
            generation: 当前代数

        Returns:
            变异后的个体
        """
        genes = individual.to_genes(self.factor_names)
        n_factors = len(self.factor_names)

        # 对每个基因添加高斯噪声
        for i in range(len(genes)):
            if self.random.random() < 0.5:  # 50% 概率变异每个基因
                noise = self.random.normal(0, self.config.mutation_strength)
                genes[i] += noise

        # 因子权重保持非负（变异后会在 Individual.__post_init__ 中归一化）
        genes[:n_factors] = np.maximum(genes[:n_factors], 0)

        # 信号阈值约束
        if n_factors < len(genes):
            genes[n_factors] = np.clip(
                genes[n_factors],
                self.config.min_signal_threshold,
                self.config.max_signal_threshold
            )

        # 出场阈值约束
        if n_factors + 1 < len(genes):
            genes[n_factors + 1] = np.clip(
                genes[n_factors + 1],
                self.config.min_exit_threshold,
                self.config.max_exit_threshold
            )

        # ATR 周期约束
        if n_factors + 2 < len(genes):
            genes[n_factors + 2] = np.clip(
                genes[n_factors + 2],
                self.config.min_atr_period,
                self.config.max_atr_period
            )

        # 从基因创建新个体
        mutated = Individual.from_genes(
            genes,
            self.factor_names,
            generation=generation,
            parent_ids=individual.parent_ids + [id(individual)]
        )

        return mutated

    def should_stop(self, generation: int) -> bool:
        """判断是否应该停止演化

        停止条件：
        1. 达到最大代数
        2. 连续 N 代无改进

        Args:
            generation: 当前代数

        Returns:
            是否停止
        """
        # 达到最大代数
        if generation >= self.config.max_generations:
            logger.info(f"达到最大代数 {generation}，停止演化")
            return True

        # 连续 N 代无改进
        if len(self.best_fitness_history) >= self.config.no_improvement_limit:
            recent_best = self.best_fitness_history[-self.config.no_improvement_limit:]
            if max(recent_best) <= self.best_fitness_history[-self.config.no_improvement_limit]:
                logger.info(
                    f"连续 {self.config.no_improvement_limit} 代无改进，停止演化"
                )
                return True

        return False

    def get_best_individual(self, population: list[Individual]) -> Individual:
        """获取种群中最优个体

        Args:
            population: 种群

        Returns:
            最优个体
        """
        return max(population, key=lambda x: x.fitness)

    def get_statistics(self) -> dict:
        """获取演化统计信息

        Returns:
            统计信息字典
        """
        return {
            "best_fitness_history": self.best_fitness_history,
            "avg_fitness_history": self.avg_fitness_history,
            "generation_stats": self.generation_stats,
            "total_generations": len(self.best_fitness_history),
        }
