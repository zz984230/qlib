"""策略池管理器

管理通过遗传算法发现的有效策略，使用 JSON 文件存储。
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import numpy as np
import pandas as pd

from src.optimizer.individual import Individual


def _convert_to_serializable(obj: Any) -> Any:
    """将numpy和pandas类型转换为Python原生类型，用于JSON序列化

    Args:
        obj: 任意对象

    Returns:
        可序列化的对象
    """
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_to_serializable(obj.tolist())
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

logger = logging.getLogger(__name__)


class StrategyPool:
    """有效策略池管理器

    将遗传算法发现的有效策略保存到文件系统，支持查询和排序。
    """

    def __init__(self, pool_dir: str = "strategies/pool"):
        """初始化策略池

        Args:
            pool_dir: 策略池目录
        """
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"策略池目录: {self.pool_dir}")

    def add(
        self,
        individual: Individual,
        symbol: str,
        backtest_results: dict,
        metadata: dict | None = None
    ) -> str:
        """添加有效策略到池中

        Args:
            individual: 策略个体
            symbol: 股票代码
            backtest_results: 回测结果
            metadata: 额外元数据

        Returns:
            策略ID
        """
        # 生成策略ID
        strategy_id = f"strategy_{uuid.uuid4().hex[:8]}"

        # 构建策略数据（包含所有 Individual 字段）
        strategy_data = {
            "id": strategy_id,
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "strategy_type": getattr(individual, "strategy_type", "turtle"),
            "factor_weights": individual.factor_weights,
            "signal_threshold": individual.signal_threshold,
            "exit_threshold": individual.exit_threshold,
            "atr_period": individual.atr_period,
            "stop_loss_atr": individual.stop_loss_atr,
            "pyramid_interval_atr": individual.pyramid_interval_atr,
            "max_pyramid_units": individual.max_pyramid_units,
            "trailing_stop_trigger": individual.trailing_stop_trigger,
            "min_adx": individual.min_adx,
            "min_trend_periods": individual.min_trend_periods,
            "use_trend_filter": individual.use_trend_filter,
            "backtest_results": backtest_results,
            "fitness": individual.fitness,
            "market_performance": getattr(individual, "market_performance", {}),
            "optimal_market": getattr(individual, "optimal_market", None),
            "generation": individual.generation,
            "parent_ids": [str(pid) for pid in individual.parent_ids],
            "metadata": metadata or {},
        }

        # 保存到文件（先转换numpy类型为Python原生类型）
        file_path = self.pool_dir / f"{strategy_id}.json"
        serializable_data = _convert_to_serializable(strategy_data)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        logger.info(f"策略已保存: {strategy_id} -> {file_path}")

        return strategy_id

    def get(self, strategy_id: str) -> dict | None:
        """获取指定策略

        Args:
            strategy_id: 策略ID

        Returns:
            策略数据，不存在则返回 None
        """
        file_path = self.pool_dir / f"{strategy_id}.json"

        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_best(
        self,
        symbol: str,
        top_n: int = 10,
        sort_by: str = "fitness"
    ) -> list[dict]:
        """获取指定股票的最优 N 个策略

        Args:
            symbol: 股票代码
            top_n: 返回数量
            sort_by: 排序字段（fitness, total_return, sharpe_ratio 等）

        Returns:
            策略列表
        """
        strategies = self.list_by_symbol(symbol)

        # 排序
        reverse = True  # 默认降序

        if sort_by == "fitness":
            strategies.sort(key=lambda x: x.get("fitness", 0), reverse=reverse)
        elif sort_by in ["total_return", "return"]:
            # 使用 1y 周期的收益率
            strategies.sort(
                key=lambda x: x.get("backtest_results", {}).get("1y", {}).get("total_return", 0),
                reverse=reverse
            )
        elif sort_by == "sharpe_ratio":
            strategies.sort(
                key=lambda x: x.get("backtest_results", {}).get("1y", {}).get("sharpe_ratio", 0),
                reverse=reverse
            )
        elif sort_by == "max_drawdown":
            # 回撤越小越好
            strategies.sort(
                key=lambda x: x.get("backtest_results", {}).get("1y", {}).get("max_drawdown", 1),
                reverse=not reverse
            )

        return strategies[:top_n]

    def list_by_symbol(self, symbol: str) -> list[dict]:
        """获取指定股票的所有策略

        Args:
            symbol: 股票代码

        Returns:
            策略列表
        """
        strategies = []

        for file_path in self.pool_dir.glob("strategy_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    strategy = json.load(f)

                if strategy.get("symbol") == symbol:
                    strategies.append(strategy)
            except Exception as e:
                logger.warning(f"读取策略文件失败 {file_path}: {e}")

        return strategies

    def list_all(self) -> list[dict]:
        """获取所有策略

        Returns:
            策略列表
        """
        strategies = []

        for file_path in self.pool_dir.glob("strategy_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    strategies.append(json.load(f))
            except Exception as e:
                logger.warning(f"读取策略文件失败 {file_path}: {e}")

        return strategies

    def exists(self, individual: Individual, symbol: str, threshold: float = 0.1) -> bool:
        """检查策略是否已存在（避免重复）

        通过比较因子权重和阈值判断是否为相似策略。

        Args:
            individual: 策略个体
            symbol: 股票代码
            threshold: 相似度阈值

        Returns:
            是否存在相似策略
        """
        strategies = self.list_by_symbol(symbol)

        for strategy in strategies:
            # 比较因子权重
            if self._is_similar(individual, strategy, threshold):
                return True

        return False

    def _is_similar(self, individual: Individual, strategy: dict, threshold: float) -> bool:
        """判断个体是否与已有策略相似

        Args:
            individual: 策略个体
            strategy: 已有策略数据
            threshold: 相似度阈值

        Returns:
            是否相似
        """
        # 比较因子权重
        strategy_weights = strategy.get("factor_weights", {})

        # 检查因子集合是否相同
        if set(individual.factor_weights.keys()) != set(strategy_weights.keys()):
            return False

        # 计算权重差异
        weight_diff = 0.0
        for factor in individual.factor_weights:
            w1 = individual.factor_weights[factor]
            w2 = strategy_weights.get(factor, 0)
            weight_diff += abs(w1 - w2)

        weight_diff /= len(individual.factor_weights)

        # 比较阈值
        threshold_diff = (
            abs(individual.signal_threshold - strategy.get("signal_threshold", 0.5))
            + abs(individual.exit_threshold - strategy.get("exit_threshold", 0.3))
        ) / 2

        # 综合判断
        similarity = weight_diff * 0.7 + threshold_diff * 0.3

        return similarity < threshold

    def remove(self, strategy_id: str) -> bool:
        """删除指定策略

        Args:
            strategy_id: 策略ID

        Returns:
            是否成功删除
        """
        file_path = self.pool_dir / f"{strategy_id}.json"

        if file_path.exists():
            file_path.unlink()
            logger.info(f"策略已删除: {strategy_id}")
            return True

        return False

    def clear(self, symbol: str | None = None) -> int:
        """清除策略

        Args:
            symbol: 股票代码，None 表示清除所有

        Returns:
            删除的数量
        """
        count = 0

        for file_path in self.pool_dir.glob("strategy_*.json"):
            try:
                if symbol is None:
                    file_path.unlink()
                    count += 1
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        strategy = json.load(f)

                    if strategy.get("symbol") == symbol:
                        file_path.unlink()
                        count += 1
            except Exception as e:
                logger.warning(f"删除策略文件失败 {file_path}: {e}")

        logger.info(f"已删除 {count} 个策略")

        return count

    def get_statistics(self) -> dict:
        """获取策略池统计信息

        Returns:
            统计信息字典
        """
        strategies = self.list_all()

        if not strategies:
            return {
                "total_strategies": 0,
                "symbols": {},
                "best_fitness": 0,
                "avg_fitness": 0,
            }

        # 按股票分组
        symbols = {}
        for strategy in strategies:
            symbol = strategy.get("symbol", "unknown")
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(strategy)

        # 计算统计
        fitness_values = [s.get("fitness", 0) for s in strategies]

        return {
            "total_strategies": len(strategies),
            "symbols": {k: len(v) for k, v in symbols.items()},
            "best_fitness": max(fitness_values) if fitness_values else 0,
            "avg_fitness": sum(fitness_values) / len(fitness_values) if fitness_values else 0,
        }

    def load_seed_individuals(self, symbol: str, top_n: int = 10) -> list[Individual]:
        """加载种子个体（用于增量进化）

        从策略池中加载指定股票的最优策略，转换为 Individual 对象。

        Args:
            symbol: 股票代码
            top_n: 加载数量

        Returns:
            Individual 对象列表
        """
        strategies = self.get_best(symbol, top_n=top_n, sort_by="fitness")

        seed_individuals = []
        for strategy in strategies:
            try:
                # 从策略数据创建 Individual 对象
                individual = Individual.from_dict(strategy)
                seed_individuals.append(individual)
            except Exception as e:
                logger.warning(f"加载种子个体失败 {strategy.get('id')}: {e}")

        logger.info(f"从策略池加载了 {len(seed_individuals)} 个种子个体")
        return seed_individuals

    def cleanup_strategies(
        self,
        symbol: str | None = None,
        min_fitness: float | None = None,
        max_age_days: int | None = None,
        max_count_per_symbol: int | None = None
    ) -> int:
        """清理策略池（淘汰失效策略）

        Args:
            symbol: 股票代码，None 表示所有股票
            min_fitness: 最低适应度，低于此值的策略将被删除
            max_age_days: 最大保留天数，超过此天数的策略将被删除
            max_count_per_symbol: 每个股票最多保留策略数量

        Returns:
            删除的策略数量
        """
        from datetime import timedelta

        strategies = self.list_all() if symbol is None else self.list_by_symbol(symbol)

        if not strategies:
            return 0

        cutoff_time = None
        if max_age_days is not None:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)

        to_remove = []
        symbol_counts = {}  # 每个股票的策略计数

        for strategy in strategies:
            strategy_id = strategy.get("id")
            strategy_symbol = strategy.get("symbol")
            strategy_timestamp = strategy.get("timestamp")
            strategy_fitness = strategy.get("fitness", 0)

            # 检查适应度阈值
            if min_fitness is not None and strategy_fitness < min_fitness:
                to_remove.append(strategy_id)
                continue

            # 检查时间阈值
            if cutoff_time is not None and strategy_timestamp:
                try:
                    strategy_time = datetime.fromisoformat(strategy_timestamp)
                    if strategy_time < cutoff_time:
                        to_remove.append(strategy_id)
                        continue
                except:
                    pass

            # 检查每个股票的最大策略数量
            if max_count_per_symbol is not None:
                if strategy_symbol not in symbol_counts:
                    symbol_counts[strategy_symbol] = 0
                symbol_counts[strategy_symbol] += 1

        # 处理按数量淘汰（保留适应度最高的）
        if max_count_per_symbol is not None:
            for sym, count in symbol_counts.items():
                if count > max_count_per_symbol:
                    # 获取该股票的所有策略，按适应度排序
                    sym_strategies = [
                        s for s in strategies
                        if s.get("symbol") == sym and s.get("id") not in to_remove
                    ]
                    sym_strategies.sort(key=lambda x: x.get("fitness", 0), reverse=True)

                    # 标记超过数量的策略为删除
                    for strategy in sym_strategies[max_count_per_symbol:]:
                        to_remove.append(strategy.get("id"))

        # 执行删除
        removed_count = 0
        for strategy_id in to_remove:
            if self.remove(strategy_id):
                removed_count += 1

        logger.info(f"清理策略池完成，删除了 {removed_count} 个策略")

        return removed_count

    def to_dataframe(self) -> "pd.DataFrame":
        """将策略池转换为 DataFrame

        Returns:
            策略DataFrame
        """
        import pandas as pd

        strategies = self.list_all()

        if not strategies:
            return pd.DataFrame()

        # 展开数据
        rows = []
        for strategy in strategies:
            row = {
                "id": strategy.get("id"),
                "symbol": strategy.get("symbol"),
                "timestamp": strategy.get("timestamp"),
                "fitness": strategy.get("fitness"),
                "generation": strategy.get("generation"),
                "signal_threshold": strategy.get("signal_threshold"),
                "exit_threshold": strategy.get("exit_threshold"),
                "atr_period": strategy.get("atr_period"),
            }

            # 添加回测结果
            backtest = strategy.get("backtest_results", {})
            for period, result in backtest.items():
                row[f"{period}_return"] = result.get("total_return")
                row[f"{period}_drawdown"] = result.get("max_drawdown")
                row[f"{period}_sharpe"] = result.get("sharpe_ratio")

            rows.append(row)

        return pd.DataFrame(rows)

    def export_to_csv(self, output_path: str) -> None:
        """导出策略池到 CSV

        Args:
            output_path: 输出文件路径
        """
        df = self.to_dataframe()
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"策略池已导出到: {output_path}")
