"""
策略选择器

比较多个策略，根据综合评分选择最佳策略
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from src.backtest.runner import BacktestRunner, BacktestResult
from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyRanking:
    """策略排名"""

    strategy_name: str
    rank: int
    composite_score: float
    metrics: dict[str, float]
    strategy_instance: Optional[BaseStrategy] = None
    backtest_result: Optional[BacktestResult] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class StrategySelector:
    """
    策略选择器

    根据多个指标综合评分，选择最佳策略
    """

    DEFAULT_SCORING_WEIGHTS = {
        "sharpe_ratio": 0.4,
        "annual_return": 0.3,
        "max_drawdown": -0.2,  # 负权重，越小越好
        "volatility": -0.1,  # 负权重，越小越好
    }

    def __init__(
        self,
        scoring_weights: Optional[dict[str, float]] = None,
        min_trades: int = 5,
    ):
        """
        初始化策略选择器

        Args:
            scoring_weights: 指标权重
            min_trades: 最小交易次数要求
        """
        self.scoring_weights = scoring_weights or self.DEFAULT_SCORING_WEIGHTS
        self.min_trades = min_trades

    def compare_strategies(
        self,
        strategies: list[BaseStrategy],
        start_date: str,
        end_date: str,
        cash: float = 1000000,
    ) -> list[StrategyRanking]:
        """
        比较多个策略

        Args:
            strategies: 策略实例列表
            start_date: 开始日期
            end_date: 结束日期
            cash: 初始资金

        Returns:
            策略排名列表（按得分降序）
        """
        logger.info(f"比较 {len(strategies)} 个策略: {start_date} -> {end_date}")

        rankings = []
        runner = BacktestRunner()

        for strategy in strategies:
            try:
                result = runner.run(strategy, start_date, end_date, cash)
                metrics = result.get_all_metrics()

                # 计算综合得分
                score = self._calculate_score(metrics)

                rankings.append(
                    StrategyRanking(
                        strategy_name=str(strategy),
                        rank=0,  # 稍后排序
                        composite_score=score,
                        metrics=metrics,
                        strategy_instance=strategy,
                        backtest_result=result,
                    )
                )

                logger.info(
                    f"策略 {strategy}: 夏普={metrics.get('sharpe_ratio', 0):.2f}, "
                    f"收益={metrics.get('annual_return', 0):.1%}, "
                    f"回撤={metrics.get('max_drawdown', 0):.1%}, "
                    f"得分={score:.3f}"
                )

            except Exception as e:
                logger.error(f"策略 {strategy} 回测失败: {e}")

        # 按得分排序
        rankings.sort(key=lambda x: x.composite_score, reverse=True)

        # 分配排名
        for i, ranking in enumerate(rankings):
            ranking.rank = i + 1

        return rankings

    def select_best(
        self,
        rankings: list[StrategyRanking],
        top_n: int = 1,
        min_score: float = 0.0,
    ) -> list[StrategyRanking]:
        """
        选择最佳策略

        Args:
            rankings: 策略排名列表
            top_n: 选择前 N 个
            min_score: 最低得分要求

        Returns:
            最佳策略列表
        """
        filtered = [r for r in rankings if r.composite_score >= min_score]
        return filtered[:top_n]

    def _calculate_score(self, metrics: dict[str, float]) -> float:
        """计算综合得分"""
        score = 0.0

        for metric, weight in self.scoring_weights.items():
            value = metrics.get(metric, 0)

            # 对于负权重指标，值越小得分越高
            if weight < 0:
                # 将负值转为正值（用于惩罚）
                # 例如：最大回撤 20% -> 0.2，权重 -0.2 -> 贡献 -0.04
                contribution = value * weight
            else:
                contribution = value * weight

            score += contribution

        return score

    def generate_comparison_report(
        self,
        rankings: list[StrategyRanking],
    ) -> pd.DataFrame:
        """生成比较报告"""
        records = []

        for ranking in rankings:
            record = {
                "rank": ranking.rank,
                "strategy": ranking.strategy_name,
                "composite_score": ranking.composite_score,
                **ranking.metrics,
            }
            records.append(record)

        return pd.DataFrame(records)


def quick_compare(
    strategy_names: list[str],
    start_date: str,
    end_date: str,
    cash: float = 1000000,
) -> pd.DataFrame:
    """
    快速比较多个策略

    Args:
        strategy_names: 策略名称列表
        start_date: 开始日期
        end_date: 结束日期
        cash: 初始资金

    Returns:
        比较结果 DataFrame
    """
    from src.strategy import get_strategy

    strategies = []
    for name in strategy_names:
        try:
            strategies.append(get_strategy(name))
        except ValueError as e:
            logger.warning(f"跳过未知策略: {name}")

    if not strategies:
        return pd.DataFrame()

    selector = StrategySelector()
    rankings = selector.compare_strategies(strategies, start_date, end_date, cash)

    return selector.generate_comparison_report(rankings)
