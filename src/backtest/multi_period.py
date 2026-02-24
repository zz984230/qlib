"""多周期回测器

支持三周期回测（1y/3m/1m），用于适应度评估。
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from src.backtest.turtle_backtest import TurtleBacktestRunner
from src.optimizer.individual import Individual

logger = logging.getLogger(__name__)


# 回测周期配置
PERIODS = {
    "1y": {"name": "近1年", "days": 365, "weight": 0.6},
    "3m": {"name": "近3月", "days": 90, "weight": 0.3},
    "1m": {"name": "近1月", "days": 30, "weight": 0.1},
}


class MultiPeriodBacktester:
    """多周期回测器

    对单个策略执行多个时间周期的回测，用于综合评估策略表现。
    """

    def __init__(
        self,
        symbol: str,
        initial_cash: float = 50000,
        commission: float = 0.0003,
    ):
        """初始化多周期回测器

        Args:
            symbol: 股票代码
            initial_cash: 初始资金
            commission: 手续费率
        """
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.commission = commission

    def run(
        self,
        data: pd.DataFrame,
        individual: Individual,
        periods: list[str] | None = None
    ) -> dict[str, "BacktestResult"]:
        """执行多周期回测

        Args:
            data: 完整的OHLCV数据
            individual: 策略个体
            periods: 要回测的周期列表，默认 ["1y", "3m", "1m"]

        Returns:
            回测结果字典 {period: BacktestResult}
        """
        if periods is None:
            periods = ["1y", "3m", "1m"]

        results = {}
        self._period_info = {}  # 存储周期信息

        for period in periods:
            period_config = PERIODS.get(period)
            if period_config is None:
                logger.warning(f"未知周期: {period}")
                continue

            # 获取周期数据（包含预热期）
            period_data = self._get_period_data(data, period_config["days"])

            if len(period_data) < 50:  # 至少需要50天数据
                logger.warning(f"周期 {period} 数据不足，跳过")
                continue

            # 执行回测
            runner = TurtleBacktestRunner(
                symbol=self.symbol,
                initial_cash=self.initial_cash,
                individual=individual,
                commission=self.commission,
            )

            result = runner.run(period_data)

            # 计算实际回测的起止日期（从第50天开始）
            warmup_days = 50
            actual_start_idx = min(warmup_days, len(period_data) - 1)
            actual_start_date = period_data.index[actual_start_idx]
            actual_end_date = period_data.index[-1]

            # 存储周期信息
            self._period_info[period] = {
                "name": period_config["name"],
                "target_days": period_config["days"],
                "weight": period_config["weight"],
                "data_start": str(period_data.index[0]),
                "data_end": str(period_data.index[-1]),
                "backtest_start": str(actual_start_date),
                "backtest_end": str(actual_end_date),
                "actual_trading_days": len(period_data) - warmup_days,
            }

            # 在结果中添加周期信息
            result.period_info = self._period_info[period]
            result.trades_list = runner.trades  # 保存交易列表

            results[period] = result

            logger.info(
                f"周期 {period} ({period_config['name']}) 回测完成: "
                f"回测区间 {actual_start_date} ~ {actual_end_date}, "
                f"收益={result.total_return:.2%}, "
                f"回撤={result.max_drawdown:.2%}"
            )

        return results

    def get_period_info(self) -> dict:
        """获取周期信息"""
        return getattr(self, '_period_info', {})

    def _get_period_data(self, data: pd.DataFrame, days: int) -> pd.DataFrame:
        """获取指定周期的数据

        注意：回测需要至少 50 天的预热数据来计算技术指标，
        所以我们返回足够的历史数据，并在 backtest_results 中
        记录实际的回测起止日期。

        Args:
            data: 完整数据
            days: 天数（回测目标周期）

        Returns:
            包含预热数据的周期数据
        """
        # 需要额外的 50 天作为预热期
        min_required = days + 50

        if len(data) <= min_required:
            # 数据不足，返回全部数据
            return data.copy()

        # 返回最近 (days + 50) 天的数据
        # 回测器会从第 50 天开始，实际回测周期为 days 天
        return data.iloc[-min_required:].copy()

    @staticmethod
    def get_period_dates(end_date: datetime, days: int) -> tuple[datetime, datetime]:
        """获取周期起止日期

        Args:
            end_date: 结束日期
            days: 天数

        Returns:
            (开始日期, 结束日期)
        """
        start_date = end_date - timedelta(days=days)
        return start_date, end_date

    @staticmethod
    def validate_periods(results: dict[str, "BacktestResult"]) -> dict:
        """验证回测结果

        检查各周期的回撤是否符合要求。

        Args:
            results: 回测结果字典

        Returns:
            验证结果字典
        """
        validation = {
            "valid": True,
            "periods": {},
            "max_drawdown": 0.0,
            "violations": [],
        }

        for period, result in results.items():
            dd = result.max_drawdown
            validation["periods"][period] = {
                "drawdown": dd,
                "return": result.total_return,
                "valid": dd <= 0.03,
            }

            if dd > validation["max_drawdown"]:
                validation["max_drawdown"] = dd

            if dd > 0.03:
                validation["valid"] = False
                validation["violations"].append(f"{period}: 回撤 {dd:.2%} > 3%")

        return validation


def create_period_datasets(
    data: pd.DataFrame,
    end_date: Optional[datetime] = None
) -> dict[str, pd.DataFrame]:
    """创建各周期的数据集

    Args:
        data: 完整数据
        end_date: 结束日期，默认为数据最后一天

    Returns:
        周期数据字典
    """
    if end_date is None:
        end_date = data.index[-1]

    datasets = {}

    for period_key, period_config in PERIODS.items():
        start_date, _ = MultiPeriodBacktester.get_period_dates(
            end_date, period_config["days"]
        )

        # 筛选日期范围
        period_data = data[(data.index >= start_date) & (data.index <= end_date)]

        if len(period_data) > 0:
            datasets[period_key] = period_data

    return datasets


def get_available_periods(data: pd.DataFrame, min_days: int = 50) -> list[str]:
    """获取数据支持的回测周期

    Args:
        data: 完整数据
        min_days: 最少天数要求

    Returns:
        可用的周期列表
    """
    available = []

    for period_key, period_config in PERIODS.items():
        if len(data) >= period_config["days"]:
            # 进一步检查实际可用数据
            period_data = data.iloc[-period_config["days"]:]
            if len(period_data) >= min_days:
                available.append(period_key)

    return available
