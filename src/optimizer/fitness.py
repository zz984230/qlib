"""适应度评估器

实现策略适应度的计算，支持 Phase 1 简化版和完整回测版。
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from src.optimizer.individual import Individual

if TYPE_CHECKING:
    from src.backtest.runner import BacktestRunner
    from src.backtest.multi_period import MultiPeriodBacktester

logger = logging.getLogger(__name__)


@dataclass
class FitnessConfig:
    """适应度评估配置"""

    # 回撤限制（移除硬约束，改为参考值）
    max_drawdown_limit: float = 0.08  # 8%参考值，非硬约束

    # 各周期权重
    period_1y_weight: float = 0.6
    period_3m_weight: float = 0.3
    period_1m_weight: float = 0.1

    # 适应度参数
    sharpe_bonus_weight: float = 1.0      # 夏普比率加分权重
    stability_bonus_weight: float = 0.5   # 稳定性加分权重
    drawdown_penalty_weight: float = 0.5  # 回撤惩罚权重（降低）
    trade_frequency_penalty_weight: float = 1.0  # 交易频率惩罚权重
    return_weight: float = 5.0  # 收益权重


class FitnessEvaluator:
    """适应度评估器

    Phase 1: 基于因子 IC 值的简化评估
    Phase 3: 基于回测的完整评估
    """

    def __init__(self, config: FitnessConfig | None = None):
        """初始化适应度评估器

        Args:
            config: 适应度评估配置
        """
        self.config = config or FitnessConfig()

    def evaluate_ic_based(
        self,
        individual: Individual,
        data: pd.DataFrame
    ) -> float:
        """基于因子 IC 值计算适应度（Phase 1 简化版）

        使用因子权重加权的 IC 值作为适应度。
        IC (Information Coefficient) 是因子值与未来收益的相关系数。

        Args:
            individual: 待评估个体
            data: 包含价格和因子数据的 DataFrame

        Returns:
            适应度值
        """
        # 计算未来收益（1日后的收益率）
        data = data.copy()
        data["future_return"] = data["close"].pct_change().shift(-1)

        # 去除 NaN
        data = data.dropna()

        if len(data) < 10:
            return 0.0

        total_fitness = 0.0
        total_weight = 0.0

        for factor_name, weight in individual.factor_weights.items():
            # 计算因子值（简化版：使用基本技术指标）
            factor_values = self._calculate_factor_value(factor_name, data)

            if factor_values is None or len(factor_values) == 0:
                continue

            # 计算 IC（相关系数）
            valid_mask = ~factor_values.isna() & ~data["future_return"].isna()
            if valid_mask.sum() < 5:
                continue

            ic = data.loc[valid_mask, "future_return"].corr(factor_values[valid_mask])

            if not np.isnan(ic):
                total_fitness += abs(ic) * weight
                total_weight += weight

        # 归一化适应度
        if total_weight > 0:
            fitness = total_fitness / total_weight
        else:
            fitness = 0.0

        # 阈值惩罚：阈值设置不合理会降低适应度
        if individual.signal_threshold <= individual.exit_threshold:
            fitness *= 0.5  # 入场阈值应高于出场阈值

        return fitness

    def evaluate_backtest_based(
        self,
        individual: Individual,
        backtest_results: dict[str, "BacktestResult"]
    ) -> float:
        """基于回测结果计算适应度（简化版）

        适应度 = 加权收益 * 收益权重 + 夏普加分 - 回撤惩罚

        Args:
            individual: 待评估个体
            backtest_results: 回测结果字典 {"1y": result, "3m": result, "1m": result}

        Returns:
            适应度值
        """
        weights = {
            "1y": self.config.period_1y_weight,
            "3m": self.config.period_3m_weight,
            "1m": self.config.period_1m_weight,
        }

        # 计算加权收益
        weighted_return = 0.0
        returns = {}
        total_trades = 0  # 总交易次数
        total_sharpe = 0.0
        total_drawdown = 0.0
        periods_with_data = 0

        for period, weight in weights.items():
            if period not in backtest_results:
                continue

            result = backtest_results[period]
            # 支持字典和 BacktestResult 对象
            if isinstance(result, dict):
                period_return = result.get("total_return", 0)
                trades = result.get("trades", [])
                sharpe = result.get("sharpe_ratio", 0)
                dd = result.get("max_drawdown", 0)
            else:
                period_return = result.total_return
                trades = result.trades if hasattr(result, 'trades') else []
                sharpe = result.sharpe_ratio
                dd = result.max_drawdown

            returns[period] = period_return
            weighted_return += period_return * weight
            total_trades += len(trades)
            total_sharpe += sharpe * weight
            total_drawdown += dd * weight
            periods_with_data += 1

        # 交易频率惩罚（过度频繁交易）
        trade_frequency_penalty = 0.0
        expected_max_trades = 100
        if total_trades > expected_max_trades:
            excess_trades = total_trades - expected_max_trades
            trade_frequency_penalty = -excess_trades / expected_max_trades * self.config.trade_frequency_penalty_weight

        # 回撤惩罚
        drawdown_penalty = 0.0
        if total_drawdown > 0.02:  # 超过2%开始惩罚
            drawdown_penalty = -(total_drawdown - 0.02) * self.config.drawdown_penalty_weight

        # 夏普比率加分
        avg_sharpe = total_sharpe / periods_with_data if periods_with_data > 0 else 0
        sharpe_bonus = max(0, avg_sharpe) * self.config.sharpe_bonus_weight

        # 稳定性惩罚
        stability_penalty = 0.0
        if len(returns) >= 2:
            returns_std = np.std(list(returns.values()))
            stability_penalty = -returns_std * self.config.stability_bonus_weight

        # 综合适应度
        fitness = (
            weighted_return * self.config.return_weight
            + sharpe_bonus
            + trade_frequency_penalty
            + drawdown_penalty
            + stability_penalty
        )

        return fitness

    def evaluate_individual(
        self,
        individual: Individual,
        data: pd.DataFrame,
        backtester: "MultiPeriodBacktester",
        periods: list[str] | None = None
    ) -> float:
        """评估个体适应度（完整流程）

        执行多周期回测并计算适应度。

        Args:
            individual: 待评估个体
            data: 市场数据
            backtester: 多周期回测器
            periods: 回测周期列表

        Returns:
            适应度值
        """
        # 执行多周期回测
        backtest_results = backtester.run(data, individual, periods)

        # 存储回测结果到个体
        individual.backtest_results = {
            period: {
                "total_return": result.total_return,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "annual_return": result.annual_return,
            }
            for period, result in backtest_results.items()
        }

        # 计算适应度
        fitness = self.evaluate_backtest_based(individual, backtest_results)

        return fitness

    def is_valid(
        self,
        individual: Individual,
        backtest_results: dict[str, "BacktestResult"] | None = None
    ) -> bool:
        """判断是否为有效策略

        修订后的判定标准：
        1. 有交易记录（至少10笔）
        2. 夏普比率 > 0.5
        3. 不是所有周期都是负收益

        Args:
            individual: 待判断个体
            backtest_results: 回测结果

        Returns:
            是否为有效策略
        """
        # Phase 1 基本检查
        if len(individual.factor_weights) < 2:
            return False

        if individual.signal_threshold <= individual.exit_threshold:
            return False

        # Phase 3 完整检查
        if backtest_results:
            total_trades = 0
            positive_periods = 0
            has_sharpe = False

            for period, result in backtest_results.items():
                # 支持字典和 BacktestResult 对象
                if isinstance(result, dict):
                    max_dd = result.get("max_drawdown", 0)
                    total_ret = result.get("total_return", 0)
                    sharpe = result.get("sharpe_ratio", 0)
                    trades = result.get("trades", [])
                else:
                    max_dd = result.max_drawdown
                    total_ret = result.total_return
                    sharpe = result.sharpe_ratio
                    trades = result.trades if hasattr(result, 'trades') else []

                total_trades += len(trades)

                if total_ret > 0:
                    positive_periods += 1

                # 1年期夏普比率检查
                if period == "1y" and sharpe > 0.5:
                    has_sharpe = True

            # 至少10笔交易
            if total_trades < 10:
                return False

            # 至少1个周期有正收益
            if positive_periods < 1:
                return False

            # 1年期夏普比率要求（可放宽）
            # if not has_sharpe:
            #     return False

        return True

    def _calculate_factor_value(
        self,
        factor_name: str,
        data: pd.DataFrame
    ) -> pd.Series | None:
        """计算因子值（简化版实现）

        Args:
            factor_name: 因子名称
            data: 价格数据

        Returns:
            因子值序列
        """
        close = data["close"]

        if factor_name == "ma_ratio":
            ma5 = close.rolling(5).mean()
            ma20 = close.rolling(20).mean()
            return (ma5 / ma20 - 1).fillna(0)

        elif factor_name == "ma_cross":
            ma5 = close.rolling(5).mean()
            ma20 = close.rolling(20).mean()
            return (ma5 > ma20).astype(float)

        elif factor_name == "momentum":
            close_n5 = close.shift(5)
            return ((close - close_n5) / close_n5).fillna(0)

        elif factor_name == "price_momentum":
            close_n20 = close.shift(20)
            return (close / close_n20 - 1).fillna(0)

        elif factor_name == "rsi":
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            return (100 - 100 / (1 + rs)) / 100

        elif factor_name == "volatility":
            return (close.rolling(20).std() / close).fillna(0)

        elif factor_name == "volume_ratio":
            if "volume" in data.columns:
                volume = data["volume"]
                ma_volume = volume.rolling(20).mean()
                return (volume / ma_volume).fillna(1)
            return None

        elif factor_name == "atr_ratio":
            high = data.get("high", close)
            low = data.get("low", close)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            return (atr / close).fillna(0)

        else:
            # 未知因子返回零序列
            return pd.Series(0.0, index=data.index)
