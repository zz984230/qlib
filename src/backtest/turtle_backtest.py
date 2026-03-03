"""海龟策略回测执行器

集成海龟交易法则的完整回测流程。
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING, Any
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.backtest.runner import BacktestResult
    from src.strategy.turtle_position import (
        TurtlePositionManager,
        TurtleRiskManager,
        Position,
        PortfolioState,
    )
    from src.strategy.turtle_signals import TurtleSignalGenerator
    from src.optimizer.individual import Individual

logger = logging.getLogger(__name__)


class TurtleBacktestRunner:
    """海龟策略回测执行器

    完整实现海龟交易法则的回测流程：
    1. 信号生成（因子组合）
    2. 仓位管理（ATR 动态仓位 + 金字塔加仓）
    3. 风险控制（双重止损）
    4. 出场管理
    """

    def __init__(
        self,
        symbol: str,
        initial_cash: float,
        individual: Any | None = None,
        commission: float = 0.0003,
    ):
        """初始化回测执行器

        Args:
            symbol: 股票代码
            initial_cash: 初始资金
            individual: 策略个体（包含因子权重和阈值）
            commission: 手续费率
        """
        # 延迟导入以避免循环依赖
        from src.strategy.turtle_position import (
            TurtlePositionManager,
            TurtleRiskManager,
            PortfolioState,
        )

        self.symbol = symbol
        self.initial_cash = initial_cash
        self.commission = commission
        self.individual = individual

        # 初始化各模块
        if individual:
            from src.strategy.turtle_signals import TurtleSignalGenerator
            self.signal_generator = TurtleSignalGenerator(individual)
        else:
            self.signal_generator = None

        self.position_manager = TurtlePositionManager(
            account_size=initial_cash,
            atr_period=individual.atr_period if individual else 20
        )
        self.risk_manager = TurtleRiskManager()

        # 回测状态
        self.portfolio = PortfolioState(cash=initial_cash)
        self.trades: list[dict] = []
        self.portfolio_values: list[float] = []
        self.dates: list = []

        # 指标缓存（预计算优化）
        self._indicator_cache = None
        self._atr_cache = None

    def run(self, data: pd.DataFrame):
        """执行回测

        Args:
            data: OHLCV 数据

        Returns:
            BacktestResult 对象
        """
        from src.backtest.runner import BacktestResult
        from src.strategy.turtle_position import PortfolioState
        from src.strategy.turtle_signals import IndicatorCache

        if self.signal_generator is None:
            raise ValueError("未设置信号生成器，请提供 Individual 参数")

        logger.info(f"开始海龟策略回测: {self.symbol}, 数据长度: {len(data)}")

        # 重置状态
        self.portfolio = PortfolioState(cash=self.initial_cash)
        self.trades = []
        self.portfolio_values = []
        self.dates = []

        # 计算指标（提前计算以加速）
        data = self._prepare_data(data)

        # [优化] 预计算所有技术指标并缓存
        self._indicator_cache = IndicatorCache(data)
        self.signal_generator.set_cache(self._indicator_cache)
        self._atr_cache = self._indicator_cache.get("atr14")

        logger.info(f"技术指标预计算完成")

        # 逐日回测（使用缓存的指标）
        for i in range(50, len(data)):  # 从第50天开始，确保指标有效
            current_date = data.index[i]
            current_price = data["close"].iloc[i]

            self._process_day_fast(i, current_date, current_price)

        # 构建结果
        return self._build_result(data)

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备数据（计算指标）

        Args:
            data: 原始数据

        Returns:
            添加了指标的数据
        """
        data = data.copy()

        # 确保数据列完整
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in data.columns:
                if col == "volume":
                    data[col] = 1000000  # 默认值
                else:
                    data[col] = data["close"]  # 使用 close 作为默认

        return data

    def _process_day_fast(
        self,
        bar_index: int,
        current_date: datetime,
        current_price: float
    ) -> None:
        """快速处理单个交易日（使用预计算指标）

        Args:
            bar_index: 当前K线索引
            current_date: 当前日期
            current_price: 当前价格
        """
        position = self.portfolio.get_position(self.symbol)

        # 1. 检查止损
        if position:
            current_atr = self._atr_cache[bar_index] if self._atr_cache is not None else 0.0
            should_exit, exit_type = self._check_stop_loss_fast(position, current_price, current_atr)
            if should_exit:
                self._close_position(position, current_price, current_date, f"止损 ({exit_type})")
                position = None  # 重置持仓引用

        # 2. 检查出场信号（重新获取持仓状态）
        if position is None:
            position = self.portfolio.get_position(self.symbol)
        if position:
            if self.signal_generator.should_exit(idx=bar_index):
                self._close_position(position, current_price, current_date, "信号出场")
                position = None  # 重置持仓引用

        # 3. 检查入场信号（重新获取持仓状态）
        if position is None:
            position = self.portfolio.get_position(self.symbol)
        if not position:
            if self.signal_generator.should_enter(idx=bar_index):
                self._open_position_fast(bar_index, current_date)
                position = self.portfolio.get_position(self.symbol)  # 获取新持仓

        # 4. 检查加仓
        if position is None:
            position = self.portfolio.get_position(self.symbol)
        if position:
            current_atr = self._atr_cache[bar_index] if self._atr_cache is not None else 0.0
            units_to_add = self.position_manager.get_pyramid_positions(
                position, current_price, current_atr
            )
            if units_to_add > 0:
                self._add_to_position_fast(position, bar_index, current_date, units_to_add)

        # 5. 更新持仓最高价
        if position is None:
            position = self.portfolio.get_position(self.symbol)
        if position:
            self.position_manager.update_position(position, current_price)

        # 6. 记录组合净值
        self._record_portfolio_value(current_date, current_price)

    def _check_stop_loss_fast(
        self,
        position: Any,
        current_price: float,
        current_atr: float
    ) -> tuple[bool, str]:
        """快速检查止损（使用预计算的ATR）"""
        return self.risk_manager.check_stop_loss(position, current_price, current_atr)

    def _open_position_fast(self, bar_index: int, current_date: datetime) -> None:
        """快速开仓（使用预计算的ATR）"""
        from src.strategy.turtle_position import Position

        current_price = float(self._indicator_cache.close[bar_index])
        current_atr = float(self._atr_cache[bar_index]) if self._atr_cache is not None and not np.isnan(self._atr_cache[bar_index]) else 0.0

        if current_atr <= 0:
            return

        # 计算单位规模
        unit_size = self.position_manager.calculate_unit_size(current_price, current_atr)

        if unit_size <= 0:
            return

        # 计算成本
        cost = unit_size * current_price * (1 + self.commission)

        # 检查资金
        if cost > self.portfolio.cash:
            unit_size = int(self.portfolio.cash / (current_price * (1 + self.commission)))
            cost = unit_size * current_price * (1 + self.commission)

        if unit_size <= 0:
            return

        # 计算止损价格
        stop_loss = self.risk_manager.calculate_stop_loss(current_price, current_atr)

        # 创建持仓
        position = Position(
            symbol=self.symbol,
            entry_price=current_price,
            units=unit_size,
            avg_price=current_price,
            stop_loss_price=stop_loss,
            entry_atr=current_atr,
            entry_date=str(current_date),
        )

        self.portfolio.cash -= cost
        self.portfolio.add_position(position)

        # 记录交易
        self.trades.append({
            "date": current_date,
            "symbol": self.symbol,
            "action": "buy",
            "price": current_price,
            "units": unit_size,
            "cost": cost,
            "reason": "信号入场",
        })

    def _add_to_position_fast(
        self,
        position: Any,
        bar_index: int,
        current_date: datetime,
        units_to_add: int
    ) -> None:
        """快速加仓（使用预计算的ATR）"""
        current_price = float(self._indicator_cache.close[bar_index])
        current_atr = float(self._atr_cache[bar_index]) if self._atr_cache is not None and not np.isnan(self._atr_cache[bar_index]) else 0.0

        if current_atr <= 0:
            return

        # 计算加仓数量
        unit_size = self.position_manager.calculate_unit_size(current_price, current_atr)
        actual_units = min(unit_size * units_to_add, int(self.portfolio.cash / current_price))

        if actual_units <= 0:
            return

        cost = actual_units * current_price * (1 + self.commission)

        # 更新持仓
        total_units = position.units + actual_units
        total_cost = position.avg_price * position.units + current_price * actual_units
        position.avg_price = total_cost / total_units
        position.units = total_units

        # 更新止损价格（使用最新入场价）
        position.stop_loss_price = self.risk_manager.calculate_stop_loss(
            current_price, current_atr, is_pyramid=True
        )

        self.portfolio.cash -= cost

        # 记录交易
        self.trades.append({
            "date": current_date,
            "symbol": self.symbol,
            "action": "buy",
            "price": current_price,
            "units": actual_units,
            "cost": cost,
            "reason": f"金字塔加仓 (第{int(total_units / max(1, unit_size))}单位)",
        })

    def _process_day(
        self,
        data: pd.DataFrame,
        current_date: datetime,
        bar_index: int
    ) -> None:
        """处理单个交易日

        Args:
            data: 历史数据
            current_date: 当前日期
            bar_index: 当前K线索引
        """
        current_row = data.iloc[-1]
        current_price = current_row["close"]

        # 1. 检查止损
        position = self.portfolio.get_position(self.symbol)
        if position:
            should_exit, exit_type = self._check_stop_loss(data, position)
            if should_exit:
                self._close_position(position, current_price, current_date, f"止损 ({exit_type})")
                # 注意：不要 return，继续记录净值

        # 2. 检查出场信号
        if position:
            if self.signal_generator.should_exit(data):
                self._close_position(position, current_price, current_date, "信号出场")
                # 注意：不要 return，继续记录净值

        # 3. 检查入场信号
        if not position:
            if self.signal_generator.should_enter(data):
                self._open_position(data, current_date)
                # 注意：不要 return，继续记录净值

        # 4. 检查加仓
        if position:
            current_atr = self.position_manager.calculate_atr(data)
            units_to_add = self.position_manager.get_pyramid_positions(
                position, current_price, current_atr
            )
            if units_to_add > 0:
                self._add_to_position(position, data, current_date, units_to_add)

        # 5. 更新持仓最高价
        if position:
            self.position_manager.update_position(position, current_price)

        # 6. 记录组合净值（每天都记录）
        self._record_portfolio_value(current_date, current_price)

    def _check_stop_loss(
        self,
        data: pd.DataFrame,
        position: Any
    ) -> tuple[bool, str]:
        """检查止损

        Args:
            data: 市场数据
            position: 持仓

        Returns:
            (是否止损, 止损类型)
        """
        current_price = data["close"].iloc[-1]
        current_atr = self.position_manager.calculate_atr(data)

        return self.risk_manager.check_stop_loss(position, current_price, current_atr)

    def _open_position(self, data: pd.DataFrame, current_date: datetime) -> None:
        """开仓

        Args:
            data: 市场数据
            current_date: 当前日期
        """
        from src.strategy.turtle_position import Position

        current_price = data["close"].iloc[-1]
        current_atr = self.position_manager.calculate_atr(data)

        # 计算单位规模
        unit_size = self.position_manager.calculate_unit_size(current_price, current_atr)

        if unit_size <= 0:
            return

        # 计算成本
        cost = unit_size * current_price * (1 + self.commission)

        # 检查资金
        if cost > self.portfolio.cash:
            unit_size = int(self.portfolio.cash / (current_price * (1 + self.commission)))
            cost = unit_size * current_price * (1 + self.commission)

        if unit_size <= 0:
            return

        # 计算止损价格
        stop_loss = self.risk_manager.calculate_stop_loss(current_price, current_atr)

        # 创建持仓
        position = Position(
            symbol=self.symbol,
            entry_price=current_price,
            units=unit_size,
            avg_price=current_price,
            stop_loss_price=stop_loss,
            entry_atr=current_atr,
            entry_date=str(current_date),
        )

        self.portfolio.cash -= cost
        self.portfolio.add_position(position)

        # 记录交易
        self.trades.append({
            "date": current_date,
            "symbol": self.symbol,
            "action": "buy",
            "price": current_price,
            "units": unit_size,
            "cost": cost,
            "reason": "信号入场",
        })

        logger.debug(f"{current_date} 开仓: {unit_size}股 @ {current_price:.2f}")

    def _add_to_position(
        self,
        position: Any,
        data: pd.DataFrame,
        current_date: datetime,
        units_to_add: int
    ) -> None:
        """加仓

        Args:
            position: 当前持仓
            data: 市场数据
            current_date: 当前日期
            units_to_add: 加仓单位数
        """
        current_price = data["close"].iloc[-1]
        current_atr = self.position_manager.calculate_atr(data)

        # 计算加仓数量
        unit_size = self.position_manager.calculate_unit_size(current_price, current_atr)
        actual_units = min(unit_size * units_to_add, int(self.portfolio.cash / current_price))

        if actual_units <= 0:
            return

        cost = actual_units * current_price * (1 + self.commission)

        # 更新持仓
        total_units = position.units + actual_units
        total_cost = position.avg_price * position.units + current_price * actual_units
        position.avg_price = total_cost / total_units
        position.units = total_units

        # 更新止损价格（使用最新入场价）
        position.stop_loss_price = self.risk_manager.calculate_stop_loss(
            current_price, current_atr, is_pyramid=True
        )

        self.portfolio.cash -= cost

        # 记录交易
        self.trades.append({
            "date": current_date,
            "symbol": self.symbol,
            "action": "buy",
            "price": current_price,
            "units": actual_units,
            "cost": cost,
            "reason": f"金字塔加仓 (第{int(total_units / unit_size)}单位)",
        })

        logger.debug(f"{current_date} 加仓: {actual_units}股 @ {current_price:.2f}")

    def _close_position(
        self,
        position: Any,
        current_price: float,
        current_date: datetime,
        reason: str
    ) -> None:
        """平仓

        Args:
            position: 持仓
            current_price: 当前价格
            current_date: 当前日期
            reason: 平仓原因
        """
        proceeds = position.units * current_price * (1 - self.commission)
        pnl = (current_price - position.avg_price) * position.units
        pnl_ratio = (current_price - position.avg_price) / position.avg_price

        self.portfolio.cash += proceeds
        self.portfolio.remove_position(self.symbol)

        # 记录交易
        self.trades.append({
            "date": current_date,
            "symbol": self.symbol,
            "action": "sell",
            "price": current_price,
            "units": position.units,
            "proceeds": proceeds,
            "pnl": pnl,
            "pnl_ratio": pnl_ratio,
            "reason": reason,
        })

        logger.debug(f"{current_date} 平仓: {position.units}股 @ {current_price:.2f}, 原因: {reason}")

    def _record_portfolio_value(self, current_date: datetime, current_price: float) -> None:
        """记录组合净值

        Args:
            current_date: 当前日期
            current_price: 当前价格
        """
        position = self.portfolio.get_position(self.symbol)

        if position:
            position_value = position.units * current_price
            total_value = self.portfolio.cash + position_value
        else:
            total_value = self.portfolio.cash

        self.portfolio_values.append(total_value)
        self.dates.append(current_date)

    def _build_result(self, data: pd.DataFrame):
        """构建回测结果

        Args:
            data: 市场数据

        Returns:
            BacktestResult
        """
        from src.backtest.runner import BacktestResult

        # 构建净值序列
        if len(self.dates) > 0:
            portfolio_value = pd.Series(self.portfolio_values, index=self.dates)
        else:
            portfolio_value = pd.Series([self.initial_cash], index=[data.index[0]])

        # 构建基准净值（买入持有）
        benchmark = self.initial_cash * (data["close"] / data["close"].iloc[0])

        # 构建交易记录
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        # 构建结果
        result = BacktestResult()
        result.portfolio_value = portfolio_value
        result.benchmark = benchmark
        result.trades = trades_df
        result.start_date = str(data.index[0]) if len(data) > 0 else ""
        result.end_date = str(data.index[-1]) if len(data) > 0 else ""
        result.strategy_name = f"Turtle_{self.symbol}"
        result.cash = self.initial_cash

        # 计算指标
        result.metrics = {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            "volatility": result.volatility,
        }

        logger.info(
            f"回测完成: 总收益={result.total_return:.2%}, "
            f"最大回撤={result.max_drawdown:.2%}, "
            f"夏普比率={result.sharpe_ratio:.2f}"
        )

        return result

    def get_trade_summary(self) -> dict:
        """获取交易汇总

        Returns:
            交易汇总字典
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
            }

        sell_trades = [t for t in self.trades if t["action"] == "sell"]

        winning = sum(1 for t in sell_trades if t.get("pnl", 0) > 0)
        losing = len(sell_trades) - winning
        total_pnl = sum(t.get("pnl", 0) for t in sell_trades)

        return {
            "total_trades": len(sell_trades),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": winning / len(sell_trades) if sell_trades else 0.0,
            "total_pnl": total_pnl,
        }
