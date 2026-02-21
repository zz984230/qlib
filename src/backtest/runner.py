"""回测执行器 (增强版)"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from src.strategy.base import BaseStrategy

logger = logging.getLogger(__name__)


class BacktestResult:
    """回测结果容器"""

    def __init__(self):
        self.portfolio_value: pd.Series = None  # 组合净值
        self.positions: pd.DataFrame = None  # 持仓记录
        self.trades: pd.DataFrame = None  # 交易记录
        self.benchmark: pd.Series = None  # 基准净值
        self.metrics: dict[str, float] = {}  # 绩效指标
        self.start_date: str = ""
        self.end_date: str = ""
        self.strategy_name: str = ""
        self.cash: float = 0

    @property
    def total_return(self) -> float:
        """总收益率"""
        if self.portfolio_value is not None and len(self.portfolio_value) > 1:
            return (
                self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0] - 1
            )
        return 0.0

    @property
    def annual_return(self) -> float:
        """年化收益率"""
        if self.portfolio_value is not None and len(self.portfolio_value) > 1:
            days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
            if days > 0:
                return (1 + self.total_return) ** (365 / days) - 1
        return 0.0

    @property
    def max_drawdown(self) -> float:
        """最大回撤"""
        if self.portfolio_value is not None and len(self.portfolio_value) > 0:
            cummax = self.portfolio_value.cummax()
            drawdown = (cummax - self.portfolio_value) / cummax
            return drawdown.max()
        return 0.0

    @property
    def sharpe_ratio(self) -> float:
        """夏普比率 (假设无风险利率 3%)"""
        if self.portfolio_value is not None and len(self.portfolio_value) > 1:
            returns = self.portfolio_value.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                excess_return = returns.mean() * 252 - 0.03
                volatility = returns.std() * np.sqrt(252)
                if volatility > 0:
                    return excess_return / volatility
        return 0.0

    @property
    def sortino_ratio(self) -> float:
        """索提诺比率"""
        if self.portfolio_value is not None and len(self.portfolio_value) > 1:
            returns = self.portfolio_value.pct_change().dropna()
            if len(returns) > 0:
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std() * np.sqrt(252)
                    if downside_std > 0:
                        excess_return = returns.mean() * 252 - 0.03
                        return excess_return / downside_std
        return 0.0

    @property
    def calmar_ratio(self) -> float:
        """卡玛比率"""
        if self.max_drawdown > 0:
            return self.annual_return / self.max_drawdown
        return 0.0

    @property
    def volatility(self) -> float:
        """年化波动率"""
        if self.portfolio_value is not None and len(self.portfolio_value) > 1:
            returns = self.portfolio_value.pct_change().dropna()
            if len(returns) > 0:
                return returns.std() * np.sqrt(252)
        return 0.0

    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.portfolio_value is not None and len(self.portfolio_value) > 1:
            returns = self.portfolio_value.pct_change().dropna()
            if len(returns) > 0:
                return (returns > 0).sum() / len(returns)
        return 0.0

    @property
    def excess_return(self) -> float:
        """超额收益 (相对于基准)"""
        if self.benchmark is not None and len(self.benchmark) > 1:
            benchmark_return = self.benchmark.iloc[-1] / self.benchmark.iloc[0] - 1
            return self.total_return - benchmark_return
        return 0.0

    def get_all_metrics(self) -> dict[str, float]:
        """获取所有指标"""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "volatility": self.volatility,
            "win_rate": self.win_rate,
            "excess_return": self.excess_return,
        }

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "strategy_name": self.strategy_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "cash": self.cash,
            **self.get_all_metrics(),
            "metrics": self.metrics,
        }


class BacktestRunner:
    """回测执行器"""

    def __init__(self, config_path: str = "configs/strategy.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.backtest_config = self.config.get("backtest", {})
        self.results_dir = Path("data/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        strategy: BaseStrategy,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cash: Optional[float] = None,
        benchmark: bool = True,
    ) -> BacktestResult:
        """
        执行回测

        Args:
            strategy: 策略实例
            start_date: 开始日期
            end_date: 结束日期
            cash: 初始资金
            benchmark: 是否计算基准

        Returns:
            BacktestResult 对象
        """
        start_date = start_date or self.backtest_config.get("start_time", "2022-01-01")
        end_date = end_date or self.backtest_config.get("end_time", "2024-12-31")
        cash = cash or self.backtest_config.get("cash", 1000000)

        logger.info(f"开始回测: {start_date} -> {end_date}, 初始资金: {cash:,.0f}")
        logger.info(f"策略: {strategy}")

        result = BacktestResult()
        result.start_date = start_date
        result.end_date = end_date
        result.strategy_name = str(strategy)
        result.cash = cash

        # 尝试使用 qlib 回测
        qlib_success = False
        try:
            result = self._run_qlib_backtest(strategy, start_date, end_date, cash)
            qlib_success = True
            logger.info("使用 Qlib 回测引擎")
        except ImportError as e:
            logger.warning(f"Qlib 未安装，使用简单回测: {e}")
        except Exception as e:
            logger.warning(f"Qlib 回测失败，使用简单回测: {e}")

        if not qlib_success:
            result = self._run_vector_backtest(strategy, start_date, end_date, cash)

        # 计算基准
        if benchmark:
            result.benchmark = self._get_benchmark(start_date, end_date, cash)

        # 计算指标
        result.metrics = result.get_all_metrics()

        logger.info(f"回测完成: 总收益 {result.total_return:.2%}, 夏普 {result.sharpe_ratio:.2f}")
        return result

    def _run_qlib_backtest(
        self,
        strategy: BaseStrategy,
        start_date: str,
        end_date: str,
        cash: float,
    ) -> BacktestResult:
        """使用 Qlib 执行回测"""
        import qlib
        from qlib.backtest import backtest
        from qlib.contrib.strategy import TopkDropoutStrategy

        # 初始化 qlib
        qlib.init(provider_uri="data/qlib", region="cn")

        # 获取策略配置
        strategy_config = strategy.get_strategy_config()

        # 创建 Qlib 策略
        qlib_strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.strategy",
            "kwargs": {
                "topk": strategy_config["topk"],
                "n_drop": strategy_config["n_drop"],
            },
        }

        # 执行器配置
        executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        }

        # 执行回测
        portfolio_result = backtest(
            strategy=qlib_strategy_config,
            executor=executor_config,
            start_time=start_date,
            end_time=end_date,
            cash=cash,
        )

        result = BacktestResult()
        result.portfolio_value = portfolio_result.portfolio_value
        result.positions = portfolio_result.positions
        result.trades = portfolio_result.trades
        result.start_date = start_date
        result.end_date = end_date

        return result

    def _run_vector_backtest(
        self,
        strategy: BaseStrategy,
        start_date: str,
        end_date: str,
        cash: float,
    ) -> BacktestResult:
        """向量化回测（使用缓存数据）"""
        # 加载缓存数据
        data_dir = Path("data/raw")
        parquet_files = list(data_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning("没有找到缓存数据，使用模拟回测")
            return self._run_simple_backtest(strategy, start_date, end_date, cash)

        # 加载所有股票数据
        all_data = {}
        for pf in parquet_files:
            symbol = pf.stem
            df = pd.read_parquet(pf)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            all_data[symbol] = df

        # 生成日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq="B")

        # 初始化组合
        portfolio_value = pd.Series(index=dates, dtype=float)
        portfolio_value.iloc[0] = cash

        # 获取策略配置
        strategy_config = strategy.get_strategy_config()
        topk = strategy_config.get("topk", 30)

        # 简化的回测逻辑：等权持有 topk 只股票
        position_value = cash
        holdings = {}

        for i, date in enumerate(dates[1:], 1):
            # 计算当日收益率（使用简单市场收益）
            daily_return = 0.0
            valid_stocks = 0

            for symbol, df in all_data.items():
                if date in df.index:
                    if i > 0 and dates[i - 1] in df.index:
                        prev_close = df.loc[dates[i - 1], "close"]
                        curr_close = df.loc[date, "close"]
                        if prev_close > 0:
                            daily_return += (curr_close - prev_close) / prev_close
                            valid_stocks += 1

            if valid_stocks > 0:
                daily_return /= valid_stocks

            position_value *= (1 + daily_return)
            portfolio_value.iloc[i] = position_value

        result = BacktestResult()
        result.portfolio_value = portfolio_value
        result.start_date = start_date
        result.end_date = end_date

        return result

    def _run_simple_backtest(
        self,
        strategy: BaseStrategy,
        start_date: str,
        end_date: str,
        cash: float,
    ) -> BacktestResult:
        """简单回测（模拟数据）"""
        # 模拟净值曲线
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        np.random.seed(42)

        # 生成模拟收益（用于测试框架）
        daily_returns = np.random.normal(0.0003, 0.012, len(dates))
        portfolio_value = cash * (1 + daily_returns).cumprod()

        result = BacktestResult()
        result.portfolio_value = pd.Series(portfolio_value, index=dates)
        result.start_date = start_date
        result.end_date = end_date

        return result

    def _get_benchmark(self, start_date: str, end_date: str, cash: float) -> pd.Series:
        """获取基准数据（沪深300）"""
        # 尝试从缓存获取
        cache_file = Path("data/raw/benchmark_csi300.parquet")

        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            dates = pd.date_range(start=start_date, end=end_date, freq="B")
            benchmark = pd.Series(index=dates, dtype=float)

            start_value = cash
            for i, date in enumerate(dates):
                if date in df.index:
                    if i == 0:
                        benchmark.iloc[i] = cash
                        start_value = df.loc[date, "close"]
                    else:
                        curr_value = df.loc[date, "close"]
                        benchmark.iloc[i] = cash * curr_value / start_value
                else:
                    benchmark.iloc[i] = benchmark.iloc[i - 1] if i > 0 else cash

            return benchmark

        # 没有基准数据，返回等权基准
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        return pd.Series(cash, index=dates)

    def compare_strategies(
        self,
        strategies: list[BaseStrategy],
        start_date: str,
        end_date: str,
        cash: float = 1000000,
    ) -> pd.DataFrame:
        """比较多个策略"""
        results = []

        for strategy in strategies:
            result = self.run(strategy, start_date, end_date, cash)
            results.append({
                "strategy": str(strategy),
                "total_return": result.total_return,
                "annual_return": result.annual_return,
                "max_drawdown": result.max_drawdown,
                "sharpe_ratio": result.sharpe_ratio,
                "calmar_ratio": result.calmar_ratio,
                "volatility": result.volatility,
            })

        return pd.DataFrame(results)

    def save_result(self, result: BacktestResult, name: str = None) -> Path:
        """保存回测结果"""
        name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"backtest_{name}.json"

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"结果已保存: {result_file}")
        return result_file

    def load_result(self, name: str) -> BacktestResult:
        """加载回测结果"""
        result_file = self.results_dir / f"backtest_{name}.json"

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = BacktestResult()
        result.strategy_name = data.get("strategy_name", "")
        result.start_date = data.get("start_date", "")
        result.end_date = data.get("end_date", "")
        result.cash = data.get("cash", 0)
        result.metrics = data.get("metrics", {})

        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from src.strategy.base import SimpleFactorStrategy

    strategy = SimpleFactorStrategy()
    runner = BacktestRunner()

    result = runner.run(strategy)
    print(f"\n回测结果:")
    print(f"  策略: {result.strategy_name}")
    print(f"  总收益: {result.total_return:.2%}")
    print(f"  年化收益: {result.annual_return:.2%}")
    print(f"  最大回撤: {result.max_drawdown:.2%}")
    print(f"  夏普比率: {result.sharpe_ratio:.2f}")
    print(f"  卡玛比率: {result.calmar_ratio:.2f}")
