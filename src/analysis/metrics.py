"""
绩效指标计算模块

包含以下指标类别：
- 收益指标: 总收益、年化收益、累计收益
- 风险指标: 最大回撤、波动率、下行风险
- 风险调整收益: 夏普比率、索提诺比率、卡玛比率
- 交易统计: 胜率、盈亏比、交易次数
- 因子分析: IC、IR、因子收益
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """绩效指标数据类"""

    # 基本信息
    start_date: str = ""
    end_date: str = ""
    total_days: int = 0
    trading_days: int = 0

    # 收益指标
    total_return: float = 0.0
    annual_return: float = 0.0
    monthly_return: float = 0.0
    daily_return_mean: float = 0.0

    # 风险指标
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    volatility: float = 0.0
    downside_volatility: float = 0.0
    var_95: float = 0.0  # 95% VaR
    cvar_95: float = 0.0  # 95% CVaR

    # 风险调整收益
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0

    # 交易统计
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # 基准比较
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    tracking_error: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0

    # 其他
    metrics_dict: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "period": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "total_days": self.total_days,
                "trading_days": self.trading_days,
            },
            "returns": {
                "total_return": self.total_return,
                "annual_return": self.annual_return,
                "monthly_return": self.monthly_return,
                "daily_return_mean": self.daily_return_mean,
            },
            "risk": {
                "max_drawdown": self.max_drawdown,
                "max_drawdown_duration": self.max_drawdown_duration,
                "volatility": self.volatility,
                "downside_volatility": self.downside_volatility,
                "var_95": self.var_95,
                "cvar_95": self.cvar_95,
            },
            "risk_adjusted": {
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
                "information_ratio": self.information_ratio,
            },
            "trading": {
                "win_rate": self.win_rate,
                "profit_loss_ratio": self.profit_loss_ratio,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
            },
            "benchmark": {
                "benchmark_return": self.benchmark_return,
                "excess_return": self.excess_return,
                "tracking_error": self.tracking_error,
                "beta": self.beta,
                "alpha": self.alpha,
            },
        }


def calculate_returns(portfolio_value: pd.Series) -> pd.Series:
    """计算收益率序列"""
    return portfolio_value.pct_change().dropna()


def calculate_total_return(portfolio_value: pd.Series) -> float:
    """计算总收益率"""
    if len(portfolio_value) < 2:
        return 0.0
    return portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1


def calculate_annual_return(portfolio_value: pd.Series) -> float:
    """计算年化收益率"""
    if len(portfolio_value) < 2:
        return 0.0

    total_return = calculate_total_return(portfolio_value)
    days = (portfolio_value.index[-1] - portfolio_value.index[0]).days

    if days <= 0:
        return 0.0

    return (1 + total_return) ** (365 / days) - 1


def calculate_max_drawdown(portfolio_value: pd.Series) -> tuple[float, int]:
    """
    计算最大回撤和持续时间

    Returns:
        (最大回撤, 持续天数)
    """
    if len(portfolio_value) < 2:
        return 0.0, 0

    cummax = portfolio_value.cummax()
    drawdown = (cummax - portfolio_value) / cummax

    max_dd = drawdown.max()

    # 计算最大回撤持续时间
    max_dd_idx = drawdown.idxmax()
    peak_idx = portfolio_value[:max_dd_idx].idxmax()

    duration = (max_dd_idx - peak_idx).days if hasattr(max_dd_idx, 'days') else 0

    return max_dd, duration


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """计算波动率"""
    if len(returns) < 2:
        return 0.0

    vol = returns.std()

    if annualize:
        vol *= np.sqrt(252)

    return vol


def calculate_downside_volatility(
    returns: pd.Series,
    target_return: float = 0.0,
    annualize: bool = True,
) -> float:
    """计算下行波动率"""
    if len(returns) < 2:
        return 0.0

    downside_returns = returns[returns < target_return]

    if len(downside_returns) < 2:
        return 0.0

    vol = downside_returns.std()

    if annualize:
        vol *= np.sqrt(252)

    return vol


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
) -> float:
    """计算夏普比率"""
    if len(returns) < 2:
        return 0.0

    excess_return = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)

    if volatility == 0:
        return 0.0

    return excess_return / volatility


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
) -> float:
    """计算索提诺比率"""
    if len(returns) < 2:
        return 0.0

    excess_return = returns.mean() * 252 - risk_free_rate
    downside_vol = calculate_downside_volatility(returns)

    if downside_vol == 0:
        return 0.0

    return excess_return / downside_vol


def calculate_calmar_ratio(
    annual_return: float,
    max_drawdown: float,
) -> float:
    """计算卡玛比率"""
    if max_drawdown == 0:
        return 0.0
    return annual_return / max_drawdown


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """计算 VaR (Value at Risk)"""
    if len(returns) < 2:
        return 0.0

    return -np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """计算 CVaR (Conditional VaR)"""
    if len(returns) < 2:
        return 0.0

    var = calculate_var(returns, confidence)
    return -returns[returns <= -var].mean()


def calculate_win_rate(returns: pd.Series) -> float:
    """计算胜率"""
    if len(returns) == 0:
        return 0.0

    wins = (returns > 0).sum()
    return wins / len(returns)


def calculate_profit_loss_ratio(returns: pd.Series) -> float:
    """计算盈亏比"""
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    avg_win = wins.mean()
    avg_loss = abs(losses.mean())

    if avg_loss == 0:
        return 0.0

    return avg_win / avg_loss


def calculate_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """计算 Beta"""
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0

    # 对齐日期
    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")

    if len(aligned_returns) < 2:
        return 0.0

    covariance = aligned_returns.cov(aligned_benchmark)
    variance = aligned_benchmark.var()

    if variance == 0:
        return 0.0

    return covariance / variance


def calculate_alpha(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    beta: float,
    risk_free_rate: float = 0.03,
) -> float:
    """计算 Alpha"""
    if len(returns) < 2:
        return 0.0

    annual_return = returns.mean() * 252
    benchmark_annual_return = benchmark_returns.mean() * 252

    return annual_return - risk_free_rate - beta * (benchmark_annual_return - risk_free_rate)


def calculate_tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """计算跟踪误差"""
    if len(returns) < 2:
        return 0.0

    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")

    if len(aligned_returns) < 2:
        return 0.0

    excess_returns = aligned_returns - aligned_benchmark
    return excess_returns.std() * np.sqrt(252)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """计算信息比率"""
    if len(returns) < 2:
        return 0.0

    aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")

    if len(aligned_returns) < 2:
        return 0.0

    excess_returns = aligned_returns - aligned_benchmark
    tracking_error = excess_returns.std() * np.sqrt(252)

    if tracking_error == 0:
        return 0.0

    return excess_returns.mean() * 252 / tracking_error


def calculate_consecutive_wins_losses(
    returns: pd.Series,
) -> tuple[int, int]:
    """计算最大连续赢/输次数"""
    if len(returns) == 0:
        return 0, 0

    wins = (returns > 0).astype(int)
    losses = (returns < 0).astype(int)

    def max_consecutive(arr):
        max_count = 0
        current_count = 0
        for val in arr:
            if val == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    return max_consecutive(wins.values), max_consecutive(losses.values)


def calculate_all_metrics(
    portfolio_value: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.03,
) -> PerformanceMetrics:
    """
    计算所有绩效指标

    Args:
        portfolio_value: 组合净值序列
        benchmark: 基准净值序列
        risk_free_rate: 无风险利率

    Returns:
        PerformanceMetrics 对象
    """
    metrics = PerformanceMetrics()

    if portfolio_value is None or len(portfolio_value) < 2:
        return metrics

    # 基本信息
    metrics.start_date = portfolio_value.index[0].strftime("%Y-%m-%d")
    metrics.end_date = portfolio_value.index[-1].strftime("%Y-%m-%d")
    metrics.total_days = (portfolio_value.index[-1] - portfolio_value.index[0]).days
    metrics.trading_days = len(portfolio_value)

    # 计算收益率
    returns = calculate_returns(portfolio_value)

    # 收益指标
    metrics.total_return = calculate_total_return(portfolio_value)
    metrics.annual_return = calculate_annual_return(portfolio_value)
    metrics.monthly_return = (1 + metrics.annual_return) ** (1 / 12) - 1
    metrics.daily_return_mean = returns.mean()

    # 风险指标
    metrics.max_drawdown, metrics.max_drawdown_duration = calculate_max_drawdown(portfolio_value)
    metrics.volatility = calculate_volatility(returns)
    metrics.downside_volatility = calculate_downside_volatility(returns)
    metrics.var_95 = calculate_var(returns, 0.95)
    metrics.cvar_95 = calculate_cvar(returns, 0.95)

    # 风险调整收益
    metrics.sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    metrics.sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate)
    metrics.calmar_ratio = calculate_calmar_ratio(metrics.annual_return, metrics.max_drawdown)

    # 交易统计
    metrics.win_rate = calculate_win_rate(returns)
    metrics.profit_loss_ratio = calculate_profit_loss_ratio(returns)

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    metrics.avg_win = wins.mean() if len(wins) > 0 else 0
    metrics.avg_loss = losses.mean() if len(losses) > 0 else 0

    metrics.max_consecutive_wins, metrics.max_consecutive_losses = calculate_consecutive_wins_losses(
        returns
    )

    # 基准比较
    if benchmark is not None and len(benchmark) >= 2:
        benchmark_returns = calculate_returns(benchmark)

        metrics.benchmark_return = calculate_total_return(benchmark)
        metrics.excess_return = metrics.total_return - metrics.benchmark_return
        metrics.tracking_error = calculate_tracking_error(returns, benchmark_returns)
        metrics.beta = calculate_beta(returns, benchmark_returns)
        metrics.alpha = calculate_alpha(returns, benchmark_returns, metrics.beta, risk_free_rate)
        metrics.information_ratio = calculate_information_ratio(returns, benchmark_returns)

    return metrics


if __name__ == "__main__":
    # 测试
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    returns = np.random.normal(0.0005, 0.015, 252)
    portfolio_value = pd.Series(1000000 * (1 + returns).cumprod(), index=dates)

    metrics = calculate_all_metrics(portfolio_value)

    print("绩效指标:")
    print(f"  总收益: {metrics.total_return:.2%}")
    print(f"  年化收益: {metrics.annual_return:.2%}")
    print(f"  最大回撤: {metrics.max_drawdown:.2%}")
    print(f"  夏普比率: {metrics.sharpe_ratio:.2f}")
    print(f"  胜率: {metrics.win_rate:.2%}")
