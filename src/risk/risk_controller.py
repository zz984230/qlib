"""
风险控制器

实时监控持仓风险，执行止损和仓位调整
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """风险操作类型"""

    NONE = "none"
    REDUCE_POSITION = "reduce_position"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    SUSPEND_TRADING = "suspend_trading"


@dataclass
class RiskCheckResult:
    """风险检查结果"""

    passed: bool
    action: RiskAction
    symbol: Optional[str] = None
    message: str = ""
    severity: str = "info"  # info, warning, critical
    suggested_action: str = ""
    stop_price: Optional[float] = None
    position_adjustment: float = 1.0  # 1.0 表示不调整


class RiskController:
    """
    风险控制器

    功能:
    - 单股止损检查
    - 组合风险监控
    - 动态止损计算
    - 交易暂停机制
    """

    def __init__(self, config_path: str = "configs/risk.yaml"):
        """初始化风险控制器"""
        # 加载配置
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {}

        # 风险告警参数
        alerts = config.get("risk_alerts", {})
        self.daily_loss_limit = alerts.get("daily_loss_limit", 0.03)
        self.drawdown_limit = alerts.get("drawdown_limit", 0.15)

        # 止损参数
        self.fixed_stop_loss = 0.08  # 固定止损 8%
        self.trailing_stop_trigger = 0.10  # 盈利 10% 后启用移动止损
        self.trailing_stop_percent = 0.05  # 移动止损 5%

        # 持仓追踪
        self._positions = {}  # symbol -> {entry_price, highest_price, shares}
        self._portfolio_high = 0.0
        self._daily_pnl = 0.0
        self._trading_suspended = False

        logger.info(
            f"风险控制器初始化: fixed_stop={self.fixed_stop_loss:.1%}, "
            f"trailing_stop={self.trailing_stop_percent:.1%}, "
            f"daily_limit={self.daily_loss_limit:.1%}"
        )

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        shares: int,
    ) -> None:
        """注册新持仓"""
        self._positions[symbol] = {
            "entry_price": entry_price,
            "highest_price": entry_price,
            "shares": shares,
        }
        logger.debug(f"注册持仓: {symbol} @ {entry_price:.2f}, {shares}股")

    def update_position_price(self, symbol: str, current_price: float) -> None:
        """更新持仓最高价（用于移动止损）"""
        if symbol in self._positions:
            if current_price > self._positions[symbol]["highest_price"]:
                self._positions[symbol]["highest_price"] = current_price

    def remove_position(self, symbol: str) -> None:
        """移除持仓"""
        self._positions.pop(symbol, None)

    def check_position_risk(
        self,
        symbol: str,
        current_price: float,
        position_value: float,
    ) -> RiskCheckResult:
        """
        检查单股风险

        Args:
            symbol: 股票代码
            current_price: 当前价格
            position_value: 持仓市值

        Returns:
            风险检查结果
        """
        if symbol not in self._positions:
            return RiskCheckResult(
                passed=True,
                action=RiskAction.NONE,
                symbol=symbol,
                message="无持仓记录",
            )

        pos = self._positions[symbol]
        entry_price = pos["entry_price"]
        highest_price = pos["highest_price"]

        # 更新最高价
        if current_price > highest_price:
            pos["highest_price"] = current_price
            highest_price = current_price

        # 计算盈亏比例
        pnl_ratio = (current_price - entry_price) / entry_price
        drawdown_from_high = (highest_price - current_price) / highest_price

        # 1. 检查移动止损（优先级最高）
        # 条件：盈利超过阈值 且 从最高点回撤超过阈值
        if pnl_ratio >= self.trailing_stop_trigger and drawdown_from_high >= self.trailing_stop_percent:
            stop_price = highest_price * (1 - self.trailing_stop_percent)
            return RiskCheckResult(
                passed=False,
                action=RiskAction.TRAILING_STOP,
                symbol=symbol,
                message=f"触发移动止损: 从高点回撤 {drawdown_from_high:.1%}",
                severity="warning",
                suggested_action=f"以 {stop_price:.2f} 卖出",
                stop_price=stop_price,
            )

        # 2. 检查固定止损
        if pnl_ratio <= -self.fixed_stop_loss:
            stop_price = entry_price * (1 - self.fixed_stop_loss)
            return RiskCheckResult(
                passed=False,
                action=RiskAction.STOP_LOSS,
                symbol=symbol,
                message=f"触发固定止损: 亏损 {abs(pnl_ratio):.1%}",
                severity="critical",
                suggested_action=f"立即以 {stop_price:.2f} 卖出",
                stop_price=stop_price,
            )

        # 3. 检查大幅回撤预警
        if drawdown_from_high >= self.trailing_stop_percent * 0.8:
            return RiskCheckResult(
                passed=True,
                action=RiskAction.NONE,
                symbol=symbol,
                message=f"接近移动止损线: 从高点回撤 {drawdown_from_high:.1%}",
                severity="warning",
                suggested_action="密切监控",
            )

        return RiskCheckResult(
            passed=True,
            action=RiskAction.NONE,
            symbol=symbol,
            message=f"持仓正常: 盈亏 {pnl_ratio:+.1%}",
        )

    def check_portfolio_risk(
        self,
        positions: dict[str, dict],
        portfolio_value: float,
        daily_pnl: float = 0.0,
    ) -> RiskCheckResult:
        """
        检查组合风险

        Args:
            positions: 持仓字典 {symbol: {price, shares, value}}
            portfolio_value: 组合总价值
            daily_pnl: 当日盈亏

        Returns:
            风险检查结果
        """
        # 更新组合高点
        if portfolio_value > self._portfolio_high:
            self._portfolio_high = portfolio_value

        # 计算组合回撤
        if self._portfolio_high > 0:
            portfolio_drawdown = (
                self._portfolio_high - portfolio_value
            ) / self._portfolio_high
        else:
            portfolio_drawdown = 0

        # 1. 检查日亏损限制
        daily_return = daily_pnl / portfolio_value if portfolio_value > 0 else 0
        if daily_return <= -self.daily_loss_limit:
            self._trading_suspended = True
            return RiskCheckResult(
                passed=False,
                action=RiskAction.SUSPEND_TRADING,
                message=f"触发日亏损限制: 当日亏损 {abs(daily_return):.1%}",
                severity="critical",
                suggested_action="暂停交易，减仓至 50%",
                position_adjustment=0.5,
            )

        # 2. 检查组合回撤限制
        if portfolio_drawdown >= self.drawdown_limit:
            self._trading_suspended = True
            return RiskCheckResult(
                passed=False,
                action=RiskAction.SUSPEND_TRADING,
                message=f"触发回撤限制: 组合回撤 {portfolio_drawdown:.1%}",
                severity="critical",
                suggested_action="暂停交易，全面减仓",
                position_adjustment=0.3,
            )

        # 3. 回撤预警
        if portfolio_drawdown >= self.drawdown_limit * 0.7:
            return RiskCheckResult(
                passed=True,
                action=RiskAction.NONE,
                message=f"接近回撤限制: 组合回撤 {portfolio_drawdown:.1%}",
                severity="warning",
                suggested_action="考虑降低仓位",
            )

        return RiskCheckResult(
            passed=True,
            action=RiskAction.NONE,
            message=f"组合正常: 回撤 {portfolio_drawdown:.1%}",
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        highest_price: Optional[float] = None,
    ) -> tuple[float, str]:
        """
        计算止损价格

        Args:
            entry_price: 入场价
            current_price: 当前价
            highest_price: 持仓期间最高价

        Returns:
            (止损价, 止损类型)
        """
        highest_price = highest_price or current_price
        pnl_ratio = (current_price - entry_price) / entry_price

        # 如果盈利超过阈值，使用移动止损
        if pnl_ratio >= self.trailing_stop_trigger:
            stop_price = highest_price * (1 - self.trailing_stop_percent)
            return stop_price, "trailing"

        # 否则使用固定止损
        stop_price = entry_price * (1 - self.fixed_stop_loss)
        return stop_price, "fixed"

    def is_trading_suspended(self) -> bool:
        """检查交易是否暂停"""
        return self._trading_suspended

    def resume_trading(self) -> None:
        """恢复交易"""
        self._trading_suspended = False
        logger.info("交易已恢复")

    def reset_daily(self) -> None:
        """每日重置（用于日度风险控制）"""
        self._daily_pnl = 0.0
        # 如果组合回撤恢复，可以恢复交易
        # 但需要谨慎，建议手动恢复

    def get_risk_report(self, portfolio_value: float) -> dict:
        """生成风险报告"""
        portfolio_drawdown = 0
        if self._portfolio_high > 0:
            portfolio_drawdown = (
                self._portfolio_high - portfolio_value
            ) / self._portfolio_high

        return {
            "trading_suspended": self._trading_suspended,
            "portfolio_high": self._portfolio_high,
            "current_drawdown": portfolio_drawdown,
            "drawdown_limit": self.drawdown_limit,
            "daily_loss_limit": self.daily_loss_limit,
            "position_count": len(self._positions),
            "risk_status": "critical"
            if self._trading_suspended
            else "warning"
            if portfolio_drawdown > self.drawdown_limit * 0.7
            else "normal",
        }
