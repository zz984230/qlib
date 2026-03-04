"""动态风险管理器 - 3%回撤硬约束 + 动态仓位控制

实现严格的回撤控制和根据市场状态的动态仓位调整。
"""

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# 市场状态类型
MarketState = Literal["strong_trend", "weak_trend", "ranging", "volatile"]


@dataclass
class RiskAlert:
    """风险警告"""
    level: str  # "warning", "critical", "emergency"
    drawdown: float
    action: str  # 建议操作


class DynamicRiskManager:
    """动态风险管理器

    实现：
    1. 3%回撤硬约束
    2. 根据市场状态动态调整仓位
    3. 预警机制（2%减仓，2.5%清仓）
    """

    # 回撤约束（硬约束，不可改变）
    MAX_DRAWDOWN_LIMIT = 0.03  # 3%

    # 预警阈值
    WARNING_DRAWDOWN = 0.02    # 2% - 减仓50%
    CRITICAL_DRAWDOWN = 0.025  # 2.5% - 清仓

    def __init__(self, initial_cash: float):
        """初始化风险管理器

        Args:
            initial_cash: 初始资金
        """
        self.initial_cash = initial_cash
        self.peak_value = initial_cash
        self.current_value = initial_cash

        # 当前回撤
        self.current_drawdown = 0.0

        # 风险状态
        self.risk_level = "normal"  # normal, warning, critical, emergency

    def update_portfolio_value(self, current_value: float) -> RiskAlert | None:
        """更新组合价值并检查风险

        Args:
            current_value: 当前组合价值

        Returns:
            如果触发预警返回 RiskAlert，否则返回 None
        """
        self.current_value = current_value

        # 更新峰值
        if current_value > self.peak_value:
            self.peak_value = current_value

        # 计算回撤
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value

        # 检查风险等级
        return self._check_risk_level()

    def _check_risk_level(self) -> RiskAlert | None:
        """检查风险等级并生成警告

        Returns:
            风险警告对象
        """
        if self.current_drawdown >= self.MAX_DRAWDOWN_LIMIT:
            self.risk_level = "emergency"
            return RiskAlert(
                level="emergency",
                drawdown=self.current_drawdown,
                action=f"触发3%回撤硬约束，立即停止交易并清仓"
            )

        if self.current_drawdown >= self.CRITICAL_DRAWDOWN:
            self.risk_level = "critical"
            return RiskAlert(
                level="critical",
                drawdown=self.current_drawdown,
                action=f"回撤{self.current_drawdown*100:.1f}%接近上限，建议清仓"
            )

        if self.current_drawdown >= self.WARNING_DRAWDOWN:
            self.risk_level = "warning"
            return RiskAlert(
                level="warning",
                drawdown=self.current_drawdown,
                action=f"回撤{self.current_drawdown*100:.1f}%，建议减仓50%"
            )

        self.risk_level = "normal"
        return None

    def calculate_position_size(
        self,
        base_size: float,
        market_state: MarketState | None = None
    ) -> float:
        """计算调整后的仓位规模

        根据市场状态和当前风险等级调整仓位。

        Args:
            base_size: 基础仓位规模（股数）
            market_state: 市场状态

        Returns:
            调整后的仓位规模
        """
        from src.optimizer.parameter_adapter import ParameterAdapter

        # 获取市场状态对应的仓位系数
        adapter = ParameterAdapter()
        position_multiplier = adapter.get_position_multiplier(market_state)

        # 基础仓位 × 市场状态系数
        adjusted_size = base_size * position_multiplier

        # 根据风险等级进一步调整
        if self.risk_level == "warning":
            adjusted_size *= 0.5  # 减仓50%
        elif self.risk_level == "critical":
            adjusted_size *= 0.0  # 清仓
        elif self.risk_level == "emergency":
            adjusted_size *= 0.0  # 清仓

        # 确保100股的整数倍（A股1手）
        adjusted_size = int(adjusted_size / 100) * 100

        if adjusted_size < 100:
            return 0  # 不足1手，无法交易

        return adjusted_size

    def can_enter_trade(self, market_state: MarketState | None = None) -> bool:
        """判断是否可以入场

        Args:
            market_state: 市场状态

        Returns:
            是否可以入场
        """
        # 异常波动市不允许入场
        if market_state == "volatile":
            logger.info("异常波动市场，禁止入场")
            return False

        # 风险过高时不允许入场
        if self.risk_level in ["critical", "emergency"]:
            logger.info(f"风险等级{self.risk_level}，禁止入场")
            return False

        # 回撤接近上限时谨慎入场
        if self.current_drawdown > self.WARNING_DRAWDOWN:
            logger.info(f"回撤{self.current_drawdown*100:.1f}%，禁止新开仓")
            return False

        return True

    def should_exit_position(
        self,
        entry_price: float,
        current_price: float,
        market_state: MarketState | None = None
    ) -> tuple[bool, str]:
        """判断是否应该离场

        综合考虑回撤控制、市场状态、止损等因素。

        Args:
            entry_price: 入场价格
            current_price: 当前价格
            market_state: 市场状态

        Returns:
            (是否离场, 原因)
        """
        # 1. 回撤硬约束检查
        if self.risk_level == "emergency":
            return True, "触发3%回撤硬约束，强制离场"

        if self.risk_level == "critical":
            return True, f"回撤{self.current_drawdown*100:.1f}%接近上限，主动离场"

        # 2. 市场状态变化
        if market_state == "volatile":
            return True, "市场进入异常波动状态，主动离场避险"

        # 3. 计算持仓盈亏
        if entry_price > 0:
            pnl_ratio = (current_price - entry_price) / entry_price

            # 亏损达到1.5%时考虑离场
            if pnl_ratio < -0.015:
                return True, f"持仓亏损{pnl_ratio*100:.1f}%，触发止损"

        return False, ""

    def get_status(self) -> dict:
        """获取当前风险状态

        Returns:
            风险状态字典
        """
        return {
            "initial_cash": self.initial_cash,
            "peak_value": self.peak_value,
            "current_value": self.current_value,
            "current_drawdown": self.current_drawdown,
            "risk_level": self.risk_level,
            "remaining_drawdown_budget": self.MAX_DRAWDOWN_LIMIT - self.current_drawdown,
        }


# 便捷函数
def check_drawdown_limit(
    peak_value: float,
    current_value: float,
    limit: float = 0.03
) -> tuple[bool, float]:
    """检查是否超过回撤限制

    Args:
        peak_value: 峰值价值
        current_value: 当前价值
        limit: 回撤限制，默认3%

    Returns:
        (是否超限, 当前回撤)
    """
    drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
    return drawdown > limit, drawdown
