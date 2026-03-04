"""海龟交易法则仓位和风险管理模块

实现海龟交易法则的核心功能：
- ATR 动态仓位管理
- 金字塔加仓
- 双重止损（固定止损 + 移动止损）
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """持仓信息"""

    symbol: str
    entry_price: float
    units: int                      # 当前单位数
    avg_price: float                # 平均成本
    stop_loss_price: float          # 止损价格
    highest_price: float = field(default_factory=lambda: 0.0)  # 最高价（用于移动止损）
    entry_atr: float = 0.0          # 入场时的 ATR
    entry_date: str = ""
    is_pyramid: bool = False        # 是否为加仓单

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price


class TurtlePositionManager:
    """海龟交易法则仓位管理器

    采用 ATR（Average True Range）动态仓位管理：
    - 1 单位 = 账户 1% / ATR
    - 金字塔加仓：最多 4 个单位，间隔 0.5 ATR
    """

    def __init__(self, account_size: float, atr_period: int = 20, risk_per_unit: float = 0.01):
        """初始化仓位管理器

        Args:
            account_size: 账户总资金
            atr_period: ATR 计算周期（海龟原版 20）
            risk_per_unit: 每单位风险比例（海龟原版 1%）
        """
        self.account_size = account_size
        self.atr_period = atr_period
        self.risk_per_unit = risk_per_unit

    def calculate_atr(self, data: pd.DataFrame) -> float:
        """计算 ATR（平均真实波幅）

        Args:
            data: 包含 high, low, close 的数据

        Returns:
            当前 ATR 值
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # 计算真实波幅
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算 ATR（使用指数移动平均）
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()

        return float(atr.iloc[-1]) if len(atr) > 0 else 0.0

    def calculate_unit_size(self, price: float, atr: float, min_shares: int = 100) -> int:
        """计算一个单位（Unit）的股数

        海龟公式：1 单位 = 账户 × 1% / ATR（直接得到股数）

        这个公式的含义是：让价格每波动1 ATR时，账户盈亏恰好为1%。

        示例：账户100000，ATR=2.3，则1单位 = 100000 × 1% / 2.3 = 435股
        当价格波动1 ATR（2.3元）时，盈亏 = 435 × 2.3 = 1000元 = 账户1%

        注意：A股最小交易单位为1手（100股），返回值会被调整为100的整数倍。

        Args:
            price: 当前价格（未使用，保留参数以兼容接口）
            atr: 当前 ATR 值
            min_shares: 最小交易单位（A股默认100股=1手）

        Returns:
            单位股数（整数，100的整数倍）
        """
        if atr <= 0:
            return 0

        # 海龟原版公式：1单位 = 账户 × 1% / ATR
        risk_amount = self.account_size * self.risk_per_unit
        unit_shares = int(risk_amount / atr)

        # 调整为100股的整数倍（A股1手=100股）
        if unit_shares >= min_shares:
            # 向下取整到100的倍数
            unit_shares = (unit_shares // min_shares) * min_shares
        else:
            # 小于100股，无法交易
            return 0

        return max(0, unit_shares)

    def get_pyramid_positions(
        self,
        position: Position,
        current_price: float,
        current_atr: float
    ) -> int:
        """获取金字塔加仓信号

        加仓规则：
        - 每次加仓间隔：0.5 ATR（海龟原版）
        - 最多加仓次数：4 次（海龟原版）
        - 必须盈利后才加仓

        Args:
            position: 当前持仓
            current_price: 当前价格
            current_atr: 当前 ATR

        Returns:
            建议加仓单位数（0 表示不加仓）
        """
        # 已达到最大单位数
        if position.units >= 4:
            return 0

        # 必须盈利才能加仓（价格高于平均成本）
        if current_price <= position.avg_price:
            return 0

        # 检查加仓间隔（0.5 ATR）
        pyramid_interval = 0.5 * current_atr
        price_progress = current_price - position.entry_price

        if price_progress >= pyramid_interval * position.units:
            # 计算可以加多少单位
            units_to_add = 4 - position.units
            return units_to_add

        return 0

    def update_position(
        self,
        position: Position,
        current_price: float
    ) -> Position:
        """更新持仓信息

        更新最高价（用于移动止损）

        Args:
            position: 当前持仓
            current_price: 当前价格

        Returns:
            更新后的持仓
        """
        if current_price > position.highest_price:
            position.highest_price = current_price

        return position

    def calculate_position_value(self, position: Position, current_price: float) -> float:
        """计算持仓市值

        Args:
            position: 持仓
            current_price: 当前价格

        Returns:
            持仓市值
        """
        return position.units * current_price

    def calculate_position_pnl(
        self,
        position: Position,
        current_price: float
    ) -> tuple[float, float]:
        """计算持仓盈亏

        Args:
            position: 持仓
            current_price: 当前价格

        Returns:
            (盈亏金额, 盈亏比例)
        """
        pnl = (current_price - position.avg_price) * position.units
        pnl_ratio = (current_price - position.avg_price) / position.avg_price

        return pnl, pnl_ratio


class TurtleRiskManager:
    """海龟交易法则风控管理器

    实现双重止损系统：
    1. 固定止损：入场价 - 2*ATR（海龟原版）
    2. 移动止损：盈利激活后，跟随最高价移动
    """

    def __init__(
        self,
        stop_loss_atr: float = 2.0,
        trailing_stop_atr: float = 2.0,
        trailing_activation: float = 1.0
    ):
        """初始化风控管理器

        Args:
            stop_loss_atr: 固定止损距离（ATR倍数），海龟原版=2N
            trailing_stop_atr: 移动止损距离（ATR倍数）
            trailing_activation: 移动止损激活价格（ATR倍数），默认盈利1N后激活
        """
        self.stop_loss_atr = stop_loss_atr
        self.trailing_stop_atr = trailing_stop_atr
        self.trailing_activation = trailing_activation

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        is_pyramid: bool = False
    ) -> float:
        """计算止损价格

        - 首次入场：entry_price - 2*ATR
        - 加仓单：各自独立止损（同首次入场）
        - 所有仓位独立止损，不加权平均

        Args:
            entry_price: 入场价格
            atr: 入场时 ATR
            is_pyramid: 是否为加仓单

        Returns:
            止损价格
        """
        stop_distance = self.stop_loss_atr * atr
        stop_loss = entry_price - stop_distance

        return stop_loss

    def calculate_trailing_stop(
        self,
        highest_price: float,
        atr: float,
        entry_price: float = None
    ) -> float:
        """计算移动止损价格

        在盈利达到激活条件后，止损价跟随最高价移动。
        使用渐进式移动止损：盈利越多，止损越紧。

        Args:
            highest_price: 持仓期间最高价
            atr: 当前 ATR
            entry_price: 入场价格（用于计算盈利比例）

        Returns:
            移动止损价格
        """
        # 基础移动止损
        trailing_stop = highest_price - self.trailing_stop_atr * atr

        # 如果提供了入场价格，根据盈利程度调整止损距离
        if entry_price is not None:
            profit_pct = (highest_price - entry_price) / entry_price

            # 盈利超过20%时，收紧止损
            if profit_pct > 0.20:
                # 使用更紧的止损距离（1.5 ATR）
                trailing_stop = highest_price - 1.5 * atr
            # 盈利超过50%时，进一步收紧
            elif profit_pct > 0.50:
                trailing_stop = highest_price - 1.0 * atr
            # 盈利超过100%时，使用最紧止损
            elif profit_pct > 1.00:
                trailing_stop = highest_price - 0.5 * atr

        return trailing_stop

    def should_use_trailing_stop(self, position: Position, current_price: float) -> bool:
        """判断是否应该使用移动止损

        移动止损激活条件：盈利 >= trailing_activation * ATR

        Args:
            position: 当前持仓
            current_price: 当前价格

        Returns:
            是否使用移动止损
        """
        profit = current_price - position.entry_price
        activation_threshold = self.trailing_activation * position.entry_atr

        return profit >= activation_threshold

    def check_stop_loss(
        self,
        position: Position,
        current_price: float,
        current_atr: float
    ) -> tuple[bool, str]:
        """检查是否触发止损

        Args:
            position: 当前持仓
            current_price: 当前价格
            current_atr: 当前 ATR

        Returns:
            (是否止损, 止损类型)
        """
        # 检查是否使用移动止损
        if self.should_use_trailing_stop(position, current_price):
            stop_loss = self.calculate_trailing_stop(
                position.highest_price,
                current_atr,
                position.entry_price  # 传递入场价格用于渐进式止损
            )
            stop_type = "trailing"
        else:
            stop_loss = position.stop_loss_price
            stop_type = "fixed"

        # 判断是否触发止损
        if current_price <= stop_loss:
            return True, stop_type

        return False, stop_type

    def calculate_position_size_limit(
        self,
        account_size: float,
        atr: float,
        price: float
    ) -> int:
        """计算单个标的最大持仓限制

        海龟规则：单标的最大不超过 4 个单位

        Args:
            account_size: 账户资金
            atr: ATR
            price: 价格

        Returns:
            最大持仓股数
        """
        position_manager = TurtlePositionManager(account_size)
        unit_size = position_manager.calculate_unit_size(price, atr)
        max_units = 4

        return unit_size * max_units


@dataclass
class PortfolioState:
    """投资组合状态"""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def total_value(self) -> float:
        """计算总资产"""
        return self.cash + sum(
            p.units * p.avg_price for p in self.positions.values()
        )

    @property
    def position_count(self) -> int:
        """持仓数量"""
        return len(self.positions)

    def add_position(self, position: Position) -> None:
        """添加持仓"""
        if position.symbol in self.positions:
            # 加仓
            existing = self.positions[position.symbol]
            total_units = existing.units + position.units
            total_cost = existing.avg_price * existing.units + position.avg_price * position.units
            existing.avg_price = total_cost / total_units
            existing.units = total_units
            existing.stop_loss_price = position.stop_loss_price
        else:
            self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> Position | None:
        """移除持仓"""
        return self.positions.pop(symbol, None)

    def get_position(self, symbol: str) -> Position | None:
        """获取持仓"""
        return self.positions.get(symbol)
