"""风险控制模块测试"""

import numpy as np
import pandas as pd
import pytest

from src.risk.position_sizer import PositionSizer, PositionSize
from src.risk.risk_controller import RiskController, RiskAction, RiskCheckResult


class TestPositionSizer:
    """仓位管理器测试"""

    def test_init(self):
        """测试初始化"""
        sizer = PositionSizer(method="equal")
        assert sizer.method == "equal"
        assert sizer.max_single_position == 0.10

    def test_equal_weight(self):
        """测试等权分配"""
        sizer = PositionSizer(method="equal")
        signals = {"stock1": 1.0, "stock2": 1.0, "stock3": 1.0}

        weights = sizer._equal_weight(signals)
        assert len(weights) == 3
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_volatility_target_weight(self):
        """测试波动率目标权重"""
        sizer = PositionSizer(method="volatility_target")

        # 创建模拟历史数据
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        historical_data = {
            "stock1": pd.DataFrame({
                "date": dates,
                "close": 100 + np.cumsum(np.random.randn(100) * 0.02),
                "high": 102 + np.cumsum(np.random.randn(100) * 0.02),
                "low": 98 + np.cumsum(np.random.randn(100) * 0.02),
            }),
            "stock2": pd.DataFrame({
                "date": dates,
                "close": 50 + np.cumsum(np.random.randn(100) * 0.03),
                "high": 51 + np.cumsum(np.random.randn(100) * 0.03),
                "low": 49 + np.cumsum(np.random.randn(100) * 0.03),
            }),
        }

        for df in historical_data.values():
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        signals = {"stock1": 1.0, "stock2": 1.0}
        weights = sizer._volatility_target_weight(signals, historical_data, 100000)

        assert len(weights) == 2
        assert all(w >= 0 for w in weights.values())

    def test_apply_limits_single_position(self):
        """测试单股仓位限制"""
        sizer = PositionSizer(method="equal")
        sizer.max_single_position = 0.10

        positions = [
            PositionSize(
                symbol="stock1",
                weight=0.15,  # 超过限制
                shares=100,
                value=10000,
                method="equal",
            )
        ]

        adjusted = sizer.apply_limits(positions)
        assert adjusted[0].weight <= 0.10
        assert adjusted[0].constrained

    def test_apply_limits_total_position(self):
        """测试总仓位限制"""
        sizer = PositionSizer(method="equal")
        sizer.max_total_position = 0.80

        positions = [
            PositionSize(symbol=f"stock{i}", weight=0.30, shares=100, value=10000, method="equal")
            for i in range(3)
        ]

        adjusted = sizer.apply_limits(positions)
        total = sum(p.weight for p in adjusted)
        assert total <= 0.80


class TestRiskController:
    """风险控制器测试"""

    def test_init(self):
        """测试初始化"""
        controller = RiskController()
        assert controller.fixed_stop_loss == 0.08
        assert controller.trailing_stop_percent == 0.05

    def test_register_position(self):
        """测试持仓注册"""
        controller = RiskController()
        controller.register_position("stock1", 100.0, 100)

        assert "stock1" in controller._positions
        assert controller._positions["stock1"]["entry_price"] == 100.0

    def test_stop_loss_trigger(self):
        """测试止损触发"""
        controller = RiskController()
        controller.register_position("stock1", 100.0, 100)

        # 价格下跌超过 8%
        result = controller.check_position_risk("stock1", 90.0, 9000)

        assert not result.passed
        assert result.action == RiskAction.STOP_LOSS

    def test_trailing_stop_trigger(self):
        """测试移动止损触发"""
        controller = RiskController()
        controller.register_position("stock1", 100.0, 100)

        # 价格先上涨 12% (触发移动止损条件)
        controller.update_position_price("stock1", 112.0)
        assert controller._positions["stock1"]["highest_price"] == 112.0

        # 从高点回落到 105，回撤约 6.25% (超过 5% trailing stop)
        # 同时盈利 5% (超过 10% 阈值)
        # 实际上 105/100 = 5% 盈利，不满足 10% 条件
        # 需要价格不低于 110 才能保持 10% 盈利

        # 价格回撤到 105.6（刚好 5% 盈利 + 5.7% 回撤）
        # 重新设置：从 112 回到 105，盈利 5%，回撤 6.25%
        # 条件不满足，因为盈利 < 10%

        # 更正测试：使用满足两个条件的价格
        # 入场价 100，需要盈利 >= 10%，所以当前价 >= 110
        # 最高价 112，回撤 >= 5%，所以当前价 <= 106.4
        # 两个条件同时满足: 110 <= 当前价 <= 106.4 不可能

        # 重新设置：让最高价更高
        controller.update_position_price("stock1", 120.0)  # 最高价 120

        # 从 120 回到 113，盈利 13% (>10%)，回撤 5.8% (>5%)
        result = controller.check_position_risk("stock1", 113.0, 11300)

        assert not result.passed
        assert result.action == RiskAction.TRAILING_STOP

    def test_portfolio_risk_daily_loss(self):
        """测试日亏损限制"""
        controller = RiskController()
        controller._portfolio_high = 100000

        positions = {
            "stock1": {"price": 50, "shares": 100, "value": 5000}
        }

        # 日亏损 4% (超过 3% 限制)
        result = controller.check_portfolio_risk(
            positions, 96000, daily_pnl=-4000
        )

        assert not result.passed
        assert result.action == RiskAction.SUSPEND_TRADING

    def test_portfolio_risk_drawdown(self):
        """测试回撤限制"""
        controller = RiskController()
        controller._portfolio_high = 100000

        positions = {
            "stock1": {"price": 50, "shares": 100, "value": 5000}
        }

        # 回撤 16% (超过 15% 限制)
        result = controller.check_portfolio_risk(
            positions, 84000, daily_pnl=-1000
        )

        assert not result.passed

    def test_calculate_stop_loss(self):
        """测试止损价格计算"""
        controller = RiskController()

        # 固定止损
        stop_price, stop_type = controller.calculate_stop_loss(100.0, 95.0)
        assert stop_type == "fixed"
        assert abs(stop_price - 92.0) < 0.01  # 100 * 0.92

        # 移动止损
        stop_price, stop_type = controller.calculate_stop_loss(100.0, 115.0, 115.0)
        assert stop_type == "trailing"
        assert abs(stop_price - 109.25) < 0.01  # 115 * 0.95

    def test_trading_suspension(self):
        """测试交易暂停"""
        controller = RiskController()

        assert not controller.is_trading_suspended()

        controller._trading_suspended = True
        assert controller.is_trading_suspended()

        controller.resume_trading()
        assert not controller.is_trading_suspended()
