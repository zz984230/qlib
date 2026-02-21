"""
仓位管理器

提供多种仓位计算方法，控制单股和组合层面的仓位风险
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """仓位信息"""

    symbol: str
    weight: float  # 权重 (0-1)
    shares: int  # 股数
    value: float  # 市值
    method: str  # 计算方法
    constrained: bool = False  # 是否经过限制调整
    original_weight: float = 0.0  # 原始权重


class PositionSizer:
    """
    仓位管理器

    支持多种仓位计算方法:
    - equal: 等权分配
    - kelly: 凯利公式
    - risk_parity: 风险平价
    - volatility_target: 波动率目标
    """

    METHODS = ["equal", "kelly", "risk_parity", "volatility_target"]

    def __init__(
        self,
        method: str = "volatility_target",
        config_path: str = "configs/risk.yaml",
    ):
        """
        初始化仓位管理器

        Args:
            method: 仓位计算方法
            config_path: 风险配置文件路径
        """
        if method not in self.METHODS:
            raise ValueError(f"未知方法: {method}. 可用方法: {self.METHODS}")

        self.method = method

        # 加载配置
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {}

        # 仓位限制
        limits = config.get("position_limits", {})
        self.max_single_position = limits.get("max_single_position", 0.10)
        self.max_sector_position = limits.get("max_sector_position", 0.30)
        self.max_total_position = limits.get("max_total_position", 0.95)

        # 波动率目标参数
        self.target_volatility = 0.15  # 目标年化波动率 15%
        self.vol_lookback = 20  # 波动率回看期

        # Kelly 参数
        self.kelly_fraction = 0.25  # 使用 25% Kelly 避免过度杠杆

        logger.info(
            f"仓位管理器初始化: method={method}, "
            f"max_single={self.max_single_position:.1%}, "
            f"max_total={self.max_total_position:.1%}"
        )

    def calculate_positions(
        self,
        signals: dict[str, float],
        portfolio_value: float,
        historical_data: dict[str, pd.DataFrame],
        prices: Optional[dict[str, float]] = None,
    ) -> list[PositionSize]:
        """
        计算各股票仓位

        Args:
            signals: 股票代码 -> 信号强度 (正值为买入)
            portfolio_value: 组合总价值
            historical_data: 股票代码 -> 历史数据 DataFrame
            prices: 股票代码 -> 当前价格 (可选，用于计算股数)

        Returns:
            仓位信息列表
        """
        # 过滤有效信号
        valid_signals = {s: v for s, v in signals.items() if v > 0}

        if not valid_signals:
            return []

        # 根据方法计算权重
        if self.method == "equal":
            weights = self._equal_weight(valid_signals)
        elif self.method == "kelly":
            weights = self._kelly_weight(valid_signals, historical_data)
        elif self.method == "risk_parity":
            weights = self._risk_parity_weight(valid_signals, historical_data)
        elif self.method == "volatility_target":
            weights = self._volatility_target_weight(
                valid_signals, historical_data, portfolio_value
            )
        else:
            weights = self._equal_weight(valid_signals)

        # 转换为仓位信息
        positions = []
        for symbol, weight in weights.items():
            if prices and symbol in prices:
                price = prices[symbol]
                value = weight * portfolio_value
                shares = int(value / price / 100) * 100  # A股以手为单位
            else:
                price = 0
                value = weight * portfolio_value
                shares = 0

            positions.append(
                PositionSize(
                    symbol=symbol,
                    weight=weight,
                    shares=shares,
                    value=value,
                    method=self.method,
                    original_weight=weight,
                )
            )

        return positions

    def apply_limits(
        self,
        positions: list[PositionSize],
        sector_map: Optional[dict[str, str]] = None,
    ) -> list[PositionSize]:
        """
        应用仓位限制

        Args:
            positions: 原始仓位列表
            sector_map: 股票代码 -> 行业映射

        Returns:
            调整后的仓位列表
        """
        if not positions:
            return positions

        total_weight = sum(p.weight for p in positions)
        adjusted_positions = []

        # 单股仓位限制
        for pos in positions:
            new_weight = min(pos.weight, self.max_single_position)
            if new_weight != pos.weight:
                pos.constrained = True
                pos.weight = new_weight
            adjusted_positions.append(pos)

        # 行业仓位限制
        if sector_map:
            sector_weights = {}
            for pos in adjusted_positions:
                sector = sector_map.get(pos.symbol, "unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + pos.weight

            for sector, weight in sector_weights.items():
                if weight > self.max_sector_position:
                    # 按比例缩减该行业仓位
                    scale = self.max_sector_position / weight
                    for pos in adjusted_positions:
                        if sector_map.get(pos.symbol, "unknown") == sector:
                            pos.weight *= scale
                            pos.constrained = True

        # 总仓位限制
        total_weight = sum(p.weight for p in adjusted_positions)
        if total_weight > self.max_total_position:
            scale = self.max_total_position / total_weight
            for pos in adjusted_positions:
                pos.weight *= scale
                pos.constrained = True

        return adjusted_positions

    def _equal_weight(self, signals: dict[str, float]) -> dict[str, float]:
        """等权分配"""
        n = len(signals)
        base_weight = 1.0 / n
        # 根据信号强度微调
        total_signal = sum(abs(v) for v in signals.values())
        if total_signal > 0:
            return {
                s: base_weight * (abs(v) / (total_signal / n))
                for s, v in signals.items()
            }
        return {s: base_weight for s in signals}

    def _kelly_weight(
        self,
        signals: dict[str, float],
        historical_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """
        Kelly 公式计算仓位

        Kelly = (p * b - q) / b
        其中 p 是胜率, q 是败率, b 是盈亏比
        """
        weights = {}

        for symbol, signal in signals.items():
            if symbol not in historical_data:
                weights[symbol] = 1.0 / len(signals)
                continue

            df = historical_data[symbol]
            if len(df) < 20:
                weights[symbol] = 1.0 / len(signals)
                continue

            # 计算历史胜率和盈亏比
            returns = df["close"].pct_change().dropna()
            if len(returns) == 0:
                weights[symbol] = 1.0 / len(signals)
                continue

            wins = returns[returns > 0]
            losses = returns[returns < 0]

            p = len(wins) / len(returns) if len(returns) > 0 else 0.5
            avg_win = wins.mean() if len(wins) > 0 else 0.01
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01
            b = avg_win / avg_loss if avg_loss > 0 else 1.0

            # Kelly 公式
            kelly = (p * b - (1 - p)) / b if b > 0 else 0

            # 使用部分 Kelly 并根据信号强度调整
            kelly = max(0, min(kelly * self.kelly_fraction * abs(signal), 0.1))
            weights[symbol] = kelly

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}

        return weights

    def _risk_parity_weight(
        self,
        signals: dict[str, float],
        historical_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """
        风险平价权重

        分配权重使每只股票对组合的风险贡献相等
        """
        volatilities = {}

        for symbol in signals:
            if symbol not in historical_data:
                volatilities[symbol] = 0.02  # 默认 2% 日波动
                continue

            df = historical_data[symbol]
            if len(df) < self.vol_lookback:
                volatilities[symbol] = 0.02
                continue

            returns = df["close"].pct_change().dropna()
            vol = returns.tail(self.vol_lookback).std()
            volatilities[symbol] = vol if vol > 0 else 0.02

        # 风险平价: 权重与波动率成反比
        inv_vols = {s: 1.0 / v for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol > 0:
            weights = {s: v / total_inv_vol for s, v in inv_vols.items()}
        else:
            weights = self._equal_weight(signals)

        return weights

    def _volatility_target_weight(
        self,
        signals: dict[str, float],
        historical_data: dict[str, pd.DataFrame],
        portfolio_value: float,
    ) -> dict[str, float]:
        """
        波动率目标权重

        调整仓位使组合波动率接近目标水平
        """
        # 首先计算风险平价权重作为基础
        base_weights = self._risk_parity_weight(signals, historical_data)

        # 计算组合预期波动率
        portfolio_var = 0.0
        for symbol, weight in base_weights.items():
            if symbol in historical_data:
                df = historical_data[symbol]
                returns = df["close"].pct_change().dropna()
                vol = returns.tail(self.vol_lookback).std()
                portfolio_var += (weight * vol) ** 2

        portfolio_vol = np.sqrt(portfolio_var)

        # 调整杠杆以达到目标波动率
        if portfolio_vol > 0:
            leverage = self.target_volatility / (portfolio_vol * np.sqrt(252))
            leverage = min(leverage, 2.0)  # 最大 2 倍杠杆
        else:
            leverage = 1.0

        # 应用杠杆调整权重
        adjusted_weights = {
            s: min(w * leverage, self.max_single_position)
            for s, w in base_weights.items()
        }

        # 确保总权重不超过限制
        total = sum(adjusted_weights.values())
        if total > self.max_total_position:
            scale = self.max_total_position / total
            adjusted_weights = {s: w * scale for s, w in adjusted_weights.items()}

        return adjusted_weights
