"""遗传算法个体（染色体）类

代表一个因子组合策略的染色体，用于海龟遗传算法优化系统。
"""

from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np
import json


@dataclass
class Individual:
    """代表一个因子组合策略的染色体

    基因编码包含因子权重和信号阈值，通过遗传算法演化寻找最优参数组合。
    """

    # ========== 策略类型 ==========
    strategy_type: Literal["turtle", "mean_reversion"] = "turtle"  # 策略类型

    # ========== 基因编码 ==========
    factor_weights: dict[str, float] = field(default_factory=dict)  # 因子权重 {"ma": 0.3, "rsi": 0.4, ...}
    signal_threshold: float = 0.5           # 入场阈值 (0-1)
    exit_threshold: float = 0.3             # 出场阈值 (0-1)

    # ========== 海龟参数 (Phase 2 使用) ==========
    atr_period: int = 20
    stop_loss_atr: float = 2.0           # 原值，范围扩展为 1.5-3.0
    pyramid_interval_atr: float = 0.5    # 原值，范围扩展为 0.3-0.7
    max_pyramid_units: int = 4
    trailing_stop_trigger: float = 1.0   # 移动止损触发点 (ATR倍数)，范围 0.5-1.5

    # ========== 趋势过滤参数 ==========
    min_adx: float = 10.0                # 最低ADX趋势强度阈值 (0-50)，默认10（非常宽松）
    min_trend_periods: int = 5           # 最少趋势周期数
    use_trend_filter: bool = False       # 是否启用趋势过滤器（默认禁用）

    # ========== 评估结果 ==========
    fitness: float = 0.0
    backtest_results: dict[str, Any] = field(default_factory=dict)

    # ========== 市场适应度（新增） ==========
    market_performance: dict[str, dict[str, float]] = field(default_factory=dict)  # 各市场状态表现
    optimal_market: str | None = None  # 最适合的市场状态

    # ========== 元数据 ==========
    generation: int = 0            # 所属代数
    parent_ids: list[str] = field(default_factory=list)  # 父代ID

    def __post_init__(self):
        """初始化后验证"""
        # 确保阈值在 [0, 1] 范围内
        self.signal_threshold = np.clip(self.signal_threshold, 0, 1)
        self.exit_threshold = np.clip(self.exit_threshold, 0, 1)

        # 归一化因子权重
        if self.factor_weights:
            total = sum(abs(w) for w in self.factor_weights.values())
            if total > 0:
                self.factor_weights = {
                    k: abs(v) / total for k, v in self.factor_weights.items()
                }

    def to_genes(self, all_factor_names: list[str] | None = None) -> np.ndarray:
        """转换为基因数组（用于遗传操作）

        Args:
            all_factor_names: 所有因子名称列表（用于确保基因长度一致）
                            如果为 None，使用个体自身的因子

        Returns:
            基因数组，格式为 [factor_weights..., signal_threshold, exit_threshold, atr_period,
                           stop_loss_atr, pyramid_interval_atr, trailing_stop_trigger, min_adx]
        """
        # 使用所有因子名称确保基因长度一致
        if all_factor_names is None:
            factor_names = sorted(self.factor_weights.keys())
        else:
            factor_names = all_factor_names

        weight_genes = [self.factor_weights.get(name, 0.0) for name in factor_names]

        # 组合所有基因
        genes = np.array([
            *weight_genes,
            self.signal_threshold,
            self.exit_threshold,
            float(self.atr_period),
            self.stop_loss_atr,
            self.pyramid_interval_atr,
            self.trailing_stop_trigger,
            self.min_adx,
        ])

        return genes

    @classmethod
    def from_genes(
        cls,
        genes: np.ndarray,
        factor_names: list[str],
        generation: int = 0,
        parent_ids: list[str] | None = None,
    ) -> "Individual":
        """从基因数组创建个体

        Args:
            genes: 基因数组
            factor_names: 因子名称列表（用于解析权重）
            generation: 所属代数
            parent_ids: 父代ID列表

        Returns:
            Individual 实例
        """
        n_factors = len(factor_names)

        # 解析因子权重
        factor_weights = {}
        for i, name in enumerate(factor_names):
            if i < len(genes):
                factor_weights[name] = float(max(0, genes[i]))

        # 解析阈值和参数
        signal_threshold = float(genes[n_factors]) if n_factors < len(genes) else 0.5
        exit_threshold = float(genes[n_factors + 1]) if n_factors + 1 < len(genes) else 0.3
        atr_period = int(genes[n_factors + 2]) if n_factors + 2 < len(genes) else 20
        stop_loss_atr = float(genes[n_factors + 3]) if n_factors + 3 < len(genes) else 2.0
        pyramid_interval_atr = float(genes[n_factors + 4]) if n_factors + 4 < len(genes) else 0.5
        trailing_stop_trigger = float(genes[n_factors + 5]) if n_factors + 5 < len(genes) else 1.0
        min_adx = float(genes[n_factors + 6]) if n_factors + 6 < len(genes) else 10.0

        return cls(
            factor_weights=factor_weights,
            signal_threshold=signal_threshold,
            exit_threshold=exit_threshold,
            atr_period=max(5, min(50, atr_period)),  # 限制 ATR 周期范围
            stop_loss_atr=max(1.5, min(3.0, stop_loss_atr)),  # 限制范围
            pyramid_interval_atr=max(0.3, min(0.7, pyramid_interval_atr)),  # 限制范围
            trailing_stop_trigger=max(0.5, min(1.5, trailing_stop_trigger)),  # 限制范围
            min_adx=max(0.0, min(50.0, min_adx)),  # 限制 ADX 范围 [0, 50]
            generation=generation,
            parent_ids=parent_ids or [],
        )

    def distance_to(self, other: "Individual") -> float:
        """计算与另一个体的基因距离

        Args:
            other: 另一个个体

        Returns:
            基因距离（欧氏距离）
        """
        genes1 = self.to_genes()
        genes2 = other.to_genes()
        return float(np.linalg.norm(genes1 - genes2))

    def is_similar_to(self, other: "Individual", threshold: float = 0.1) -> bool:
        """判断是否与另一个体相似

        Args:
            other: 另一个个体
            threshold: 相似度阈值

        Returns:
            是否相似
        """
        return self.distance_to(other) < threshold

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "strategy_type": self.strategy_type,
            "factor_weights": self.factor_weights,
            "signal_threshold": self.signal_threshold,
            "exit_threshold": self.exit_threshold,
            "atr_period": self.atr_period,
            "stop_loss_atr": self.stop_loss_atr,
            "pyramid_interval_atr": self.pyramid_interval_atr,
            "max_pyramid_units": self.max_pyramid_units,
            "trailing_stop_trigger": self.trailing_stop_trigger,
            "min_adx": self.min_adx,
            "min_trend_periods": self.min_trend_periods,
            "use_trend_filter": self.use_trend_filter,
            "fitness": self.fitness,
            "backtest_results": self.backtest_results,
            "market_performance": self.market_performance,
            "optimal_market": self.optimal_market,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Individual":
        """从字典创建个体（用于反序列化）"""
        return cls(
            strategy_type=data.get("strategy_type", "turtle"),
            factor_weights=data.get("factor_weights", {}),
            signal_threshold=data.get("signal_threshold", 0.5),
            exit_threshold=data.get("exit_threshold", 0.3),
            atr_period=data.get("atr_period", 20),
            stop_loss_atr=data.get("stop_loss_atr", 2.0),
            pyramid_interval_atr=data.get("pyramid_interval_atr", 0.5),
            max_pyramid_units=data.get("max_pyramid_units", 4),
            trailing_stop_trigger=data.get("trailing_stop_trigger", 1.0),
            min_adx=data.get("min_adx", 10.0),
            min_trend_periods=data.get("min_trend_periods", 5),
            use_trend_filter=data.get("use_trend_filter", False),
            fitness=data.get("fitness", 0.0),
            backtest_results=data.get("backtest_results", {}),
            market_performance=data.get("market_performance", {}),
            optimal_market=data.get("optimal_market", None),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
        )

    def __repr__(self) -> str:
        """字符串表示"""
        top_factors = sorted(
            self.factor_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        factor_str = ", ".join([f"{k}={v:.2f}" for k, v in top_factors])
        return (
            f"<Individual fitness={self.fitness:.4f} "
            f"factors=[{factor_str}] "
            f"sig_th={self.signal_threshold:.2f} "
            f"exit_th={self.exit_threshold:.2f}>"
        )


# 因子池定义（25个核心因子）
# 分类参考量化交易因子挖掘最佳实践：动量、技术、波动、量价、趋势、风险调整
FACTOR_POOL = {
    # ========== 动量因子 ==========
    "ma_ratio": "MA5 / MA20 - 1",                    # 均线偏离度
    "ma_cross": "MA5 > MA20",                        # 金叉死叉
    "momentum": "(close - close_n5) / close_n5",    # 短期动量
    "price_momentum": "close / close_n20 - 1",      # 价格动量
    "roc": "变动率 ROC(10)",                         # 变动率

    # ========== 技术指标 ==========
    "rsi": "RSI / 100",                             # RSI相对强弱
    "macd": "MACD / MACD_signal",                   # MACD
    "kdj": "(K - D) / 100",                         # KDJ
    "williams_r": "威廉指标 %R",                    # 威廉指标

    # ========== 波动率 ==========
    "volatility": "STD(close, 20) / close",         # 波动率
    "atr_ratio": "ATR / close",                     # ATR比率

    # ========== 成交量 ==========
    "volume_ratio": "volume / MA(volume, 20)",      # 量比
    "volume_price": "volume * (close / close_n1 - 1)",  # 量价配合
    "volume_trend": "MA(volume, 5) / MA(volume, 20)",   # 成交量趋势

    # ========== 趋势 ==========
    "adx": "ADX / 100",                             # 趋势强度
    "cci": "CCI / 200",                             # CCI

    # ========== 能量/资金流 ==========
    "obv": "OBV / MA(OBV, 20)",                     # OBV能量潮
    "money_flow": "典型价格与成交量关联",            # 资金流向

    # ========== 布林带 ==========
    "bb_ratio": "布林带位置 (close - lower) / (upper - lower)",

    # ========== 风险调整因子 (新增) ==========
    "risk_adj_momentum": "动量 / 波动率",           # 风险调整动量
    "relative_strength": "close / MA(close, 50) - 1",  # 相对强度
    "vol_adj_return": "(close - close_n1) / ATR",   # 波动调整收益

    # ========== 趋势质量因子 (新增) ==========
    "trend_consistency": "趋势方向一致性",           # 趋势一致性
    "higher_highs": "连续新高计数",                 # 突破强度

    # ========== 价格行为因子 (右侧交易) ==========
    "new_high_count": "过去N日创新高次数",           # 突破强度
    "new_low_count": "过去N日创新低次数",            # 跌破强度
    "consecutive_up": "连续阳线天数",                # 上涨惯性
    "consecutive_down": "连续阴线天数",              # 下跌惯性
    "gap_up": "向上跳空 (open > prev_high)",         # 强势信号
    "gap_down": "向下跳空 (open < prev_low)",        # 弱势信号
}


def get_factor_names() -> list[str]:
    """获取因子池中所有因子名称"""
    return list(FACTOR_POOL.keys())


def get_factor_expression(name: str) -> str | None:
    """获取因子表达式"""
    return FACTOR_POOL.get(name)
