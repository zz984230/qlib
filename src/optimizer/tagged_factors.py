"""标签化因子池 - 市场适应性标签系统

为30个因子定义市场适应性标签，遗传算法根据市场状态调整因子选择权重。
"""

from typing import Dict, List

# 市场状态标签类型
MARKET_TAGS = {
    "trend_strong": "强趋势市场（ADX>30），适合动量突破类因子",
    "trend_weak": "弱趋势市场（20<ADX<=30），适合趋势跟随类因子",
    "ranging": "震荡市场（ADX<=20），适合均值回归类因子",
    "volatile": "异常波动（波动率突增），适合波动率类因子",
    "universal": "全场景适用，权重不受市场状态影响",
}

# 因子标签映射
# 基于现有30个因子，按其技术特性和市场适应性进行标签
FACTOR_TAGS: Dict[str, List[str]] = {
    # ========== 动量因子 ==========
    "ma_ratio": ["trend_strong", "trend_weak"],           # 均线偏离度
    "ma_cross": ["trend_weak"],                           # 金叉死叉
    "momentum": ["trend_strong", "trend_weak"],           # 短期动量
    "price_momentum": ["trend_strong"],                   # 价格动量
    "roc": ["trend_weak"],                                # 变动率

    # ========== 技术指标 ==========
    "rsi": ["ranging"],                                   # RSI相对强弱
    "macd": ["trend_weak", "ranging"],                    # MACD
    "kdj": ["ranging"],                                   # KDJ
    "williams_r": ["ranging"],                            # 威廉指标

    # ========== 波动率 ==========
    "volatility": ["volatile", "ranging"],                # 波动率
    "atr_ratio": ["volatile", "trend_strong"],            # ATR比率

    # ========== 成交量 ==========
    "volume_ratio": ["universal"],                        # 量比
    "volume_price": ["volatile", "trend_strong"],         # 量价配合
    "volume_trend": ["universal"],                        # 成交量趋势

    # ========== 趋势 ==========
    "adx": ["trend_strong", "trend_weak"],                # 趋势强度
    "cci": ["ranging"],                                   # CCI

    # ========== 能量/资金流 ==========
    "obv": ["universal"],                                 # OBV能量潮
    "money_flow": ["universal"],                          # 资金流向

    # ========== 布林带 ==========
    "bb_ratio": ["ranging", "volatile"],                  # 布林带位置

    # ========== 风险调整因子 ==========
    "risk_adj_momentum": ["trend_strong", "trend_weak"],  # 风险调整动量
    "relative_strength": ["trend_strong", "trend_weak"],  # 相对强度
    "vol_adj_return": ["volatile", "trend_strong"],       # 波动调整收益

    # ========== 趋势质量因子 ==========
    "trend_consistency": ["trend_strong", "trend_weak"],  # 趋势一致性
    "higher_highs": ["trend_strong"],                     # 突破强度

    # ========== 价格行为因子 ==========
    "new_high_count": ["trend_strong"],                   # 突破强度
    "new_low_count": ["trend_weak", "ranging"],           # 跌破强度
    "consecutive_up": ["trend_strong", "trend_weak"],     # 上涨惯性
    "consecutive_down": ["trend_weak", "ranging"],        # 下跌惯性
    "gap_up": ["trend_strong", "volatile"],               # 强势信号
    "gap_down": ["volatile", "trend_weak"],               # 弱势信号
}


def get_factor_tags(factor_name: str) -> List[str]:
    """获取因子的市场适应性标签

    Args:
        factor_name: 因子名称

    Returns:
        标签列表，如果因子不存在返回 ["universal"]
    """
    return FACTOR_TAGS.get(factor_name, ["universal"])


def get_factors_by_tag(tag: str) -> List[str]:
    """获取具有特定标签的所有因子

    Args:
        tag: 市场标签（trend_strong, trend_weak, ranging, volatile, universal）

    Returns:
        因子名称列表
    """
    return [
        factor_name
        for factor_name, tags in FACTOR_TAGS.items()
        if tag in tags
    ]


def calculate_tag_weights(
    base_weights: Dict[str, float],
    market_state: str
) -> Dict[str, float]:
    """根据市场状态调整因子权重

    Args:
        base_weights: 基础因子权重字典
        market_state: 当前市场状态（strong_trend, weak_trend, ranging, volatile）

    Returns:
        调整后的因子权重字典
    """
    # 市场状态到标签的映射
    state_to_primary_tag = {
        "strong_trend": "trend_strong",
        "weak_trend": "trend_weak",
        "ranging": "ranging",
        "volatile": "volatile",
    }

    # 权重倍数配置
    multipliers = {
        "strong_trend": {
            "trend_strong": 2.0,
            "trend_weak": 1.5,
            "ranging": 0.3,
            "volatile": 0.5,
            "universal": 1.0,
        },
        "weak_trend": {
            "trend_strong": 1.0,
            "trend_weak": 2.0,
            "ranging": 0.7,
            "volatile": 0.8,
            "universal": 1.0,
        },
        "ranging": {
            "trend_strong": 0.3,
            "trend_weak": 0.6,
            "ranging": 2.0,
            "volatile": 1.5,
            "universal": 1.0,
        },
        "volatile": {
            "trend_strong": 0.2,
            "trend_weak": 0.3,
            "ranging": 0.5,
            "volatile": 3.0,
            "universal": 1.0,
        },
    }

    if market_state not in multipliers:
        # 未知市场状态，返回原权重
        return base_weights.copy()

    multiplier_config = multipliers[market_state]
    adjusted_weights = {}

    # 应用权重倍数
    for factor_name, weight in base_weights.items():
        factor_tags = get_factor_tags(factor_name)
        max_multiplier = 1.0

        # 找到该因子在当前市场下的最大倍数
        for tag in factor_tags:
            if tag in multiplier_config:
                max_multiplier = max(max_multiplier, multiplier_config[tag])

        adjusted_weights[factor_name] = weight * max_multiplier

    # 重新归一化
    total = sum(abs(w) for w in adjusted_weights.values())
    if total > 0:
        adjusted_weights = {
            k: abs(v) / total
            for k, v in adjusted_weights.items()
        }

    return adjusted_weights


def get_factor_selection_bias(
    all_factors: List[str],
    market_state: str
) -> Dict[str, float]:
    """获取因子选择偏向（用于遗传算法初始化时）

    返回每个因子被选中的概率权重，适合当前市场状态的因子概率更高。

    Args:
        all_factors: 所有可选因子名称列表
        market_state: 当前市场状态

    Returns:
        因子名称 -> 选择概率权重的字典
    """
    # 市场状态到标签的映射
    state_to_primary_tag = {
        "strong_trend": "trend_strong",
        "weak_trend": "trend_weak",
        "ranging": "ranging",
        "volatile": "volatile",
    }

    primary_tag = state_to_primary_tag.get(market_state)

    # 基础权重
    base_weights = {}
    for factor in all_factors:
        tags = get_factor_tags(factor)
        if primary_tag and primary_tag in tags:
            base_weights[factor] = 3.0  # 主要标签因子高权重
        elif "universal" in tags:
            base_weights[factor] = 2.0  # 通用因子中权重
        else:
            base_weights[factor] = 1.0  # 其他因子低权重

    # 归一化
    total = sum(base_weights.values())
    if total > 0:
        base_weights = {
            k: v / total
            for k, v in base_weights.items()
        }

    return base_weights
