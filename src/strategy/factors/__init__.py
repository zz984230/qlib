"""
因子库 - 包含常用量化因子

因子分类:
- 动量因子 (Momentum): 价格趋势相关
- 波动率因子 (Volatility): 价格波动相关
- 成交量因子 (Volume): 交易活跃度相关
- 估值因子 (Value): 估值水平相关
- 质量因子 (Quality): 财务质量相关
- 技术因子 (Technical): 技术指标相关
"""

from src.strategy.factors.momentum import (
    MomentumFactor,
    ROCFactor,
    MACDFactor,
)
from src.strategy.factors.volatility import (
    VolatilityFactor,
    ATRFactor,
)
from src.strategy.factors.volume import (
    VolumeRatioFactor,
    OBVFactor,
)
from src.strategy.factors.technical import (
    MAFactor,
    RSIFactor,
    BollingerBandsFactor,
)

__all__ = [
    # 动量因子
    "MomentumFactor",
    "ROCFactor",
    "MACDFactor",
    # 波动率因子
    "VolatilityFactor",
    "ATRFactor",
    # 成交量因子
    "VolumeRatioFactor",
    "OBVFactor",
    # 技术因子
    "MAFactor",
    "RSIFactor",
    "BollingerBandsFactor",
]

# 因子注册表
FACTOR_REGISTRY = {
    # 动量
    "momentum": MomentumFactor,
    "roc": ROCFactor,
    "macd": MACDFactor,
    # 波动率
    "volatility": VolatilityFactor,
    "atr": ATRFactor,
    # 成交量
    "volume_ratio": VolumeRatioFactor,
    "obv": OBVFactor,
    # 技术
    "ma": MAFactor,
    "rsi": RSIFactor,
    "bollinger": BollingerBandsFactor,
}


def get_factor(name: str, **params):
    """获取因子实例"""
    if name not in FACTOR_REGISTRY:
        raise ValueError(f"Unknown factor: {name}. Available: {list(FACTOR_REGISTRY.keys())}")
    return FACTOR_REGISTRY[name](**params)


def list_factors() -> list[str]:
    """列出所有可用因子"""
    return list(FACTOR_REGISTRY.keys())
