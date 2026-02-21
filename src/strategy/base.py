"""策略基类"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import yaml


class BaseStrategy(ABC):
    """策略基类，所有策略都需要继承此类"""

    def __init__(self, config_path: str = "configs/strategy.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.strategy_config = self.config.get("strategy", {})
        self.model_config = self.config.get("model", {})
        self.factors_config = self.config.get("factors", [])

    @abstractmethod
    def get_factors(self) -> list[str]:
        """
        返回因子表达式列表

        Returns:
            因子表达式列表，如 ["$close / Mean($close, 20)", "($close - $open) / $open"]
        """
        raise NotImplementedError

    def get_factor_names(self) -> list[str]:
        """获取因子名称列表"""
        return [f.get("name", f"factor_{i}") for i, f in enumerate(self.factors_config)]

    def get_model_config(self) -> dict[str, Any]:
        """
        返回模型配置

        Returns:
            模型配置字典
        """
        return {
            "class": self.model_config.get("class", "LGBModel"),
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": self.model_config.get("params", {}),
        }

    def get_strategy_config(self) -> dict[str, Any]:
        """
        返回策略配置

        Returns:
            策略配置字典
        """
        return {
            "topk": self.strategy_config.get("topk", 30),
            "n_drop": self.strategy_config.get("n_drop", 5),
        }

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        生成交易信号

        Args:
            data: 包含因子数据的 DataFrame

        Returns:
            信号数组，正值表示买入信号，负值表示卖出信号
        """
        raise NotImplementedError

    def get_backtest_config(self) -> dict[str, Any]:
        """获取回测配置"""
        return self.config.get("backtest", {})

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.strategy_config.get('name', 'unnamed')}>"


class SimpleFactorStrategy(BaseStrategy):
    """简单的因子策略示例"""

    def get_factors(self) -> list[str]:
        """返回因子表达式"""
        return [
            "($close - Ref($close, 1)) / Ref($close, 1)",  # 收益率
            "$volume / Mean($volume, 20)",  # 成交量比率
            "($high - $low) / $close",  # 振幅
        ]

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        基于因子生成信号

        简单策略：因子值的加权平均作为信号
        """
        # 获取因子列
        factor_cols = [col for col in data.columns if col.startswith("factor_")]

        if not factor_cols:
            # 如果没有预处理因子，使用价格数据计算
            close = data["close"].values
            volume = data["volume"].values

            # 简单动量信号
            momentum = np.zeros(len(close))
            momentum[1:] = (close[1:] - close[:-1]) / close[:-1]

            # 成交量异常信号
            vol_signal = np.zeros(len(volume))
            if len(volume) > 20:
                vol_ma = np.convolve(volume, np.ones(20) / 20, mode="valid")
                vol_signal[20:] = volume[20:] / vol_ma[: len(volume) - 20]

            # 组合信号
            signals = 0.7 * momentum + 0.3 * vol_signal
        else:
            # 使用预处理因子
            signals = data[factor_cols].mean(axis=1).values

        return signals


class MomentumStrategy(BaseStrategy):
    """动量策略"""

    def __init__(self, lookback_period: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.lookback_period = lookback_period

    def get_factors(self) -> list[str]:
        return [
            f"Ref($close, -{self.lookback_period}) / $close - 1",  # 过去N日收益率
        ]

    def generate_signals(self, data: pd.DataFrame) -> np.ndarray:
        close = data["close"].values
        signals = np.zeros(len(close))

        # 计算动量
        signals[self.lookback_period :] = (
            close[self.lookback_period :] - close[: -self.lookback_period]
        ) / close[: -self.lookback_period]

        return signals


if __name__ == "__main__":
    # 测试策略
    strategy = SimpleFactorStrategy()
    print(f"策略: {strategy}")
    print(f"因子: {strategy.get_factors()}")
    print(f"模型配置: {strategy.get_model_config()}")
    print(f"策略配置: {strategy.get_strategy_config()}")
