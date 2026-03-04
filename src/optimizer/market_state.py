"""市场状态管理器 - 缓存和更新市场状态

管理市场状态的缓存、更新和平滑处理。
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
import pandas as pd
import json

from src.strategy.market_classifier import MarketClassifier, MarketState

logger = logging.getLogger(__name__)


class MarketStateManager:
    """市场状态管理器

    功能：
    1. 识别和缓存市场状态
    2. 状态平滑处理（避免频繁切换）
    3. 冷却期管理（状态切换后等待期）
    """

    def __init__(self, cache_dir: str = "data/cache/market_state"):
        """初始化管理器

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = MarketClassifier()

        # 冷却期设置（状态切换后等待天数）
        self.cooldown_days = 3

        # 平滑窗口（用于状态平滑）
        self.smoothing_window = 5

    def get_market_state(
        self,
        symbol: str,
        data: pd.DataFrame,
        force_refresh: bool = False
    ) -> MarketState:
        """获取当前市场状态

        优先从缓存读取，如果缓存过期则重新计算。
        应用平滑处理避免频繁切换。

        Args:
            symbol: 股票代码
            data: 价格数据
            force_refresh: 是否强制刷新

        Returns:
            当前市场状态
        """
        cache_file = self.cache_dir / f"{symbol}.json"

        # 检查缓存
        if not force_refresh and cache_file.exists():
            cached_state = self._load_cache(cache_file)
            if cached_state and not self._is_cache_expired(cached_state):
                # 检查冷却期
                if not self._is_in_cooldown(cached_state):
                    return cached_state["state"]

        # 计算新的市场状态
        current_state = self.classifier.classify(data)

        # 应用平滑处理
        if cache_file.exists():
            cached_state = self._load_cache(cache_file)
            if cached_state:
                current_state = self._smooth_state_transition(
                    cached_state["state"],
                    current_state
                )

        # 保存到缓存
        self._save_cache(cache_file, current_state, data)

        logger.info(f"市场状态 ({symbol}): {current_state}")
        return current_state

    def _load_cache(self, cache_file: Path) -> dict | None:
        """加载缓存数据

        Args:
            cache_file: 缓存文件路径

        Returns:
            缓存数据字典，如果文件不存在或解析失败返回 None
        """
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return None

    def _save_cache(
        self,
        cache_file: Path,
        state: MarketState,
        data: pd.DataFrame
    ) -> None:
        """保存缓存数据

        Args:
            cache_file: 缓存文件路径
            state: 市场状态
            data: 价格数据（用于记录日期）
        """
        cache_data = {
            "state": state,
            "date": data.index[-1].strftime('%Y-%m-%d') if hasattr(data.index[-1], 'strftime') else str(data.index[-1]),
            "timestamp": datetime.now().isoformat(),
            "cooldown_until": (datetime.now() + timedelta(days=self.cooldown_days)).isoformat(),
        }

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def _is_cache_expired(self, cached_data: dict) -> bool:
        """检查缓存是否过期

        缓存超过1天视为过期。

        Args:
            cached_data: 缓存数据

        Returns:
            是否过期
        """
        try:
            cache_time = datetime.fromisoformat(cached_data["timestamp"])
            return (datetime.now() - cache_time) > timedelta(days=1)
        except Exception:
            return True

    def _is_in_cooldown(self, cached_data: dict) -> bool:
        """检查是否在冷却期内

        Args:
            cached_data: 缓存数据

        Returns:
            是否在冷却期
        """
        try:
            cooldown_end = datetime.fromisoformat(cached_data.get("cooldown_until", ""))
            return datetime.now() < cooldown_end
        except Exception:
            return False

    def _smooth_state_transition(
        self,
        old_state: MarketState,
        new_state: MarketState
    ) -> MarketState:
        """平滑状态转换

        避免状态频繁切换，只有在明确且持续的状态变化时才更新。

        Args:
            old_state: 旧状态
            new_state: 新状态

        Returns:
            平滑后的状态
        """
        # 异常波动优先（风险控制）
        if new_state == "volatile":
            return "volatile"

        # 状态优先级（用于平滑）
        state_priority = {
            "volatile": 4,
            "strong_trend": 3,
            "weak_trend": 2,
            "ranging": 1,
        }

        # 如果新状态优先级更高，允许切换
        if state_priority[new_state] > state_priority[old_state]:
            return new_state

        # 如果新状态优先级相同或更低，保持旧状态（平滑）
        # 这样可以避免 ranging <-> weak_trend 之间的频繁切换
        return old_state

    def clear_cache(self, symbol: str | None = None) -> None:
        """清除缓存

        Args:
            symbol: 股票代码，如果为 None 则清除所有缓存
        """
        if symbol:
            cache_file = self.cache_dir / f"{symbol}.json"
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"已清除 {symbol} 的市场状态缓存")
        else:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("已清除所有市场状态缓存")


# 便捷函数
def get_market_state(
    symbol: str,
    data: pd.DataFrame,
    force_refresh: bool = False
) -> MarketState:
    """便捷函数：获取市场状态

    Args:
        symbol: 股票代码
        data: 价格数据
        force_refresh: 是否强制刷新

    Returns:
        市场状态字符串
    """
    manager = MarketStateManager()
    return manager.get_market_state(symbol, data, force_refresh)
