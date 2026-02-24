"""海龟策略数据加载辅助类

提供按周期加载股票数据的功能，集成 AkshareLoader。
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd

from src.data.akshare_loader import AkshareLoader

logger = logging.getLogger(__name__)

# 缓存有效期（天）- 如果缓存数据的最新日期在这个范围内，就认为缓存有效
CACHE_VALID_DAYS = 3  # 允许缓存数据最多延迟3天


class TurtleDataLoader:
    """海龟策略数据加载器

    提供方便的数据加载和缓存功能，支持按周期获取数据。

    缓存策略：
    - 每个标的一个缓存文件: stock_{symbol}_full.parquet
    - 如果缓存数据足够新（最新数据在 CACHE_VALID_DAYS 天内），直接使用
    - 否则从网络获取最新数据并更新缓存
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        use_cache: bool = True
    ):
        """初始化数据加载器

        Args:
            cache_dir: 缓存目录
            use_cache: 是否使用缓存
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.use_cache = use_cache

        # 初始化 AkshareLoader
        self.loader = AkshareLoader()
        # 覆盖 cache_dir
        self.loader.cache_dir = self.cache_dir / "raw"
        self.loader.cache_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存 - 同一次运行中避免重复加载
        self._memory_cache: dict[str, pd.DataFrame] = {}

    def load_stock_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        reload: bool = False
    ) -> pd.DataFrame:
        """加载股票数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            reload: 是否重新加载（强制从网络获取）

        Returns:
            OHLCV 数据
        """
        try:
            # 计算默认日期范围（最近2年）
            if end_date is None:
                end_date = datetime.now()
            else:
                end_date = pd.to_datetime(end_date)

            if start_date is None:
                start_date = end_date - timedelta(days=730)  # 2年
            else:
                start_date = pd.to_datetime(start_date)

            # 1. 检查内存缓存
            cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            if self.use_cache and not reload and cache_key in self._memory_cache:
                logger.info(f"从内存缓存加载数据: {symbol}")
                return self._memory_cache[cache_key].copy()

            # 2. 检查磁盘缓存
            cached_data = self._load_from_disk_cache(symbol, start_date, end_date, reload)

            if cached_data is not None and len(cached_data) > 0:
                # 存入内存缓存
                self._memory_cache[cache_key] = cached_data
                return cached_data.copy()

            # 3. 从 akshare 加载
            logger.info(f"从 akshare 加载数据: {symbol}")

            # 获取更长时间范围的数据用于缓存
            fetch_start = start_date - timedelta(days=30)  # 多获取30天用于后续使用
            data = self._fetch_from_akshare(symbol, fetch_start, end_date)

            if len(data) > 0:
                # 保存全量缓存
                if self.use_cache:
                    full_cache_file = self.cache_dir / f"stock_{symbol}_full.parquet"
                    full_cache_file.parent.mkdir(parents=True, exist_ok=True)
                    data.to_parquet(full_cache_file)
                    logger.info(f"数据已缓存: {full_cache_file}")

                # 裁剪到所需范围
                mask = data.index >= start_date
                result = data[mask].copy()

                # 存入内存缓存
                self._memory_cache[cache_key] = result

                return result

            return data

        except Exception as e:
            logger.error(f"加载数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _load_from_disk_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        reload: bool
    ) -> pd.DataFrame | None:
        """从磁盘缓存加载数据

        优先级：
        1. 新格式缓存: stock_{symbol}_full.parquet
        2. 旧格式缓存: stock_{symbol}_{start}_{end}.parquet（兼容）

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            reload: 是否强制刷新

        Returns:
            缓存数据或 None
        """
        if not self.use_cache or reload:
            return None

        # 1. 尝试新格式缓存
        full_cache_file = self.cache_dir / f"stock_{symbol}_full.parquet"

        if full_cache_file.exists():
            cached_data = pd.read_parquet(full_cache_file)

            if len(cached_data) > 0:
                cache_latest = cached_data.index[-1]
                cache_earliest = cached_data.index[0]
                days_old = (datetime.now() - cache_latest.to_pydatetime()).days

                # 检查缓存数据是否覆盖所需范围（允许1天误差）
                covers_start = cache_earliest <= start_date + timedelta(days=1)
                covers_end = cache_latest >= end_date - timedelta(days=CACHE_VALID_DAYS + 1)

                if days_old <= CACHE_VALID_DAYS and covers_start and covers_end:
                    logger.info(f"从磁盘缓存加载数据: {symbol} (缓存最新: {cache_latest.strftime('%Y-%m-%d')}, {days_old}天前)")

                    # 裁剪到所需范围
                    mask = (cached_data.index >= start_date) & (cached_data.index <= end_date)
                    return cached_data[mask].copy()
                else:
                    logger.info(f"缓存过期或数据不足: {symbol} (最新: {cache_latest.strftime('%Y-%m-%d')}, 需要: {end_date.strftime('%Y-%m-%d')})")

        # 2. 尝试旧格式缓存（兼容）
        old_cache_files = list(self.cache_dir.glob(f"stock_{symbol}_*.parquet"))
        old_cache_files = [f for f in old_cache_files if "full" not in f.name]

        best_match = None
        best_match_file = None

        for cache_file in old_cache_files:
            try:
                cached_data = pd.read_parquet(cache_file)

                if len(cached_data) > 0:
                    cache_latest = cached_data.index[-1]
                    cache_earliest = cached_data.index[0]

                    # 检查是否覆盖所需范围（允许1天误差）
                    covers_start = cache_earliest <= start_date + timedelta(days=1)
                    covers_end = cache_latest >= end_date - timedelta(days=CACHE_VALID_DAYS + 1)

                    # 放宽条件：只要数据足够多且不太旧就使用
                    has_enough_data = len(cached_data) >= 100

                    if covers_start and covers_end and has_enough_data:
                        days_old = (datetime.now() - cache_latest.to_pydatetime()).days
                        logger.info(f"从旧格式缓存加载数据: {cache_file.name} ({days_old}天前)")

                        # 裁剪到所需范围
                        mask = (cached_data.index >= start_date) & (cached_data.index <= end_date)
                        result = cached_data[mask].copy()

                        # 迁移到新格式缓存
                        full_cache_file = self.cache_dir / f"stock_{symbol}_full.parquet"
                        cached_data.to_parquet(full_cache_file)
                        logger.info(f"已迁移到新格式缓存: {full_cache_file}")

                        return result

                    # 记录最佳匹配（即使不完全满足条件）
                    if has_enough_data and (best_match is None or cache_latest > best_match['latest']):
                        best_match = {
                            'data': cached_data,
                            'latest': cache_latest,
                            'earliest': cache_earliest,
                            'file': cache_file
                        }

            except Exception as e:
                logger.warning(f"读取旧缓存失败 {cache_file}: {e}")
                continue

        # 3. 如果没有完全匹配的，使用最佳匹配（数据足够多）
        if best_match is not None:
            cached_data = best_match['data']
            cache_latest = best_match['latest']
            cache_earliest = best_match['earliest']
            cache_file = best_match['file']

            days_old = (datetime.now() - cache_latest.to_pydatetime()).days
            logger.info(f"使用最佳匹配缓存: {cache_file.name} (数据 {len(cached_data)} 条, {days_old}天前)")

            # 裁剪到可用范围
            actual_start = max(start_date, cache_earliest)
            actual_end = min(end_date, cache_latest)

            mask = (cached_data.index >= actual_start) & (cached_data.index <= actual_end)
            result = cached_data[mask].copy()

            # 迁移到新格式缓存
            full_cache_file = self.cache_dir / f"stock_{symbol}_full.parquet"
            cached_data.to_parquet(full_cache_file)
            logger.info(f"已迁移到新格式缓存: {full_cache_file}")

            return result

        return None

    def _fetch_from_akshare(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """从 akshare 获取数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            OHLCV 数据
        """
        try:
            # 使用 stock_zh_a_hist 获取历史数据
            import akshare as ak

            data = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                adjust="qfq"  # 前复权
            )

            if data is None or len(data) == 0:
                logger.warning(f"未获取到数据: {symbol}")
                return pd.DataFrame()

            # 重命名列
            column_map = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
            }

            data = data.rename(columns=column_map)

            # 设置日期索引
            if "date" in data.columns:
                data["date"] = pd.to_datetime(data["date"])
                data = data.set_index("date")

            # 选择需要的列
            required_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [c for c in required_cols if c in data.columns]

            if not available_cols:
                logger.error(f"数据列不完整: {data.columns}")
                return pd.DataFrame()

            data = data[available_cols].copy()

            # 去除空值
            data = data.dropna()

            logger.info(f"成功加载 {len(data)} 条数据")
            return data

        except ImportError:
            logger.error("akshare 未安装，请运行: uv pip install akshare")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return pd.DataFrame()

    def load_period_data(
        self,
        symbol: str,
        period: str = "1y",
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """加载指定周期的数据

        Args:
            symbol: 股票代码
            period: 周期（1y/3m/1m）
            end_date: 结束日期

        Returns:
            周期数据
        """
        period_days = {
            "1y": 365,
            "3m": 90,
            "1m": 30,
        }.get(period, 365)

        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=period_days + 50)  # 多加载50天用于指标计算

        # 加载数据
        data = self.load_stock_data(
            symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        # 裁剪到精确周期
        cutoff_date = end_date - timedelta(days=period_days)
        period_data = data[data.index >= cutoff_date]

        return period_data

    def load_multi_period_data(
        self,
        symbol: str,
        end_date: Optional[datetime] = None
    ) -> dict[str, pd.DataFrame]:
        """加载多周期数据

        Args:
            symbol: 股票代码
            end_date: 结束日期

        Returns:
            多周期数据字典
        """
        periods = ["1y", "3m", "1m"]
        data_dict = {}

        for period in periods:
            data = self.load_period_data(symbol, period, end_date)
            if len(data) > 0:
                data_dict[period] = data

        return data_dict

    def clear_cache(self, symbol: str | None = None) -> None:
        """清除缓存

        Args:
            symbol: 股票代码，None 表示清除所有缓存
        """
        # 清除内存缓存
        if symbol is None:
            self._memory_cache.clear()
        else:
            keys_to_remove = [k for k in self._memory_cache if k.startswith(symbol)]
            for k in keys_to_remove:
                del self._memory_cache[k]

        # 清除磁盘缓存
        if symbol is None:
            # 清除所有缓存
            for file in self.cache_dir.glob("stock_*.parquet"):
                file.unlink()
                logger.info(f"已删除缓存: {file}")
        else:
            # 清除指定股票的缓存（包括旧格式和新格式）
            for file in self.cache_dir.glob(f"stock_{symbol}_*.parquet"):
                file.unlink()
                logger.info(f"已删除缓存: {file}")

    def get_cache_info(self, symbol: str) -> dict:
        """获取缓存信息

        Args:
            symbol: 股票代码

        Returns:
            缓存信息字典
        """
        cache_file = self.cache_dir / f"stock_{symbol}_full.parquet"

        if not cache_file.exists():
            return {"exists": False}

        try:
            data = pd.read_parquet(cache_file)
            return {
                "exists": True,
                "file": str(cache_file),
                "rows": len(data),
                "start_date": str(data.index[0]),
                "end_date": str(data.index[-1]),
                "days": (data.index[-1] - data.index[0]).days,
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}


def get_stock_data(
    symbol: str,
    days: int = 365,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """便捷函数：获取股票数据

    Args:
        symbol: 股票代码
        days: 天数
        end_date: 结束日期

    Returns:
        OHLCV 数据
    """
    loader = TurtleDataLoader()

    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=days + 50)

    return loader.load_stock_data(
        symbol,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
