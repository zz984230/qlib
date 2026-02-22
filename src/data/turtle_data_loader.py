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


class TurtleDataLoader:
    """海龟策略数据加载器

    提供方便的数据加载和缓存功能，支持按周期获取数据。
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
            reload: 是否重新加载

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

            # 尝试从缓存加载
            cache_file = self.cache_dir / f"stock_{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.parquet"

            if self.use_cache and cache_file.exists() and not reload:
                logger.info(f"从缓存加载数据: {cache_file}")
                return pd.read_parquet(cache_file)

            # 从 akshare 加载
            logger.info(f"从 akshare 加载数据: {symbol}")

            # 使用 AkshareLoader 获取数据
            data = self._fetch_from_akshare(symbol, start_date, end_date)

            # 保存到缓存
            if self.use_cache and len(data) > 0:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                data.to_parquet(cache_file)
                logger.info(f"数据已缓存: {cache_file}")

            return data

        except Exception as e:
            logger.error(f"加载数据失败 {symbol}: {e}")
            return pd.DataFrame()

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
        if symbol is None:
            # 清除所有缓存
            for file in self.cache_dir.glob("stock_*.parquet"):
                file.unlink()
                logger.info(f"已删除缓存: {file}")
        else:
            # 清除指定股票的缓存
            for file in self.cache_dir.glob(f"stock_{symbol}_*.parquet"):
                file.unlink()
                logger.info(f"已删除缓存: {file}")


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
