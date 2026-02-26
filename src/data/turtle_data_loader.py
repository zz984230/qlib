"""海龟策略数据加载辅助类

提供按周期加载股票数据的功能，支持按天粒度缓存。
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TurtleDataLoader:
    """海龟策略数据加载器

    提供方便的数据加载和缓存功能，支持按周期获取数据。

    缓存策略（天粒度）：
    - 每个交易日一个缓存文件: data/cache/daily/{symbol}/{YYYY-MM-DD}.parquet
    - 加载时检查缓存，存在则直接加载，不存在则从接口获取并缓存
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

        # 天粒度缓存目录
        self.daily_cache_dir = self.cache_dir / "daily"
        self.daily_cache_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存 - 同一次运行中避免重复加载
        self._memory_cache: dict[str, pd.DataFrame] = {}

    def load_stock_data(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        reload: bool = False
    ) -> pd.DataFrame:
        """加载股票数据（优先使用天粒度缓存）

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            reload: 是否重新加载（强制从网络获取）

        Returns:
            OHLCV 数据
        """
        try:
            # 计算默认日期范围（最近2年，满足优化器400条数据要求）
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

            # 2. 从天粒度缓存加载（合并缓存和网络数据）
            if self.use_cache and not reload:
                cached_data = self._load_from_daily_cache(symbol, start_date, end_date)
                if cached_data is not None and len(cached_data) > 0:
                    self._memory_cache[cache_key] = cached_data
                    return cached_data.copy()

            # 3. 从 akshare 加载并按天缓存（增量模式）
            logger.info(f"从 akshare 加载数据: {symbol}")
            data = self._fetch_and_cache_daily_incremental(symbol, start_date, end_date)

            if len(data) > 0:
                # 存入内存缓存
                self._memory_cache[cache_key] = data
                return data.copy()

            return data

        except Exception as e:
            logger.error(f"加载数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def _load_from_daily_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame | None:
        """从天粒度缓存加载数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            缓存数据或 None（如果不完整）
        """
        daily_dir = self.daily_cache_dir / symbol
        if not daily_dir.exists():
            return None

        # 收集日期范围内的缓存文件
        dfs = []
        missing_dates = []

        current_date = start_date
        while current_date <= end_date:
            # 跳过周末
            if current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            cache_file = daily_dir / f"{current_date.strftime('%Y-%m-%d')}.parquet"
            if cache_file.exists():
                try:
                    df = pd.read_parquet(cache_file)
                    if len(df) > 0:
                        dfs.append(df)
                except Exception as e:
                    logger.warning(f"读取缓存失败 {cache_file}: {e}")
                    missing_dates.append(current_date)
            else:
                missing_dates.append(current_date)

            current_date += timedelta(days=1)

        # 如果有缺失日期，返回 None 触发网络获取
        if missing_dates:
            logger.info(f"缓存缺失 {len(missing_dates)} 个交易日，需要从网络获取")
            return None

        if not dfs:
            return None

        # 合并所有数据
        result = pd.concat(dfs).sort_index()
        logger.info(f"从天粒度缓存加载 {len(result)} 条数据: {symbol}")

        return result

    def _fetch_and_cache_daily(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """从 akshare 获取数据并按天缓存

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            OHLCV 数据
        """
        # 获取数据
        data = self._fetch_from_akshare(symbol, start_date, end_date)

        if data is None or len(data) == 0:
            return pd.DataFrame()

        # 按天缓存
        if self.use_cache:
            self._save_daily_cache(symbol, data)

        return data

    def _fetch_and_cache_daily_incremental(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """从 akshare 获取数据并增量缓存（合并已有缓存）

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            OHLCV 数据
        """
        # 1. 读取已有缓存
        existing_data = self._load_all_cached_data(symbol)

        # 2. 获取网络数据
        new_data = self._fetch_from_akshare(symbol, start_date, end_date)

        if new_data is None or len(new_data) == 0:
            # 网络获取失败，返回已有缓存
            if existing_data is not None and len(existing_data) > 0:
                logger.warning(f"网络获取失败，使用已有缓存: {len(existing_data)} 条")
                return existing_data
            return pd.DataFrame()

        # 3. 合并数据
        if existing_data is not None and len(existing_data) > 0:
            # 合并并去重（保留新数据）
            combined = pd.concat([existing_data, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            logger.info(f"合并缓存: 原有 {len(existing_data)} 条 + 新获取 {len(new_data)} 条 = {len(combined)} 条")
        else:
            combined = new_data

        # 4. 按天缓存合并后的数据
        if self.use_cache:
            self._save_daily_cache(symbol, combined)

        return combined

    def _load_all_cached_data(self, symbol: str) -> pd.DataFrame | None:
        """加载所有已缓存的数据

        Args:
            symbol: 股票代码

        Returns:
            所有缓存数据或 None
        """
        daily_dir = self.daily_cache_dir / symbol
        if not daily_dir.exists():
            return None

        cache_files = list(daily_dir.glob("*.parquet"))
        if not cache_files:
            return None

        dfs = []
        for cache_file in cache_files:
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 0:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"读取缓存失败 {cache_file}: {e}")

        if not dfs:
            return None

        result = pd.concat(dfs).sort_index()
        result = result[~result.index.duplicated(keep='last')]

        return result

    def _save_daily_cache(self, symbol: str, data: pd.DataFrame) -> None:
        """按天保存缓存

        Args:
            symbol: 股票代码
            data: OHLCV 数据
        """
        daily_dir = self.daily_cache_dir / symbol
        daily_dir.mkdir(parents=True, exist_ok=True)

        # 按日期分组保存
        for date, group in data.groupby(data.index.date):
            cache_file = daily_dir / f"{date.strftime('%Y-%m-%d')}.parquet"
            group.to_parquet(cache_file)

        logger.info(f"已缓存 {len(data)} 条数据到 {daily_dir} (按天粒度)")

    def _fetch_from_akshare(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """从 akshare 获取数据（支持多数据源备用）

        数据源优先级：
        1. stock_zh_a_hist (东方财富)
        2. stock_zh_a_daily (新浪)
        3. stock_zh_a_hist_163 (网易)

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            OHLCV 数据
        """
        try:
            import akshare as ak
        except ImportError:
            logger.error("akshare 未安装，请运行: uv pip install akshare")
            return pd.DataFrame()

        # 尝试多个数据源
        data_sources = [
            ("东方财富 (stock_zh_a_hist)", lambda: self._fetch_from_eastmoney(ak, symbol, start_date, end_date)),
            ("新浪 (stock_zh_a_daily)", lambda: self._fetch_from_sina(ak, symbol, start_date, end_date)),
            ("网易 (stock_zh_a_hist_163)", lambda: self._fetch_from_netease(ak, symbol, start_date, end_date)),
        ]

        for source_name, fetch_func in data_sources:
            try:
                logger.info(f"尝试数据源: {source_name}")
                data = fetch_func()

                if data is not None and len(data) > 0:
                    logger.info(f"成功从 {source_name} 加载 {len(data)} 条数据")
                    return data

            except Exception as e:
                logger.warning(f"{source_name} 获取失败: {e}")
                continue

        logger.error(f"所有数据源均获取失败: {symbol}")
        return pd.DataFrame()

    def _fetch_from_eastmoney(
        self,
        ak,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame | None:
        """从东方财富获取数据（主数据源）

        Args:
            ak: akshare 模块
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            OHLCV 数据或 None
        """
        data = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust="qfq"  # 前复权
        )

        if data is None or len(data) == 0:
            return None

        return self._normalize_akshare_data(data, {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        })

    def _fetch_from_sina(
        self,
        ak,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame | None:
        """从新浪获取数据（备用数据源1）

        Args:
            ak: akshare 模块
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            OHLCV 数据或 None
        """
        try:
            # stock_zh_a_daily 使用 sh/sz 前缀
            # 判断股票代码所属交易所
            if symbol.startswith("6"):
                full_symbol = f"sh{symbol}"
            else:
                full_symbol = f"sz{symbol}"

            data = ak.stock_zh_a_daily(symbol=full_symbol, adjust="qfq")

            if data is None or len(data) == 0:
                return None

            # 筛选日期范围
            data = self._normalize_akshare_data(data, {
                "date": "date",
                "open": "open",
                "close": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
            })

            if data is not None and len(data) > 0:
                # 筛选日期范围
                mask = (data.index >= start_date) & (data.index <= end_date)
                data = data[mask]

            return data if len(data) > 0 else None

        except Exception as e:
            logger.debug(f"新浪数据源异常: {e}")
            return None

    def _fetch_from_netease(
        self,
        ak,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame | None:
        """从网易获取数据（备用数据源2）

        Args:
            ak: akshare 模块
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            OHLCV 数据或 None
        """
        try:
            # 使用 163 数据源
            data = ak.stock_zh_a_hist_163(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                adjust="qfq"
            )

            if data is None or len(data) == 0:
                return None

            return self._normalize_akshare_data(data, {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
            })

        except Exception as e:
            logger.debug(f"网易数据源异常: {e}")
            return None

    def _normalize_akshare_data(
        self,
        data: pd.DataFrame,
        column_map: dict[str, str]
    ) -> pd.DataFrame | None:
        """标准化 akshare 数据格式

        Args:
            data: 原始数据
            column_map: 列名映射

        Returns:
            标准化后的数据或 None
        """
        try:
            if data is None or len(data) == 0:
                return None

            # 重命名列
            data = data.rename(columns=column_map)

            # 设置日期索引
            if "date" in data.columns:
                data["date"] = pd.to_datetime(data["date"])
                data = data.set_index("date")

            # 选择需要的列
            required_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [c for c in required_cols if c in data.columns]

            if not available_cols:
                logger.warning(f"数据列不完整: {data.columns}")
                return None

            data = data[available_cols].copy()

            # 去除空值
            data = data.dropna()

            return data if len(data) > 0 else None

        except Exception as e:
            logger.warning(f"数据标准化失败: {e}")
            return None

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

    def refresh_cache(
        self,
        symbol: str,
        days: int = 365
    ) -> dict:
        """刷新缓存数据（重新获取并按天缓存）

        Args:
            symbol: 股票代码
            days: 获取最近多少天的数据

        Returns:
            刷新结果信息
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"刷新缓存: {symbol}, 日期范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

        # 清除内存缓存
        keys_to_remove = [k for k in self._memory_cache if k.startswith(symbol)]
        for k in keys_to_remove:
            del self._memory_cache[k]

        # 强制从网络获取
        data = self._fetch_and_cache_daily(symbol, start_date, end_date)

        return {
            "symbol": symbol,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "rows": len(data),
            "cache_dir": str(self.daily_cache_dir / symbol),
        }

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

        # 清除磁盘缓存（天粒度）
        if symbol is None:
            # 清除所有缓存
            for stock_dir in self.daily_cache_dir.iterdir():
                if stock_dir.is_dir():
                    for file in stock_dir.glob("*.parquet"):
                        file.unlink()
                    logger.info(f"已删除缓存: {stock_dir}")
        else:
            # 清除指定股票的缓存
            stock_dir = self.daily_cache_dir / symbol
            if stock_dir.exists():
                for file in stock_dir.glob("*.parquet"):
                    file.unlink()
                logger.info(f"已删除缓存: {stock_dir}")

    def get_cache_info(self, symbol: str) -> dict:
        """获取缓存信息

        Args:
            symbol: 股票代码

        Returns:
            缓存信息字典
        """
        stock_dir = self.daily_cache_dir / symbol

        if not stock_dir.exists():
            return {"exists": False, "daily_files": 0}

        try:
            cache_files = list(stock_dir.glob("*.parquet"))
            if not cache_files:
                return {"exists": False, "daily_files": 0}

            # 获取日期范围
            dates = sorted([f.stem for f in cache_files])

            return {
                "exists": True,
                "cache_dir": str(stock_dir),
                "daily_files": len(cache_files),
                "start_date": dates[0] if dates else None,
                "end_date": dates[-1] if dates else None,
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
