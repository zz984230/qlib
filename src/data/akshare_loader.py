"""Akshare 数据加载器"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class AkshareLoader:
    """Akshare 数据加载器，用于获取 A 股市场数据"""

    def __init__(self, config_path: str = "configs/data.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.cache_dir = Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.akshare_config = self.config.get("akshare", {})
        self.retry_times = self.akshare_config.get("retry_times", 3)
        self.retry_delay = self.akshare_config.get("retry_delay", 1)

    def get_stock_list(self, market: str = "csi300") -> pd.DataFrame:
        """
        获取股票池列表

        Args:
            market: 市场指数代码
                - csi300: 沪深300
                - csi500: 中证500
                - csi1000: 中证1000

        Returns:
            包含股票代码和名称的 DataFrame
        """
        market_map = {
            "csi300": "000300",
            "csi500": "000905",
            "csi1000": "000852",
        }

        symbol = market_map.get(market, "000300")

        try:
            df = ak.index_stock_cons(symbol=symbol)
            logger.info(f"获取 {market} 成分股: {len(df)} 只")
            return df
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise

    def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票的历史行情数据

        Args:
            symbol: 股票代码 (如 "000001")
            start_date: 开始日期 (格式: "2020-01-01")
            end_date: 结束日期 (格式: "2024-01-01")
            adjust: 复权类型 ("qfq" 前复权, "hfq" 后复权, "" 不复权)

        Returns:
            包含 OHLCV 数据的 DataFrame
        """
        # 转换日期格式 (2020-01-01 -> 20200101)
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")

        for attempt in range(self.retry_times):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust=adjust,
                )

                if df is not None and len(df) > 0:
                    # 标准化列名
                    column_map = {
                        "日期": "date",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "volume",
                        "成交额": "amount",
                        "振幅": "amplitude",
                        "涨跌幅": "pct_change",
                        "涨跌额": "change",
                        "换手率": "turnover",
                    }
                    df = df.rename(columns=column_map)
                    df["date"] = pd.to_datetime(df["date"])
                    df["symbol"] = symbol

                    logger.debug(f"获取 {symbol} 数据: {len(df)} 条")
                    return df

            except Exception as e:
                logger.warning(
                    f"获取 {symbol} 数据失败 (尝试 {attempt + 1}/{self.retry_times}): {e}"
                )
                if attempt < self.retry_times - 1:
                    import time

                    time.sleep(self.retry_delay)

        logger.error(f"获取 {symbol} 数据最终失败")
        return None

    def update_all(self, universe: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
        """
        批量更新股票池数据

        Args:
            universe: 股票代码列表，为 None 时使用配置中的股票池

        Returns:
            股票代码到数据的映射字典
        """
        if universe is None:
            market = self.config.get("universe", "csi300")
            stock_list = self.get_stock_list(market)
            universe = stock_list["股票代码"].tolist()

        start_date = self.config.get("start_date", "2020-01-01")
        end_date = self.config.get("end_date") or datetime.now().strftime("%Y-%m-%d")
        adjust = self.akshare_config.get("adjust", "qfq")

        results = {}
        total = len(universe)

        for i, symbol in enumerate(universe, 1):
            logger.info(f"更新数据 [{i}/{total}]: {symbol}")

            data = self.get_stock_data(symbol, start_date, end_date, adjust)
            if data is not None:
                results[symbol] = data
                # 保存到缓存
                cache_file = self.cache_dir / f"{symbol}.parquet"
                data.to_parquet(cache_file, index=False)

        logger.info(f"数据更新完成: {len(results)}/{total} 只股票")
        return results

    def load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据"""
        cache_file = self.cache_dir / f"{symbol}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return None


if __name__ == "__main__":
    # 测试数据加载
    logging.basicConfig(level=logging.INFO)

    loader = AkshareLoader()

    # 测试获取股票列表
    stock_list = loader.get_stock_list("csi300")
    print(f"\n沪深300 成分股 (前10只):")
    print(stock_list.head(10))

    # 测试获取单只股票数据
    if len(stock_list) > 0:
        symbol = stock_list.iloc[0]["股票代码"]
        data = loader.get_stock_data(symbol, "2024-01-01", "2024-12-31")
        if data is not None:
            print(f"\n{symbol} 数据 (前5条):")
            print(data.head())
