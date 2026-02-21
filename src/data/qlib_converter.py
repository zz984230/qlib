"""Qlib 数据格式转换器 (增强版)"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class QlibConverter:
    """将 Akshare 数据转换为 Qlib 格式"""

    def __init__(self, config_path: str = "configs/qlib.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.qlib_dir = Path(self.config.get("provider_uri", "data/qlib"))
        self.qlib_dir.mkdir(parents=True, exist_ok=True)

        # Qlib 数据存储结构
        self.features_dir = self.qlib_dir / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # 日志目录
        self.log_dir = self.qlib_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def convert_from_akshare(
        self,
        data_dict: dict[str, pd.DataFrame],
        market: str = "cn",
    ) -> dict:
        """
        将 Akshare 数据转换为 Qlib 格式

        Qlib 数据格式要求:
        - 每只股票一个目录，目录名为股票代码
        - 每个特征一个文件，文件名为特征名.bin
        - 数据为二进制 float32 格式
        - 需要一个 instruments 文件记录股票列表和日期范围

        Returns:
            转换统计信息
        """
        logger.info(f"开始转换 {len(data_dict)} 只股票数据...")

        # 收集所有股票的日期范围
        instruments_info = []
        stats = {
            "total": len(data_dict),
            "success": 0,
            "failed": 0,
            "stocks": [],
        }

        for symbol, df in data_dict.items():
            if df is None or len(df) == 0:
                stats["failed"] += 1
                continue

            try:
                # 创建股票目录
                stock_dir = self.features_dir / symbol
                stock_dir.mkdir(parents=True, exist_ok=True)

                # 转换日期为数值 (用于排序)
                df = df.sort_values("date").reset_index(drop=True)
                dates = df["date"].dt.strftime("%Y-%m-%d").tolist()

                # 保存核心 OHLCV 特征
                feature_columns = {
                    "open": "$open",
                    "close": "$close",
                    "high": "$high",
                    "low": "$low",
                    "volume": "$volume",
                    "amount": "$amount",
                }

                for col, _ in feature_columns.items():
                    if col in df.columns:
                        values = df[col].values.astype(np.float32)
                        bin_file = stock_dir / f"{col}.bin"
                        values.tofile(str(bin_file))

                # 保存日期索引
                date_file = stock_dir / "date.bin"
                date_values = df["date"].astype(np.int64).values
                date_values.tofile(str(date_file))

                # 记录日期范围
                instruments_info.append(
                    {
                        "symbol": symbol,
                        "start_date": dates[0],
                        "end_date": dates[-1],
                    }
                )

                stats["success"] += 1
                stats["stocks"].append(symbol)

            except Exception as e:
                logger.error(f"转换 {symbol} 失败: {e}")
                stats["failed"] += 1

        # 生成 instruments 文件
        self._write_instruments(instruments_info, market)

        # 生成日历文件
        self._generate_calendar(instruments_info)

        # 保存转换日志
        self._save_conversion_log(stats)

        logger.info(
            f"数据转换完成: {stats['success']}/{stats['total']} 只股票成功"
        )

        return stats

    def _write_instruments(
        self,
        instruments_info: list[dict],
        market: str = "cn",
    ) -> None:
        """生成 Qlib instruments 文件"""
        instruments_file = self.qlib_dir / f"instruments_{market}.txt"

        with open(instruments_file, "w", encoding="utf-8") as f:
            for info in instruments_info:
                # Qlib instruments 格式: symbol\tstart_date\tend_date
                f.write(f"{info['symbol']}\t{info['start_date']}\t{info['end_date']}\n")

        logger.info(f"生成 instruments 文件: {instruments_file}")

    def _generate_calendar(self, instruments_info: list[dict]) -> None:
        """生成交易日历文件"""
        if not instruments_info:
            return

        # 收集所有日期
        all_dates = set()
        for info in instruments_info:
            start = datetime.strptime(info["start_date"], "%Y-%m-%d")
            end = datetime.strptime(info["end_date"], "%Y-%m-%d")

            # 生成日期范围 (工作日)
            dates = pd.date_range(start=start, end=end, freq="B")
            all_dates.update(dates.strftime("%Y-%m-%d").tolist())

        # 排序并写入文件
        sorted_dates = sorted(all_dates)
        calendar_file = self.qlib_dir / "calendars" / "day.txt"
        calendar_file.parent.mkdir(parents=True, exist_ok=True)

        with open(calendar_file, "w", encoding="utf-8") as f:
            for date in sorted_dates:
                f.write(f"{date}\n")

        logger.info(f"生成日历文件: {calendar_file} ({len(sorted_dates)} 个交易日)")

    def _save_conversion_log(self, stats: dict) -> None:
        """保存转换日志"""
        import json

        log_file = self.log_dir / f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def convert_from_parquet_cache(
        self,
        cache_dir: str = "data/raw",
        market: str = "cn",
    ) -> dict:
        """从 parquet 缓存文件转换"""
        cache_path = Path(cache_dir)
        data_dict = {}

        for parquet_file in cache_path.glob("*.parquet"):
            symbol = parquet_file.stem
            try:
                df = pd.read_parquet(parquet_file)
                if len(df) > 0:
                    data_dict[symbol] = df
            except Exception as e:
                logger.warning(f"读取 {parquet_file} 失败: {e}")

        if data_dict:
            return self.convert_from_akshare(data_dict, market)
        else:
            logger.warning("缓存目录中没有找到数据文件")
            return {"total": 0, "success": 0, "failed": 0}

    def verify_data(self) -> bool:
        """验证 Qlib 数据是否可用"""
        # 检查文件结构
        instruments_file = self.qlib_dir / "instruments_cn.txt"
        calendar_file = self.qlib_dir / "calendars" / "day.txt"

        if not instruments_file.exists():
            logger.warning("instruments 文件不存在")
            return False

        if not calendar_file.exists():
            logger.warning("日历文件不存在")
            return False

        # 检查至少有一只股票的数据
        stock_dirs = list(self.features_dir.glob("*"))
        if not stock_dirs:
            logger.warning("没有找到股票数据")
            return False

        # 检查特征文件
        required_features = ["close"]
        for stock_dir in stock_dirs[:5]:  # 检查前5只
            for feature in required_features:
                if not (stock_dir / f"{feature}.bin").exists():
                    logger.warning(f"{stock_dir.name} 缺少 {feature} 特征")
                    return False

        logger.info(f"数据验证成功: {len(stock_dirs)} 只股票")
        return True

    def get_data_summary(self) -> dict:
        """获取数据摘要信息"""
        summary = {
            "total_stocks": 0,
            "date_range": {"start": None, "end": None},
            "features": [],
            "data_size_mb": 0,
        }

        # 统计股票数量和数据大小
        stock_dirs = list(self.features_dir.glob("*"))
        summary["total_stocks"] = len(stock_dirs)

        total_size = 0
        if stock_dirs:
            first_stock = stock_dirs[0]
            summary["features"] = [f.stem for f in first_stock.glob("*.bin")]

            # 计算总数据大小
            for stock_dir in stock_dirs:
                for bin_file in stock_dir.glob("*.bin"):
                    total_size += bin_file.stat().st_size

        summary["data_size_mb"] = round(total_size / 1024 / 1024, 2)

        # 读取 instruments 文件获取日期范围
        instruments_file = self.qlib_dir / "instruments_cn.txt"
        if instruments_file.exists():
            with open(instruments_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    dates = []
                    for line in lines:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            dates.extend([parts[1], parts[2]])
                    if dates:
                        summary["date_range"]["start"] = min(dates)
                        summary["date_range"]["end"] = max(dates)

        return summary

    def clear_data(self) -> None:
        """清除所有数据"""
        import shutil

        if self.qlib_dir.exists():
            shutil.rmtree(self.qlib_dir)
            logger.info("数据已清除")

        self.qlib_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    converter = QlibConverter()

    # 从缓存转换数据
    stats = converter.convert_from_parquet_cache()
    print(f"\n转换统计: {stats}")

    # 验证数据
    if converter.verify_data():
        print("\n数据转换成功!")
        summary = converter.get_data_summary()
        print(f"数据摘要: {summary}")
