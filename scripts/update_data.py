#!/usr/bin/env python
"""数据更新脚本"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.akshare_loader import AkshareLoader
from src.data.qlib_converter import QlibConverter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="更新股票数据")
    parser.add_argument(
        "--market",
        type=str,
        default="csi300",
        choices=["csi300", "csi500", "csi1000"],
        help="股票池 (default: csi300)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="开始日期 (default: 2020-01-01)",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="仅转换已有数据，不重新下载",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="转换后验证数据",
    )
    args = parser.parse_args()

    logger.info(f"开始数据更新流程，股票池: {args.market}")

    # Step 1: 下载数据 (如果需要)
    if not args.convert_only:
        logger.info("Step 1: 从 akshare 下载数据...")
        loader = AkshareLoader()

        # 获取股票列表
        stock_list = loader.get_stock_list(args.market)
        universe = stock_list["股票代码"].tolist()

        # 批量下载
        data_dict = loader.update_all(universe)
        logger.info(f"下载完成: {len(data_dict)} 只股票")
    else:
        logger.info("Step 1: 跳过下载，使用缓存数据")

    # Step 2: 转换为 Qlib 格式
    logger.info("Step 2: 转换为 Qlib 格式...")
    converter = QlibConverter()
    converter.convert_from_parquet_cache()

    # Step 3: 验证数据 (如果需要)
    if args.verify:
        logger.info("Step 3: 验证数据...")
        if converter.verify_data():
            summary = converter.get_data_summary()
            logger.info("数据验证成功!")
            logger.info(f"  股票数量: {summary['total_stocks']}")
            logger.info(f"  日期范围: {summary['date_range']['start']} -> {summary['date_range']['end']}")
            logger.info(f"  特征: {summary['features']}")
        else:
            logger.error("数据验证失败!")
            return 1

    logger.info("数据更新流程完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
