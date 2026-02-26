#!/usr/bin/env python3
"""刷新股票数据缓存

按天粒度缓存股票数据，用于后续计算脚本快速加载。
"""

import argparse
import logging
import sys
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="刷新股票数据缓存")
    parser.add_argument("--symbol", type=str, required=True, help="股票代码，如 601138")
    parser.add_argument("--days", type=int, default=365, help="获取最近多少天的数据（默认365天）")
    parser.add_argument("--clear", action="store_true", help="清除现有缓存后再刷新")
    args = parser.parse_args()

    from src.data.turtle_data_loader import TurtleDataLoader

    loader = TurtleDataLoader()

    # 显示当前缓存状态
    cache_info = loader.get_cache_info(args.symbol)
    logger.info(f"当前缓存状态: {cache_info}")

    # 清除现有缓存
    if args.clear:
        logger.info(f"清除现有缓存: {args.symbol}")
        loader.clear_cache(args.symbol)

    # 刷新缓存
    logger.info(f"开始刷新缓存: {args.symbol}, 最近 {args.days} 天")
    result = loader.refresh_cache(args.symbol, args.days)

    if result["rows"] > 0:
        logger.info(f"[OK] 缓存刷新成功!")
        logger.info(f"  - 股票代码: {result['symbol']}")
        logger.info(f"  - 日期范围: {result['start_date']} ~ {result['end_date']}")
        logger.info(f"  - 数据条数: {result['rows']}")
        logger.info(f"  - 缓存目录: {result['cache_dir']}")
        return 0
    else:
        logger.error(f"[ERROR] 缓存刷新失败，未获取到数据")
        return 1


if __name__ == "__main__":
    sys.exit(main())
