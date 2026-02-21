#!/usr/bin/env python
"""运行回测脚本 (增强版)"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.runner import BacktestRunner
from src.strategy import get_strategy, list_strategies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="运行策略回测")
    parser.add_argument(
        "--strategy",
        type=str,
        default="simple",
        help=f"策略类型 (可选: {', '.join(list_strategies())})",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-01-01",
        help="回测开始日期",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="回测结束日期",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=1000000,
        help="初始资金 (default: 1000000)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存回测结果",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="结果文件名称",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="比较所有策略",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用策略",
    )
    args = parser.parse_args()

    # 列出策略
    if args.list:
        print("可用策略:")
        for name in list_strategies():
            print(f"  - {name}")
        return 0

    # 比较所有策略
    if args.compare:
        runner = BacktestRunner()
        strategies = [get_strategy(name) for name in list_strategies()]
        comparison = runner.compare_strategies(
            strategies, args.start_date, args.end_date, args.cash
        )
        print("\n策略比较结果:")
        print(comparison.to_string())
        return 0

    # 单策略回测
    logger.info(f"初始化策略: {args.strategy}")

    try:
        strategy = get_strategy(args.strategy)
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info(f"开始回测: {args.start_date} -> {args.end_date}")
    runner = BacktestRunner()
    result = runner.run(
        strategy=strategy,
        start_date=args.start_date,
        end_date=args.end_date,
        cash=args.cash,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"策略: {result.strategy_name}")
    print(f"时间段: {result.start_date} -> {result.end_date}")
    print(f"初始资金: {result.cash:,.0f}")
    print("-" * 60)
    print(f"总收益率: {result.total_return:>12.2%}")
    print(f"年化收益率: {result.annual_return:>10.2%}")
    print(f"最大回撤: {result.max_drawdown:>12.2%}")
    print(f"夏普比率: {result.sharpe_ratio:>12.2f}")
    print(f"索提诺比率: {result.sortino_ratio:>10.2f}")
    print(f"卡玛比率: {result.calmar_ratio:>12.2f}")
    print(f"年化波动率: {result.volatility:>10.2%}")
    print(f"胜率: {result.win_rate:>15.2%}")

    if result.excess_return != 0:
        print(f"超额收益: {result.excess_return:>11.2%}")

    print("=" * 60)

    # 保存结果
    if args.save:
        runner.save_result(result, args.name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
