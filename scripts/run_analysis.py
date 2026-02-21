#!/usr/bin/env python
"""完整分析流水线脚本"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.runner import BacktestRunner
from src.analysis.report import ReportGenerator
from src.strategy import get_strategy, list_strategies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="运行回测并生成报告")
    parser.add_argument(
        "--strategy",
        type=str,
        default="dual_ma",
        help=f"策略类型 (可选: {', '.join(list_strategies())})",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
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
        help="初始资金",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="报告文件名",
    )
    args = parser.parse_args()

    logger.info(f"策略: {args.strategy}")
    logger.info(f"回测区间: {args.start_date} -> {args.end_date}")

    # 运行回测
    strategy = get_strategy(args.strategy)
    runner = BacktestRunner()
    result = runner.run(
        strategy=strategy,
        start_date=args.start_date,
        end_date=args.end_date,
        cash=args.cash,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print("回测结果摘要")
    print("=" * 60)
    print(f"总收益率: {result.total_return:>12.2%}")
    print(f"年化收益率: {result.annual_return:>10.2%}")
    print(f"最大回撤: {result.max_drawdown:>12.2%}")
    print(f"夏普比率: {result.sharpe_ratio:>12.2f}")
    print("=" * 60)

    # 保存结果
    result_file = runner.save_result(result)
    logger.info(f"结果已保存: {result_file}")

    # 生成报告
    if result.portfolio_value is not None:
        logger.info("生成 PDF 报告...")
        generator = ReportGenerator()

        output_name = args.output or f"{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        report_path = generator.generate_report(
            portfolio_value=result.portfolio_value,
            strategy_name=str(result),
            benchmark=result.benchmark,
            output_filename=output_name,
        )

        print(f"\nPDF 报告已生成: {report_path}")
    else:
        logger.warning("没有净值数据，跳过报告生成")

    return 0


if __name__ == "__main__":
    sys.exit(main())
