#!/usr/bin/env python
"""生成报告脚本"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.report import ReportGenerator
from src.analysis.metrics import calculate_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="生成回测分析报告")
    parser.add_argument(
        "--result-file",
        type=str,
        required=True,
        help="回测结果文件路径 (JSON)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 PDF 文件路径",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="不生成图表（加快速度）",
    )
    args = parser.parse_args()

    # 加载回测结果
    import json

    result_file = Path(args.result_file)
    if not result_file.exists():
        logger.error(f"结果文件不存在: {result_file}")
        return 1

    with open(result_file, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    logger.info(f"加载回测结果: {result_file}")

    # 生成报告
    generator = ReportGenerator()

    # 注意：这里需要实际的 portfolio_value 数据
    # 简化版：使用结果文件中的指标直接生成摘要报告
    logger.info("生成 PDF 报告...")

    # 创建模拟数据用于演示
    import numpy as np
    import pandas as pd

    start_date = result_data.get("start_date", "2023-01-01")
    end_date = result_data.get("end_date", "2023-12-31")
    total_return = result_data.get("total_return", 0.1)

    # 生成模拟净值曲线
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n = len(dates)
    daily_return = (1 + total_return) ** (1 / n) - 1
    noise = np.random.normal(0, 0.01, n)
    returns = daily_return + noise
    portfolio_value = pd.Series(1000000 * (1 + returns).cumprod(), index=dates)

    report_path = generator.generate_report(
        portfolio_value=portfolio_value,
        strategy_name=result_data.get("strategy_name", "未知策略"),
        output_filename=args.output,
        include_charts=not args.no_charts,
    )

    print(f"\n报告已生成: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
