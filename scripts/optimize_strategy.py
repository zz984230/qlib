#!/usr/bin/env python
"""
策略优化脚本

交互式策略优化工具
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.tools import (
    analyze_backtest_result,
    suggest_optimizations,
)
from src.backtest.runner import BacktestRunner
from src.strategy import get_strategy, list_strategies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def interactive_optimize(strategy_name: str, start_date: str, end_date: str):
    """交互式优化"""

    print(f"\n{'=' * 60}")
    print(f"策略优化: {strategy_name}")
    print(f"{'=' * 60}\n")

    # 获取策略
    strategy = get_strategy(strategy_name)
    print(f"当前策略参数: {strategy.get_strategy_config()}\n")

    # 运行回测
    print("运行回测...")
    runner = BacktestRunner()
    result = runner.run(strategy, start_date, end_date)

    print(f"\n--- 回测结果 ---")
    print(f"总收益: {result.total_return:.2%}")
    print(f"年化收益: {result.annual_return:.2%}")
    print(f"最大回撤: {result.max_drawdown:.2%}")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")

    # 分析
    print("\n--- 问题诊断 ---")
    analysis = analyze_backtest_result(result.to_dict())

    for issue in analysis["diagnosis"]["issues"]:
        severity = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}.get(issue["severity"], "[?]")
        print(f"{severity} {issue['message']}")

    # 优化建议
    print("\n--- 优化建议 ---")
    suggestions = suggest_optimizations(strategy_name, analysis, strategy.get_strategy_config())

    if suggestions["parameter_adjustments"]:
        print("\n参数调整:")
        for param, adj in suggestions["parameter_adjustments"].items():
            print(f"  - {param}: {adj['current']} -> {adj['suggested']}")
            print(f"    原因: {adj['reason']}")

    if suggestions["factor_changes"]["add"]:
        print(f"\n添加因子: {', '.join(suggestions['factor_changes']['add'])}")
        if suggestions["factor_changes"].get("reason"):
            print(f"  原因: {suggestions['factor_changes']['reason']}")

    if suggestions["risk_controls"]:
        print("\n风险控制:")
        for control, value in suggestions["risk_controls"].items():
            print(f"  - {control}: {value}")

    print(f"\n预期改进:")
    exp = suggestions["expected_improvement"]
    print(f"  - 夏普变化: {exp.get('sharpe_change', 0):+.2f}")
    print(f"  - 回撤降低: {exp.get('drawdown_reduction', 0):.1%}")

    # 保存结果
    runner.save_result(result)
    print(f"\n结果已保存")

    return suggestions


def main():
    parser = argparse.ArgumentParser(description="策略优化工具")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help=f"策略名称 (可选: {', '.join(list_strategies())})",
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
        "--apply",
        action="store_true",
        help="应用优化建议",
    )
    args = parser.parse_args()

    suggestions = interactive_optimize(
        args.strategy,
        args.start_date,
        args.end_date,
    )

    if args.apply:
        print("\n应用优化建议...")
        # 实际应用需要修改配置文件或策略代码
        print("注意: 需要手动更新配置文件或策略代码")

    return 0


if __name__ == "__main__":
    sys.exit(main())
