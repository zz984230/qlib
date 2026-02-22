#!/usr/bin/env python
"""海龟遗传算法优化器 - 命令行入口

使用遗传算法自动探索海龟交易策略的最优参数组合。

示例:
    # 基本用法
    python scripts/turtle_genetic_optimizer.py --symbol 601138 --name "工业富联"

    # 自定义参数
    python scripts/turtle_genetic_optimizer.py \\
        --symbol 601138 \\
        --name "工业富联" \\
        --cash 100000 \\
        --max-generations 50 \\
        --population-size 30 \\
        --no-interactive
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer.turtle_optimizer import TurtleGeneticOptimizer
from src.optimizer.genetic_engine import GeneticConfig
from src.optimizer.fitness import FitnessConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="海龟遗传算法优化器 - 自动探索最优交易策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --symbol 601138 --name "工业富联"
  %(prog)s --symbol 601138 --cash 100000 --max-generations 50
  %(prog)s --symbol 601138 --no-interactive --max-drawdown 0.05
        """
    )

    # 必需参数
    parser.add_argument(
        "--symbol",
        required=True,
        help="股票代码 (如: 601138)"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="股票名称 (如: 工业富联)"
    )

    # 资金参数
    parser.add_argument(
        "--cash",
        type=float,
        default=50000,
        help="初始资金 (默认: 50000)"
    )

    # 遗传算法参数
    parser.add_argument(
        "--max-generations",
        type=int,
        default=100,
        help="最大演化代数 (默认: 100)"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="种群大小 (默认: 50)"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.1,
        help="变异率 (默认: 0.1)"
    )
    parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="交叉率 (默认: 0.7)"
    )
    parser.add_argument(
        "--elite-size",
        type=int,
        default=5,
        help="精英保留数量 (默认: 5)"
    )

    # 风控参数
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.03,
        help="最大回撤限制 (默认: 0.03)"
    )

    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/html",
        help="HTML 报告输出目录 (默认: reports/html)"
    )
    parser.add_argument(
        "--pool-dir",
        type=str,
        default="strategies/pool",
        help="策略池目录 (默认: strategies/pool)"
    )

    # 运行参数
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="非交互模式运行（不暂停等待用户输入）"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="数据开始日期 (格式: YYYY-MM-DD)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="随机种子（用于可复现性）"
    )

    # 日志参数
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细日志输出"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="安静模式（只输出错误）"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # 创建必要目录
    Path("logs").mkdir(exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.pool_dir).mkdir(parents=True, exist_ok=True)

    # 配置遗传算法
    genetic_config = GeneticConfig(
        population_size=args.population_size,
        max_generations=args.max_generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_size=args.elite_size,
    )

    # 配置适应度评估
    fitness_config = FitnessConfig(
        max_drawdown_limit=args.max_drawdown,
    )

    # 创建优化器
    optimizer = TurtleGeneticOptimizer(
        symbol=args.symbol,
        name=args.name,
        initial_cash=args.cash,
        output_dir=args.output_dir,
        pool_dir=args.pool_dir,
        genetic_config=genetic_config,
        fitness_config=fitness_config,
        interactive=not args.no_interactive,
    )

    # 运行优化
    logger.info("=" * 60)
    logger.info("海龟遗传算法优化器启动")
    logger.info(f"股票: {args.name} ({args.symbol})")
    logger.info(f"初始资金: {args.cash:,.0f}")
    logger.info(f"最大回撤限制: {args.max_drawdown:.1%}")
    logger.info(f"最大代数: {args.max_generations}")
    logger.info(f"种群大小: {args.population_size}")
    logger.info("=" * 60)

    try:
        result = optimizer.run(start_date=args.start_date)

        # 输出结果
        if result["status"] == "completed":
            logger.info("=" * 60)
            logger.info("优化完成!")
            logger.info(f"总代数: {result['generations']}")
            logger.info(f"发现策略: {result['valid_strategies']}")
            logger.info(f"最优适应度: {result['best_fitness']:.4f}")
            logger.info(f"报告路径: {result['report_path']}")
            logger.info("=" * 60)
            return 0

        elif result["status"] == "cancelled":
            logger.info("优化被用户取消")
            return 1

        else:
            logger.error(f"优化失败: {result.get('message', '未知错误')}")
            return 1

    except KeyboardInterrupt:
        logger.info("\n检测到中断，正在停止...")
        return 130

    except Exception as e:
        logger.exception(f"优化过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
