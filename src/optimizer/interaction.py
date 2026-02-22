"""交互式用户交互模块

提供用户交互功能，用于在发现有效策略时暂停并等待用户确认。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def confirm_continue(
    strategy_count: int,
    best_fitness: float,
    generation: int
) -> str:
    """等待用户确认是否继续优化

    Args:
        strategy_count: 发现的策略数量
        best_fitness: 最优适应度
        generation: 当前代数

    Returns:
        用户选择 ('c', 's', 'a')
    """
    print("\n" + "=" * 60)
    print(f"发现有效策略！代数: {generation}, 策略数: {strategy_count}, 最优适应度: {best_fitness:.4f}")
    print("=" * 60)
    print("请选择操作:")
    print("  [c]ontinue - 继续优化")
    print("  [s]top    - 停止并生成最终报告")
    print("  [a]djust  - 调整参数后继续")
    print()

    while True:
        try:
            choice = input("您的选择 (c/s/a): ").strip().lower()
            if choice in ["c", "s", "a"]:
                return choice
            print("无效选择，请输入 c, s 或 a")
        except (EOFError, KeyboardInterrupt):
            print("\n检测到中断，将停止优化")
            return "s"


def display_strategy_summary(
    strategies: list[dict],
    top_n: int = 5
) -> None:
    """显示策略摘要

    Args:
        strategies: 策略列表
        top_n: 显示前 N 个
    """
    print("\n最优策略 (Top {}):".format(min(top_n, len(strategies))))

    for i, strategy in enumerate(strategies[:top_n], 1):
        print(f"\n  [{i}] 策略 ID: {strategy.get('id', 'N/A')}")

        # 因子权重
        factors = strategy.get("factor_weights", {})
        print("      因子权重:")
        for factor, weight in sorted(factors.items(), key=lambda x: -x[1])[:5]:
            print(f"        - {factor}: {weight:.2f}")

        # 阈值
        print(f"      入场阈值: {strategy.get('signal_threshold', 0):.2f}")
        print(f"      出场阈值: {strategy.get('exit_threshold', 0):.2f}")

        # 适应度
        print(f"      适应度: {strategy.get('fitness', 0):.4f}")

        # 回测结果
        backtest = strategy.get("backtest_results", {})
        if backtest:
            print("      回测结果:")
            for period, result in backtest.items():
                print(f"        - {period}: 收益 {result.get('total_return', 0)*100:.1f}%, "
                      f"回撤 {result.get('max_drawdown', 0)*100:.2f}%")


def pause_and_wait(
    message: str = "按 Enter 继续...",
    timeout: Optional[int] = None
) -> str:
    """暂停并等待用户输入

    Args:
        message: 提示消息
        timeout: 超时时间（秒），None 表示无限等待

    Returns:
        用户输入
    """
    print(f"\n{message}")
    try:
        return input("> ")
    except (EOFError, KeyboardInterrupt):
        return ""


def display_progress(
    generation: int,
    max_generations: int,
    best_fitness: float,
    avg_fitness: float,
    valid_count: int
) -> None:
    """显示优化进度

    Args:
        generation: 当前代数
        max_generations: 最大代数
        best_fitness: 最优适应度
        avg_fitness: 平均适应度
        valid_count: 有效策略数
    """
    progress = generation / max_generations * 100

    print(f"\r[进度: {progress:5.1f}%] "
          f"代数: {generation}/{max_generations} | "
          f"最优: {best_fitness:.4f} | "
          f"平均: {avg_fitness:.4f} | "
          f"有效: {valid_count}",
          end="", flush=True)


def display_generation_summary(
    generation: int,
    population_size: int,
    best_fitness: float,
    avg_fitness: float,
    valid_strategies: int
) -> None:
    """显示一代的汇总信息

    Args:
        generation: 代数
        population_size: 种群大小
        best_fitness: 最优适应度
        avg_fitness: 平均适应度
        valid_strategies: 有效策略数
    """
    print()
    print("-" * 60)
    print(f"代数 {generation} 完成")
    print(f"  种群大小: {population_size}")
    print(f"  最优适应度: {best_fitness:.4f}")
    print(f"  平均适应度: {avg_fitness:.4f}")
    print(f"  有效策略数: {valid_strategies}")
    print("-" * 60)


def should_pause_on_valid_strategy(
    valid_strategies_count: int,
    pause_every: int = 1
) -> bool:
    """判断是否应该暂停

    Args:
        valid_strategies_count: 当前有效策略数
        pause_every: 每发现 N 个策略暂停一次

    Returns:
        是否暂停
    """
    return valid_strategies_count > 0 and valid_strategies_count % pause_every == 0


def get_user_adjustment() -> dict:
    """获取用户调整的参数

    Returns:
        调整后的参数字典
    """
    print("\n参数调整:")
    print("直接按 Enter 使用默认值")

    adjustments = {}

    try:
        max_gen = input(f"最大代数 [当前: 默认]: ").strip()
        if max_gen:
            adjustments["max_generations"] = int(max_gen)

        pop_size = input(f"种群大小 [当前: 默认]: ").strip()
        if pop_size:
            adjustments["population_size"] = int(pop_size)

        mutation = input(f"变异率 [当前: 默认]: ").strip()
        if mutation:
            adjustments["mutation_rate"] = float(mutation)

    except (ValueError, EOFError, KeyboardInterrupt):
        print("输入无效或被中断，使用默认参数")

    return adjustments


def confirm_save_strategy(strategy: dict) -> bool:
    """确认是否保存策略

    Args:
        strategy: 策略数据

    Returns:
        是否保存
    """
    print("\n发现新策略，是否保存到策略池？")
    print(f"  适应度: {strategy.get('fitness', 0):.4f}")

    factors = strategy.get("factor_weights", {})
    if factors:
        print("  主要因子:")
        for factor, weight in sorted(factors.items(), key=lambda x: -x[1])[:3]:
            print(f"    - {factor}: {weight:.2f}")

    try:
        choice = input("保存策略? (y/n): ").strip().lower()
        return choice == "y"
    except (EOFError, KeyboardInterrupt):
        return True  # 默认保存


def display_final_report(
    total_generations: int,
    total_strategies: int,
    best_fitness: float,
    report_path: str
) -> None:
    """显示最终报告摘要

    Args:
        total_generations: 总代数
        total_strategies: 总策略数
        best_fitness: 最优适应度
        report_path: 报告路径
    """
    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)
    print(f"  总代数: {total_generations}")
    print(f"  发现策略: {total_strategies}")
    print(f"  最优适应度: {best_fitness:.4f}")
    print(f"  报告路径: {report_path}")
    print("=" * 60)


def confirm_run_optimization(symbol: str, cash: float) -> bool:
    """确认是否运行优化

    Args:
        symbol: 股票代码
        cash: 初始资金

    Returns:
        是否运行
    """
    print(f"\n即将运行海龟遗传算法优化:")
    print(f"  股票代码: {symbol}")
    print(f"  初始资金: {cash:,.0f}")
    print(f"  最大回撤限制: 3%")
    print()

    try:
        choice = input("确认开始? (y/n): ").strip().lower()
        return choice == "y"
    except (EOFError, KeyboardInterrupt):
        return False
