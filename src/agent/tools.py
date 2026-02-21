"""
AI Agent 工具函数

提供策略分析、优化建议和策略搜索的工具
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.analysis.metrics import calculate_all_metrics
from src.backtest.runner import BacktestRunner
from src.strategy import get_strategy, list_strategies

logger = logging.getLogger(__name__)


def analyze_backtest_result(
    result_data: dict[str, Any],
    output_format: str = "text",
) -> dict[str, Any]:
    """
    分析回测结果

    Args:
        result_data: 回测结果数据
        output_format: 输出格式 (text/json)

    Returns:
        分析结果字典
    """
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "strategy_name": result_data.get("strategy_name", "Unknown"),
        "period": {
            "start": result_data.get("start_date", ""),
            "end": result_data.get("end_date", ""),
        },
        "performance": {},
        "diagnosis": {},
        "recommendations": [],
    }

    # 提取关键指标
    metrics = {
        "total_return": result_data.get("total_return", 0),
        "annual_return": result_data.get("annual_return", 0),
        "max_drawdown": result_data.get("max_drawdown", 0),
        "sharpe_ratio": result_data.get("sharpe_ratio", 0),
        "volatility": result_data.get("metrics", {}).get("volatility", 0),
        "win_rate": result_data.get("metrics", {}).get("win_rate", 0),
    }
    analysis["performance"] = metrics

    # 诊断问题
    diagnosis = []

    # 收益诊断
    if metrics["total_return"] < 0:
        diagnosis.append({
            "type": "negative_return",
            "severity": "high",
            "message": "策略总收益为负，需要重新评估策略逻辑",
        })
    elif metrics["annual_return"] < 0.05:
        diagnosis.append({
            "type": "low_return",
            "severity": "medium",
            "message": "年化收益率较低，可能不如固定收益产品",
        })

    # 风险诊断
    if metrics["max_drawdown"] > 0.2:
        diagnosis.append({
            "type": "high_drawdown",
            "severity": "high",
            "message": f"最大回撤 {metrics['max_drawdown']:.1%} 过高，风险控制需要加强",
        })

    # 夏普比率诊断
    if metrics["sharpe_ratio"] < 0:
        diagnosis.append({
            "type": "negative_sharpe",
            "severity": "high",
            "message": "夏普比率为负，策略表现不如无风险收益",
        })
    elif metrics["sharpe_ratio"] < 1.0:
        diagnosis.append({
            "type": "low_sharpe",
            "severity": "medium",
            "message": "夏普比率低于 1，风险调整后收益不理想",
        })

    # 胜率诊断
    if metrics["win_rate"] < 0.4:
        diagnosis.append({
            "type": "low_win_rate",
            "severity": "medium",
            "message": f"胜率 {metrics['win_rate']:.1%} 较低，需要提高信号质量",
        })

    analysis["diagnosis"] = {
        "issues": diagnosis,
        "issue_count": len(diagnosis),
        "severity_summary": {
            "high": sum(1 for d in diagnosis if d["severity"] == "high"),
            "medium": sum(1 for d in diagnosis if d["severity"] == "medium"),
            "low": sum(1 for d in diagnosis if d["severity"] == "low"),
        },
    }

    # 生成建议
    recommendations = []

    if any(d["type"] in ["negative_return", "negative_sharpe"] for d in diagnosis):
        recommendations.append({
            "priority": 1,
            "action": "review_strategy",
            "description": "重新审视策略逻辑，考虑暂停使用或重大调整",
        })

    if any(d["type"] == "high_drawdown" for d in diagnosis):
        recommendations.append({
            "priority": 2,
            "action": "add_risk_control",
            "description": "添加止损机制和仓位控制",
            "suggestions": [
                "设置单股最大仓位限制",
                "添加组合止损线",
                "使用波动率调整仓位",
            ],
        })

    if any(d["type"] == "low_sharpe" for d in diagnosis):
        recommendations.append({
            "priority": 2,
            "action": "optimize_parameters",
            "description": "优化策略参数提高风险调整收益",
        })

    if any(d["type"] == "low_win_rate" for d in diagnosis):
        recommendations.append({
            "priority": 3,
            "action": "improve_signals",
            "description": "改进信号生成逻辑，提高预测准确率",
        })

    analysis["recommendations"] = recommendations

    return analysis


def suggest_optimizations(
    strategy_name: str,
    analysis_result: dict[str, Any],
    current_params: Optional[dict] = None,
    optimization_history: Optional[list] = None,
) -> dict[str, Any]:
    """
    生成优化建议

    Args:
        strategy_name: 策略名称
        analysis_result: 分析结果
        current_params: 当前参数
        optimization_history: 优化历史

    Returns:
        优化建议
    """
    suggestions = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy_name,
        "parameter_adjustments": {},
        "factor_changes": {"add": [], "remove": []},
        "risk_controls": {},
        "expected_improvement": {},
        "rationale": "",
    }

    # 获取当前参数
    if current_params is None:
        try:
            strategy = get_strategy(strategy_name)
            current_params = strategy.get_strategy_config()
        except Exception:
            current_params = {}

    # 基于诊断生成参数调整建议
    diagnosis = analysis_result.get("diagnosis", {}).get("issues", [])

    # 高回撤调整
    if any(d["type"] == "high_drawdown" for d in diagnosis):
        if "window" in current_params:
            # 增加窗口期以减少交易频率
            suggestions["parameter_adjustments"]["window"] = {
                "current": current_params.get("window", 20),
                "suggested": current_params.get("window", 20) + 5,
                "reason": "增加窗口期减少交易频率，降低回撤",
            }

        suggestions["risk_controls"]["stop_loss"] = 0.05
        suggestions["risk_controls"]["position_limit"] = 0.1

    # 低夏普调整
    if any(d["type"] in ["low_sharpe", "negative_sharpe"] for d in diagnosis):
        if "topk" in current_params:
            suggestions["parameter_adjustments"]["topk"] = {
                "current": current_params.get("topk", 30),
                "suggested": min(current_params.get("topk", 30) - 5, 20),
                "reason": "减少持仓数量集中优质信号",
            }

    # 低胜率调整
    if any(d["type"] == "low_win_rate" for d in diagnosis):
        suggestions["factor_changes"]["add"] = ["rsi", "bollinger"]
        suggestions["factor_changes"]["reason"] = "添加 RSI 和布林带因子过滤假信号"

    # 生成预期改进
    suggestions["expected_improvement"] = {
        "sharpe_change": 0.3,
        "drawdown_reduction": 0.03,
        "win_rate_improvement": 0.05,
    }

    # 生成说明
    suggestion_parts = []
    if suggestions["parameter_adjustments"]:
        param_str = ", ".join(
            f"{k}: {v['current']} -> {v['suggested']}"
            for k, v in suggestions["parameter_adjustments"].items()
        )
        suggestion_parts.append(f"参数调整: {param_str}")

    if suggestions["factor_changes"]["add"]:
        suggestion_parts.append(f"添加因子: {', '.join(suggestions['factor_changes']['add'])}")

    if suggestions["risk_controls"]:
        suggestion_parts.append(f"风险控制: 止损 {suggestions['risk_controls'].get('stop_loss', 'N/A')}")

    suggestions["rationale"] = " | ".join(suggestion_parts) if suggestion_parts else "无需调整"

    return suggestions


def search_strategies(
    strategy_types: Optional[list[str]] = None,
    exclude_existing: bool = True,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """
    搜索新策略

    Args:
        strategy_types: 策略类型列表
        exclude_existing: 是否排除已有策略
        max_results: 最大返回数量

    Returns:
        策略推荐列表
    """
    # 策略库
    strategy_database = [
        {
            "name": "均线偏离策略",
            "type": "mean_reversion",
            "description": "利用价格偏离均线的程度进行反向交易，价格大幅偏离时预期回归",
            "factors": ["price_to_ma", "z_score"],
            "advantages": ["逻辑简单", "适合震荡市", "风险可控"],
            "risks": ["趋势市场中表现不佳", "需要严格止损"],
            "complexity": "low",
            "source": "经典技术分析",
        },
        {
            "name": "多因子动量策略",
            "type": "momentum",
            "description": "结合价格动量、成交量动量和分析师预期变化的多因子策略",
            "factors": ["price_momentum", "volume_momentum", "earnings_revision"],
            "advantages": ["多维度验证", "信号更可靠"],
            "risks": ["因子相关性", "过拟合风险"],
            "complexity": "medium",
            "source": "学术研究 (Jegadeesh 1993)",
        },
        {
            "name": "波动率突破策略",
            "type": "volatility",
            "description": "基于波动率收缩后的价格突破进行交易",
            "factors": ["atr", "volatility_ratio", "breakout"],
            "advantages": ["捕捉大行情", "风险收益比高"],
            "risks": ["假突破", "需要及时止损"],
            "complexity": "medium",
            "source": "期货交易实践",
        },
        {
            "name": "行业轮动策略",
            "type": "sector_rotation",
            "description": "基于行业景气度和资金流向的行业轮动策略",
            "factors": ["sector_momentum", "fund_flow", "relative_strength"],
            "advantages": ["捕捉结构性机会", "分散风险"],
            "risks": ["行业判断错误", "切换成本"],
            "complexity": "high",
            "source": "资产配置研究",
        },
        {
            "name": "价值质量复合策略",
            "type": "value_quality",
            "description": "结合价值因子和质量因子，选择低估优质股票",
            "factors": ["pe_ratio", "roe", "debt_ratio", "cash_flow"],
            "advantages": ["长期有效", "符合价值投资理念"],
            "risks": ["短期可能跑输", "需要财务数据"],
            "complexity": "medium",
            "source": "Fama-French 研究",
        },
    ]

    # 获取已有策略
    existing = list_strategies() if exclude_existing else []

    # 过滤
    filtered = []
    for strategy in strategy_database:
        # 类型过滤
        if strategy_types and strategy["type"] not in strategy_types:
            continue

        # 排除已有
        if exclude_existing:
            strategy_name_lower = strategy["name"].lower().replace(" ", "_")
            if any(strategy_name_lower in ex or ex in strategy_name_lower for ex in existing):
                continue

        filtered.append(strategy)

    # 按复杂度排序（优先推荐简单策略）
    complexity_order = {"low": 0, "medium": 1, "high": 2}
    filtered.sort(key=lambda x: complexity_order.get(x["complexity"], 1))

    return filtered[:max_results]


def run_optimization_cycle(
    strategy_name: str,
    start_date: str,
    end_date: str,
    cash: float = 1000000,
    max_iterations: int = 50,
    optimization_method: str = "bayesian",
    param_grid: Optional[dict[str, list]] = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """
    运行优化循环

    Args:
        strategy_name: 策略名称
        start_date: 开始日期
        end_date: 结束日期
        cash: 初始资金
        max_iterations: 最大迭代次数
        optimization_method: 优化方法 (grid/random/bayesian)
        param_grid: 参数网格，为 None 时使用推荐网格
        dry_run: 是否为试运行

    Returns:
        优化结果
    """
    from src.optimization.auto_optimizer import AutoOptimizer, suggest_param_grid
    from src.optimization.param_optimizer import ParameterOptimizer

    optimization_log = {
        "start_time": datetime.now().isoformat(),
        "strategy": strategy_name,
        "method": optimization_method,
        "iterations": [],
        "final_result": None,
    }

    # 获取策略类
    try:
        strategy = get_strategy(strategy_name)
        strategy_class = strategy.__class__
    except ValueError:
        logger.error(f"未知策略: {strategy_name}")
        optimization_log["error"] = f"未知策略: {strategy_name}"
        return optimization_log

    # 获取参数网格
    if param_grid is None:
        param_grid = suggest_param_grid(strategy_name)

    if not param_grid:
        logger.warning(f"策略 {strategy_name} 没有推荐的参数网格")
        optimization_log["error"] = "没有可优化的参数"
        return optimization_log

    # 创建优化器
    try:
        auto_optimizer = AutoOptimizer(
            optimization_method=optimization_method,
            max_evaluations=max_iterations,
            oos_ratio=0.2,
        )

        # 加载数据（简化：使用空 DataFrame，实际应从缓存加载）
        data = pd.DataFrame()

        # 运行优化循环
        result = auto_optimizer.run_cycle(
            strategy_class,
            param_grid,
            data,
            start_date,
            end_date,
            cash,
            dry_run,
        )

        optimization_log["end_time"] = datetime.now().isoformat()
        optimization_log["final_result"] = {
            "best_params": result.best_params,
            "in_sample_sharpe": result.in_sample_metrics.get("sharpe_ratio", 0),
            "out_of_sample_sharpe": result.out_of_sample_metrics.get("sharpe_ratio", 0),
            "is_overfitted": result.is_overfitted,
            "degradation": result.degradation,
            "recommendation": result.recommendation,
            "optimization_details": result.optimization_details,
        }

        # 记录详细信息
        logger.info(f"优化完成:")
        logger.info(f"  最佳参数: {result.best_params}")
        logger.info(f"  样本内夏普: {result.in_sample_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  样本外夏普: {result.out_of_sample_metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  是否过拟合: {result.is_overfitted}")
        logger.info(f"  建议: {result.recommendation}")

    except Exception as e:
        logger.error(f"优化失败: {e}")
        optimization_log["error"] = str(e)

        # 回退到简单迭代优化
        return _run_simple_optimization(
            strategy_name, start_date, end_date, cash, max_iterations
        )

    return optimization_log


def _run_simple_optimization(
    strategy_name: str,
    start_date: str,
    end_date: str,
    cash: float,
    max_iterations: int,
) -> dict[str, Any]:
    """简单迭代优化（回退方案）"""
    optimization_log = {
        "start_time": datetime.now().isoformat(),
        "strategy": strategy_name,
        "iterations": [],
        "final_result": None,
    }

    best_sharpe = -float("inf")
    best_params = None

    for i in range(max_iterations):
        logger.info(f"优化迭代 {i + 1}/{max_iterations}")

        try:
            strategy = get_strategy(strategy_name)
        except ValueError:
            logger.error(f"未知策略: {strategy_name}")
            break

        runner = BacktestRunner()
        result = runner.run(strategy, start_date, end_date, cash)

        result_data = result.to_dict()
        analysis = analyze_backtest_result(result_data)

        iteration_record = {
            "iteration": i + 1,
            "sharpe_ratio": result.sharpe_ratio,
            "total_return": result.total_return,
            "max_drawdown": result.max_drawdown,
            "params": strategy.get_strategy_config(),
            "diagnosis": analysis["diagnosis"]["severity_summary"],
        }
        optimization_log["iterations"].append(iteration_record)

        if result.sharpe_ratio > best_sharpe:
            best_sharpe = result.sharpe_ratio
            best_params = strategy.get_strategy_config()

        if result.sharpe_ratio >= 1.5 and result.max_drawdown <= 0.15:
            logger.info("达到优化目标")
            break

        suggestions = suggest_optimizations(
            strategy_name,
            analysis,
            strategy.get_strategy_config(),
        )

        logger.info(f"优化建议: {suggestions['rationale']}")

    optimization_log["end_time"] = datetime.now().isoformat()
    optimization_log["final_result"] = {
        "best_sharpe": best_sharpe,
        "best_params": best_params,
    }

    return optimization_log


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 测试分析工具
    test_result = {
        "strategy_name": "dual_ma",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "total_return": -0.05,
        "annual_return": -0.05,
        "max_drawdown": 0.18,
        "sharpe_ratio": -0.5,
        "metrics": {"win_rate": 0.42, "volatility": 0.15},
    }

    print("=== 回测分析 ===")
    analysis = analyze_backtest_result(test_result)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))

    print("\n=== 优化建议 ===")
    suggestions = suggest_optimizations("dual_ma", analysis)
    print(json.dumps(suggestions, indent=2, ensure_ascii=False))

    print("\n=== 策略搜索 ===")
    new_strategies = search_strategies(max_results=3)
    for s in new_strategies:
        print(f"- {s['name']}: {s['description']}")
