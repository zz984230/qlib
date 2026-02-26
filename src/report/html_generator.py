"""HTML 报告生成器

生成海龟遗传算法优化的 HTML 报告，包含图表和详细分析。
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np

try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    logging.warning("Jinja2 未安装，HTML 报告功能受限")

from src.optimizer.individual import Individual

logger = logging.getLogger(__name__)

# 因子描述字典（用于报告中的 tooltip）
FACTOR_DESCRIPTIONS = {
    "ma_ratio": "均线偏离度, MA5/MA20-1, 正值表示短期均线上方",
    "ma_cross": "金叉死叉信号, MA5>MA20时为1, 否则为0",
    "momentum": "5日价格动量, (当前价-5日前价格)/5日前价格",
    "price_momentum": "20日价格动量, (当前价-20日前价格)/20日前价格",
    "rsi": "RSI相对强弱指标/100, >0.7超买, <0.3超卖",
    "macd": "MACD/Signal比值, 正值表示多头趋势",
    "kdj": "(K-D)/100, K线上穿D线为金叉",
    "volatility": "20日价格波动率, 标准差/收盘价",
    "atr_ratio": "ATR/收盘价, 真实波幅占比",
    "volume_ratio": "量比, 当前成交量/20日平均成交量",
    "volume_price": "量价配合度, 成交量变化*价格变化",
    "adx": "ADX趋势强度/100, >0.25表示有趋势",
    "cci": "CCI商品通道指标/200, >1超买, <-1超卖",
    "obv": "OBV能量潮/MA(OBV,20)-1, 资金流向指标",
    "money_flow": "资金流向强度, 典型价格变化*成交量强度",
    "bb_ratio": "布林带位置, (close-lower)/(upper-lower), >0.8超买, <0.2超卖",
    "roc": "变动率ROC(10), (close-close_10)/close_10, 正值上涨趋势",
    "williams_r": "威廉指标%R, -100~0, >-20超买, <-80超卖",
}


class HtmlReportGenerator:
    """HTML 报告生成器

    使用 Jinja2 模板生成包含图表和分析的 HTML 报告。
    """

    def __init__(self, template_dir: str = "templates/reports"):
        """初始化报告生成器

        Args:
            template_dir: 模板目录
        """
        self.template_dir = Path(template_dir)

        if JINJA2_AVAILABLE:
            self.env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=True
            )
        else:
            self.env = None
            logger.warning("Jinja2 不可用，请运行: uv pip install jinja2")

    def generate_optimization_report(
        self,
        symbol: str,
        generation: int,
        population: list[Individual],
        valid_strategies: list[Individual],
        evolution_history: list[dict],
        output_path: str,
        metadata: dict | None = None
    ) -> str:
        """生成优化报告

        Args:
            symbol: 股票代码
            generation: 当前代数
            population: 当前种群
            valid_strategies: 有效策略列表
            evolution_history: 演化历史
            output_path: 输出文件路径
            metadata: 额外元数据

        Returns:
            输出文件路径
        """
        if not JINJA2_AVAILABLE:
            logger.error("Jinja2 不可用，无法生成 HTML 报告")
            return ""

        # 准备模板数据
        template_data = self._prepare_template_data(
            symbol, generation, population, valid_strategies, evolution_history, metadata
        )

        # 加载模板
        template = self.env.get_template("optimization_report.html")

        # 渲染 HTML
        html_content = template.render(**template_data)

        # 保存文件
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML 报告已生成: {output_file}")

        return str(output_file)

    def _prepare_template_data(
        self,
        symbol: str,
        generation: int,
        population: list[Individual],
        valid_strategies: list[Individual],
        evolution_history: list[dict],
        metadata: dict | None
    ) -> dict[str, Any]:
        """准备模板数据

        Args:
            symbol: 股票代码
            generation: 当前代数
            population: 当前种群
            valid_strategies: 有效策略列表
            evolution_history: 演化历史
            metadata: 额外元数据

        Returns:
            模板数据字典
        """
        # 基本统计
        fitness_values = [ind.fitness for ind in population]
        best_fitness = max(fitness_values) if fitness_values else 0
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0

        # 最优策略
        best_individual = max(population, key=lambda x: x.fitness) if population else None

        # 演化历史数据 - 确保数据正确提取
        generations = []
        best_fitness_history = []
        avg_fitness_history = []

        for i, h in enumerate(evolution_history):
            generations.append(h.get("generation", i + 1))
            best_fitness_history.append(h.get("best_fitness", 0))
            avg_fitness_history.append(h.get("avg_fitness", 0))

        # 如果没有历史数据，添加一个默认值以避免图表为空
        if not generations:
            generations = [1]
            best_fitness_history = [best_fitness]
            avg_fitness_history = [avg_fitness]

        # 因子权重分析
        factor_analysis = self._analyze_factor_weights(population)

        # 有效策略详情 - 包含完整的回测信息
        valid_strategies_details = []
        for ind in valid_strategies[:10]:  # 最多显示 10 个
            strategy_detail = {
                "id": id(ind),
                "fitness": ind.fitness,
                "factors": ind.factor_weights,
                "signal_threshold": ind.signal_threshold,
                "exit_threshold": ind.exit_threshold,
                "backtest_results": ind.backtest_results,
            }

            # 提取每个周期的交易详情
            trades_by_period = {}
            period_info = {}
            if ind.backtest_results:
                for period, result in ind.backtest_results.items():
                    if isinstance(result, dict):
                        trades_by_period[period] = result.get("trades", [])
                        period_info[period] = result.get("period_info", {})

            strategy_detail["trades_by_period"] = trades_by_period
            strategy_detail["period_info"] = period_info
            valid_strategies_details.append(strategy_detail)

        # 为最优策略准备详细信息
        best_individual_detail = None
        if best_individual:
            best_individual_detail = {
                "factor_weights": best_individual.factor_weights,
                "signal_threshold": best_individual.signal_threshold,
                "exit_threshold": best_individual.exit_threshold,
                "atr_period": best_individual.atr_period,
                "stop_loss_atr": getattr(best_individual, 'stop_loss_atr', 2.0),
                "fitness": best_individual.fitness,
                "backtest_results": best_individual.backtest_results,
            }

            # 提取交易详情和周期信息
            trades_by_period = {}
            period_info = {}
            if best_individual.backtest_results:
                for period, result in best_individual.backtest_results.items():
                    if isinstance(result, dict):
                        trades_by_period[period] = result.get("trades", [])
                        period_info[period] = result.get("period_info", {})

            best_individual_detail["trades_by_period"] = trades_by_period
            best_individual_detail["period_info"] = period_info

        # 准备基准对比数据
        benchmark_comparison = {}
        if best_individual and best_individual.backtest_results:
            for period, result in best_individual.backtest_results.items():
                if isinstance(result, dict):
                    strategy_series = result.get("strategy_series", {})
                    benchmark_series = result.get("benchmark_series", {})

                    # 使用 len() 判断是否为空，避免 pandas Series 布尔判断问题
                    if len(strategy_series) > 0 and len(benchmark_series) > 0:
                        # 转换为图表数据格式
                        dates = sorted(set(strategy_series.keys()) & set(benchmark_series.keys()))
                        if len(dates) > 0:
                            # 处理 benchmark_series 可能是 dict 或 numpy array 的情况
                            if isinstance(benchmark_series, dict):
                                benchmark_values_list = list(benchmark_series.values())
                            else:
                                # numpy array 或 pandas Series
                                benchmark_values_list = list(benchmark_series)

                            benchmark_comparison[period] = {
                                "dates": [str(d)[:10] for d in dates],  # 截取日期部分
                                "strategy_values": [strategy_series.get(d, 0) for d in dates],
                                "benchmark_values": [benchmark_series.get(d, 0) if isinstance(benchmark_series, dict) else benchmark_series[list(benchmark_series).index(d)] if d in list(benchmark_series) else 0 for d in dates],
                                "strategy_return": result.get("total_return", 0),
                                "benchmark_return": (benchmark_values_list[-1] / benchmark_values_list[0] - 1) if len(benchmark_values_list) >= 2 else 0,
                            }

        return {
            "symbol": symbol,
            "generation": generation,
            "valid_count": len(valid_strategies),
            "best_fitness": best_fitness,
            "avg_fitness": avg_fitness,
            "population_size": len(population),
            "best_individual": best_individual_detail,
            "valid_strategies": valid_strategies_details,
            "evolution_history": evolution_history,
            "generations": generations,
            "best_fitness_history": best_fitness_history,
            "avg_fitness_history": avg_fitness_history,
            "factor_analysis": factor_analysis,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata or {},
            "FACTOR_DESCRIPTIONS": FACTOR_DESCRIPTIONS,
            "benchmark_comparison": benchmark_comparison,
        }

    def _analyze_factor_weights(self, population: list[Individual]) -> dict:
        """分析因子权重分布

        Args:
            population: 种群

        Returns:
            因子分析结果
        """
        if not population:
            return {}

        # 收集所有因子
        all_factors = set()
        for ind in population:
            all_factors.update(ind.factor_weights.keys())

        # 计算每个因子的统计信息
        factor_stats = {}
        for factor in all_factors:
            weights = [
                ind.factor_weights.get(factor, 0)
                for ind in population
                if factor in ind.factor_weights
            ]

            if weights:
                factor_stats[factor] = {
                    "mean": float(np.mean(weights)),
                    "std": float(np.std(weights)),
                    "min": float(np.min(weights)),
                    "max": float(np.max(weights)),
                    "usage_rate": len(weights) / len(population),
                }

        # 按使用率排序
        sorted_factors = dict(
            sorted(factor_stats.items(), key=lambda x: x[1]["usage_rate"], reverse=True)
        )

        return sorted_factors

    def generate_simple_report(
        self,
        title: str,
        content: str,
        output_path: str
    ) -> str:
        """生成简单 HTML 报告

        Args:
            title: 报告标题
            content: 报告内容（HTML）
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-6">{title}</h1>
        <div class="bg-white p-6 rounded shadow">
            {content}
        </div>
        <p class="text-gray-500 text-sm mt-4">
            生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </div>
</body>
</html>"""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"简单报告已生成: {output_file}")

        return str(output_file)


def generate_summary_html(
    symbol: str,
    results: dict,
    output_path: str
) -> str:
    """生成摘要报告 HTML

    Args:
        symbol: 股票代码
        results: 结果字典
        output_path: 输出文件路径

    Returns:
        输出文件路径
    """
    generator = HtmlReportGenerator()

    content = f"""
    <h2 class="text-xl font-semibold mb-4">优化结果摘要</h2>

    <div class="grid grid-cols-2 gap-4 mb-6">
        <div class="bg-blue-50 p-4 rounded">
            <h3 class="font-semibold">股票代码</h3>
            <p class="text-2xl">{symbol}</p>
        </div>
        <div class="bg-green-50 p-4 rounded">
            <h3 class="font-semibold">最优适应度</h3>
            <p class="text-2xl">{results.get('best_fitness', 0):.4f}</p>
        </div>
    </div>

    <h3 class="text-lg font-semibold mb-2">回测结果</h3>
    <table class="min-w-full border-collapse border">
        <thead>
            <tr class="bg-gray-100">
                <th class="border p-2">周期</th>
                <th class="border p-2">收益率</th>
                <th class="border p-2">最大回撤</th>
                <th class="border p-2">夏普比率</th>
            </tr>
        </thead>
        <tbody>
    """

    for period, result in results.get("backtest_results", {}).items():
        content += f"""
            <tr>
                <td class="border p-2">{period}</td>
                <td class="border p-2">{result.get('total_return', 0):.2%}</td>
                <td class="border p-2">{result.get('max_drawdown', 0):.2%}</td>
                <td class="border p-2">{result.get('sharpe_ratio', 0):.2f}</td>
            </tr>
        """

    content += """
        </tbody>
    </table>
    """

    return generator.generate_simple_report(
        f"{symbol} 优化结果",
        content,
        output_path
    )
