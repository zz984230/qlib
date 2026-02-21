"""
AI 提示模板

包含用于策略分析、优化建议和策略搜索的提示模板
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class PromptTemplate:
    """提示模板基类"""

    name: str
    description: str
    template: str

    def render(self, **kwargs) -> str:
        """渲染模板"""
        return self.template.format(**kwargs)


class StrategyAnalysisPrompt:
    """策略分析提示"""

    SYSTEM_PROMPT = """你是一个专业的量化投资策略分析师。
你的任务是分析回测结果，识别策略的优势和劣势，并提供专业的诊断意见。

分析框架：
1. 收益分析：评估绝对收益和相对基准的表现
2. 风险分析：评估最大回撤、波动率等风险指标
3. 风险调整收益：评估夏普比率、索提诺比率等
4. 交易分析：评估胜率、盈亏比、交易频率等
5. 稳定性分析：评估滚动指标的时间稳定性

请用专业的语言进行分析，指出问题和改进方向。"""

    TEMPLATE = """## 策略分析任务

### 策略信息
- 策略名称: {strategy_name}
- 回测区间: {start_date} 至 {end_date}
- 初始资金: {cash:,.0f}

### 核心指标
| 指标 | 数值 |
|------|------|
| 总收益率 | {total_return:.2%} |
| 年化收益率 | {annual_return:.2%} |
| 最大回撤 | {max_drawdown:.2%} |
| 夏普比率 | {sharpe_ratio:.2f} |
| 胜率 | {win_rate:.2%} |

### 详细指标
```json
{metrics_json}
```

### 分析要求
1. 诊断策略当前存在的主要问题
2. 分析导致表现不佳/优异的可能原因
3. 提出具体的改进方向

请进行分析："""

    @classmethod
    def create(cls, context: dict[str, Any]) -> str:
        """创建分析提示"""
        import json

        metrics = context.get("metrics", {})
        metrics_json = json.dumps(metrics, indent=2, ensure_ascii=False)

        return cls.TEMPLATE.format(
            strategy_name=context.get("strategy_name", "Unknown"),
            start_date=context.get("start_date", ""),
            end_date=context.get("end_date", ""),
            cash=context.get("cash", 1000000),
            total_return=context.get("total_return", 0),
            annual_return=context.get("annual_return", 0),
            max_drawdown=context.get("max_drawdown", 0),
            sharpe_ratio=context.get("sharpe_ratio", 0),
            win_rate=context.get("win_rate", 0),
            metrics_json=metrics_json,
        )


class OptimizationSuggestionPrompt:
    """优化建议提示"""

    SYSTEM_PROMPT = """你是一个量化策略优化专家。
基于策略分析结果，你需要提出具体的参数调整和策略改进建议。

优化原则：
1. 简单优先：优先尝试简单的参数调整
2. 风险控制：优先降低风险而非追求高收益
3. 过拟合防范：避免过度优化历史数据
4. 可解释性：建议应该有合理的经济逻辑支持

输出格式为 JSON，包含具体的参数调整建议。"""

    TEMPLATE = """## 策略优化任务

### 当前策略
- 策略类型: {strategy_type}
- 当前参数: {current_params}

### 问题诊断
{diagnosis}

### 历史优化记录
{optimization_history}

### 优化约束
- 最大因子数量: {max_factors}
- 最大回撤容忍: {max_drawdown:.0%}
- 最小夏普比率: {min_sharpe}

请提出具体的优化建议，包括：
1. 参数调整建议（具体数值）
2. 因子增删建议
3. 风险控制建议
4. 预期改进效果

以 JSON 格式输出：
```json
{{
  "parameter_adjustments": {{
    "param_name": {{"current": x, "suggested": y, "reason": "..."}}
  }},
  "factor_changes": {{
    "add": ["factor1"],
    "remove": ["factor2"],
    "reason": "..."
  }},
  "risk_controls": {{
    "stop_loss": 0.05,
    "position_limit": 0.1
  }},
  "expected_improvement": {{
    "sharpe_change": 0.5,
    "drawdown_reduction": 0.05
  }}
}}
```"""

    @classmethod
    def create(cls, context: dict[str, Any]) -> str:
        import json

        current_params = context.get("current_params", {})
        history = context.get("optimization_history", [])

        return cls.TEMPLATE.format(
            strategy_type=context.get("strategy_type", "Unknown"),
            current_params=json.dumps(current_params, indent=2),
            diagnosis=context.get("diagnosis", "未提供诊断信息"),
            optimization_history=json.dumps(history[-5:], indent=2) if history else "无历史记录",
            max_factors=context.get("max_factors", 10),
            max_drawdown=context.get("max_drawdown", 0.15),
            min_sharpe=context.get("min_sharpe", 1.0),
        )


class StrategySearchPrompt:
    """策略搜索提示"""

    SYSTEM_PROMPT = """你是一个量化策略研究员。
你的任务是从学术研究和行业实践中发现有价值的新策略思路。

评估标准：
1. 理论基础：策略是否有学术支持或经济逻辑
2. 实用性：策略是否适合 A 股市场
3. 复杂度：策略实现难度是否适中
4. 原创性：策略是否有独特价值

输出格式为结构化的策略描述。"""

    TEMPLATE = """## 策略搜索任务

### 搜索范围
- 策略类型: {strategy_types}
- 市场适应性: A股市场
- 数据依赖: {data_sources}

### 当前策略池
{existing_strategies}

### 性能基准
- 目标夏普比率: {target_sharpe}
- 最大回撤容忍: {max_drawdown:.0%}

### 搜索任务
请搜索并推荐 1-3 个新策略，每个策略包含：
1. 策略名称和类型
2. 核心逻辑描述
3. 关键因子/指标
4. 预期优势和风险
5. 实现复杂度评估

输出格式：
```json
{{
  "recommendations": [
    {{
      "name": "策略名称",
      "type": "momentum/value/etc",
      "description": "核心逻辑",
      "factors": ["因子1", "因子2"],
      "advantages": ["优势1"],
      "risks": ["风险1"],
      "complexity": "low/medium/high",
      "source": "来源参考"
    }}
  ]
}}
```"""

    @classmethod
    def create(cls, context: dict[str, Any]) -> str:
        import json

        existing = context.get("existing_strategies", [])
        strategy_types = context.get("strategy_types", ["momentum", "value", "quality", "technical"])

        return cls.TEMPLATE.format(
            strategy_types=", ".join(strategy_types),
            data_sources="日K线 (OHLCV)",
            existing_strategies=json.dumps(existing, indent=2) if existing else "无现有策略",
            target_sharpe=context.get("target_sharpe", 1.5),
            max_drawdown=context.get("max_drawdown", 0.15),
        )


class ReportSummaryPrompt:
    """报告摘要提示"""

    TEMPLATE = """## 投资策略报告摘要

### 报告信息
- 生成时间: {timestamp}
- 策略: {strategy_name}
- 分析周期: {period}

### 核心结论
{summary}

### 关键指标
{key_metrics}

### 主要发现
{findings}

### 建议行动
{actions}

---
*本报告由 AI 系统自动生成*"""

    @classmethod
    def create(cls, context: dict[str, Any]) -> str:
        return cls.TEMPLATE.format(
            timestamp=context.get("timestamp", ""),
            strategy_name=context.get("strategy_name", ""),
            period=context.get("period", ""),
            summary=context.get("summary", ""),
            key_metrics=context.get("key_metrics", ""),
            findings=context.get("findings", ""),
            actions=context.get("actions", ""),
        )


if __name__ == "__main__":
    # 测试提示模板
    context = {
        "strategy_name": "双均线策略",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "cash": 1000000,
        "total_return": 0.15,
        "annual_return": 0.15,
        "max_drawdown": 0.08,
        "sharpe_ratio": 1.2,
        "win_rate": 0.55,
        "metrics": {"volatility": 0.12, "sortino": 1.5},
    }

    prompt = StrategyAnalysisPrompt.create(context)
    print("策略分析提示:")
    print(prompt)
