# 海龟遗传算法优化器增强设计

## 概述

本设计文档描述如何增强海龟遗传算法优化器，主要包括：
1. 添加策略与买入持有基准的收益比较折线图
2. 从因子、风控、适应度三方面优化以提高收益

## 第一部分：基准比较折线图

### 目标

在 HTML 报告中显示策略净值与买入持有基准的对比折线图，支持三个时间周期（1y/3m/1m）。

### 架构

```
TurtleBacktestRunner.run()
    -> 计算基准净值 (benchmark = initial_cash * close/close[0])
    -> BacktestResult.benchmark

MultiPeriodBacktester.run()
    -> 收集各周期 benchmark 数据
    -> BacktestResult.period_info + benchmark_series

HtmlReportGenerator._prepare_template_data()
    -> 提取 benchmark 数据
    -> 转换为 JSON 格式

optimization_report.html
    -> Chart.js 渲染策略 vs 基准折线图
```

### 修改文件

1. **src/backtest/multi_period.py**
   - 在返回结果中保存 `benchmark` Series
   - 添加基准相关的统计信息

2. **src/report/html_generator.py**
   - 在 `_prepare_template_data()` 中提取各周期基准数据
   - 转换为 JSON 格式的日期-净值对

3. **templates/reports/optimization_report.html**
   - 新增三个周期的对比折线图区域
   - 使用 Chart.js 绘制双线图（策略线 + 基准线）

### 数据流

```python
# 回测结果中新增字段
result.benchmark_series = benchmark  # pd.Series

# 报告模板数据
{
    "benchmark_comparison": {
        "1y": {
            "dates": ["2025-02-01", "2025-02-02", ...],
            "strategy_values": [50000, 50100, ...],
            "benchmark_values": [50000, 49900, ...],
        },
        "3m": {...},
        "1m": {...}
    }
}
```

## 第二部分：遗传算法优化

### 2.1 因子优化

**新增因子** (src/strategy/turtle_signals.py):

| 因子名 | 描述 | 计算公式 |
|--------|------|----------|
| `bb_ratio` | 布林带位置 | (close - lower) / (upper - lower) |
| `roc` | 变动率 | (close - close_n) / close_n * 100 |
| `williams_r` | 威廉指标 | (high_n - close) / (high_n - low_n) * -100 |

**信号生成优化**:
- 使用因子加权得分：`score = sum(factor * weight)`
- 入场条件：`score > signal_threshold`
- 出场条件：`score < exit_threshold`

### 2.2 风控参数优化

**参数范围扩展** (src/optimizer/individual.py):

| 参数 | 原值 | 新范围 |
|------|------|--------|
| `stop_loss_atr` | 固定 2.0 | 1.5 - 3.0 |
| `trailing_stop_trigger` | 无 | 新增 0.5 - 1.5 ATR |
| `pyramid_interval` | 固定 0.5 | 0.3 - 0.7 ATR |

### 2.3 适应度函数优化

**改进适应度计算** (src/optimizer/fitness.py):

```python
fitness = (
    weighted_return                                    # 加权收益
    + drawdown_penalty                                 # 回撤惩罚
    + sharpe_bonus                                     # 夏普加分
    + stability_bonus                                  # 稳定性加分
    + excess_return_bonus    # 新增：超额收益加分
)

# 超额收益加分
excess_return = strategy_return - benchmark_return
excess_return_bonus = max(0, excess_return) * 5.0
```

**平滑回撤惩罚**:
- 原逻辑：回撤 > 3% 时大幅惩罚
- 新逻辑：线性惩罚，回撤每增加 1% 惩罚增加

## 实现计划

### 阶段1：基准比较功能
1. 修改 multi_period.py 保存基准数据
2. 修改 html_generator.py 提取基准数据
3. 修改模板添加对比图表

### 阶段2：因子优化
1. 在 turtle_signals.py 添加新因子计算
2. 更新 individual.py 的因子初始化范围

### 阶段3：风控和适应度优化
1. 扩展参数搜索范围
2. 改进适应度函数

## 验收标准

1. 报告中正确显示三个周期的策略 vs 基准对比折线图
2. 新增因子能被遗传算法正确探索
3. 优化后的策略收益不低于原策略
4. 最大回撤仍控制在 3% 以内
