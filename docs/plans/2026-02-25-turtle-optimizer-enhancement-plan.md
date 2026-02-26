# 海龟遗传算法优化器增强实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 添加策略与买入持有基准的收益比较折线图，并从因子、风控、适应度三方面优化以提高收益。

**Architecture:** 基于现有海龟遗传算法优化器，扩展回测结果数据流以包含基准净值序列，在 HTML 报告中使用 Chart.js 绘制对比折线图；同时扩展因子池、参数范围和适应度函数。

**Tech Stack:** Python, pandas, numpy, Jinja2, Chart.js, HTML/Tailwind CSS

---

## Task 1: 添加基准数据到多周期回测

**Files:**
- Modify: `src/backtest/multi_period.py:90-125`

**Step 1: 修改 multi_period.py 保存基准净值序列**

在 `MultiPeriodBacktester.run()` 方法中，将 `result.benchmark` 保存到 `backtest_results` 字典。

```python
# 在 line 113 之后添加
result.benchmark_series = benchmark.loc[actual_start_date:]  # 只保留回测期间的基准
```

同时在 `individual.backtest_results` 中添加基准数据：

```python
# 修改 src/optimizer/turtle_optimizer.py 的 _evaluate_individual 方法
# 在 line 230-241 的 backtest_results 字典中添加:
individual.backtest_results[period] = {
    "total_return": result.total_return,
    "max_drawdown": result.max_drawdown,
    "sharpe_ratio": result.sharpe_ratio,
    "annual_return": result.annual_return,
    "period_info": getattr(result, 'period_info', {}),
    "trades": getattr(result, 'trades_list', []),
    # 新增基准数据
    "benchmark_series": getattr(result, 'benchmark_series', None),
    "strategy_series": result.portfolio_value.to_dict() if hasattr(result.portfolio_value, 'to_dict') else {},
}
```

**Step 2: 验证数据流**

运行: `uv run python -c "from src.backtest.multi_period import MultiPeriodBacktester; print('OK')"`
预期: 输出 "OK"

**Step 3: Commit**

```bash
git add src/backtest/multi_period.py src/optimizer/turtle_optimizer.py
git commit -m "feat: add benchmark series to backtest results"
```

---

## Task 2: 更新 HTML 报告生成器提取基准数据

**Files:**
- Modify: `src/report/html_generator.py:117-233`

**Step 1: 在 _prepare_template_data 中添加基准数据处理**

在 `_prepare_template_data` 方法的返回字典中添加 `benchmark_comparison` 数据：

```python
# 在 line 232 的 return 语句之前添加:

# 准备基准对比数据
benchmark_comparison = {}
if best_individual and best_individual.get("backtest_results"):
    for period, result in best_individual["backtest_results"].items():
        if isinstance(result, dict):
            strategy_series = result.get("strategy_series", {})
            benchmark_series = result.get("benchmark_series", {})

            if strategy_series and benchmark_series:
                # 转换为图表数据格式
                dates = sorted(set(strategy_series.keys()) & set(benchmark_series.keys()))
                benchmark_comparison[period] = {
                    "dates": [str(d)[:10] for d in dates],  # 截取日期部分
                    "strategy_values": [strategy_series.get(d, 0) for d in dates],
                    "benchmark_values": [benchmark_series.get(d, 0) for d in dates],
                    "strategy_return": result.get("total_return", 0),
                    "benchmark_return": (list(benchmark_series.values())[-1] / list(benchmark_series.values())[0] - 1) if benchmark_series else 0,
                }
```

然后在返回字典中添加:

```python
return {
    # ... 现有字段 ...
    "benchmark_comparison": benchmark_comparison,
}
```

**Step 2: 验证代码语法**

运行: `uv run python -c "from src.report.html_generator import HtmlReportGenerator; print('OK')"`
预期: 输出 "OK"

**Step 3: Commit**

```bash
git add src/report/html_generator.py
git commit -m "feat: extract benchmark data for report comparison charts"
```

---

## Task 3: 更新 HTML 模板添加基准对比图表

**Files:**
- Modify: `templates/reports/optimization_report.html`

**Step 1: 在模板中添加基准对比图表区域**

在最优策略部分之后添加新的图表区域：

```html
<!-- 基准对比图表区域 -->
{% if benchmark_comparison %}
<section class="mb-8">
    <h2 class="text-2xl font-bold mb-4 text-cyan-400">策略与买入持有基准对比</h2>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {% for period, data in benchmark_comparison.items() %}
        <div class="bg-gray-800/50 rounded-lg p-4 border border-indigo-500/20">
            <h3 class="text-lg font-semibold mb-2 text-white">
                {% if period == '1y' %}近1年{% elif period == '3m' %}近3月{% elif period == '1m' %}近1月{% else %}{{ period }}{% endif %}
            </h3>
            <div class="text-sm text-gray-400 mb-3">
                策略收益: <span class="text-green-400">{{ (data.strategy_return * 100)|round(2) }}%</span> |
                基准收益: <span class="text-blue-400">{{ (data.benchmark_return * 100)|round(2) }}%</span>
            </div>
            <canvas id="benchmark-chart-{{ period }}" height="200"></canvas>
        </div>
        {% endfor %}
    </div>
</section>

<script>
// 绘制基准对比图表
{% for period, data in benchmark_comparison.items() %}
(function() {
    const ctx = document.getElementById('benchmark-chart-{{ period }}');
    if (!ctx) return;

    const labels = {{ data.dates | tojson }};
    const strategyValues = {{ data.strategy_values | tojson }};
    const benchmarkValues = {{ data.benchmark_values | tojson }};

    // 采样数据点（最多50个点）
    const maxPoints = 50;
    const step = Math.max(1, Math.floor(labels.length / maxPoints));
    const sampledLabels = labels.filter((_, i) => i % step === 0);
    const sampledStrategy = strategyValues.filter((_, i) => i % step === 0);
    const sampledBenchmark = benchmarkValues.filter((_, i) => i % step === 0);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: sampledLabels,
            datasets: [
                {
                    label: '策略净值',
                    data: sampledStrategy,
                    borderColor: '#22d3ee',
                    backgroundColor: 'rgba(34, 211, 238, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: '买入持有',
                    data: sampledBenchmark,
                    borderColor: '#f472b6',
                    backgroundColor: 'rgba(244, 114, 182, 0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                    borderDash: [5, 5],
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#64748b', maxTicksLimit: 5 },
                    grid: { color: 'rgba(99, 102, 241, 0.1)' }
                },
                y: {
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(99, 102, 241, 0.1)' }
                }
            }
        }
    });
})();
{% endfor %}
</script>
{% endif %}
```

**Step 2: 验证模板语法**

运行: `uv run python -c "from jinja2 import Environment, FileSystemLoader; env = Environment(loader=FileSystemLoader('templates/reports')); env.get_template('optimization_report.html'); print('Template OK')"`
预期: 输出 "Template OK"

**Step 3: Commit**

```bash
git add templates/reports/optimization_report.html
git commit -m "feat: add benchmark comparison charts to report template"
```

---

## Task 4: 添加新因子到因子池

**Files:**
- Modify: `src/optimizer/individual.py:195-225`
- Modify: `src/strategy/turtle_signals.py:294-433`

**Step 1: 在 individual.py 的 FACTOR_POOL 中添加新因子**

```python
# 在 FACTOR_POOL 字典中添加 (line 224 之后):
FACTOR_POOL = {
    # ... 现有因子 ...

    # 新增因子
    "bb_ratio": "布林带位置 (close - lower) / (upper - lower)",
    "roc": "变动率 ROC(10)",
    "williams_r": "威廉指标 %R",
}
```

**Step 2: 在 turtle_signals.py 中实现新因子计算**

在 `_calculate_factor_value` 方法中添加新因子的计算逻辑 (line 424 之前):

```python
elif factor_name == "bb_ratio":
    # 布林带位置: (close - lower) / (upper - lower)
    period = 20
    if len(close) < period:
        return 0.5
    ma = _sma(close, period)
    std = _std(close, period)
    if len(ma) > 0 and not np.isnan(ma[-1]) and std[-1] > 0:
        upper = ma[-1] + 2 * std[-1]
        lower = ma[-1] - 2 * std[-1]
        return float((close[-1] - lower) / (upper - lower + 1e-10))
    return 0.5

elif factor_name == "roc":
    # 变动率 ROC(10)
    period = 10
    if len(close) < period + 1:
        return 0.0
    return float((close[-1] - close[-period-1]) / (close[-period-1] + 1e-10))

elif factor_name == "williams_r":
    # 威廉指标 %R
    period = 14
    if len(close) < period:
        return -50.0
    highest = np.max(high[-period:])
    lowest = np.min(low[-period:])
    if highest - lowest < 1e-10:
        return -50.0
    return float(-100 * (highest - close[-1]) / (highest - lowest))
```

**Step 3: 添加新因子的归一化逻辑**

在 `_normalize_factor` 方法中添加 (line 470 之前):

```python
elif factor_name == "bb_ratio":
    # 布林带位置已在 [0, 1]，但可能有超出
    return np.clip(value, 0, 1)

elif factor_name == "roc":
    # ROC 变动率，使用 tanh 归一化
    return (np.tanh(value * 5) + 1) / 2

elif factor_name == "williams_r":
    # 威廉指标范围 [-100, 0]，归一化到 [0, 1]
    return np.clip((value + 100) / 100, 0, 1)
```

**Step 4: 验证新因子**

运行: `uv run python -c "from src.optimizer.individual import FACTOR_POOL; print('bb_ratio' in FACTOR_POOL, 'roc' in FACTOR_POOL, 'williams_r' in FACTOR_POOL)"`
预期: 输出 "True True True"

**Step 5: Commit**

```bash
git add src/optimizer/individual.py src/strategy/turtle_signals.py
git commit -m "feat: add bb_ratio, roc, williams_r factors"
```

---

## Task 5: 扩展风控参数范围

**Files:**
- Modify: `src/optimizer/individual.py:24-29`

**Step 1: 扩展 Individual 的参数范围**

在 `Individual` dataclass 中添加新参数:

```python
@dataclass
class Individual:
    # ... 现有字段 ...

    # ========== 海龟参数 (扩展) ==========
    atr_period: int = 20
    stop_loss_atr: float = 2.0           # 原值，范围扩展为 1.5-3.0
    pyramid_interval_atr: float = 0.5    # 原值，范围扩展为 0.3-0.7
    max_pyramid_units: int = 4
    trailing_stop_trigger: float = 1.0   # 新增：移动止损触发点 (ATR倍数)，范围 0.5-1.5
```

**Step 2: 更新 to_genes 和 from_genes 方法**

修改 `to_genes` 方法 (line 52-78):

```python
def to_genes(self, all_factor_names: list[str] | None = None) -> np.ndarray:
    # ... 现有代码 ...
    genes = np.array([
        *weight_genes,
        self.signal_threshold,
        self.exit_threshold,
        float(self.atr_period),
        self.stop_loss_atr,
        self.pyramid_interval_atr,
        self.trailing_stop_trigger,
    ])
    return genes
```

修改 `from_genes` 方法 (line 80-119):

```python
@classmethod
def from_genes(cls, genes: np.ndarray, factor_names: list[str], generation: int = 0, parent_ids: list[str] | None = None) -> "Individual":
    n_factors = len(factor_names)

    # ... 现有代码 ...

    stop_loss_atr = float(genes[n_factors + 3]) if n_factors + 3 < len(genes) else 2.0
    pyramid_interval_atr = float(genes[n_factors + 4]) if n_factors + 4 < len(genes) else 0.5
    trailing_stop_trigger = float(genes[n_factors + 5]) if n_factors + 5 < len(genes) else 1.0

    return cls(
        factor_weights=factor_weights,
        signal_threshold=signal_threshold,
        exit_threshold=exit_threshold,
        atr_period=max(5, min(50, atr_period)),
        stop_loss_atr=max(1.5, min(3.0, stop_loss_atr)),  # 限制范围
        pyramid_interval_atr=max(0.3, min(0.7, pyramid_interval_atr)),  # 限制范围
        trailing_stop_trigger=max(0.5, min(1.5, trailing_stop_trigger)),  # 限制范围
        generation=generation,
        parent_ids=parent_ids or [],
    )
```

**Step 3: 更新 to_dict 和 from_dict 方法**

在 `to_dict` 中添加新字段:

```python
def to_dict(self) -> dict[str, Any]:
    return {
        # ... 现有字段 ...
        "trailing_stop_trigger": self.trailing_stop_trigger,
    }
```

在 `from_dict` 中添加:

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "Individual":
    return cls(
        # ... 现有字段 ...
        trailing_stop_trigger=data.get("trailing_stop_trigger", 1.0),
    )
```

**Step 4: Commit**

```bash
git add src/optimizer/individual.py
git commit -m "feat: extend risk control parameter ranges"
```

---

## Task 6: 优化适应度函数

**Files:**
- Modify: `src/optimizer/fitness.py:114-188`

**Step 1: 添加超额收益加分项**

修改 `evaluate_backtest_based` 方法:

```python
def evaluate_backtest_based(self, individual: Individual, backtest_results: dict[str, "BacktestResult"]) -> float:
    # ... 现有的加权收益计算 ...

    # 新增：计算超额收益加分
    excess_return_bonus = 0.0
    for period, result in backtest_results.items():
        if isinstance(result, dict):
            strategy_ret = result.get("total_return", 0)
            benchmark_series = result.get("benchmark_series", {})
            if benchmark_series:
                values = list(benchmark_series.values())
                if len(values) >= 2:
                    benchmark_ret = values[-1] / values[0] - 1
                    excess = strategy_ret - benchmark_ret
                    # 只有跑赢基准才加分
                    excess_return_bonus += max(0, excess) * 5.0

    # 平滑回撤惩罚（线性惩罚替代突变惩罚）
    drawdown_penalty = 0.0
    for period, result in backtest_results.items():
        if isinstance(result, dict):
            dd = result.get("max_drawdown", 0)
        else:
            dd = result.max_drawdown

        # 线性惩罚：回撤每增加 1%，惩罚增加
        if dd > 0.01:  # 超过 1% 开始惩罚
            drawdown_penalty -= (dd - 0.01) * 5.0
        if dd > self.config.max_drawdown_limit:
            drawdown_penalty -= (dd - self.config.max_drawdown_limit) * 10.0  # 超限额外惩罚

    # ... 现有的夏普比率和稳定性计算 ...

    # 综合适应度
    fitness = (
        weighted_return
        + drawdown_penalty
        + sharpe_bonus
        + stability_bonus
        + excess_return_bonus  # 新增
    )

    return fitness
```

**Step 2: Commit**

```bash
git add src/optimizer/fitness.py
git commit -m "feat: add excess return bonus and smooth drawdown penalty"
```

---

## Task 7: 更新 FACTOR_DESCRIPTIONS 字典

**Files:**
- Modify: `src/report/html_generator.py:24-41`

**Step 1: 添加新因子描述**

```python
FACTOR_DESCRIPTIONS = {
    # ... 现有因子 ...
    "bb_ratio": "布林带位置, (close-lower)/(upper-lower), >0.8超买, <0.2超卖",
    "roc": "变动率ROC(10), (close-close_10)/close_10, 正值上涨趋势",
    "williams_r": "威廉指标%R, -100~0, >-20超买, <-80超卖",
}
```

**Step 2: Commit**

```bash
git add src/report/html_generator.py
git commit -m "docs: add descriptions for new factors"
```

---

## Task 8: 集成测试

**Step 1: 运行单元测试**

```bash
uv run pytest tests/ -v
```

预期: 所有测试通过

**Step 2: 运行优化器验证**

```bash
uv run python scripts/turtle_genetic_optimizer.py --symbol 601138 --name "工业富联" --max-generations 5 --population-size 10 --no-interactive
```

预期:
- 优化器正常运行
- 生成 HTML 报告
- 报告中包含基准对比图表

**Step 3: 检查报告**

打开生成的 HTML 报告，验证:
1. 三个周期的基准对比折线图正确显示
2. 策略线和基准线清晰区分
3. 新因子在因子分析中显示

**Step 4: Final Commit**

```bash
git add -A
git commit -m "feat: complete turtle optimizer enhancement with benchmark comparison"
```

---

## 实现进度

**更新时间**: 2026-02-26

### 已完成 (Task 1-7)

| 任务 | 状态 | 提交 |
|------|------|------|
| Task 1: 添加基准数据到多周期回测 | ✅ 完成 | edba5c5 |
| Task 2: 更新 HTML 报告生成器提取基准数据 | ✅ 完成 | cabd4c1 |
| Task 3: 更新 HTML 模板添加基准对比图表 | ✅ 完成 | f6f9d38 |
| Task 4: 添加新因子到因子池 | ✅ 完成 | 255f5d2 |
| Task 5: 扩展风控参数范围 | ✅ 完成 | 8f39e6b |
| Task 6: 优化适应度函数 | ✅ 完成 | - |
| Task 7: 更新 FACTOR_DESCRIPTIONS 字典 | ✅ 完成 | - |

### 已完成 (Task 8)

| 任务 | 状态 | 说明 |
|------|------|------|
| Task 8: 集成测试 | ✅ 完成 | 88 个单元测试通过，优化器集成测试通过 |

---

## 验收标准

1. [x] HTML 报告中正确显示三个周期的策略 vs 基准对比折线图
2. [x] 新增因子 (bb_ratio, roc, williams_r) 能被遗传算法正确探索
3. [x] 风控参数范围已扩展 (stop_loss_atr: 1.5-3.0, trailing_stop_trigger: 0.5-1.5)
4. [x] 适应度函数包含超额收益加分和平滑回撤惩罚
5. [x] 所有现有测试通过 (88/88)
6. [x] 优化器可正常运行并生成报告

## 额外优化

- [x] 添加 akshare 多数据源备用 (东方财富/新浪/网易)
