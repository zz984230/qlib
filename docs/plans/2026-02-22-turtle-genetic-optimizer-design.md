# 海龟遗传算法优化系统 - 设计文档

## 概述

基于微软 Qlib 构建的海龟交易法则优化系统，通过遗传算法自动探索量化因子组合，在 3% 严格回撤约束下最大化收益。

## 核心目标

| 目标 | 说明 |
|------|------|
| 回撤控制 | 最大回撤 <= 3% |
| 信号生成 | 通过量化因子组合探索入场/出场信号 |
| 仓位管理 | 采用海龟交易法则的 ATR 动态仓位 |
| 加仓规则 | 海龟金字塔加仓 |
| 止损策略 | 海龟双重止损（固定止损 + 移动止损） |
| 探索方式 | 遗传算法演化因子组合 |
| 交互模式 | 每找到有效组合即暂停，等待用户确认 |
| 报告输出 | 完整版 HTML（含演化历史、探索日志、AI 分析） |
| 策略存储 | 文件存储到 `strategies/pool/` |
| 回测周期 | 默认三种：近1年、近3月、近1月 |

## 架构设计

### 目录结构

```
qlib/
├── scripts/
│   ├── low_drawdown_report.py      # 保留：基线报告
│   └── turtle_genetic_optimizer.py # 新增：主优化器入口
│
├── src/
│   ├── strategy/
│   │   ├── low_drawdown.py          # 新增：从 low_drawdown_report.py 提取
│   │   ├── turtle_signals.py        # 新增：因子信号生成器
│   │   └── turtle_position.py       # 新增：海龟仓位/加仓/止损
│   │
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── genetic_engine.py        # 遗传算法引擎
│   │   ├── strategy_pool.py         # 策略池管理（JSON 文件存储）
│   │   └── fitness.py               # 适应度评估
│   │
│   └── report/
│       ├── __init__.py
│       └── html_generator.py        # HTML 报告生成器
│
├── strategies/
│   └── pool/                        # 有效策略存储目录
│       ├── strategy_001.json
│       ├── strategy_002.json
│       └── ...
│
├── reports/
│   └── html/                        # HTML 报告输出目录
│       └── turtle_optimization_YYYYMMDD_HHMMSS.html
│
├── templates/
│   └── reports/
│       └── optimization_report.html # Jinja2 模板
│
└── docs/
    └── plans/
        └── 2026-02-22-turtle-genetic-optimizer-design.md
```

## 核心模块设计

### 1. TurtleSignalGenerator（因子信号生成器）

```python
class TurtleSignalGenerator:
    """海龟策略信号生成器 - 使用因子组合"""

    def __init__(self, factor_config: dict):
        """
        Args:
            factor_config: {
                "factors": ["ma", "rsi", "momentum", ...],
                "weights": [0.3, 0.3, 0.4, ...],  # 归一化权重
                "params": {"ma_short": 5, "ma_long": 20, ...}
            }
        """

    def generate_entry_signal(self, data: pd.DataFrame) -> float:
        """生成入场信号强度 (0-1)
        综合多个因子计算信号强度
        """

    def generate_exit_signal(self, data: pd.DataFrame) -> float:
        """生成出场信号强度 (0-1)"""
```

### 2. TurtlePositionManager（海龟仓位管理器）

```python
class TurtlePositionManager:
    """海龟交易法则仓位管理"""

    def __init__(self, account_size: float, atr_period: int = 20):
        self.account_size = account_size
        self.atr_period = atr_period

    def calculate_unit_size(self, price: float, atr: float) -> int:
        """计算一个单位（Unit）的股数
        海龟公式：1单位 = 账户1% / ATR
        """

    def get_pyramid_positions(self, entry_price: float, current_price: float,
                              atr: float, current_units: int) -> int:
        """获取金字塔加仓信号
        - 每次加仓间隔：0.5 ATR（海龟原版）或可配置
        - 最多加仓次数：4次（海龟原版）
        返回：建议加仓单位数
        """
```

### 3. TurtleRiskManager（海龟风控管理器）

```python
class TurtleRiskManager:
    """海龟交易法则风控"""

    def __init__(self, stop_loss_atr: float = 2.0, trailing_stop: bool = True):
        # stop_loss_atr: 止损距离（ATR倍数），海龟原版=2N
        self.stop_loss_atr = stop_loss_atr

    def calculate_stop_loss(self, entry_price: float, atr: float,
                           is_pyramid: bool = False) -> float:
        """计算止损价格
        - 首次入场：entry_price - 2*ATR
        - 加仓单：各自独立止损
        """

    def calculate_trailing_stop(self, highest_price: float, atr: float) -> float:
        """计算移动止损（可选）"""
```

### 4. Individual（个体/策略染色体）

```python
@dataclass
class Individual:
    """代表一个因子组合策略"""

    # 基因编码
    factor_weights: dict[str, float]      # 因子权重 {"ma": 0.3, "rsi": 0.4, ...}
    signal_threshold: float               # 入场阈值 (0-1)
    exit_threshold: float                 # 出场阈值 (0-1)

    # 海龟参数
    atr_period: int                       # ATR周期
    stop_loss_atr: float                  # 止损ATR倍数
    pyramid_interval_atr: float           # 加仓间隔ATR倍数
    max_pyramid_units: int                # 最大加仓次数

    # 评估结果
    fitness: float = 0.0                  # 适应度得分
    backtest_results: dict = None         # 回测结果 {period: result}

    def to_genes(self) -> np.ndarray:
        """转换为基因数组（用于遗传操作）"""

    @classmethod
    def from_genes(cls, genes: np.ndarray) -> "Individual":
        """从基因数组创建个体"""
```

### 5. GeneticEngine（遗传算法引擎）

```python
class GeneticEngine:
    """遗传算法优化引擎"""

    def __init__(self, config: GeneticConfig):
        """
        GeneticConfig:
            - population_size: 种群大小（如 50）
            - mutation_rate: 变异率（如 0.1）
            - crossover_rate: 交叉率（如 0.7）
            - elite_size: 精英保留数量（如 5）
            - max_drawdown_limit: 回撤限制（3%）
        """

    def evolve(self, generation: int) -> Population:
        """执行一代演化
        1. 适应度评估
        2. 选择
        3. 交叉
        4. 变异
        """

    def crossover(self, parent1: Individual, parent2: Individual) -> tuple:
        """单点交叉"""

    def mutate(self, individual: Individual) -> Individual:
        """高斯变异"""

    def select(self, population: list[Individual]) -> list[Individual]:
        """轮盘赌选择 + 精英保留"""

    def should_stop(self, generation: int, best_fitness: float) -> bool:
        """判断是否停止演化"""
```

### 6. FitnessEvaluator（适应度评估器）

```python
class FitnessEvaluator:
    """适应度评估器"""

    def __init__(self, backtest_runner, max_drawdown_limit: float = 0.03):

    def evaluate(self, individual: Individual, symbol: str,
                 periods: list[str]) -> float:
        """评估个体适应度
        回测三个周期（1y/3m/1m），计算综合适应度

        适应度函数：
            - 基础分：加权收益（1y权重 60%，3m权重 30%，1m权重 10%）
            - 回撤惩罚：超过3%大幅扣分
            - 夏普比率加分
            - 稳定性加分：三个周期表现一致性
        """

    def is_valid(self, individual: Individual) -> bool:
        """判断是否为有效策略
        条件：三个周期回撤都 <=3% 且至少两个周期有正收益
        """
```

### 7. StrategyPool（策略池管理器）

```python
class StrategyPool:
    """有效策略池管理器"""

    def __init__(self, pool_dir: str = "strategies/pool"):

    def add(self, individual: Individual, symbol: str,
            metadata: dict) -> str:
        """添加有效策略到池中
        返回策略ID

        文件格式：
        {
            "id": "strategy_001",
            "symbol": "601138",
            "timestamp": "2026-02-22T10:30:00",
            "factor_weights": {...},
            "turtle_params": {...},
            "backtest_results": {
                "1y": {...},
                "3m": {...},
                "1m": {...}
            },
            "fitness": 0.85,
            "metadata": {
                "generation": 5,
                "parent_ids": ["strategy_000", ...]
            }
        }
        """

    def get_best(self, symbol: str, top_n: int = 10) -> list[dict]:
        """获取指定股票的最优N个策略"""

    def exists(self, individual: Individual) -> bool:
        """检查策略是否已存在（避免重复）"""
```

### 8. HtmlReportGenerator（HTML报告生成器）

```python
class HtmlReportGenerator:
    """HTML报告生成器"""

    def __init__(self, template_dir: str = "templates/reports"):

    def generate_optimization_report(
        self,
        symbol: str,
        generation: int,
        population: list[Individual],
        valid_strategies: list[Individual],
        evolution_history: list[dict],
        output_path: str
    ) -> str:
        """生成优化报告

        报告包含：
        1. 概览：当前代数、有效策略数
        2. 最优策略详情：参数、回测结果、交易记录
        3. 遗传算法演化历史：各代适应度曲线
        4. 因子权重分析：热力图
        5. 三周期对比：1y/3m/1m 绩效表
        6. 交易明细：完整交割单
        7. 探索日志：每代发现的改进点
        8. AI分析建议：基于当前结果的建议
        """
```

### 9. TurtleGeneticOptimizer（主优化器）

```python
class TurtleGeneticOptimizer:
    """海龟遗传算法主优化器"""

    def __init__(self, symbol: str, initial_cash: float = 50000):
        self.symbol = symbol
        self.cash = initial_cash

        # 初始化各模块
        self.genetic_engine = GeneticEngine(...)
        self.fitness_evaluator = FitnessEvaluator(...)
        self.strategy_pool = StrategyPool(...)
        self.report_generator = HtmlReportGenerator(...)
        self.evolution_history = []

    def run(self) -> dict:
        """执行优化主流程
        返回：最终优化结果摘要
        """

    def _handle_valid_strategies(
        self,
        valid_strategies: list[Individual],
        generation: int,
        population: list[Individual]
    ) -> None:
        """处理有效策略 - 交互暂停点
        1. 保存到策略池
        2. 生成临时报告
        3. 显示摘要并等待用户输入
        """

    def _show_summary_and_wait(
        self,
        strategies: list[Individual],
        report_path: str
    ) -> None:
        """显示策略摘要并等待用户确认
        交互命令：
        - [c]ontinue: 继续优化
        - [s]top: 停止并生成最终报告
        - [a]djust: 调整参数后继续
        """
```

## 回测周期配置

```python
PERIODS = {
    "1y": {"name": "近1年", "days": 365, "weight": 0.6},
    "3m": {"name": "近3月", "days": 90, "weight": 0.3},
    "1m": {"name": "近1月", "days": 30, "weight": 0.1},
}
```

## 遗传算法参数

```python
GENETIC_CONFIG = {
    "population_size": 50,        # 种群大小
    "max_generations": 100,       # 最大演化代数
    "mutation_rate": 0.1,         # 变异率
    "crossover_rate": 0.7,        # 交叉率
    "elite_size": 5,              # 精英保留数量
    "tournament_size": 3,         # 锦标赛选择规模
    "no_improvement_limit": 20,   # 连续N代无改进则停止
}
```

## 适应度函数

```python
def calculate_fitness(individual: Individual) -> float:
    """综合适应度计算"""

    # 各周期收益加权
    returns = {
        "1y": individual.backtest_results["1y"]["total_return"],
        "3m": individual.backtest_results["3m"]["total_return"],
        "1m": individual.backtest_results["1m"]["total_return"],
    }
    weighted_return = (
        returns["1y"] * 0.6 +
        returns["3m"] * 0.3 +
        returns["1m"] * 0.1
    )

    # 回撤惩罚
    max_dd = max(
        individual.backtest_results["1y"]["max_drawdown"],
        individual.backtest_results["3m"]["max_drawdown"],
        individual.backtest_results["1m"]["max_drawdown"],
    )
    drawdown_penalty = 0
    if max_dd > 0.03:
        drawdown_penalty = -(max_dd - 0.03) * 10  # 每超1%扣0.1

    # 夏普比率加分
    sharpe = individual.backtest_results["1y"]["sharpe_ratio"]
    sharpe_bonus = sharpe * 0.1

    # 稳定性加分
    returns_std = np.std(list(returns.values()))
    stability_bonus = -returns_std * 5  # 收益波动越小加分越多

    return weighted_return + drawdown_penalty + sharpe_bonus + stability_bonus
```

## HTML 报告结构

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>海龟遗传算法优化报告</title>
    <!-- Tailwind CSS -->
    <!-- Chart.js -->
    <style>
        /* 自定义样式 */
    </style>
</head>
<body class="bg-gray-50">
    <div class="container mx-auto px-4 py-8">
        <!-- 标题区 -->
        <header class="mb-8">
            <h1>工业富联 (601138) - 海龟遗传算法优化报告</h1>
            <p class="text-gray-600">生成时间: 2026-02-22 10:30:00</p>
        </header>

        <!-- 概览卡片 -->
        <section class="grid grid-cols-4 gap-4 mb-8">
            <div class="bg-white p-4 rounded shadow">
                <h3>当前代数</h3>
                <p class="text-2xl font-bold">5</p>
            </div>
            <div class="bg-white p-4 rounded shadow">
                <h3>有效策略数</h3>
                <p class="text-2xl font-bold text-green-600">3</p>
            </div>
            <div class="bg-white p-4 rounded shadow">
                <h3>最优适应度</h3>
                <p class="text-2xl font-bold text-blue-600">0.85</p>
            </div>
            <div class="bg-white p-4 rounded shadow">
                <h3>平均适应度</h3>
                <p class="text-2xl font-bold">0.42</p>
            </div>
        </section>

        <!-- 最优策略详情 -->
        <section class="bg-white p-6 rounded shadow mb-8">
            <h2>最优策略详情</h2>
            <!-- 策略参数、回测结果 -->
        </section>

        <!-- 演化历史图表 -->
        <section class="bg-white p-6 rounded shadow mb-8">
            <h2>遗传算法演化历史</h2>
            <canvas id="evolutionChart"></canvas>
        </section>

        <!-- 因子权重热力图 -->
        <section class="bg-white p-6 rounded shadow mb-8">
            <h2>因子权重分析</h2>
            <canvas id="factorHeatmap"></canvas>
        </section>

        <!-- 三周期绩效对比表 -->
        <section class="bg-white p-6 rounded shadow mb-8">
            <h2>三周期绩效对比</h2>
            <table class="min-w-full">
                <!-- 对比表格 -->
            </table>
        </section>

        <!-- 交易明细 -->
        <section class="bg-white p-6 rounded shadow mb-8">
            <h2>交易明细（交割单）</h2>
            <table class="min-w-full">
                <!-- 交易记录 -->
            </table>
        </section>

        <!-- 探索日志 -->
        <section class="bg-white p-6 rounded shadow mb-8">
            <h2>探索日志</h2>
            <div class="timeline">
                <!-- 每代日志 -->
            </div>
        </section>

        <!-- AI建议 -->
        <section class="bg-white p-6 rounded shadow mb-8">
            <h2>AI分析建议</h2>
            <div class="suggestions">
                <!-- 建议内容 -->
            </div>
        </section>
    </div>

    <script>
        // Chart.js 图表渲染
        // 演化历史曲线
        // 因子权重热力图
        // 交互式数据筛选
    </script>
</body>
</html>
```

## 命令行接口

```bash
# 运行优化
python scripts/turtle_genetic_optimizer.py \
    --symbol 601138 \
    --name "工业富联" \
    --cash 50000 \
    --max-generations 100 \
    --population-size 50 \
    --max-drawdown 0.03
```

## 知识管理

### Skill 创建

创建专用 skill：`skills/quant-strategy-optimization.md`

记录内容：
- 海龟交易法则参数调优经验
- 遗传算法参数设置指南
- 回测过拟合防范方法
- A股市场特性与适应
- 常见错误与解决方案

### 通用 Skill 更新

更新以下 skills：
- `superpowers:systematic-debugging`: 添加回测调试技巧
- `superpowers:brainstorming`: 添加量化策略设计要点

## 风险控制

### 回测过拟合防范

```python
RISK_CONTROLS = {
    "out_of_sample_ratio": 0.2,      # 20% 样本外测试
    "min_trading_days": 30,          # 最少交易日数
    "max_trade_frequency": 0.5,      # 最大换手率
    "transaction_cost": 0.003,       # 交易成本 0.3%
}
```

### 实盘风险限制

```python
POSITION_LIMITS = {
    "max_single_position": 0.10,     # 单股最大 10%
    "max_total_position": 0.95,      # 最大总仓位 95%
    "max_pyramid_units": 4,          # 最大加仓次数
}

RISK_ALERTS = {
    "daily_loss_limit": 0.03,        # 单日亏损 3%
    "drawdown_limit": 0.03,          # 回撤 3%
}
```

## 实施路线图

### Phase 1: 基础框架
- 创建目录结构
- 实现 Individual 类和基因编码
- 实现基础的遗传算法引擎
- 实现简单的适应度评估

### Phase 2: 海龟模块
- 实现 TurtlePositionManager（ATR 仓位）
- 实现 TurtleRiskManager（双重止损）
- 实现金字塔加仓逻辑
- 实现 TurtleSignalGenerator（因子信号）

### Phase 3: 回测集成
- 集成现有的 BacktestRunner
- 实现三周期回测（1y/3m/1m）
- 实现适应度函数
- 实现有效策略判断

### Phase 4: 策略池与报告
- 实现 StrategyPool（文件存储）
- 实现 HTML 报告生成器
- 创建 Jinja2 模板
- 集成 Chart.js 图表

### Phase 5: 交互与优化
- 实现主优化器流程
- 实现用户交互暂停
- 实现命令行接口
- 性能优化与测试

## 成功标准

1. 能成功运行遗传算法并发现有效策略
2. 有效策略满足：回撤 <=3%，且至少两个周期有正收益
3. 每次发现有效策略能正确暂停并等待用户确认
4. HTML 报告包含所有设计的章节且正确渲染
5. 策略池正确存储且可查询

---

*文档创建日期: 2026-02-22*
*设计确认: 已通过用户确认*
