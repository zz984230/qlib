# Qlib AI Strategy - 设计文档

## 概述

基于微软 qlib 构建的 AI 驱动量化投资策略系统，专注于 A 股市场，通过 Claude Code 实现策略的自动化挖掘与持续优化。

## 核心需求

| 需求 | 描述 |
|------|------|
| 底座 | 微软 qlib，uv 管理 |
| 数据源 | akshare，仅 A 股市场 |
| 交易模式 | 模拟盘交易 |
| AI 自动化 | 完全自主模式 |
| 策略类型 | AI 自动探索 + 网络策略发现 |
| 报告输出 | PDF 格式详细分析报告 |

## 技术方案

### 方案选择：轻量级 Agent 架构

Claude Code 作为主控制器，通过脚本和配置驱动整个流程。

```
Claude Code (主控制器)
    │
    ├── 调用 scripts/ 执行任务
    ├── 读写 configs/ 修改配置
    ├── 分析 reports/ 获取反馈
    └── 修改 src/strategy/ 优化策略
```

## 项目结构

```
qlib-ai-strategy/
├── pyproject.toml              # uv 项目配置
├── .python-version             # Python 版本锁定
│
├── configs/
│   ├── qlib.yaml              # qlib 配置
│   ├── data.yaml              # 数据源配置
│   └── strategy.yaml          # 策略参数配置
│
├── data/
│   ├── raw/                   # akshare 原始数据缓存
│   ├── qlib/                  # qlib 格式数据
│   └── results/               # 回测结果存储
│
├── src/
│   ├── data/
│   │   ├── akshare_loader.py  # akshare 数据加载
│   │   └── qlib_converter.py  # 转换为 qlib 格式
│   │
│   ├── strategy/
│   │   ├── base.py            # 策略基类
│   │   └── factors/           # 因子库
│   │
│   ├── backtest/
│   │   └── runner.py          # 回测执行器
│   │
│   ├── analysis/
│   │   ├── metrics.py         # 指标计算
│   │   └── report.py          # 报告生成
│   │
│   └── agent/
│       ├── prompts.py         # AI 提示模板
│       └── tools.py           # Claude Code 调用工具
│
├── reports/                    # PDF 报告输出
│
├── scripts/
│   ├── update_data.py         # 数据更新脚本
│   ├── run_backtest.py        # 运行回测
│   └── generate_report.py     # 生成报告
│
└── docs/
    └── plans/                  # 设计文档
```

## 数据流程

```
akshare → 数据转换器 → qlib存储 → 回测引擎 → 结果分析 → PDF报告
            │                                      │
            └──────── AI 优化决策循环 ◀────────────┘
```

### 数据范围配置

```yaml
# configs/data.yaml
market: "cn"
universe: "csi300"          # 沪深300 (可扩展)
frequency: "day"
start_date: "2020-01-01"
fields: [open, high, low, close, volume, amount]
```

## AI 交互机制

### 工作循环

1. **数据更新**: 从 akshare 获取最新数据
2. **策略回测**: 使用 qlib 执行回测
3. **结果分析**: 计算各项指标
4. **报告生成**: 输出 PDF 报告
5. **AI 优化**: Claude Code 分析并改进策略
   - 调整因子
   - 搜索网络策略
   - 修改配置参数

### 成本控制

```python
TRIGGER_CONDITIONS = {
    "max_drawdown_exceeds": 0.15,
    "sharpe_below": 1.0,
    "days_since_last_optimization": 7,
    "manual_trigger": True,
}
```

## 报告结构

PDF 报告包含以下章节：

1. **执行摘要** - 关键指标、AI 评估结论
2. **回测详情** - 收益曲线、回撤分析、月度收益
3. **风险分析** - 最大回撤、波动率、持仓分布
4. **交易分析** - 换手率、胜率、持仓周期
5. **因子分析** - 因子贡献度、相关性、IC 值
6. **AI 优化建议** - 问题诊断、改进方向、行动项

## 技术栈

| 组件 | 选择 | 说明 |
|------|------|------|
| 包管理 | uv | 现代化 Python 包管理 |
| 量化框架 | pyqlib | 微软开源量化平台 |
| 数据源 | akshare | A 股免费数据 |
| 模型 | LightGBM | qlib 内置支持 |
| 图表 | matplotlib + seaborn | 可视化 |
| PDF | reportlab | 报告生成 |
| 模板 | jinja2 | 报告模板 |

### 核心依赖

```toml
[project]
requires-python = ">=3.10"

dependencies = [
    "pyqlib",
    "akshare",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "lightgbm",
    "jinja2",
    "reportlab",
    "pyyaml",
]
```

## 实施路线图

### Phase 1: 基础框架 (MVP)
- 项目初始化 + uv 配置
- akshare 数据采集
- qlib 数据格式转换
- 简单回测跑通

**里程碑 M1**: 能用 akshare 数据跑通 qlib 回测

### Phase 2: 策略系统
- 策略基类设计
- 基础因子库 (5-10个常用因子)
- 回测执行器完善
- 结果存储机制

**里程碑 M2**: 至少实现一个完整策略并回测

### Phase 3: 分析报告
- 指标计算模块
- 可视化图表
- PDF 报告生成
- 报告模板设计

**里程碑 M3**: 能生成包含图表的 PDF 报告

### Phase 4: AI 集成
- 优化决策循环
- 策略改进流程
- 网络策略搜索
- 自动化工作流

**里程碑 M4**: 能基于报告自动提出改进建议

## 关键接口设计

```python
# src/strategy/base.py
class BaseStrategy:
    def get_factors(self) -> list[str]:
        """返回因子表达式列表"""
        raise NotImplementedError

    def get_model_config(self) -> dict:
        """返回模型配置"""
        return {"class": "LGBModel"}

    def generate_signals(self, data) -> np.ndarray:
        """生成交易信号"""
        raise NotImplementedError
```

```python
# src/backtest/runner.py
def run_backtest(
    strategy: BaseStrategy,
    start_date: str,
    end_date: str,
    config: dict | None = None,
) -> BacktestResult:
    """执行回测并返回结果"""
    ...
```

---

*文档创建日期: 2026-02-21*
