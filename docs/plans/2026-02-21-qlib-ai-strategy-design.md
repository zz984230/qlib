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

## 风险管理

### 回测过拟合防范

```python
# configs/risk.yaml
risk_controls:
  # 样本外测试要求
  out_of_sample_ratio: 0.2       # 20% 数据保留样本外
  walk_forward_windows: 5         # 滚动窗口数

  # 参数复杂度限制
  max_factor_count: 10           # 最大因子数量
  max_model_depth: 5             # 模型最大深度

  # 交易成本模拟
  commission: 0.0003             # 万三佣金
  slippage: 0.0001               # 滑点
  stamp_duty: 0.001              # 印花税（卖出）
```

### 实盘风险限制

```yaml
position_limits:
  max_single_position: 0.10      # 单股最大仓位 10%
  max_sector_position: 0.30      # 单行业最大仓位 30%
  max_total_position: 0.95       # 最大总仓位

risk_alerts:
  daily_loss_limit: 0.03         # 单日亏损 3% 止损
  drawdown_limit: 0.15           # 回撤 15% 暂停交易
```

## 监控与日志

### 日志结构

```
logs/
├── data/              # 数据更新日志
├── backtest/          # 回测执行日志
├── analysis/          # 分析日志
└── agent/             # AI 决策日志
```

### 监控指标

```python
# 监控关键指标并记录
METRICS_TO_TRACK = {
    "数据更新": ["last_update_time", "data_quality_score"],
    "回测执行": ["execution_time", "memory_usage", "success_rate"],
    "策略表现": ["sharpe_ratio", "max_drawdown", "annual_return"],
    "AI 优化": ["optimization_count", "improvement_rate"],
}
```

## Phase 1 详细实施步骤

### 1.1 项目初始化

```bash
# 创建项目
mkdir qlib-ai-strategy && cd qlib-ai-strategy

# 初始化 uv
uv init --python 3.10
uv pip install pyqlib akshare

# 创建目录结构
mkdir -p {configs,data/{raw,qlib,results},src/{data,strategy/factors,backtest,analysis,agent},reports,scripts,docs/plans}
```

### 1.2 数据采集模块

**文件**: `src/data/akshare_loader.py`

```python
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

class AkshareLoader:
    """akshare 数据加载器"""

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = cache_dir

    def get_stock_list(self, market: str = "沪深300") -> pd.DataFrame:
        """获取股票列表"""
        if market == "沪深300":
            return ak.index_stock_cons(symbol="000300")
        # 扩展其他指数...

    def get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取单股票历史数据"""
        return ak.stock_zh_a_hist(
            symbol=symbol,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq",  # 前复权
        )

    def update_all(self, universe: list[str]) -> dict:
        """批量更新股票池数据"""
        results = {}
        for symbol in universe:
            try:
                results[symbol] = self.get_stock_data(symbol, ...)
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
        return results
```

### 1.3 Qlib 数据转换

**文件**: `src/data/qlib_converter.py`

```python
import qlib
from qlib.data import D

class QlibConverter:
    """转换为 qlib 格式"""

    def __init__(self, qlib_dir: str = "data/qlib"):
        qlib.init(provider_uri=qlib_dir, region="cn")
        self.data_dir = qlib_dir

    def convert_from_akshare(self, raw_data: dict) -> None:
        """转换 akshare 数据到 qlib 格式"""
        # qlib 需要特定的目录结构和文件格式
        # 实现数据格式转换逻辑
        pass

    def verify_data(self) -> bool:
        """验证数据完整性"""
        instruments = D.instruments(market="cn")
        return len(instruments) > 0
```

### 1.4 简单回测

**文件**: `src/backtest/runner.py`

```python
from qlib.backtest import backtest, executor
from qlib.contrib.strategy import TopkDropoutStrategy

def run_simple_backtest():
    """运行简单回测验证数据"""

    strategy = TopkDropoutStrategy(
        topk=30,
        n_drop=5,
    )

    # 执行回测
    portfolio_result = backtest(
        strategy=strategy,
        executor=executor,
        start_time="2020-01-01",
        end_time="2023-12-31",
    )

    return portfolio_result
```

### 1.5 验证脚本

**文件**: `scripts/validate_setup.py`

```python
#!/usr/bin/env python
"""验证基础设置是否正确"""

def check_dependencies():
    """检查依赖"""
    import qlib, akshare, pandas
    print("✓ 依赖包安装完成")

def check_data():
    """检查数据"""
    from src.data.qlib_converter import QlibConverter
    converter = QlibConverter()
    assert converter.verify_data(), "数据验证失败"
    print("✓ 数据加载成功")

def check_backtest():
    """检查回测"""
    from src.backtest.runner import run_simple_backtest
    result = run_simple_backtest()
    assert result is not None, "回测失败"
    print("✓ 回测执行成功")

if __name__ == "__main__":
    check_dependencies()
    check_data()
    check_backtest()
    print("\n🎉 里程碑 M1 达成：基础框架验证通过！")
```

## 测试策略

### 单元测试

```python
# tests/test_data_loader.py
def test_akshare_loader():
    loader = AkshareLoader()
    df = loader.get_stock_data("000001", "2023-01-01", "2023-12-31")
    assert len(df) > 200
    assert "close" in df.columns

# tests/test_strategy.py
def test_strategy_signals():
    strategy = BaseStrategy()
    signals = strategy.generate_signals(test_data)
    assert signals.shape == test_data.shape
```

### 集成测试

```python
# tests/integration/test_full_pipeline.py
def test_full_pipeline():
    """测试完整流程"""
    # 1. 加载数据
    # 2. 运行回测
    # 3. 生成报告
    # 4. 验证结果
    pass
```

## 成本估算

### 开发成本

| 阶段 | 预计工时 | 说明 |
|------|----------|------|
| Phase 1 | 8-12h | 基础框架搭建 |
| Phase 2 | 16-24h | 策略系统开发 |
| Phase 3 | 12-16h | 分析报告系统 |
| Phase 4 | 16-20h | AI 集成优化 |
| **总计** | **52-72h** | 约 2-3 周 |

### 运行成本

```
数据获取: 免费 (akshare)
计算资源: 本地运行，无额外成本
API 调用: Claude Code 按使用计费
存储: ~500MB (历史数据)
```

## 参考资源

### 核心文档

- [Qlib 官方文档](https://qlib.readthedocs.io/)
- [Akshare 文档](https://akshare.akfamily.xyz/)
- [Claude Code 文档](https://github.com/anthropics/claude-code)

### 策略参考

- Qlib 自带示例策略
- 量化社区常见因子
- 学术论文: Alpha360, Alpha158 等

### 风险参考

- [量化交易风险清单](https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-I/)

---

*文档创建日期: 2026-02-21*
*最后更新: 2026-02-21*
