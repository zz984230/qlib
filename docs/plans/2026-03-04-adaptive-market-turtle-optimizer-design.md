# 自适应市场风格的海龟遗传优化系统设计

**日期**: 2026-03-04
**状态**: 设计完成，待实施

---

## 1. 需求概述

### 核心需求
1. 添加市场趋势判断来选择合适策略的功能
2. 丰富适应不同市场风格的策略因子组合，并能够持续进化优化
3. 3%的回撤约束不能改变（硬约束）

### 设计目标
- 实现4种市场状态的自动识别和分类
- 建立30个因子的市场适应性标签体系
- 根据市场状态动态调整策略参数（阈值、止损、仓位）
- 保持3%回撤硬约束的同时，最大化不同市场环境下的收益

---

## 2. 架构设计

### 2.1 系统架构

```
市场数据 -> 市场状态识别 -> 参数适配器 -> 遗传算法优化 -> 策略执行 -> 回测评估 -> 反馈
    ^                                                                    |
    |                                                                    v
    └────────────────── 策略池（带市场标签的因子组合） <───────────────────┘
```

### 2.2 核心模块

| 模块 | 职责 |
|------|------|
| `MarketStateClassifier` | 使用 ADX + ATR + 价格动量识别四种市场状态 |
| `TaggedFactorPool` | 30个因子带市场适应性标签，支持权重动态调整 |
| `ParameterAdapter` | 根据市场状态提供参数配置（阈值、止损、仓位） |
| `DynamicRiskManager` | 确保3%回撤硬约束，动态仓位控制 |

---

## 3. 市场状态分类

### 3.1 四种市场模式

| 状态 | 判定条件 | 特征 |
|------|----------|------|
| **强趋势** | ADX > 30 | 价格单向运动，动量强 |
| **弱趋势** | 20 < ADX <= 30 | 有趋势但动量一般 |
| **震荡** | ADX <= 20 | 无明确方向，横盘整理 |
| **异常波动** | 波动率突增 > 2倍 | 价格剧烈波动，风险高 |

### 3.2 状态识别逻辑

```python
def classify_market_state(data: DataFrame) -> str:
    """基于最近60天数据识别市场状态"""
    adx = calculate_adx(data, period=14)
    atr = calculate_atr(data, period=14)
    volatility = atr / data['close']

    recent_adx = adx.iloc[-1]
    vol_spike = volatility.iloc[-1] / volatility.iloc[-5:].mean()

    # 异常波动优先判断
    if vol_spike > 2.0:
        return "volatile"

    # 趋势判断
    if recent_adx > 30:
        return "strong_trend"
    elif recent_adx > 20:
        return "weak_trend"
    else:
        return "ranging"
```

---

## 4. 因子标签化设计

### 4.1 因子标签映射

| 标签 | 适用因子 | 权重调整 |
|------|----------|----------|
| `trend_strong` | ma_ratio, price_momentum, adx, higher_highs, new_high_count | 强趋势时×2 |
| `trend_weak` | ma_cross, momentum, roc, relative_strength | 弱趋势时×2 |
| `ranging` | rsi, kdj, williams_r, bb_ratio, cci | 震荡时×2 |
| `volatile` | volatility, atr_ratio, volume_price, vol_adj_return | 异常时×3 |
| `universal` | volume_ratio, volume_trend, obv, money_flow | 权重不变 |

### 4.2 权重调整算法

```python
def adjust_factor_weights(base_weights: dict, market_state: str) -> dict:
    """根据市场状态调整因子选择概率"""
    multipliers = {
        "strong_trend": {"trend_strong": 2.0, "trend_weak": 1.5, "ranging": 0.5, "volatile": 1.0},
        "weak_trend": {"trend_strong": 1.0, "trend_weak": 2.0, "ranging": 1.0, "volatile": 1.0},
        "ranging": {"trend_strong": 0.3, "trend_weak": 0.5, "ranging": 2.0, "volatile": 1.5},
        "volatile": {"trend_strong": 0.2, "trend_weak": 0.3, "ranging": 0.5, "volatile": 3.0},
    }
    # 应用权重倍数后重新归一化
```

---

## 5. 参数适配器设计

### 5.1 市场参数配置

```python
@dataclass
class MarketParameters:
    signal_threshold: tuple[float, float]  # 信号阈值范围
    exit_threshold: tuple[float, float]    # 出场阈值范围
    stop_loss_atr: tuple[float, float]     # 止损ATR倍数
    trailing_stop_trigger: tuple[float, float]
    position_multiplier: float             # 仓位系数
    max_units: int                         # 最大加仓单位
    factor_count_range: tuple[int, int]    # 因子数量范围
```

### 5.2 四种市场状态参数表

| 参数 | 强趋势 | 弱趋势 | 震荡 | 异常波动 |
|------|--------|--------|------|----------|
| 信号阈值 | 0.05-0.15 | 0.10-0.20 | 0.15-0.30 | 0.25-0.40 |
| 出场阈值 | 0.05-0.15 | 0.08-0.18 | 0.10-0.20 | 0.15-0.25 |
| 止损ATR | 2.5-3.5 | 2.0-2.5 | 1.0-1.5 | 0.5-1.0 |
| 仓位系数 | 100% | 60% | 25% | 0% |
| 最大单位 | 4 | 3 | 2 | 1 |

### 5.3 3%回撤硬约束

- 所有市场状态下 `max_drawdown_limit = 0.03` 不变
- 通过仓位系数（position_multiplier）控制风险暴露
- 异常波动市直接空仓（0%），避免亏损

---

## 6. 文件结构

```
src/
├── optimizer/
│   ├── market_state.py          [NEW] 市场状态分类器
│   ├── parameter_adapter.py     [NEW] 参数适配器
│   ├── tagged_factors.py        [NEW] 因子标签定义
│   ├── dynamic_risk_manager.py  [NEW] 动态风险管理器
│   ├── turtle_optimizer.py      [MOD] 集成新模块
│   ├── genetic_engine.py        [MOD] 支持市场状态参数
│   └── individual.py            [MOD] 添加市场适应度字段
├── strategy/
│   └── market_classifier.py     [NEW] ADX/ATR/波动率计算
└── data/
    └── cache/
        └── market_state/        [NEW] 市场状态缓存
```

---

## 7. 执行流程

### 7.1 初始化阶段
1. 加载策略池（已保存的策略）
2. 识别当前市场状态（基于最近60天）
3. 获取对应市场状态的参数配置

### 7.2 种子生成阶段
1. 从策略池加载top 10策略作为种子
2. 根据市场状态调整种子参数：
   - 因子权重 × 标签倍数
   - 阈值映射到市场配置范围
   - 仓位系数应用

### 7.3 遗传演化阶段
1. 生成新个体使用市场状态对应的参数范围
2. 因子选择概率带标签权重
3. 适应度评估保持3%回撤硬约束

### 7.4 策略保存阶段
1. 保存策略时附加市场适应度标签
2. 记录该策略在四种市场状态下的表现
3. 后续可根据当前市场状态筛选最优策略

---

## 8. 错误处理

| 场景 | 处理策略 |
|------|----------|
| 市场状态识别失败 | 默认使用"ranging"（最保守参数） |
| 状态频繁切换 | 使用5日平滑窗口 |
| 异常波动后快速恢复 | 添加3个交易日冷却期 |
| 策略池为空 | 随机初始化，应用市场状态参数约束 |
| 仓位<100股 | 跳过交易，记录日志 |
| 回撤>2% | 减仓50%；>2.5%清仓 |

---

## 9. 实施计划

### Phase 1: 因子标签化（1-2小时）
- 创建 `tagged_factors.py`
- 为30个因子定义市场标签
- 更新 `genetic_engine.py` 支持标签权重

### Phase 2: 市场状态识别（1-2小时）
- 创建 `market_classifier.py`
- 创建 `market_state.py`
- 添加市场状态缓存

### Phase 3: 参数适配器（1小时）
- 创建 `parameter_adapter.py`
- 定义4种市场状态参数配置
- 更新 `genetic_engine.py`

### Phase 4: 动态风险管理（1小时）
- 创建 `dynamic_risk_manager.py`
- 实现3%回撤硬约束
- 更新 `turtle_position.py`

### Phase 5: 主优化器集成（1小时）
- 更新 `turtle_optimizer.py`
- 更新 `individual.py`
- 添加市场状态记录

### Phase 6: 测试与验证（1-2小时）
- 单元测试：市场状态分类器
- 单元测试：参数适配器
- 集成测试：完整优化流程
- 回测验证：3%回撤约束

### Phase 7: 报告与文档（30分钟）
- 更新 skill.md
- 添加市场状态可视化
- 更新 CLAUDE.md

---

## 10. 预期效果

1. **适应性提升**：策略根据市场状态自动调整参数，避免一刀切
2. **风险控制**：3%回撤硬约束，异常波动市自动空仓
3. **收益优化**：强趋势市满仓运行，震荡市轻仓降低磨损
4. **持续进化**：遗传算法在不同市场状态下独立优化因子组合
