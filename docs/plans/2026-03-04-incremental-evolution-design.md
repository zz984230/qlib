# 增量进化与价格行为因子设计

## 背景

用户需求：
1. 量化因子持续通过遗传算法进行优化（增量进化模式）
2. 添加海龟交易法则的右侧交易策略（价格行为因子）
3. 更新 turtle_optimizer skill 文档

## 设计

### 1. 增量进化机制

**目标**：让新的优化运行能够加载历史优秀策略作为初始种群的种子。

**流程**：
1. 加载策略池：从 `strategies/pool/*.json` 筛选 fitness > threshold 的策略
2. 种子注入：优秀策略作为初始种群的一部分（10-20%），剩余随机生成
3. 演化标记：策略记录 `evolution_lineage` 和累加代数

**新增参数**：
- `--seed-from-pool`: 从策略池加载种子
- `--seed-ratio`: 种子占初始种群比例（默认 0.2）
- `--min-seed-fitness`: 种子最低适应度阈值（默认 -0.5）

### 2. 价格行为因子

**新增因子**：

| 因子名 | 描述 | 计算方式 |
|--------|------|----------|
| `new_high_count` | 新高计数 | 过去 N 日内创新高的次数 / N |
| `new_low_count` | 新低计数 | 过去 N 日内创新低的次数 / N |
| `consecutive_up` | 连续阳线 | 连续上涨天数 / 10 |
| `consecutive_down` | 连续阴线 | 连续下跌天数 / 10 |
| `gap_up` | 向上跳空 | open > prev_high (0/1) |
| `gap_down` | 向下跳空 | open < prev_low (0/1) |

**归一化**：所有因子归一化到 [0, 1] 区间。

### 3. Skill 更新

更新 `turtle_optimizer/skill.md`：
- 添加增量进化使用说明
- 添加价格行为因子文档

## 实现计划

1. **价格行为因子实现**
   - 修改 `src/optimizer/individual.py` 添加新因子到 FACTOR_POOL
   - 修改 `src/strategy/turtle_signals.py` 添加因子计算逻辑

2. **增量进化实现**
   - 修改 `src/optimizer/genetic_engine.py` 支持种子注入
   - 修改 `scripts/turtle_genetic_optimizer.py` 添加命令行参数

3. **Skill 更新**
   - 更新 `.claude/skills/turtle_optimizer/skill.md`

## 风险控制

- 3% 最大回撤限制保持不变
- 新因子需通过单元测试验证计算正确性
