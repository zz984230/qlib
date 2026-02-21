# Qlib AI Strategy

基于微软 Qlib 构建的 AI 驱动量化投资策略系统。

## 功能特性

- 数据采集：使用 Akshare 获取 A 股市场数据
- 策略框架：灵活的策略基类，支持自定义因子和模型
- 回测引擎：集成 Qlib 回测系统
- 报告生成：PDF 格式详细分析报告（Phase 3）
- AI 集成：Claude Code 自动优化策略（Phase 4）

## 快速开始

### 安装依赖

```bash
# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -e .
```

### 更新数据

```bash
python scripts/update_data.py --market csi300 --verify
```

### 运行回测

```bash
python scripts/run_backtest.py --strategy simple --save
```

## 项目结构

```
qlib-ai-strategy/
├── configs/           # 配置文件
├── data/              # 数据存储
├── src/               # 源代码
│   ├── data/          # 数据加载和转换
│   ├── strategy/      # 策略模块
│   ├── backtest/      # 回测引擎
│   └── analysis/      # 分析报告
├── scripts/           # 脚本工具
├── reports/           # 输出报告
└── tests/             # 测试代码
```

## 实施路线

- [x] Phase 1: 基础框架 (MVP)
- [ ] Phase 2: 策略系统
- [ ] Phase 3: 分析报告
- [ ] Phase 4: AI 集成

## License

MIT
