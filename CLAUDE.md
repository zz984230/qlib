# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-driven quantitative investment strategy system for A-share market (China stocks), built on Microsoft Qlib. Uses akshare for data, supports automated strategy optimization via Claude Code.

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run single test file
uv run pytest tests/test_strategy.py -v

# Run single test
uv run pytest tests/test_strategy.py::TestSimpleFactorStrategy::test_init -v

# Lint
uv run ruff check src/
uv run black src/ --check

# Format
uv run black src/
```

## Architecture

### Data Flow
```
akshare → AkshareLoader → QlibConverter → qlib format → BacktestRunner → Analysis → PDF Report
                                                               ↓
                                    AI Agent ← analyze results ←─┘
```

### Core Modules

**src/data/**: Data layer
- `AkshareLoader`: Fetches A-share data from akshare API, caches as parquet
- `QlibConverter`: Transforms akshare format to qlib binary format

**src/strategy/**: Strategy layer with registry pattern
- `BaseStrategy`: Abstract base class, implements `get_factors()`, `generate_signals()`, `get_model_config()`
- `get_strategy(name)`: Factory function to instantiate strategies by name
- `src/strategy/factors/`: Factor library with `FACTOR_REGISTRY` and `get_factor(name)` factory

**src/backtest/**: Backtesting
- `BacktestRunner.run()`: Falls back from qlib backtest to vectorized backtest if qlib unavailable
- `BacktestResult`: Contains portfolio_value (Series), metrics dict, calculates sharpe/return/drawdown properties

**src/analysis/**: Analysis & Reporting
- `PerformanceMetrics`: Dataclass with 30+ metrics (returns, risk, risk-adjusted)
- `BacktestVisualizer`: Generates nav curves, drawdown plots, monthly heatmaps
- `ReportGenerator`: Creates PDF reports with charts embedded

**src/agent/**: AI automation
- `analyze_backtest_result()`: Diagnoses strategy issues, returns severity-ranked recommendations
- `suggest_optimizations()`: Generates parameter adjustment suggestions
- `search_strategies()`: Searches strategy database for new ideas

### Configuration

All configs in `configs/` are YAML:
- `qlib.yaml`: Qlib provider path, region
- `data.yaml`: Market universe (csi300), date range, fields
- `strategy.yaml`: Strategy params (topk, n_drop), model config, factor expressions
- `risk.yaml`: Position limits, stop loss, optimization triggers

### Adding New Strategy

1. Create class inheriting `BaseStrategy` in `src/strategy/advanced.py`
2. Implement `get_factors()` returning list of factor expressions
3. Implement `generate_signals(data: DataFrame)` returning np.ndarray
4. Register in `STRATEGY_REGISTRY` dict and export in `__init__.py`

### Adding New Factor

1. Create class inheriting `BaseFactor` in appropriate `src/strategy/factors/` submodule
2. Implement `calculate(data: DataFrame)` returning np.ndarray
3. Optionally implement `to_qlib_expression()` for qlib native execution
4. Register in `FACTOR_REGISTRY` in `factors/__init__.py`

## Scripts

```bash
# Update data from akshare
python scripts/update_data.py --market csi300 --verify

# Run backtest
python scripts/run_backtest.py --strategy dual_ma --start-date 2023-01-01 --save

# Generate analysis report with charts
python scripts/run_analysis.py --strategy rsi

# Interactive strategy optimization
python scripts/optimize_strategy.py --strategy dual_ma

# AI workflow automation
python scripts/run_agent.py --mode daily     # Daily workflow
python scripts/run_agent.py --mode search    # Search new strategies
```

## Key Patterns

- Registry pattern for strategies and factors (`STRATEGY_REGISTRY`, `FACTOR_REGISTRY`)
- Factory functions (`get_strategy()`, `get_factor()`)
- YAML configuration driven behavior
- Fallback from qlib native to simple backtest when qlib unavailable
