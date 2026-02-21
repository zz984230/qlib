# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-driven quantitative investment strategy system for A-share market (China stocks), built on Microsoft Qlib. Uses akshare for data, supports automated strategy optimization via Claude Code.

## Commands

```bash
# Install dependencies
uv sync

# Install PDF report dependencies (optional)
uv pip install reportlab matplotlib

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

### Low Drawdown Strategy

For strict risk control (target max drawdown <= 3%), use `low_drawdown_report.py`:

**Risk Control Parameters:**
- Stop Loss: 1.5%
- Trailing Stop: 1.0% (triggered after 1% profit)
- Daily Loss Limit: 1.0%
- Portfolio Drawdown Limit: 2.4%
- Position Size: 20% per trade
- Cooldown Period: 3 days after stop loss

**Signal Conditions:**
- Price must be above long-term MA
- Short-term MA trending up
- RSI not overbought (< 65)
- Golden cross or RSI oversold entry

### Adding New Factor

1. Create class inheriting `BaseFactor` in appropriate `src/strategy/factors/` submodule
2. Implement `calculate(data: DataFrame)` returning np.ndarray
3. Optionally implement `to_qlib_expression()` for qlib native execution
4. Register in `FACTOR_REGISTRY` in `factors/__init__.py`

## Scripts

```bash
# Update data from akshare
uv run python scripts/update_data.py --market csi300 --verify

# Run standard backtest
uv run python scripts/run_backtest.py --strategy dual_ma --start-date 2023-01-01 --save

# Run analysis pipeline (backtest + report)
uv run python scripts/run_analysis.py --strategy rsi --start-date 2023-01-01

# Low drawdown strategy backtest with PDF report (recommended for risk control)
uv run python scripts/low_drawdown_report.py --symbol 601138 --period 1y    # 近1年
uv run python scripts/low_drawdown_report.py --symbol 601138 --period 3m    # 近3月
uv run python scripts/low_drawdown_report.py --symbol 601138 --period 1m    # 近1月

# Interactive strategy optimization
uv run python scripts/optimize_strategy.py --strategy dual_ma

# AI workflow automation
uv run python scripts/run_agent.py --mode daily     # Daily workflow
uv run python scripts/run_agent.py --mode search    # Search new strategies
```

### Script Descriptions

| Script | Description |
|--------|-------------|
| `update_data.py` | Fetch stock data from akshare, save as parquet cache |
| `run_backtest.py` | Standard strategy backtest with qlib/vectorized engine |
| `run_analysis.py` | Complete analysis pipeline: backtest + performance report |
| `low_drawdown_report.py` | Low drawdown strategy (target 3%) + PDF report generation |
| `optimize_strategy.py` | Interactive parameter optimization |
| `run_agent.py` | AI-powered automation for strategy analysis |

## Key Patterns

- Registry pattern for strategies and factors (`STRATEGY_REGISTRY`, `FACTOR_REGISTRY`)
- Factory functions (`get_strategy()`, `get_factor()`)
- YAML configuration driven behavior
- Fallback from qlib native to simple backtest when qlib unavailable

## Coding Standards

### Report Language

**IMPORTANT: All generated reports must be in Chinese (Simplified).**

This applies to:
- PDF reports generated by `low_drawdown_report.py`
- Chart labels and titles
- Table headers and content
- Conclusions and recommendations

### No Emoji Policy

**IMPORTANT: Do NOT use emoji in any code, scripts, comments, or documentation files.**

Emoji characters cause encoding errors on Windows systems (GBK codec issues) and should be avoided entirely. This includes:
- Python scripts (.py)
- Markdown files (.md)
- Shell scripts (.sh)
- Configuration files
- All skill-related files

Use plain text alternatives instead:
- Use `[OK]`, `[WARN]`, `[ERROR]` instead of emoji status indicators
- Use `->`, `=>`, `-->` instead of arrow emoji
- Use `*`, `-`, `+` for bullet points instead of emoji bullets
