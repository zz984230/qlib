# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-driven quantitative investment strategy system for A-share market (China stocks), built on Microsoft Qlib. Uses akshare for data, supports automated strategy optimization via Claude Code.

## Commands

```bash
# Install dependencies
uv sync

# Install HTML report dependencies (optional)
uv pip install jinja2 plotly kaleido

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
akshare → AkshareLoader → QlibConverter → qlib format → BacktestRunner → Analysis → HTML Report
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
- `HtmlReportGenerator`: Creates HTML reports with interactive charts (using Jinja2 + Chart.js/Plotly)

**src/optimizer/**: Strategy Optimization (NEW)
- `GeneticEngine`: Genetic algorithm for factor combination optimization
- `StrategyPool`: Manages valid strategies (JSON file storage in `strategies/pool/`)
- `FitnessEvaluator`: Evaluates strategy fitness with multi-period backtesting

**src/report/**: HTML Report Generation (NEW)
- `html_generator.py`: Generates comprehensive HTML reports with:
  - Evolution history charts
  - Factor weight analysis
  - Three-period performance comparison (1y/3m/1m)
  - Trade details (delivery notes)
  - Exploration logs
  - AI analysis recommendations

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

### Data Caching Patterns

**AkshareLoader**: Market-level data caching
- Fetches entire market (e.g., csi300) data from akshare
- Caches as parquet files in `data/` directory
- Used by standard backtest scripts

**TurtleDataLoader**: Symbol-level daily-granularity caching
- Fetches individual stock data with daily granularity
- Incremental cache updates: preserves existing data, only fetches new days
- Cache location: `data/cache/turtle/{symbol}/`
- Key methods: `load_data()`, `refresh_cache()`, `get_cache_info()`
- Used by turtle genetic optimizer and low drawdown strategy

Choose the appropriate loader based on your use case:
- Market-wide backtesting: Use `AkshareLoader`
- Single-stock optimization: Use `TurtleDataLoader` (faster incremental updates)

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

## Turtle Genetic Optimizer (NEW)

For automated strategy optimization using genetic algorithm combined with Turtle Trading rules:

**Overview:**
- Uses genetic algorithm to explore factor combinations for entry/exit signals
- Implements Turtle Trading rules for position sizing, pyramiding, and risk management
- Target: Maximize returns under 3% maximum drawdown constraint
- Interactive: Pauses on each valid strategy discovery for user confirmation

**Core Components:**

1. **TurtleSignalGenerator**: Factor-based signal generation
   - Explores combinations of: MA, RSI, Momentum, Volatility, Volume, etc.
   - Genetic algorithm evolves optimal factor weights

2. **TurtlePositionManager**: ATR-based dynamic position sizing
   - Unit size = 1% of account / ATR
   - Pyramid adding: Up to 4 units at 0.5 ATR intervals

3. **TurtleRiskManager**: Dual stop-loss system
   - Fixed stop: Entry price - 2*ATR
   - Trailing stop: Activates after 1N profit

4. **GeneticEngine**: Evolutionary optimization
   - Population: 50 individuals per generation
   - Selection: Tournament + Elite preservation
   - Crossover: Single-point
   - Mutation: Gaussian

5. **FitnessEvaluator**: Multi-period fitness calculation
   - Backtests on 3 periods: 1y (60% weight), 3m (30%), 1m (10%)
   - Penalty for drawdown > 3%
   - Bonus for Sharpe ratio and stability

**Valid Strategy Criteria:**
- Maximum drawdown <= 3% across all three periods
- Positive returns in at least 2 of 3 periods

**Usage:**
```bash
# Run turtle genetic optimizer
uv run python scripts/turtle_genetic_optimizer.py \
    --symbol 601138 \
    --name "工业富联" \
    --cash 50000 \
    --max-generations 100 \
    --population-size 50
```

**Strategy Pool:**
Valid strategies are saved to `strategies/pool/` as JSON files with complete parameters and backtest results.

**HTML Report:**
Comprehensive report includes:
- Evolution history charts
- Factor weight heatmap
- Three-period performance comparison
- Trade details (delivery notes)
- Exploration logs
- AI analysis recommendations

## Scripts

```bash
# Update data from akshare (market-level)
uv run python scripts/update_data.py --market csi300 --verify

# Refresh daily-granularity cache for single stock (symbol-level)
uv run python scripts/refresh_cache.py --symbol 601138 --days 365
uv run python scripts/refresh_cache.py --symbol 601138 --clear  # clear cache first

# Run standard backtest
uv run python scripts/run_backtest.py --strategy dual_ma --start-date 2023-01-01 --save

# Run analysis pipeline (backtest + report)
uv run python scripts/run_analysis.py --strategy rsi --start-date 2023-01-01

# Low drawdown strategy backtest with HTML report (recommended for risk control)
# Generates a combined HTML report with 3 time periods (1y/3m/1m)
uv run python scripts/low_drawdown_report.py --symbol 601138 --name "工业富联"

# Turtle genetic optimizer (NEW - automated strategy exploration)
uv run python scripts/turtle_genetic_optimizer.py --symbol 601138 --name "工业富联"

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
| `refresh_cache.py` | Refresh daily-granularity cache for stock data with incremental updates |
| `run_backtest.py` | Standard strategy backtest with qlib/vectorized engine |
| `run_analysis.py` | Complete analysis pipeline: backtest + performance report |
| `low_drawdown_report.py` | Low drawdown strategy (target 3%) + HTML report generation |
| `turtle_genetic_optimizer.py` | Genetic algorithm optimizer with Turtle Trading rules (NEW) |
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
- HTML reports generated by `low_drawdown_report.py` and `turtle_genetic_optimizer.py`
- Chart labels and titles
- Table headers and content
- Conclusions and recommendations

### Report Format

**IMPORTANT: All backtest reports must be generated as a single combined HTML file containing 3 time periods.**

The report must include:
- Period 1: Near 1 year (1y)
- Period 2: Near 3 months (3m)
- Period 3: Near 1 month (1m)

Each period should have its own performance metrics, charts, and trade summary, all consolidated into one HTML file with interactive charts.

**HTML Report Technology:**
- Jinja2 templating engine
- Chart.js or Plotly for interactive charts
- Tailwind CSS for styling
- Single self-contained HTML file (no external dependencies required)

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
