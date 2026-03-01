---
name: turtle_optimizer
description: Automated quantitative trading strategy optimization using genetic algorithm combined with Turtle Trading rules. Use when user wants to explore optimal factor combinations for entry/exit signals, optimize position sizing and risk management parameters, or continuously improve trading strategy performance. Triggers when user mentions "turtle optimizer", "genetic algorithm optimization", "factor exploration", "strategy optimization", or wants to run the turtle_genetic_optimizer.py script.
---

# Turtle Genetic Optimizer

Automated strategy optimization using genetic algorithm combined with Turtle Trading rules for A-share market quantitative trading.

## Core Workflow

```
1. Data Preparation     -->  Ensure stock data is cached (auto-fetch if missing)
2. Run Optimization     -->  Execute genetic algorithm to explore factor combinations
3. Review Strategies    -->  Examine valid strategies from strategy pool
4. Analyze Report       -->  Review HTML report for evolution history and performance
5. Iterate             -->  Continue optimization with refined parameters
```

## Quick Start

### Basic Optimization Run

```bash
# Run optimizer for a specific stock
uv run python scripts/turtle_genetic_optimizer.py --symbol 601138 --name "Industrial Fulian"

# With custom parameters
uv run python scripts/turtle_genetic_optimizer.py \
    --symbol 601138 \
    --name "Industrial Fulian" \
    --cash 100000 \
    --max-generations 50 \
    --population-size 30 \
    --max-drawdown 0.03
```

### Non-Interactive Mode (for automation)

```bash
uv run python scripts/turtle_genetic_optimizer.py \
    --symbol 601138 \
    --name "Industrial Fulian" \
    --no-interactive \
    --max-generations 100
```

## Key Parameters

### Genetic Algorithm Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--population-size` | 50 | Number of individuals per generation |
| `--max-generations` | 100 | Maximum evolution generations |
| `--mutation-rate` | 0.1 | Probability of mutation |
| `--crossover-rate` | 0.7 | Probability of crossover |
| `--elite-size` | 5 | Number of elites preserved |

### Risk Control Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-drawdown` | 0.03 | Maximum drawdown limit (3%) |
| `--cash` | 50000 | Initial capital |

### Output Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | reports/html | HTML report directory |
| `--pool-dir` | strategies/pool | Strategy pool directory |

## Factor Pool

The optimizer explores combinations of 18 factors:

**Momentum Factors:**
- `ma_ratio`: MA5 / MA20 - 1 (moving average deviation)
- `ma_cross`: MA5 > MA20 (golden cross/death cross)
- `momentum`: (close - close_n5) / close_n5
- `price_momentum`: close / close_n20 - 1

**Technical Indicators:**
- `rsi`: RSI / 100 (relative strength)
- `macd`: MACD / MACD_signal
- `kdj`: (K - D) / 100

**Volatility:**
- `volatility`: STD(close, 20) / close
- `atr_ratio`: ATR / close

**Volume:**
- `volume_ratio`: volume / MA(volume, 20)
- `volume_price`: volume * (close / close_n1 - 1)

**Trend:**
- `adx`: ADX / 100 (trend strength)
- `cci`: CCI / 200

**Energy/Flow:**
- `obv`: OBV / MA(OBV, 20)
- `money_flow`: typical price-volume correlation
- `bb_ratio`: Bollinger Band position
- `roc`: Rate of Change(10)
- `williams_r`: Williams %R

## Valid Strategy Criteria

A strategy is considered valid when:
1. Maximum drawdown <= 3% across all three periods (1y, 3m, 1m)
2. Positive returns in at least 2 of 3 periods

## Strategy Pool Management

Valid strategies are automatically saved to `strategies/pool/` as JSON files.

### View Strategy Pool

```bash
# List all strategies for a symbol
ls strategies/pool/strategy_*.json

# View a specific strategy
cat strategies/pool/strategy_xxxxxxxx.json | python -m json.tool
```

### Strategy Data Structure

Each strategy JSON contains:
- `id`: Unique strategy identifier
- `symbol`: Stock code
- `factor_weights`: Factor weights dictionary
- `signal_threshold`: Entry signal threshold
- `exit_threshold`: Exit signal threshold
- `atr_period`: ATR calculation period
- `stop_loss_atr`: Stop loss in ATR multiples
- `pyramid_interval_atr`: Pyramiding interval
- `backtest_results`: Performance metrics for 1y/3m/1m
- `fitness`: Fitness score

## Data Cache Management

Stock data is cached at daily granularity in `data/cache/daily/{symbol}/`.

### Cache Structure
```
data/cache/daily/
  601138/
    2024-01-02.parquet
    2024-01-03.parquet
    ...
```

### Refresh Cache

If data is missing or outdated, it's automatically fetched. To force refresh:

```python
from src.data.turtle_data_loader import TurtleDataLoader

loader = TurtleDataLoader()
loader.refresh_cache("601138", days=365)
```

## Continuous Optimization Workflow

### 1. Initial Run
```bash
# Start with default parameters
uv run python scripts/turtle_genetic_optimizer.py --symbol 601138 --name "Stock Name"
```

### 2. Review Results
- Check HTML report in `reports/html/`
- Review valid strategies in `strategies/pool/`
- Note best fitness score and factor weights

### 3. Refine Parameters
Based on results, adjust:
- Increase `--population-size` for more exploration
- Adjust `--max-drawdown` based on risk tolerance
- Use `--random-seed` for reproducibility

### 4. Continue Optimization
```bash
# Run again with refined parameters
uv run python scripts/turtle_genetic_optimizer.py \
    --symbol 601138 \
    --name "Stock Name" \
    --population-size 100 \
    --max-generations 200
```

### 5. Compare Strategies
New valid strategies are added to the pool. Compare with previous runs to track improvement.

## HTML Report Contents

The generated report includes:
1. **Evolution History**: Fitness improvement over generations
2. **Factor Weight Heatmap**: Distribution of factor weights across strategies
3. **Three-Period Performance**: 1y/3m/1m comparison with benchmark
4. **Trade Details**: Entry/exit points, position sizing
5. **Exploration Logs**: Generation-by-generation progress

## Troubleshooting

### Data Insufficient Error
```
Error: Data insufficient: X items
```
Solution: The optimizer requires at least 400 trading days. Let the data fetch complete or check akshare API availability.

### No Valid Strategies Found
- Increase `--max-generations` for longer exploration
- Relax `--max-drawdown` constraint slightly
- Ensure stock has sufficient volatility for trading signals

### Slow Performance
- Reduce `--population-size` for faster iterations
- Use `--no-interactive` mode for unattended runs
- Check network connection for data fetching

## Programmatic Usage

```python
from src.optimizer.turtle_optimizer import TurtleGeneticOptimizer
from src.optimizer.genetic_engine import GeneticConfig
from src.optimizer.fitness import FitnessConfig

# Configure genetic algorithm
genetic_config = GeneticConfig(
    population_size=50,
    max_generations=100,
    mutation_rate=0.1,
)

# Configure fitness evaluation
fitness_config = FitnessConfig(
    max_drawdown_limit=0.03,
)

# Create and run optimizer
optimizer = TurtleGeneticOptimizer(
    symbol="601138",
    name="Industrial Fulian",
    initial_cash=50000,
    genetic_config=genetic_config,
    fitness_config=fitness_config,
    interactive=False,
)

result = optimizer.run()
print(f"Found {result['valid_strategies']} valid strategies")
```
