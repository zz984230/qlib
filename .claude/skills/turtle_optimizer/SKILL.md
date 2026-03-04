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
| `--max-drawdown` | 0.08 | Maximum drawdown reference (8%, non-hard constraint) |
| `--cash` | 100000 | Initial capital |

### Output Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output-dir` | reports/html | HTML report directory |
| `--pool-dir` | strategies/pool | Strategy pool directory |

## Factor Pool

The optimizer explores combinations of 30 factors across 9 categories:

**Momentum Factors:**
- `ma_ratio`: MA5 / MA20 - 1 (moving average deviation)
- `ma_cross`: MA5 > MA20 (golden cross/death cross)
- `momentum`: (close - close_n5) / close_n5
- `price_momentum`: close / close_n20 - 1
- `roc`: Rate of Change(10)

**Technical Indicators:**
- `rsi`: RSI / 100 (relative strength)
- `macd`: MACD / MACD_signal
- `kdj`: (K - D) / 100
- `williams_r`: Williams %R

**Volatility:**
- `volatility`: STD(close, 20) / close
- `atr_ratio`: ATR / close

**Volume:**
- `volume_ratio`: volume / MA(volume, 20)
- `volume_price`: volume * (close / close_n1 - 1)
- `volume_trend`: MA(volume, 5) / MA(volume, 20)

**Trend:**
- `adx`: ADX / 100 (trend strength)
- `cci`: CCI / 200

**Energy/Flow:**
- `obv`: OBV / MA(OBV, 20)
- `money_flow`: typical price-volume correlation

**Bollinger Bands:**
- `bb_ratio`: Bollinger Band position (close - lower) / (upper - lower)

**Risk-Adjusted Factors (NEW):**
- `risk_adj_momentum`: momentum / volatility (risk-adjusted return)
- `relative_strength`: close / MA(close, 50) - 1 (medium-term strength)
- `vol_adj_return`: (close - close_n1) / ATR (volatility-adjusted return)

**Trend Quality Factors (NEW):**
- `trend_consistency`: percentage of up days in last 10 days
- `higher_highs`: frequency of new highs in last 5 days

**Price Behavior Factors (Right-Side Trading NEW):**
- `new_high_count`: count of new highs in past N days / N
- `new_low_count`: count of new lows in past N days / N
- `consecutive_up`: consecutive up days / 10
- `consecutive_down`: consecutive down days / 10
- `gap_up`: upward gap (open > previous high)
- `gap_down`: downward gap (open < previous low)

## Advanced Features (NEW)

### Trend Strength Filter
Each strategy now includes an `min_adx` parameter that filters entry signals based on trend strength:
- **ADX < 20**: Weak trend, avoid trading
- **ADX 20-30**: Moderate trend, acceptable
- **ADX > 30**: Strong trend, ideal for turtle strategy

The genetic algorithm automatically optimizes this parameter as part of the strategy evolution.

### Market Environment Recognition
The system now automatically detects market conditions and only enters trades during trending markets:
- **Uptrend**: Price consistently rising with strong momentum
- **Downtrend**: Price consistently falling (avoid long positions)
- **Ranging**: Price oscillating without clear direction (avoid trading)

This prevents the strategy from getting chopped up in sideways markets.

### Progressive Trailing Stop
Improved stop-loss logic that tightens as profits increase:
- **Base trailing stop**: 2 ATR below highest price
- **After 20% profit**: 1.5 ATR (tighter protection)
- **After 50% profit**: 1.0 ATR (aggressive profit protection)
- **After 100% profit**: 0.5 ATR (maximum profit lock-in)

This allows profits to run while protecting against reversals.

## Market-Adaptive Strategy Selection (REVISED)

The system now dynamically selects strategies based on market state:

### Strategy Selection Matrix

| Market State | Strategy | Position Size | Stop Loss |
|---------------|----------|---------------|-----------|
| **Strong Trend** | Turtle Trend Following | 100% | 2.0-2.5 ATR |
| **Weak Trend** | Turtle Trend Following | 60% | 1.8-2.2 ATR |
| **Ranging** | Mean Reversion | 25% | 1.0-1.5 ATR |
| **Volatile** | No Trading | 0% | N/A |

### Mean Reversion Strategy (NEW)

For ranging markets (ADX <= 20), the optimizer now uses a mean reversion strategy:

**Entry Signals:**
- RSI < 30 (oversold) or RSI > 70 (overbought)
- Price touches Bollinger Bands (lower/upper)
- Price deviates significantly from moving average

**Exit Signals:**
- Price returns to mean (MA20)
- RSI returns to neutral range (40-60)
- Target profit level reached

**Characteristics:**
- Smaller position size (25%)
- Tighter stop loss (1.0-1.5 ATR)
- No pyramiding (disabled)
- Buy low, sell high approach

### Key Concept Change

**OLD (Incorrect)**: Maximum portfolio drawdown of 3%
**NEW (Correct)**: Per-trade stop loss of 2% account value

The Turtle Trading risk control is based on:
- **Unit Size**: 1 Unit = Account × 1% / ATR
- **Stop Loss**: Entry Price - 2×ATR (approximately 2% of account)
- **Maximum Loss**: 2% of account value per trade
- **No Hard Drawdown Limit**: Portfolio drawdown emerges from trading results

This means the optimizer will find strategies that:
- Respect the 2% per-trade stop loss rule
- May have higher portfolio drawdowns during strong trends
- Focus on returns rather than strict drawdown limits

### Four Market Modes

| State | Condition | Characteristics |
|-------|-----------|-----------------|
| **Strong Trend** | ADX > 30 | Strong directional movement, high momentum |
| **Weak Trend** | 20 < ADX <= 30 | Moderate trend, medium momentum |
| **Ranging** | ADX <= 20 | Sideways, no clear direction |
| **Volatile** | Volatility spike > 2x | High volatility, high risk |

### Adaptive Parameters by Market State

| Parameter | Strong Trend | Weak Trend | Ranging | Volatile |
|-----------|--------------|------------|---------|----------|
| Signal Threshold | 0.05-0.15 | 0.10-0.20 | 0.15-0.30 | 0.25-0.40 |
| Stop Loss ATR | 2.5-3.5 | 2.0-2.5 | 1.0-1.5 | 0.5-1.0 |
| Position Size | 100% | 60% | 25% | 0% (avoid) |
| Max Units | 4 | 3 | 2 | 1 |

### Tagged Factor System

Each factor has market adaptation tags that adjust selection probability:

- **trend_strong**: ma_ratio, price_momentum, adx, higher_highs (x2 weight in strong trend)
- **trend_weak**: ma_cross, momentum, roc, relative_strength (x2 weight in weak trend)
- **ranging**: rsi, kdj, williams_r, bb_ratio (x2 weight in ranging markets)
- **volatile**: volatility, atr_ratio, volume_price (x3 weight in volatile markets)
- **universal**: volume_ratio, volume_trend, obv (unchanged weight)

### Dynamic Risk Management

The 3% drawdown hard constraint is enforced through:

- **Position multiplier**: Reduces position size based on market state
- **Warning system**: 2% drawdown triggers 50% position reduction
- **Critical system**: 2.5% drawdown triggers position exit
- **Emergency stop**: 3% drawdown forces immediate liquidation

## Turtle Trading Rules Implementation (REVISED)

The optimizer implements authentic Turtle Trading rules with A-share market adaptations:

### Position Sizing (ATR-Based)
- **Unit Size Formula**: `1 Unit = Account × 1% / ATR`
- **Risk Per Trade**: Maximum 2% of account value (1 unit = 1%, stop at 2×ATR below entry)
- **A-share Constraint**: Units are rounded to multiples of 100 shares (1 lot)
- **Minimum Trade**: 100 shares (A-share market requirement)

### Pyramiding Rules
- **Max Units**: 4 units per position (Turtle original)
- **Add Interval**: 0.5 ATR above previous entry price
- **Requirement**: Must be profitable before adding
- **Independent Stops**: Each unit has its own stop loss (2 ATR below entry)

### Stop Loss System
- **Fixed Stop**: Entry price - 2×ATR (original Turtle rule)
- **Trailing Stop**: Activates after 1×ATR profit, follows highest price
- **Per-Trade Risk**: Maximum 2% account value loss

### Entry Filters
- **Signal Threshold**: Entry only when composite signal >= threshold
- **Market State**: System automatically adjusts parameters based on market conditions

## Incremental Evolution (NEW)

The optimizer automatically loads seed individuals from strategy pool for continuous improvement:

### Seed Loading
- **Automatic**: Each run loads top 10 strategies from pool as seeds
- **Seed Ratio**: 20% of initial population comes from seeds, 80% random
- **Benefits**:
  - Preserves discovered good gene combinations
  - Accelerates convergence to better solutions
  - Enables continuous improvement over multiple runs

### Usage
```bash
# First run - builds initial strategy pool
uv run python scripts/turtle_genetic_optimizer.py --symbol 601138 --name "Stock A"

# Subsequent runs - automatically uses seeds from pool
uv run python scripts/turtle_genetic_optimizer.py --symbol 601138 --name "Stock A"

# Each run improves upon previous discoveries
```

## Strategy Pool Cleanup (NEW)

The optimizer automatically culls underperforming strategies after each run:

### Cleanup Rules
- **Minimum Fitness**: Removes strategies with fitness < -5.0
- **Max per Symbol**: Keeps only top 50 strategies per stock
- **Automatic**: Runs after every optimization completion

### Manual Cleanup
```python
from src.optimizer.strategy_pool import StrategyPool

pool = StrategyPool("strategies/pool")

# Remove low fitness strategies
pool.cleanup_strategies(symbol="601138", min_fitness=-5.0)

# Remove old strategies (older than 30 days)
pool.cleanup_strategies(max_age_days=30)

# Limit pool size (keep top 20 per symbol)
pool.cleanup_strategies(max_count_per_symbol=20)
```

## Valid Strategy Criteria

A strategy is considered valid when:
1. **Minimum 10 trades** across all periods
2. **At least 1 period with positive returns**
3. **Sharpe ratio > 0.5** in 1-year period (preferred but not required)

**Note**: The system no longer uses a strict 3% maximum drawdown constraint. Instead, it focuses on:
- Per-trade risk management (2% account value stop loss)
- Return consistency across periods
- Risk-adjusted returns (Sharpe ratio)

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
- `min_adx`: Minimum ADX trend strength filter (NEW)
- `min_trend_periods`: Minimum trend periods (NEW)
- `use_trend_filter`: Enable trend filtering (NEW)
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

# Configure fitness evaluation (no hard drawdown constraint)
fitness_config = FitnessConfig(
    max_drawdown_limit=0.08,  # Reference value, not hard constraint
    sharpe_bonus_weight=1.0,
    stability_bonus_weight=0.5,
    drawdown_penalty_weight=0.5,
    return_weight=5.0,
)

# Create and run optimizer
optimizer = TurtleGeneticOptimizer(
    symbol="601138",
    name="Industrial Fulian",
    initial_cash=100000,
    genetic_config=genetic_config,
    fitness_config=fitness_config,
    interactive=False,
)

result = optimizer.run()
print(f"Found {result['valid_strategies']} valid strategies")
```

## Dynamic Strategy Selection

The optimizer automatically selects strategies based on market state:

```python
from src.optimizer.strategy_selector import StrategySelector

selector = StrategySelector()

# Get strategy for current market
strategy = selector.select_strategy(market_state="weak_trend")
# Returns: "turtle" for trend markets

# Get position multiplier
position_size = selector.get_position_multiplier(market_state="ranging")
# Returns: 0.25 (25% position size for ranging markets)

# Check if trading is allowed
should_trade = selector.should_trade(market_state="volatile")
# Returns: False (no trading in volatile markets)
```
