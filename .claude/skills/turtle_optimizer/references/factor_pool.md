# Factor Pool Reference

## Overview

The turtle optimizer explores combinations of 18 quantitative factors to find optimal entry/exit signals.

## Factor Categories

### 1. Momentum Factors

| Factor | Expression | Description |
|--------|-----------|-------------|
| `ma_ratio` | MA5 / MA20 - 1 | Moving average deviation. Positive when short MA above long MA. |
| `ma_cross` | MA5 > MA20 | Golden cross (1) / death cross (0). Binary signal. |
| `momentum` | (close - close_n5) / close_n5 | 5-day price momentum. |
| `price_momentum` | close / close_n20 - 1 | 20-day price momentum. |

### 2. Technical Indicators

| Factor | Expression | Description |
|--------|-----------|-------------|
| `rsi` | RSI / 100 | Relative Strength Index normalized to 0-1. Overbought > 0.7, oversold < 0.3. |
| `macd` | MACD / MACD_signal | MACD histogram ratio. Positive when MACD above signal line. |
| `kdj` | (K - D) / 100 | KDJ oscillator. K above D is bullish. |

### 3. Volatility Factors

| Factor | Expression | Description |
|--------|-----------|-------------|
| `volatility` | STD(close, 20) / close | 20-day volatility coefficient. Higher = more volatile. |
| `atr_ratio` | ATR / close | Average True Range ratio. Used for position sizing. |

### 4. Volume Factors

| Factor | Expression | Description |
|--------|-----------|-------------|
| `volume_ratio` | volume / MA(volume, 20) | Volume relative to 20-day average. >1 = above average volume. |
| `volume_price` | volume * (close / close_n1 - 1) | Volume-price coordination. Positive on up days with high volume. |

### 5. Trend Factors

| Factor | Expression | Description |
|--------|-----------|-------------|
| `adx` | ADX / 100 | Average Directional Index. >0.25 indicates strong trend. |
| `cci` | CCI / 200 | Commodity Channel Index. Normalized for signal generation. |

### 6. Energy/Flow Factors

| Factor | Expression | Description |
|--------|-----------|-------------|
| `obv` | OBV / MA(OBV, 20) | On-Balance Volume ratio. Measures buying/selling pressure. |
| `money_flow` | typical price * volume | Money flow indicator. |
| `bb_ratio` | (close - lower) / (upper - lower) | Bollinger Band position. 0 = lower band, 1 = upper band. |
| `roc` | ROC(10) | 10-day Rate of Change. |
| `williams_r` | %R | Williams %R oscillator. Oversold < -80, overbought > -20. |

## Signal Generation

### Combined Signal Calculation

```
signal = sum(factor_value[i] * weight[i]) for all factors
```

### Entry Signal
```
if signal > signal_threshold:
    generate_buy_signal()
```

### Exit Signal
```
if signal < exit_threshold:
    generate_sell_signal()
```

## Parameter Ranges

| Parameter | Range | Default |
|-----------|-------|---------|
| factor_weight | 0.0 - 1.0 | (normalized) |
| signal_threshold | 0.0 - 1.0 | 0.5 |
| exit_threshold | 0.0 - 1.0 | 0.3 |
| atr_period | 5 - 50 | 20 |
| stop_loss_atr | 1.5 - 3.0 | 2.0 |
| pyramid_interval_atr | 0.3 - 0.7 | 0.5 |
| trailing_stop_trigger | 0.5 - 1.5 | 1.0 |

## Factor Selection Tips

1. **Trending Markets**: Favor `ma_cross`, `adx`, `price_momentum`
2. **Range-bound Markets**: Favor `rsi`, `bb_ratio`, `williams_r`
3. **High Volatility**: Favor `volatility`, `atr_ratio` for position sizing
4. **Volume Confirmation**: Always include `volume_ratio` or `obv`
