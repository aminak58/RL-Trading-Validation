# Feature Engineering for Exit-Focused RL

Best practices for creating features that help RL agents make intelligent exit decisions.

## Table of Contents

1. [Exit vs Entry Features](#exit-vs-entry-features)
2. [Feature Categories](#feature-categories)
3. [Implementation Cookbook](#implementation-cookbook)
4. [Feature Selection](#feature-selection)
5. [Common Pitfalls](#common-pitfalls)

---

## Exit vs Entry Features

### Critical Distinction

**Entry features** (classic TA): "Is now a good time to enter?"
- Trend alignment
- Momentum strength
- Support/resistance levels
- Volume confirmation

**Exit features** (RL focus): "Should I close this position NOW?"
- Profit deterioration
- Momentum exhaustion
- Reversal signals
- Time-in-position effects

### Why Exit Features are Different

Your MtfScalper_RL_Hybrid uses **classic entry signals** (3-timeframe alignment), but relies on **RL for exits**. The RL agent needs different information:

```python
# Entry context (provided by classic strategy)
"Is the trend strong?" → ADX, EMA alignment

# Exit context (what RL needs)
"Is the trend weakening?" → Momentum acceleration, divergence
"Have I held too long?" → Position duration proxies
"Is profit at risk?" → Drawdown from peak
"Is liquidity good?" → Volume ratios
```

---

## Feature Categories

### Category 1: Position-Aware Features (Critical!)

These features track the health of the current position.

#### 1.1 Profit Metrics

```python
# Current unrealized profit (if tracked in environment)
dataframe["%-current_profit"] = (
    dataframe["close"] - dataframe["entry_price"]
) / dataframe["entry_price"]

# Distance from session high/low
dataframe["%-dist_from_high_20"] = (
    dataframe["high"].rolling(20).max() - dataframe["close"]
) / dataframe["close"]

dataframe["%-dist_from_low_20"] = (
    dataframe["close"] - dataframe["low"].rolling(20).min()
) / dataframe["close"]

# Profit erosion indicator
dataframe["%-profit_erosion"] = (
    dataframe["close"].rolling(20).max() - dataframe["close"]
) / dataframe["close"]
```

**Why this matters**: RL agent learns "I was up 3% but now only 1.5% - should exit!"

#### 1.2 Time-Based Features

```python
# RL environments track time internally, but features help

# Trend persistence (how long has trend been aligned?)
dataframe["%-trend_age_long"] = (
    (dataframe["ema_fast"] > dataframe["ema_slow"])
    .rolling(50)
    .sum()
)

dataframe["%-trend_age_short"] = (
    (dataframe["ema_fast"] < dataframe["ema_slow"])
    .rolling(50)
    .sum()
)

# Candles since local extreme
def candles_since_high(df, window=50):
    highs = df["high"].rolling(window).max()
    is_high = df["high"] == highs
    candles_since = pd.Series(index=df.index, dtype=float)

    last_high_idx = 0
    for i in range(len(df)):
        if is_high.iloc[i]:
            last_high_idx = i
        candles_since.iloc[i] = i - last_high_idx

    return candles_since

dataframe["%-candles_since_high"] = candles_since_high(dataframe)
```

**Why this matters**: "Trend is 45 candles old, likely exhausted"

#### 1.3 Momentum Features (Your Strong Performers)

```python
# Short-term momentum (you have this)
dataframe["%-momentum_5"] = dataframe["close"].pct_change(5)
dataframe["%-momentum_10"] = dataframe["close"].pct_change(10)
dataframe["%-momentum_20"] = dataframe["close"].pct_change(20)

# Acceleration (rate of change of momentum)
dataframe["%-acceleration"] = dataframe["%-momentum_5"].diff()

# Momentum divergence (key for exits!)
dataframe["%-momentum_divergence"] = (
    dataframe["%-momentum_5"] - dataframe["%-momentum_20"]
)

# Jerk (third derivative - advanced)
dataframe["%-jerk"] = dataframe["%-acceleration"].diff()
```

**Why this matters**: Exit when momentum turns negative (divergence signal)

### Category 2: Reversal Indicators

#### 2.1 Divergence Detection

```python
# RSI Divergence (you have this)
price_higher = dataframe["close"] > dataframe["close"].shift(10)
rsi_lower = dataframe["rsi"] < dataframe["rsi"].shift(10)
dataframe["%-bearish_divergence"] = (price_higher & rsi_lower).astype(int)

price_lower = dataframe["close"] < dataframe["close"].shift(10)
rsi_higher = dataframe["rsi"] > dataframe["rsi"].shift(10)
dataframe["%-bullish_divergence"] = (price_lower & rsi_higher).astype(int)

# Volume Divergence
volume_decreasing = dataframe["volume"] < dataframe["volume"].rolling(10).mean()
price_increasing = dataframe["close"] > dataframe["close"].shift(5)
dataframe["%-volume_price_divergence"] = (
    volume_decreasing & price_increasing
).astype(int)

# MACD Divergence
macd = ta.MACD(dataframe)
macd_higher = macd["macd"] > macd["macd"].shift(10)
dataframe["%-macd_divergence"] = (
    (price_higher & ~macd_higher).astype(int)
)
```

**Why this matters**: Classic early warning of trend reversal

#### 2.2 Exhaustion Signals

```python
# Overbought/Oversold extremes
dataframe["%-rsi_extreme_high"] = (dataframe["rsi"] > 75).astype(int)
dataframe["%-rsi_extreme_low"] = (dataframe["rsi"] < 25).astype(int)

# Bollinger Band position
bb_width = dataframe["bb_upper"] - dataframe["bb_lower"]
dataframe["%-bb_position"] = (
    (dataframe["close"] - dataframe["bb_lower"]) / (bb_width + 1e-10)
)
# 0 = at lower band, 1 = at upper band

dataframe["%-bb_squeeze"] = (
    bb_width / dataframe["bb_middle"]
)  # Low values = consolidation

# Stochastic RSI extremes
stoch_rsi = ta.STOCHRSI(dataframe)
dataframe["%-stochrsi_overbought"] = (stoch_rsi["fastk"] > 80).astype(int)
dataframe["%-stochrsi_oversold"] = (stoch_rsi["fastk"] < 20).astype(int)
```

**Why this matters**: Exit before reversal happens

### Category 3: Market Microstructure (Exit Quality)

#### 3.1 Volume Analysis

```python
# Volume ratios (you have this)
dataframe["%-volume_ratio_5"] = (
    dataframe["volume"] / dataframe["volume"].rolling(5).mean()
)
dataframe["%-volume_ratio_20"] = (
    dataframe["volume"] / dataframe["volume"].rolling(20).mean()
)

# Volume trend
dataframe["%-volume_trend"] = (
    dataframe["volume"].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )
)

# Volume exhaustion
dataframe["%-volume_exhaustion"] = (
    dataframe["volume"].rolling(5).std() /
    (dataframe["volume"].rolling(20).std() + 1e-10)
)

# Smart money indicator (money flow)
typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
money_flow = typical_price * dataframe["volume"]
dataframe["%-money_flow_ratio"] = (
    money_flow.rolling(5).mean() / money_flow.rolling(20).mean()
)
```

**Why this matters**: Low volume = poor exit liquidity

#### 3.2 Spread and Slippage Proxies

```python
# Bid-ask spread proxy (you have this)
dataframe["%-spread_proxy"] = (
    (dataframe["high"] - dataframe["low"]) / dataframe["close"]
)

dataframe["%-spread_ma_ratio"] = (
    dataframe["%-spread_proxy"] /
    dataframe["%-spread_proxy"].rolling(20).mean()
)

# Candle wicks (rejection strength)
dataframe["%-upper_wick"] = (
    (dataframe["high"] - np.maximum(dataframe["open"], dataframe["close"])) /
    dataframe["close"]
)

dataframe["%-lower_wick"] = (
    (np.minimum(dataframe["open"], dataframe["close"]) - dataframe["low"]) /
    dataframe["close"]
)

# Wick ratio (where is price in candle?)
dataframe["%-wick_ratio"] = (
    dataframe["%-upper_wick"] /
    (dataframe["%-lower_wick"] + 1e-10)
)
```

**Why this matters**: Wide spreads = worse exit execution

### Category 4: Multi-Timeframe Context

#### 4.1 Higher Timeframe Signals

```python
# 15m trend change (you have partial implementation)
if "ema_fast_15m" in dataframe.columns:
    # Trend alignment
    dataframe["%-trend_aligned_15m"] = (
        (dataframe["ema_fast_15m"] > dataframe["ema_slow_15m"]).astype(int)
    )

    # Momentum change
    dataframe["%-momentum_15m"] = (
        dataframe["close_15m"].pct_change(3)  # 3 periods = 45 minutes
    )

    # RSI exit signals
    dataframe["%-rsi_15m_exit_long"] = (
        (dataframe["rsi_15m"] > 70).astype(int)
    )
    dataframe["%-rsi_15m_exit_short"] = (
        (dataframe["rsi_15m"] < 30).astype(int)
    )

# 1h trend change (you have this)
if "ema_fast_1h" in dataframe.columns:
    dataframe["%-ema_cross_bearish_1h"] = (
        (dataframe["ema_fast_1h"].shift(1) > dataframe["ema_slow_1h"].shift(1)) &
        (dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"])
    ).astype(int)
```

**Why this matters**: Don't hold long when 1h trend flips

#### 4.2 Timeframe Alignment Score

```python
def calculate_mtf_alignment(dataframe):
    """
    Composite score: how aligned are all timeframes?
    """
    score = 0

    # Base timeframe (5m)
    if dataframe["ema_fast"] > dataframe["ema_slow"]:
        score += 1
    else:
        score -= 1

    # 15m
    if "ema_fast_15m" in dataframe.columns:
        if dataframe["ema_fast_15m"] > dataframe["ema_slow_15m"]:
            score += 2  # Weight higher TF more
        else:
            score -= 2

    # 1h
    if "ema_fast_1h" in dataframe.columns:
        if dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]:
            score += 3  # Weight highest TF most
        else:
            score -= 3

    return score  # Range: -6 to +6

dataframe["%-mtf_alignment_score"] = calculate_mtf_alignment(dataframe)
```

**Why this matters**: Exit when alignment breaks down

### Category 5: Risk & Volatility

#### 5.1 Volatility Regime

```python
# ATR-based volatility
dataframe["%-atr_ratio"] = dataframe["atr"] / dataframe["close"]

# Volatility regime (your risk_score component)
dataframe["%-volatility_regime"] = (
    dataframe["atr"] / dataframe["atr"].rolling(50).mean()
)
# > 1.5 = high volatility, < 0.7 = low volatility

# Rolling volatility
dataframe["%-volatility_5"] = dataframe["close"].pct_change().rolling(5).std()
dataframe["%-volatility_20"] = dataframe["close"].pct_change().rolling(20).std()

# Volatility trend
dataframe["%-volatility_trend"] = (
    dataframe["%-volatility_5"] / (dataframe["%-volatility_20"] + 1e-10)
)
```

**Why this matters**: High volatility = exit sooner to protect profits

#### 5.2 Risk Score (Your Implementation)

```python
# Composite risk measure (you have this)
dataframe["%-risk_score"] = (
    dataframe["%-spread_proxy"] * 0.3 +
    (1 / (dataframe["%-volume_ratio_5"] + 0.1)) * 0.3 +
    dataframe["atr"] / dataframe["close"] * 0.4
)

# Enhanced version with more components
dataframe["%-risk_score_v2"] = (
    dataframe["%-spread_proxy"] * 0.25 +
    (1 / (dataframe["%-volume_ratio_5"] + 0.1)) * 0.25 +
    dataframe["%-volatility_regime"] * 0.25 +
    dataframe["%-profit_erosion"] * 0.25
)
```

### Category 6: Support/Resistance Levels

#### 6.1 Pivot Points (Your Implementation)

```python
# Standard pivots (you have this)
dataframe["%-pivot"] = (
    dataframe["high"] + dataframe["low"] + dataframe["close"]
) / 3
dataframe["%-r1"] = 2 * dataframe["%-pivot"] - dataframe["low"]
dataframe["%-s1"] = 2 * dataframe["%-pivot"] - dataframe["high"]

# Distance to levels
dataframe["%-dist_to_r1"] = (
    (dataframe["%-r1"] - dataframe["close"]) / dataframe["close"]
)
dataframe["%-dist_to_s1"] = (
    (dataframe["close"] - dataframe["%-s1"]) / dataframe["close"]
)
```

#### 6.2 Dynamic S/R Levels

```python
# Recent swing highs/lows
def find_swing_highs(df, window=10):
    """Find local maxima."""
    swing_highs = df["high"].rolling(window * 2 + 1, center=True).max()
    is_swing_high = df["high"] == swing_highs
    return is_swing_high

dataframe["%-is_swing_high"] = find_swing_highs(dataframe).astype(int)

# Distance to nearest swing high
def dist_to_nearest_swing(df, window=50):
    is_swing = find_swing_highs(df)
    swing_prices = df.loc[is_swing, "high"]

    distances = []
    for i in range(len(df)):
        current_price = df.iloc[i]["close"]
        recent_swings = swing_prices.iloc[max(0, i-window):i]

        if len(recent_swings) > 0:
            nearest_swing = recent_swings.iloc[-1]
            dist = (nearest_swing - current_price) / current_price
        else:
            dist = 0

        distances.append(dist)

    return pd.Series(distances, index=df.index)

dataframe["%-dist_to_swing_high"] = dist_to_nearest_swing(dataframe)
```

**Why this matters**: Exit near resistance, avoid near support

---

## Implementation Cookbook

### Recipe 1: Momentum Exhaustion Detector

```python
def add_momentum_exhaustion_features(df):
    """
    Detects when momentum is fading - ideal exit signal.
    """
    # 1. Price momentum
    df["%-mom_5"] = df["close"].pct_change(5)
    df["%-mom_10"] = df["close"].pct_change(10)

    # 2. Momentum divergence
    df["%-mom_divergence"] = df["%-mom_5"] - df["%-mom_10"]
    # Negative = momentum slowing

    # 3. Acceleration
    df["%-acceleration"] = df["%-mom_5"].diff()
    # Negative = deceleration

    # 4. Composite exhaustion score
    exhaustion = 0

    # Component 1: Divergence
    if df["%-mom_divergence"] < -0.005:
        exhaustion += 1

    # Component 2: Deceleration
    if df["%-acceleration"] < 0:
        exhaustion += 1

    # Component 3: RSI peak
    if df["rsi"] > 70 and df["rsi"] < df["rsi"].shift(1):
        exhaustion += 1

    df["%-exhaustion_score"] = exhaustion  # 0-3

    return df
```

### Recipe 2: Profit Protection Features

```python
def add_profit_protection_features(df):
    """
    Tracks profit deterioration - when to secure gains.
    """
    # Rolling maximum (peak profit potential)
    df["%-rolling_max_20"] = df["close"].rolling(20).max()

    # Profit erosion
    df["%-profit_erosion"] = (
        (df["%-rolling_max_20"] - df["close"]) / df["close"]
    )

    # Erosion rate (how fast is profit disappearing?)
    df["%-erosion_rate"] = df["%-profit_erosion"].diff()

    # Binary: is profit eroding fast?
    df["%-fast_erosion"] = (
        df["%-erosion_rate"] > 0.01
    ).astype(int)

    # Exit score: higher = should exit
    exit_score = 0
    if df["%-profit_erosion"] > 0.02:  # Down 2% from peak
        exit_score += 2
    if df["%-fast_erosion"] == 1:
        exit_score += 3

    df["%-profit_exit_score"] = exit_score

    return df
```

### Recipe 3: Volume Confirmation

```python
def add_volume_confirmation_features(df):
    """
    Validates exit quality through volume analysis.
    """
    # Volume moving averages
    df["%-vol_ma5"] = df["volume"].rolling(5).mean()
    df["%-vol_ma20"] = df["volume"].rolling(20).mean()

    # Volume ratio
    df["%-vol_ratio"] = df["volume"] / df["%-vol_ma20"]

    # Volume trend
    df["%-vol_trend"] = (
        df["%-vol_ma5"] / df["%-vol_ma20"]
    )

    # Exit quality: good volume for execution?
    quality = 0

    if df["%-vol_ratio"] > 1.5:  # Strong volume
        quality += 2
    elif df["%-vol_ratio"] > 1.0:
        quality += 1
    elif df["%-vol_ratio"] < 0.5:  # Weak volume
        quality -= 2

    df["%-exit_quality"] = quality  # -2 to +2

    return df
```

### Recipe 4: Multi-Timeframe Divergence

```python
def add_mtf_divergence_features(df):
    """
    Detects when timeframes disagree - early exit warning.
    """
    # 5m trend
    df["%-trend_5m"] = (
        df["ema_fast"] > df["ema_slow"]
    ).astype(int)

    # 15m trend
    if "ema_fast_15m" in df.columns:
        df["%-trend_15m"] = (
            df["ema_fast_15m"] > df["ema_slow_15m"]
        ).astype(int)
    else:
        df["%-trend_15m"] = df["%-trend_5m"]

    # 1h trend
    if "ema_fast_1h" in df.columns:
        df["%-trend_1h"] = (
            df["ema_fast_1h"] > df["ema_slow_1h"]
        ).astype(int)
    else:
        df["%-trend_1h"] = df["%-trend_5m"]

    # Alignment score
    df["%-alignment"] = (
        df["%-trend_5m"] +
        df["%-trend_15m"] * 2 +  # Weight higher TFs
        df["%-trend_1h"] * 3
    )  # Range: 0 (all down) to 6 (all up)

    # Divergence: lower TF disagrees with higher TF
    df["%-tf_divergence"] = (
        (df["%-trend_5m"] != df["%-trend_15m"]).astype(int) +
        (df["%-trend_15m"] != df["%-trend_1h"]).astype(int)
    )  # 0 = aligned, 1-2 = divergence

    return df
```

---

## Feature Selection

### How Many Features?

```python
# Your current setup: ~25-30 features
# Network capacity: [256, 256, 128]

# Guideline:
first_layer_size = 256
recommended_features = first_layer_size / 10
# → ~25 features (you're at the sweet spot!)

# If adding features:
if num_features > 40:
    # Increase first layer
    net_arch = [512, 256, 128]
```

### Feature Importance Analysis

```python
# After training, analyze which features matter:
from scripts.feature_importance import calculate_feature_importance

importance = calculate_feature_importance(
    model=your_trained_model,
    dataframe=test_data,
    n_samples=1000
)

# Remove features with importance < 0.01
low_importance = importance[importance < 0.01].index
dataframe_filtered = dataframe.drop(columns=low_importance)
```

### Correlation Check

```python
# Remove highly correlated features (> 0.95)
feature_cols = [col for col in dataframe.columns if col.startswith("%-")]
corr_matrix = dataframe[feature_cols].corr().abs()

# Find pairs with high correlation
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

# Remove one from each pair (keep more important one)
```

---

## Common Pitfalls

### ❌ Pitfall 1: Lookahead Bias

```python
# BAD: Uses future information
dataframe["%-next_high"] = dataframe["high"].shift(-5)  # DON'T!

# GOOD: Only past information
dataframe["%-prev_high"] = dataframe["high"].rolling(20).max()
```

### ❌ Pitfall 2: Not Normalizing

```python
# BAD: Features on different scales
dataframe["%-price"] = dataframe["close"]  # Range: 20000-70000
dataframe["%-rsi"] = dataframe["rsi"]      # Range: 0-100
dataframe["%-vol_ratio"] = vol_ratio       # Range: 0-5

# GOOD: Normalized features
dataframe["%-price_norm"] = dataframe["close"] / dataframe["close"].rolling(100).mean()
# All features now in similar ranges (0.5-1.5)
```

### ❌ Pitfall 3: Too Many NaN Values

```python
# BAD: Creates NaN in first 200 rows
dataframe["%-long_ma"] = dataframe["close"].rolling(200).mean()

# GOOD: Shorter window or fill strategy
dataframe["%-long_ma"] = dataframe["close"].rolling(50).mean().fillna(method="bfill")
```

### ❌ Pitfall 4: Ignoring Stationarity

```python
# BAD: Non-stationary feature
dataframe["%-price"] = dataframe["close"]  # Trends over time

# GOOD: Stationary feature
dataframe["%-price_change"] = dataframe["close"].pct_change()
dataframe["%-price_zscore"] = (
    (dataframe["close"] - dataframe["close"].rolling(50).mean()) /
    dataframe["close"].rolling(50).std()
)
```

### ❌ Pitfall 5: Redundant Features

```python
# BAD: Essentially the same information
dataframe["%-rsi_14"] = ta.RSI(dataframe, 14)
dataframe["%-rsi_15"] = ta.RSI(dataframe, 15)  # Redundant!

# GOOD: Different aspects
dataframe["%-rsi_14"] = ta.RSI(dataframe, 14)
dataframe["%-rsi_change"] = dataframe["%-rsi_14"].diff()  # Rate of change
```

---

## Recommended Feature Set

Based on your strategy and exit focus, here's an optimal 25-30 feature set:

```python
# Tier 1: Critical (must have)
"%-momentum_5"
"%-momentum_10"
"%-acceleration"
"%-dist_from_high_20"
"%-dist_from_low_20"
"%-volume_ratio_5"
"%-spread_proxy"
"%-risk_score"

# Tier 2: Very Important
"%-bearish_divergence"
"%-bullish_divergence"
"%-profit_erosion"
"%-volatility_regime"
"%-rsi"
"%-mtf_alignment_score"
"%-ema_cross_bearish_1h"

# Tier 3: Nice to Have
"%-volume_exhaustion"
"%-momentum_divergence"
"%-timing_score" (based on percentile)
"%-candles_since_high"
"%-bb_position"
"%-stochrsi_overbought"

# Tier 4: Experimental
"%-jerk" (third derivative)
"%-money_flow_ratio"
"%-wick_ratio"
"%-volume_trend"
"%-erosion_rate"
```

Total: ~25 features (optimal for [256, 256, 128] architecture)

---

## Testing New Features

```python
def test_feature_quality(df, feature_name, lookforward=10):
    """
    Quick test: does feature correlate with future returns?
    """
    df["future_return"] = df["close"].shift(-lookforward) / df["close"] - 1

    correlation = df[feature_name].corr(df["future_return"])

    print(f"Feature: {feature_name}")
    print(f"Correlation with future return: {correlation:.4f}")

    if abs(correlation) > 0.1:
        print("✓ Strong signal - keep this feature!")
    elif abs(correlation) > 0.05:
        print("→ Moderate signal - test in model")
    else:
        print("✗ Weak signal - probably not useful")

    return abs(correlation)

# Example usage:
test_feature_quality(dataframe, "%-momentum_divergence", lookforward=10)
```

Run this for every new feature before adding to your model.
