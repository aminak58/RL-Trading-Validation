# Reward Function Design Patterns

Comprehensive guide to reward function architectures for exit-focused RL trading models.

## Table of Contents

1. [Fundamental Principles](#fundamental-principles)
2. [Core Patterns](#core-patterns)
3. [Component Designs](#component-designs)
4. [Anti-Patterns](#anti-patterns)
5. [Real-World Examples](#real-world-examples)

---

## Fundamental Principles

### The Three Laws of RL Trading Rewards

1. **Asymmetric Penalties**: Punish losses harder than you reward gains
   - Loss at -2% should hurt more than gain at +2% feels good
   - Prevents the agent from taking excessive risks

2. **Temporal Decay**: Time is money, penalize stale positions
   - Encourages efficient capital utilization
   - Prevents "hope trading" (holding losers hoping for recovery)

3. **Multi-Objective Balance**: No single metric tells the full story
   - Profit alone → risky behavior
   - Win rate alone → agent takes only tiny profits
   - Use weighted combinations

### Reward Function Anatomy

```python
total_reward = Σ(weight_i × component_i)

# Standard decomposition:
total_reward = (
    w_profit × profit_score +
    w_risk × risk_score +
    w_timing × timing_score +
    w_efficiency × efficiency_score
)
```

**Key insight**: Components should be normalized to similar scales (-10 to +10) so weights actually matter.

---

## Core Patterns

### Pattern 1: Stepped Rewards (Your Current Implementation)

```python
def stepped_profit_reward(profit: float) -> float:
    """
    Discretized profit zones with clear thresholds.
    Good for: Learning clear profit targets.
    """
    if profit <= -0.05:      # Catastrophic loss
        return -5.0
    elif profit <= -0.02:    # Bad loss
        return -2.0
    elif profit <= -0.005:   # Small loss
        return -0.5
    elif profit <= 0:        # Breakeven zone
        return profit * 20   # Smooth near zero
    elif profit < 0.002:     # Tiny profit
        return 0.5 + profit * 100
    elif profit < 0.01:      # Small profit
        return 1.0 + profit * 150
    elif profit < 0.02:      # Good profit
        return 3.0 + profit * 100
    elif profit < 0.04:      # Excellent profit
        return 5.0 + profit * 50
    else:                    # Outstanding profit
        return min(15.0, 7.0 + profit * 30)  # Capped
```

**Pros**:
- Clear learning targets (agent knows 2% is a goal)
- Prevents exploitation (capped at 15.0)
- Easy to reason about

**Cons**:
- Discontinuities at boundaries can cause learning instability
- May encourage "threshold hunting" behavior

**When to use**: Starting point for new models, markets with clear profit targets.

### Pattern 2: Smooth Exponential

```python
def smooth_exponential_reward(profit: float) -> float:
    """
    Smooth, continuously differentiable reward.
    Good for: Stable gradient-based learning (PPO).
    """
    if profit >= 0:
        # Asymptotic approach to max reward
        return 10 * (1 - np.exp(-profit * 50))
    else:
        # Faster decay for losses
        return -10 * (1 - np.exp(profit * 30))

# Example values:
# profit = 0.01 → reward ≈ 3.93
# profit = 0.02 → reward ≈ 6.32
# profit = -0.02 → reward ≈ -4.51
```

**Pros**:
- Smooth gradients everywhere
- Natural diminishing returns at high profits
- Mathematically elegant

**Cons**:
- Less interpretable ("why did agent exit at 1.8%?")
- Parameters (50, 30) require tuning
- Can be too smooth (no clear targets)

**When to use**: Stable training environments, continuous action spaces (if extended).

### Pattern 3: Risk-Adjusted Sharpe-Like

```python
def sharpe_inspired_reward(profit: float, volatility: float) -> float:
    """
    Penalizes profits achieved through high volatility.
    Good for: Encouraging consistent, stable exits.
    """
    # Sharpe-like ratio
    risk_adjusted = profit / (volatility + 1e-6)

    # Squash to reasonable range
    reward = np.tanh(risk_adjusted * 5) * 10

    # Bonus for positive exits in low volatility
    if profit > 0 and volatility < 0.01:
        reward += 2.0

    return reward
```

**Pros**:
- Discourages risky exits
- Matches professional trading criteria
- Good for risk-averse strategies

**Cons**:
- Requires accurate volatility estimation
- May be too conservative (misses volatile opportunities)
- Complex interaction with other components

**When to use**: Real money trading, risk-sensitive environments.

### Pattern 4: Percentile-Based

```python
def percentile_reward(profit: float, profit_distribution: np.array) -> float:
    """
    Reward based on how profit compares to historical distribution.
    Good for: Adaptive markets, avoiding overfitting to specific ranges.
    """
    # Calculate percentile of current profit
    percentile = scipy.stats.percentileofscore(profit_distribution, profit)

    # Map to reward scale
    if profit < 0:
        # Below median losses → harsh penalty
        reward = -10 * (1 - percentile / 50)
    else:
        # Above median profits → increasing reward
        reward = 10 * ((percentile - 50) / 50)

    return reward
```

**Pros**:
- Adapts to market conditions
- No hardcoded profit targets
- Robust to regime changes

**Cons**:
- Requires maintaining distribution buffer
- Less stable during training
- Cold start problem (no distribution initially)

**When to use**: Multi-market strategies, long training periods with regime changes.

### Pattern 5: Multi-Stage Composite (Recommended)

```python
def multi_stage_reward(
    profit: float,
    max_profit_seen: float,
    position_duration: int,
    drawdown: float,
    volume_ratio: float
) -> float:
    """
    Comprehensive reward combining multiple signals.
    Based on your MtfScalperRLModel implementation.
    """

    # Component 1: Base profit score (35%)
    profit_score = _calculate_profit_score(profit)

    # Component 2: Drawdown control (25%)
    if drawdown < 0.005:
        drawdown_score = 5.0
    elif drawdown < 0.01:
        drawdown_score = 2.0
    elif drawdown < 0.02:
        drawdown_score = 0.0
    else:
        drawdown_score = -5.0 * drawdown

    # Component 3: Timing quality (20%)
    # Reward exiting near peak profit
    profit_capture_ratio = profit / (max_profit_seen + 1e-6) if max_profit_seen > 0 else 1.0
    if profit_capture_ratio > 0.8:
        timing_score = 5.0
    elif profit_capture_ratio > 0.6:
        timing_score = 2.0
    else:
        timing_score = -1.0

    # Component 4: Risk/Reward efficiency (20%)
    if position_duration < 50:
        efficiency_score = 3.0  # Quick profit
    elif position_duration < 150:
        efficiency_score = 1.0  # Acceptable
    else:
        efficiency_score = -2.0  # Too long

    # Liquidity penalty
    if volume_ratio < 0.3:
        efficiency_score -= 3.0

    # Weighted combination
    total = (
        0.35 * profit_score +
        0.25 * drawdown_score +
        0.20 * timing_score +
        0.20 * efficiency_score
    )

    return total
```

**Pros**:
- Comprehensive view of exit quality
- Tunable through weights
- Captures multiple objectives

**Cons**:
- Many parameters to tune
- Components must be balanced
- Can be opaque (which component is failing?)

**When to use**: Production systems, well-understood markets, after simpler patterns tested.

---

## Component Designs

### Profit Component

#### Linear (Simple)
```python
# Don't use: unbounded, no asymmetry
reward = profit * 100  # Bad!
```

#### Piecewise Linear (Better)
```python
if profit > 0:
    reward = min(10, profit * 200)  # Capped gains
else:
    reward = max(-15, profit * 300)  # Steeper losses
```

#### Logarithmic (Best for large ranges)
```python
if profit > 0:
    reward = 5 * np.log(1 + profit * 100)
else:
    reward = -7 * np.log(1 - profit * 100)
```

### Drawdown Component

```python
def drawdown_score(current_profit: float, max_profit_seen: float) -> float:
    """
    Penalize letting profits erode.
    """
    if max_profit_seen <= 0:
        return 0.0

    # Profit erosion ratio
    erosion = (max_profit_seen - current_profit) / max_profit_seen

    if erosion < 0.1:  # Less than 10% erosion
        return 5.0
    elif erosion < 0.3:  # 10-30% erosion
        return 2.0
    elif erosion < 0.5:  # 30-50% erosion
        return -2.0
    else:  # More than 50% erosion
        return -10.0

# Alternative: Continuous
def drawdown_score_smooth(erosion: float) -> float:
    return 5 * (1 - erosion) ** 2 - 5  # Parabola, max at 0 erosion
```

### Timing Component

```python
def timing_score(
    current_price: float,
    recent_high: float,
    recent_low: float,
    position_type: int  # 1 = long, -1 = short
) -> float:
    """
    Reward exits near local extremes.
    """
    price_range = recent_high - recent_low
    if price_range < 1e-8:
        return 0.0

    if position_type == 1:  # Long exit
        # Percentile in recent range (1.0 = at high)
        percentile = (current_price - recent_low) / price_range

        if percentile > 0.8:
            return 5.0
        elif percentile > 0.6:
            return 2.0
        elif percentile > 0.4:
            return 0.0
        else:
            return -2.0

    else:  # Short exit
        percentile = (current_price - recent_low) / price_range

        if percentile < 0.2:
            return 5.0
        elif percentile < 0.4:
            return 2.0
        elif percentile < 0.6:
            return 0.0
        else:
            return -2.0
```

### Position Duration Penalty

```python
def time_penalty(duration_candles: int, max_duration: int = 200) -> float:
    """
    Encourage timely exits.
    """
    ratio = duration_candles / max_duration

    if ratio < 0.25:  # Very quick
        return 2.0
    elif ratio < 0.5:  # Fast
        return 1.0
    elif ratio < 0.75:  # Normal
        return 0.0
    elif ratio < 1.0:  # Slow
        return -1.0
    else:  # Over limit
        return -5.0 * (ratio - 1.0)  # Escalating penalty
```

---

## Anti-Patterns

### ❌ Anti-Pattern 1: Unbounded Rewards

```python
# BAD: No upper limit
reward = profit * 1000  # Agent will hold forever chasing infinite reward
```

**Fix**: Always cap rewards
```python
reward = min(10.0, profit * 1000)
```

### ❌ Anti-Pattern 2: Zero-Sum Trap

```python
# BAD: Entry and exit rewards cancel out
reward_entry = -5
reward_exit = +5  # Net zero, agent doesn't care about profit!
```

**Fix**: Make exit rewards dependent on outcome
```python
reward_entry = -2  # Small entry cost
reward_exit = profit * 100  # Outcome-dependent
```

### ❌ Anti-Pattern 3: Sparse Rewards

```python
# BAD: Only reward at trade end
def calculate_reward(action):
    if action == EXIT:
        return profit * 100
    else:
        return 0  # No signal during position!
```

**Fix**: Provide continuous feedback
```python
def calculate_reward(action):
    if action == EXIT:
        return profit * 100
    elif action == HOLD and in_position:
        return current_profit * 10 - time_penalty  # Continuous signal
    else:
        return 0
```

### ❌ Anti-Pattern 4: Conflicting Objectives

```python
# BAD: Contradictory signals
reward = profit_score + win_rate_bonus

# profit_score encourages big wins
# win_rate_bonus encourages exiting early for high win%
# Agent gets confused!
```

**Fix**: Align objectives or use hierarchical rewards
```python
# Profit-adjusted win rate
if profit > 0:
    reward = profit * 100 + 5  # Bonus for winning
else:
    reward = profit * 150  # Larger penalty for losing
```

### ❌ Anti-Pattern 5: Scale Mismatch

```python
# BAD: Components on different scales
reward = (
    profit * 1000 +      # Range: -50 to +50
    drawdown * 0.01 +    # Range: -0.1 to 0
    timing * 10          # Range: -50 to +50
)
# Drawdown component is invisible!
```

**Fix**: Normalize components
```python
profit_norm = np.clip(profit * 100, -10, 10)
drawdown_norm = np.clip(drawdown * 100, -10, 10)
timing_norm = np.clip(timing, -10, 10)

reward = 0.5 * profit_norm + 0.3 * drawdown_norm + 0.2 * timing_norm
```

---

## Real-World Examples

### Example 1: Conservative Scalper

**Goal**: Many small wins, cut losses fast

```python
def conservative_scalper_reward(profit, duration, drawdown):
    # Heavy penalty for losses
    if profit < -0.005:
        return -20.0

    # Reward small quick profits
    if profit > 0 and duration < 20:
        return 10.0 + profit * 500
    elif profit > 0:
        return 5.0 + profit * 200

    # Penalize holding with drawdown
    if drawdown > 0.01:
        return -10.0

    return 0.0
```

**Result**: Win rate ~60%, avg profit ~0.5%, rare large losses

### Example 2: Trend Rider

**Goal**: Let winners run, cut losers fast

```python
def trend_rider_reward(profit, max_profit, duration):
    # Allow long holds for trending profits
    if profit > 0.02 and profit > max_profit * 0.7:
        return 15.0 + profit * 100  # Big reward for capturing trends

    # Cut losses quickly
    if profit < -0.01:
        return -15.0

    # Penalize early exits in profit
    if profit > 0 and profit < max_profit * 0.5:
        return -5.0  # You exited too early!

    return profit * 100
```

**Result**: Win rate ~45%, avg profit ~2%, some large winners

### Example 3: Mean Reversion

**Goal**: Exit at extremes, avoid momentum traps

```python
def mean_reversion_reward(profit, rsi, price_percentile):
    base_reward = profit * 100

    # Bonus for exiting at RSI extremes
    if rsi > 70 or rsi < 30:
        base_reward += 5.0

    # Bonus for exiting near price extremes
    if price_percentile > 0.9 or price_percentile < 0.1:
        base_reward += 5.0

    # Penalty for exiting in middle (mean)
    if 0.4 < price_percentile < 0.6:
        base_reward -= 3.0

    return base_reward
```

**Result**: High timing quality, works in ranging markets

### Example 4: Adaptive Risk (Advanced)

**Goal**: Adjust behavior based on market regime

```python
def adaptive_risk_reward(profit, volatility, volume_ratio, market_regime):
    """
    market_regime: 0 = low vol, 1 = normal, 2 = high vol
    """

    # Base profit score
    profit_score = np.tanh(profit * 50) * 10

    # Regime-specific adjustments
    if market_regime == 0:  # Low volatility
        # Encourage holding for larger moves
        if profit > 0.02:
            profit_score += 5.0

    elif market_regime == 1:  # Normal
        # Standard behavior (no adjustment)
        pass

    elif market_regime == 2:  # High volatility
        # Encourage quick exits
        if profit > 0.01:
            profit_score += 3.0  # Take profits earlier

        # Harsh penalty for drawdown in high vol
        if drawdown > 0.02:
            profit_score -= 10.0

    # Volume confirmation
    if volume_ratio > 1.5:  # Strong volume
        profit_score += 2.0
    elif volume_ratio < 0.5:  # Weak volume
        profit_score -= 3.0  # Bad liquidity

    return profit_score
```

---

## Tuning Guide

### Step 1: Start Simple

Begin with single-component reward:
```python
reward = profit * 100
```

**If agent learns basic profit-seeking**, proceed to Step 2.
**If agent doesn't learn**, you have environment issues (not reward issues).

### Step 2: Add Asymmetry

```python
if profit > 0:
    reward = profit * 100
else:
    reward = profit * 150  # Losses hurt more
```

**Tune the ratio** (100 vs 150) based on desired win rate.

### Step 3: Add Time Penalty

```python
reward = (profit * 100) - (duration * 0.05)
```

**Tune the coefficient** (0.05) based on desired position duration.

### Step 4: Add Drawdown Control

```python
reward = (profit * 100) - (duration * 0.05) - (max_drawdown * 200)
```

**Tune the coefficient** (200) based on risk tolerance.

### Step 5: Weight and Balance

```python
profit_score = calculate_profit_score(profit)
time_score = calculate_time_score(duration)
drawdown_score = calculate_drawdown_score(max_drawdown)

reward = 0.5 * profit_score + 0.3 * drawdown_score + 0.2 * time_score
```

**Tune weights** through grid search or Bayesian optimization.

---

## Testing Your Reward Function

```python
# Sanity check tests
def test_reward_function(reward_fn):
    """
    Every reward function should pass these tests.
    """

    # Test 1: Profitable exits should be positive
    assert reward_fn(profit=0.02) > 0

    # Test 2: Large losses should be heavily penalized
    assert reward_fn(profit=-0.05) < -5

    # Test 3: Asymmetry (losses hurt more)
    assert abs(reward_fn(profit=-0.02)) > abs(reward_fn(profit=0.02))

    # Test 4: Bounded (no infinity)
    assert abs(reward_fn(profit=1.0)) < 100

    # Test 5: Monotonic in profit (when other factors constant)
    assert reward_fn(profit=0.01) < reward_fn(profit=0.02)

    # Test 6: Time matters
    r1 = reward_fn(profit=0.01, duration=10)
    r2 = reward_fn(profit=0.01, duration=100)
    assert r1 > r2

    print("✓ All reward function tests passed!")
```

Run this test suite every time you modify your reward function.
