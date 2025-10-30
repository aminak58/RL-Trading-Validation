# FreqAI RL Optimizer

Professional skill for optimizing FreqAI Reinforcement Learning trading strategies, specifically designed for exit-focused RL models like MtfScalper_RL_Hybrid.

## Quick Start Diagnostics

Run this checklist when your RL model isn't performing:

```bash
# 1. Check training data quality
[ ] Sufficient data? (30+ days for 5m timeframe)
[ ] No major gaps in OHLCV data?
[ ] All features calculated correctly (no NaN)?
[ ] Price data normalized properly?

# 2. Check RL environment health
[ ] Episodes completing successfully?
[ ] Reward variance reasonable (-10 to +10 range)?
[ ] Actions distributed across all 5 options?
[ ] Position entry/exit balance maintained?

# 3. Check model convergence
[ ] Loss decreasing over time?
[ ] Policy entropy declining (but not to zero)?
[ ] Value function learning (explained variance > 0)?
[ ] No exploding gradients?

# 4. Check backtest results
[ ] Win rate > 45%?
[ ] Avg profit per trade > 0.5%?
[ ] Max position duration reasonable?
[ ] Exit quality better than random?
```

## Current Configuration Analysis

Your MtfScalper_RL_Hybrid setup:
- **Model**: PPO with 5-action space (Hold, Enter Long, Enter Short, Exit Long, Exit Short)
- **Architecture**: [256, 256, 128] - 3-layer MLP
- **Learning Rate**: 3e-4 (moderate)
- **Reward Weights**: Profit 35%, Drawdown 25%, Timing 20%, Risk/Reward 20%
- **Training**: 30 cycles, 2048 steps per rollout

## Hyperparameter Optimization Guide

### Priority 1: Reward Function Weights

Your current weights are balanced but may need tuning:

```python
# Current (balanced approach)
reward_weights = {
    "profit": 0.35,
    "drawdown_control": 0.25,
    "timing_quality": 0.20,
    "risk_reward_ratio": 0.20
}

# For aggressive profit seeking
reward_weights = {
    "profit": 0.50,
    "drawdown_control": 0.20,
    "timing_quality": 0.15,
    "risk_reward_ratio": 0.15
}

# For conservative risk management
reward_weights = {
    "profit": 0.25,
    "drawdown_control": 0.40,
    "timing_quality": 0.20,
    "risk_reward_ratio": 0.15
}
```

**Test protocol**: Run 3-month backtest with each configuration, compare Sharpe ratio.

### Priority 2: PPO Learning Parameters

```python
# Your current settings (good defaults)
learning_rate = 3e-4      # ✓ Standard
n_steps = 2048            # ✓ Good for 5m timeframe
batch_size = 64           # Consider increasing to 128-256
n_epochs = 10             # ✓ Standard
gamma = 0.99              # ✓ Standard

# If model is unstable:
learning_rate = 1e-4      # Reduce by 3x
clip_range = 0.1          # Reduce from 0.2

# If learning too slow:
learning_rate = 5e-4      # Increase slightly
n_epochs = 15             # More gradient steps
batch_size = 128          # Larger batches
```

### Priority 3: Network Architecture

```python
# Current: [256, 256, 128]
# Good for: ~20-30 features

# If you have 40+ features:
net_arch = [512, 256, 128]

# If experiencing overfitting:
net_arch = [128, 128]  # Smaller network

# For better feature extraction:
net_arch = [256, 256, 256, 128]  # Deeper network
```

**Rule of thumb**: First layer = 8-10x number of input features

### Priority 4: Entry/Exit Penalties

Critical for exit-focused models:

```python
# Current setup
entry_penalty_multiplier = 15.0     # Strong entry discouragement
classic_signal_reward = 2.0         # Modest entry reward

# If RL agent enters too much:
entry_penalty_multiplier = 25.0     # Stronger penalty
classic_signal_reward = 1.0         # Lower reward

# If agent never enters (only exits):
entry_penalty_multiplier = 10.0     # Lighter penalty
classic_signal_reward = 5.0         # Higher reward
```

## Feature Engineering Best Practices

### Exit-Specific Features (Your Current Implementation)

**Strong performers** (keep these):
```python
# Momentum indicators
"%-momentum_5"           # Short-term price velocity
"%-momentum_10"          # Medium-term trend
"%-acceleration"         # Rate of change in momentum

# Exit timing indicators
"%-dist_from_high_20"    # How far from recent peak?
"%-dist_from_low_20"     # How far from recent low?

# Risk indicators
"%-volume_ratio_5"       # Liquidity for exit
"%-spread_proxy"         # Slippage estimate
"%-risk_score"           # Composite risk measure

# Divergence signals
"%-bearish_divergence"   # Price up, RSI down
"%-bullish_divergence"   # Price down, RSI up
```

### Recommended Additions

```python
# 1. Profit deterioration indicator
dataframe["%-profit_erosion"] = (
    dataframe["close"].rolling(20).max() - dataframe["close"]
) / dataframe["close"]

# 2. Volume exhaustion
dataframe["%-volume_exhaustion"] = (
    dataframe["volume"].rolling(5).std() /
    dataframe["volume"].rolling(20).std()
)

# 3. Volatility regime
dataframe["%-volatility_regime"] = (
    dataframe["atr"] / dataframe["atr"].rolling(50).mean()
)

# 4. Time-in-position proxy (for state awareness)
# This requires tracking, but you can approximate:
dataframe["%-trend_age"] = (
    dataframe["ema_fast"] > dataframe["ema_slow"]
).rolling(50).sum()
```

### Features to Avoid

❌ **Don't use**:
- Future-looking indicators (introduces lookahead bias)
- Highly correlated features (causes multicollinearity)
- Categorical data without encoding
- Features with >30% NaN values

## Reward Function Design Patterns

### Pattern 1: Stepped Profit Rewards (Current)

Your implementation uses thresholds:

```python
if profit <= -0.05: return -5.0
elif profit <= -0.02: return -2.0
elif profit < 0.002: return 0.5 + profit * 100
elif profit < 0.02: return 3.0 + profit * 100
else: return 5.0 + profit * 50
```

✓ **Pros**: Clear learning targets, prevents exploitation
✗ **Cons**: Discontinuities can confuse learning

### Pattern 2: Smooth Exponential (Alternative)

```python
def smooth_profit_reward(profit):
    if profit >= 0:
        return 10 * (1 - np.exp(-profit * 50))
    else:
        return -10 * (1 - np.exp(profit * 30))
```

✓ **Pros**: Smooth gradients, better convergence
✗ **Cons**: Harder to interpret, needs tuning

### Pattern 3: Risk-Adjusted Returns

```python
def risk_adjusted_reward(profit, volatility):
    sharpe_like = profit / (volatility + 1e-6)
    return np.tanh(sharpe_like * 5) * 10
```

✓ **Pros**: Penalizes risky profits
✗ **Cons**: May discourage profitable but volatile exits

### Recommended: Hybrid Approach

Combine your stepped rewards with smoothing:

```python
def improved_profit_score(profit):
    # Base score (your current logic)
    if profit <= -0.05:
        base = -5.0
    elif profit < 0:
        base = profit * 20
    elif profit < 0.01:
        base = 1.0 + profit * 100
    elif profit < 0.03:
        base = 3.0 + profit * 80
    else:
        base = min(12.0, 5.0 + profit * 60)

    # Add smooth bonus for excellent exits
    if profit > 0.02:
        bonus = 2 * np.tanh((profit - 0.02) * 50)
        base += bonus

    return base
```

## Step-by-Step Debugging Workflow

### Stage 1: Environment Validation (10 min)

```bash
# Run minimal training to check environment
freqtrade backtesting \
  --strategy MtfScalper_RL_Hybrid \
  --timerange 20240101-20240107 \
  --freqai-train-enabled \
  --freqai-backtest-enabled

# Check logs for:
# - "RL environment created for X with Y data points"
# - "Training completed for PAIR"
# - No errors about NaN or invalid states
```

**Common issues**:
- `ValueError: No 'close' column found` → Check feature_engineering_standard()
- `NaN values in close price` → Add data cleaning in _create_env()
- `Environment crashes` → Reduce window_size from 30 to 20

### Stage 2: Reward Analysis (30 min)

```bash
# Create test script to evaluate rewards
python scripts/reward_backtest.py \
  --pair BTC/USDT:USDT \
  --timerange 20240101-20240131 \
  --config configs/config_rl_hybrid.json
```

**Look for**:
- Average episode reward > 0
- Reward variance reasonable (std < 5x mean)
- Exit rewards higher than hold rewards on average

**Red flags**:
- All rewards negative → Entry penalties too high
- Rewards always near zero → No learning signal
- Huge variance (std > 100) → Clip rewards or normalize

### Stage 3: Training Convergence (2-4 hours)

```bash
# Full training run with tensorboard
freqtrade backtesting \
  --strategy MtfScalper_RL_Hybrid \
  --timerange 20240101-20240401 \
  --freqai-train-enabled \
  --tensorboard

# Monitor in real-time
tensorboard --logdir ./tensorboard/
```

**Watch these metrics**:
- `rollout/ep_rew_mean`: Should increase over time
- `train/policy_loss`: Should decrease and stabilize
- `train/entropy_loss`: Should slowly decrease (not crash to zero)
- `train/explained_variance`: Should be > 0.5 after 50k steps

**Intervention points**:
- If loss explodes → Reduce learning rate by 5x
- If entropy → 0 quickly → Increase ent_coef from 0.01 to 0.05
- If explained_variance < 0 → Check reward function, may have wrong sign

### Stage 4: Backtest Quality Check (1 hour)

```bash
# Full backtest with trained model
freqtrade backtesting \
  --strategy MtfScalper_RL_Hybrid \
  --timerange 20240401-20240701 \
  --freqai-backtest-enabled

# Analyze results
python scripts/feature_importance.py \
  --results user_data/backtest_results/backtest-result.json
```

**Success criteria**:
- Win rate: 45-65% (exit optimization focus)
- Avg profit/trade: > 0.5%
- Max position duration: < 200 candles (fits your 300 limit)
- Profit factor: > 1.3
- Sharpe ratio: > 0.5

**Failure patterns**:
- Win rate < 40% → Exits too early, reduce timing_quality weight
- Win rate > 70% but low profit → Holding losers, increase drawdown_control
- Very short positions → Entry/exit penalties imbalanced
- Never exits → Exit rewards too weak, increase magnitude

### Stage 5: Walk-Forward Validation (Critical!)

```bash
# Test on completely unseen data
freqtrade backtesting \
  --strategy MtfScalper_RL_Hybrid \
  --timerange 20240701-20241001 \
  --freqai-backtest-enabled

# Compare metrics to training period
# Acceptable degradation: 20-30%
# If performance drops > 50%, you have overfitting
```

## Model Evaluation Metrics

### Beyond Profit: RL-Specific Metrics

```python
# 1. Exit Quality Score
exit_quality = (profit_at_exit / max_profit_during_trade)
# Target: > 0.70 (exiting at 70%+ of max profit seen)

# 2. Decision Consistency
action_entropy = -sum(action_probs * log(action_probs))
# Target: 0.5 - 1.5 (not random, not deterministic)

# 3. Risk-Adjusted Returns
calmar_ratio = annual_return / max_drawdown
# Target: > 2.0

# 4. Position Duration Efficiency
avg_duration_winners = mean(duration[profit > 0])
avg_duration_losers = mean(duration[profit < 0])
# Target: Winners last longer than losers

# 5. Adaptive Behavior
correlation(volatility, position_duration)
# Target: Negative correlation (shorter holds in high volatility)
```

### Comparing Configurations

Use this table template:

```
Configuration    | Sharpe | Calmar | Win% | Avg Profit | Max DD | Exit Quality
-----------------|--------|--------|------|------------|--------|-------------
Baseline         | 0.42   | 1.8    | 48%  | 0.8%       | 12%    | 0.65
Profit-focused   | 0.51   | 2.1    | 52%  | 1.1%       | 15%    | 0.72
Risk-focused     | 0.38   | 2.5    | 44%  | 0.6%       | 8%     | 0.61
Timing-focused   | 0.46   | 1.9    | 50%  | 0.9%       | 11%    | 0.78
```

**Decision criteria**:
- **For live trading**: Choose highest Calmar ratio (risk-adjusted)
- **For competitions**: Choose highest Sharpe ratio
- **For scalping**: Choose highest Exit Quality + Win Rate

## Quick Reference Commands

```bash
# Analyze training logs
python scripts/analyze_training.py \
  --tensorboard-dir ./tensorboard/ \
  --output-dir ./analysis/

# Calculate feature importance
python scripts/feature_importance.py \
  --model-dir user_data/models/MtfScalperRL_v1/ \
  --pair BTC/USDT:USDT

# Test reward function changes
python scripts/reward_backtest.py \
  --config configs/config_rl_hybrid.json \
  --reward-config configs/reward_test.json \
  --compare

# Hyperparameter optimization
python scripts/hyperparameter_scanner.py \
  --strategy MtfScalper_RL_Hybrid \
  --params learning_rate,n_steps,gamma \
  --ranges 1e-5:1e-3,1024:4096,0.95:0.99 \
  --trials 20
```

## Next Steps Based on Performance

### If Win Rate < 40%
1. Increase `classic_signal_reward` from 2.0 to 5.0
2. Add momentum confirmation to exit conditions
3. Check if position entries are aligned with market regime

### If Avg Profit < 0.5%
1. Increase `profit` weight from 0.35 to 0.45
2. Reduce `timing_quality` weight to allow longer holds
3. Check if stop loss is triggering too early

### If Max Drawdown > 15%
1. Increase `drawdown_control` weight from 0.25 to 0.35
2. Add emergency exit at -2% instead of -3%
3. Reduce position sizing (risk_per_trade from 0.02 to 0.015)

### If Exit Quality < 0.60
1. Add profit deterioration detection features
2. Increase `timing_quality` weight to 0.30
3. Review `_calculate_timing_score()` thresholds

## References

See detailed guides in `references/`:
- `reward-patterns.md` - Comprehensive reward function designs
- `hyperparameter-tuning.md` - Deep dive into PPO parameters
- `feature-engineering.md` - Exit-focused feature catalog
- `troubleshooting.md` - Common issues and solutions
- `evaluation-metrics.md` - Performance assessment frameworks

## Scripts

Utility scripts in `scripts/`:
- `analyze_training.py` - Parse tensorboard logs and generate reports
- `feature_importance.py` - SHAP analysis for RL models
- `reward_backtest.py` - Simulate different reward functions
- `hyperparameter_scanner.py` - Automated optimization grid search
