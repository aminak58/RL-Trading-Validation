# Troubleshooting Guide

Common issues with FreqAI RL models and their solutions.

## Quick Diagnostic Index

- [Training Issues](#training-issues)
- [Environment Issues](#environment-issues)
- [Performance Issues](#performance-issues)
- [Data Issues](#data-issues)
- [Integration Issues](#integration-issues)

---

## Training Issues

### Issue 1: Loss Explodes to NaN

**Symptoms**:
```
Step 5000: loss = 2.45
Step 5100: loss = 12.87
Step 5200: loss = NaN
Training crashed
```

**Root Causes**:

1. **Learning rate too high**
   ```python
   # Solution: Reduce by 5-10x
   learning_rate = 1e-4  # Was 5e-4 or 1e-3
   ```

2. **Reward values unbounded**
   ```python
   # Problem in your calculate_reward():
   return profit * 10000  # Can explode to +50000

   # Solution: Clip rewards
   reward = np.clip(profit * 100, -20, 20)
   ```

3. **Gradient explosion**
   ```python
   # Solution: Reduce max_grad_norm
   max_grad_norm = 0.5  # Was 1.0 (default)
   ```

4. **NaN in observations**
   ```python
   # Problem: Division by zero or invalid operations
   profit = (close - entry) / entry  # entry could be 0!

   # Solution: Add epsilon
   profit = (close - entry) / (entry + 1e-10)
   ```

**Quick Fix** (apply all):
```python
learning_rate = 1e-4
max_grad_norm = 0.5

# In calculate_reward():
reward = np.clip(reward, -20, 20)

# In feature engineering:
dataframe = dataframe.replace([np.inf, -np.inf], 0).fillna(0)
```

---

### Issue 2: Model Not Learning (Flat Rewards)

**Symptoms**:
```
Episode reward: 0.32
Episode reward: 0.28
Episode reward: 0.31
... (no improvement after 50k steps)
```

**Diagnosis Steps**:

**Step 1**: Check reward scale
```python
# Print reward statistics during training
print(f"Reward mean: {np.mean(rewards)}")
print(f"Reward std: {np.std(rewards)}")

# Expected:
# mean: -5 to +5
# std: 2 to 10

# If mean ≈ 0 and std < 1:
# Problem: Rewards too small/sparse
```

**Step 2**: Verify reward function fires
```python
# Add logging in calculate_reward()
def calculate_reward(self, action):
    reward = ...

    # Debug print every 100 steps
    if self._current_tick % 100 == 0:
        print(f"Action: {action}, Reward: {reward:.2f}")

    return reward

# If always seeing same action/reward:
# Problem: Agent found local minimum
```

**Solutions**:

1. **Increase reward magnitude**
   ```python
   # Multiply all rewards by 10x
   profit_score = profit * 1000  # Was profit * 100
   ```

2. **Add exploration bonus**
   ```python
   ent_coef = 0.05  # Was 0.01
   # Forces more diverse actions
   ```

3. **Check if entry/exit balanced**
   ```python
   # Count actions during training
   action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
   # If one action dominates (>90%), reward structure is broken
   ```

4. **Verify environment resets**
   ```python
   def reset(self):
       # Must reset ALL state variables
       self.position_start_price = None
       self.position_start_step = None
       self.max_profit_seen = 0.0
       # If missing: agent gets confused between episodes
   ```

---

### Issue 3: Policy Entropy Crashes to Zero

**Symptoms**:
```
Step 1000: entropy = 1.45
Step 5000: entropy = 0.82
Step 8000: entropy = 0.12
Step 10000: entropy = 0.001  # Agent became deterministic!
```

**Problem**: Agent stopped exploring, may have found bad local optimum.

**Solutions**:

1. **Increase entropy coefficient**
   ```python
   ent_coef = 0.05  # Was 0.01
   # Or even higher: 0.1 for very sparse rewards
   ```

2. **Add entropy bonus to reward**
   ```python
   # In environment:
   action_probs = self.get_action_probabilities()  # From policy
   entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

   bonus = entropy * 0.1  # Reward diverse actions
   total_reward = base_reward + bonus
   ```

3. **Restart with lower learning rate**
   ```python
   # Fast convergence → low exploration
   learning_rate = 1e-4  # Was 3e-4
   ```

---

### Issue 4: Value Function Not Learning

**Symptoms**:
```
explained_variance: -0.15  # Should be > 0!
value_loss: Not decreasing
```

**Problem**: Value function can't predict returns.

**Solutions**:

1. **Increase value function weight**
   ```python
   vf_coef = 1.0  # Was 0.5
   # Give value function more importance
   ```

2. **Check reward consistency**
   ```python
   # In calculate_reward(), make sure:
   # - Same state → same reward
   # - Rewards don't depend on random factors

   # BAD:
   reward = profit + np.random.randn()  # Random noise

   # GOOD:
   reward = profit * 100  # Deterministic
   ```

3. **Verify gamma setting**
   ```python
   # If episodes very long:
   gamma = 0.995  # Was 0.99
   # Helps value function account for distant rewards
   ```

4. **Separate policy and value networks**
   ```python
   # Advanced: custom PPO config
   policy_kwargs = dict(
       net_arch=[dict(pi=[256, 256], vf=[256, 256])]
       # Separate networks learn better
   )
   ```

---

## Environment Issues

### Issue 5: Environment Crashes Mid-Episode

**Symptoms**:
```
Episode step 150
Episode step 151
ValueError: index 152 out of bounds
Environment crashed
```

**Root Causes**:

1. **Episode extends beyond data**
   ```python
   # In step():
   self._current_tick += 1

   # Problem: _current_tick exceeds len(dataframe)

   # Solution: Add boundary check
   if self._current_tick >= len(self.df) - 1:
       done = True
       return obs, reward, done, info
   ```

2. **Window size too large**
   ```python
   # Your current setting:
   window_size = 30

   # If dataframe has <30 rows at start:
   # → Index error

   # Solution: Check in __init__
   if len(self.df) < self.window_size:
       self.window_size = len(self.df) - 1
   ```

3. **Invalid action**
   ```python
   # In step(action):
   if action == Actions.Long_exit and self._position != 1:
       # Trying to exit long when not in long position
       return obs, -10.0, False, {}  # Penalty, don't crash
   ```

---

### Issue 6: "No 'close' Column Found"

**Symptoms**:
```
ValueError: No 'close' column found in dataframe for RL environment
```

**Your Code** (MtfScalperRLModel.py:488-502):
```python
# Already handles this well!
close_col = None
if "close" in df.columns:
    close_col = "close"
elif "%-close" in df.columns:
    close_col = "%-close"

if close_col != "close":
    df = df.copy()
    df["close"] = df[close_col]
```

**Additional Check**:
```python
# In feature_engineering_standard(), ensure:
dataframe["%-raw_close"] = dataframe["close"]
dataframe["%-raw_open"] = dataframe["open"]
# ... etc

# These are required for RL environment price access
```

---

### Issue 7: NaN Values in Environment State

**Symptoms**:
```
obs: [1.23, 0.45, nan, 0.78, ...]
Policy receives NaN → outputs NaN → training fails
```

**Solutions**:

1. **Fill NaN in data preprocessing** (your current approach):
   ```python
   # In _create_env():
   df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')

   # For feature columns:
   feature_columns = [col for col in df_clean.columns if col.startswith("%")]
   for col in feature_columns:
       df_clean[col] = df_clean[col].fillna(0)
   ```

2. **Check for inf values**:
   ```python
   df_clean = df_clean.replace([np.inf, -np.inf], 0)
   ```

3. **Add validation before returning observation**:
   ```python
   def _get_observation(self):
       obs = ...  # Build observation

       # Validate
       if np.any(np.isnan(obs)):
           logger.error(f"NaN in observation at step {self._current_tick}")
           obs = np.nan_to_num(obs, nan=0.0)

       return obs
   ```

---

## Performance Issues

### Issue 8: Win Rate < 40%

**Problem**: Agent exits too early or holds losers too long.

**Diagnosis**:
```python
# Analyze exit decisions
wins = trades[trades['profit'] > 0]
losses = trades[trades['profit'] < 0]

print(f"Avg winner duration: {wins['duration'].mean()}")
print(f"Avg loser duration: {losses['duration'].mean()}")

# If loser_duration > winner_duration:
# Problem: Holding losers hoping for recovery
```

**Solutions**:

1. **Increase loss penalty asymmetry**:
   ```python
   # In _calculate_profit_score():
   if profit <= -0.02:
       return -5.0  # Was -2.0 (harsher)
   ```

2. **Add time penalty for losing positions**:
   ```python
   def _calculate_holding_reward(self, current_profit):
       if current_profit < 0 and position_duration > 50:
           return -5.0  # Force exit of losers
   ```

3. **Check entry penalty not too high**:
   ```python
   # If agent never enters:
   entry_penalty_multiplier = 10.0  # Was 15.0
   classic_signal_reward = 3.0  # Was 2.0
   ```

---

### Issue 9: Average Profit < 0.5%

**Problem**: Agent taking profits too quickly.

**Solutions**:

1. **Increase profit thresholds in reward**:
   ```python
   # In _calculate_profit_score():
   elif profit < 0.01:  # Was 0.005
       return 1.0 + profit * 100

   # Encourages holding for larger gains
   ```

2. **Reduce timing_quality weight**:
   ```python
   reward_weights = {
       "profit": 0.45,  # Was 0.35
       "timing_quality": 0.10,  # Was 0.20
       # ...
   }
   ```

3. **Add holding bonus for profitable positions**:
   ```python
   if current_profit > 0.01 and position_duration < 100:
       return 0.5  # Small reward for patience
   ```

---

### Issue 10: Max Drawdown > 15%

**Problem**: Agent not respecting risk limits.

**Solutions**:

1. **Increase drawdown_control weight**:
   ```python
   reward_weights = {
       "drawdown_control": 0.40,  # Was 0.25
       "profit": 0.30,  # Reduced
       # ...
   }
   ```

2. **Harsher drawdown penalties**:
   ```python
   # In _calculate_drawdown_score():
   if drawdown > 0.02:
       return -10.0 * drawdown  # Was -5.0
   ```

3. **Add circuit breaker in custom_exit()**:
   ```python
   # In strategy:
   if current_profit < -0.025:  # -2.5%
       return "circuit_breaker"
   ```

---

## Data Issues

### Issue 11: Insufficient Training Data

**Symptoms**:
```
Training on 5,000 candles
Poor generalization to validation
```

**Solution**:
```python
# For 5m timeframe:
# Minimum: 1 month = 8,640 candles
# Recommended: 3 months = 25,920 candles
# Optimal: 6 months = 51,840 candles

# Check your config:
"train_period_days": 90,  # 3 months
```

---

### Issue 12: Data Gaps

**Symptoms**:
```
2024-01-15 10:00
2024-01-15 10:05
2024-01-15 15:30  # Gap of 5+ hours!
```

**Solution**:
```python
# Check for gaps before training
def check_data_gaps(df, timeframe='5m'):
    expected_delta = pd.Timedelta(timeframe)
    gaps = df.index.to_series().diff() > expected_delta * 2

    if gaps.any():
        print(f"Found {gaps.sum()} gaps in data!")
        print(df[gaps].head())

        # Fill gaps
        df = df.asfreq(expected_delta, method='ffill')

    return df

dataframe = check_data_gaps(dataframe)
```

---

## Integration Issues

### Issue 13: FreqAI Not Providing Predictions

**Symptoms**:
```python
# In populate_exit_trend():
if "&-action" in dataframe.columns:
    # This never executes!
```

**Solutions**:

1. **Check FreqAI enabled**:
   ```python
   # In strategy __init__:
   self.freqai_enabled = True

   # In populate_indicators():
   if hasattr(self, 'freqai') and self.freqai:
       self.freqai.start(dataframe, metadata, self)
   ```

2. **Verify model trained**:
   ```bash
   # Check for model files:
   ls user_data/models/MtfScalperRL_v1/

   # Should see:
   # - trained_model.zip
   # - data_kitchen_*
   ```

3. **Check predictions format**:
   ```python
   # In predict(), must return:
   pred_df = DataFrame({
       "&-action": actions,  # Note the & prefix!
       "&-action_confidence": confidences
   }, index=filtered_df.index)
   ```

---

### Issue 14: Backtest vs Live Mismatch

**Symptoms**:
```
Backtest: +15% profit
Live trading: -5% loss
```

**Root Causes**:

1. **Lookahead bias in features**:
   ```python
   # BAD:
   dataframe["%-future_high"] = dataframe["high"].shift(-5)

   # Audit all features for .shift(negative)
   ```

2. **Unrealistic assumptions**:
   ```python
   # Backtest assumes instant execution
   # Live has slippage, delays

   # Add slippage buffer:
   fee = 0.001  # 0.1%
   slippage = 0.0005  # 0.05%
   total_cost = fee + slippage  # 0.15% per trade
   ```

3. **Training/test data overlap**:
   ```python
   # Ensure walk-forward validation:
   "train_period_days": 30,
   "backtest_period_days": 7,
   # Training: Day 1-30
   # Testing: Day 31-37
   # No overlap!
   ```

---

## Emergency Fixes

### Quick Fix Checklist

If training completely broken, apply ALL of these:

```python
# 1. Conservative learning rate
learning_rate = 1e-4

# 2. Clip rewards
def calculate_reward(self, action):
    reward = ...
    return np.clip(reward, -20, 20)

# 3. Fill NaN
dataframe = dataframe.fillna(0).replace([np.inf, -np.inf], 0)

# 4. Increase exploration
ent_coef = 0.05

# 5. Simpler architecture
net_arch = [128, 128]

# 6. Larger batch size (stability)
batch_size = 128

# 7. Fewer epochs (prevent overfitting)
n_epochs = 5

# 8. Check episode termination
def step(self, action):
    if self._current_tick >= len(self.df) - 10:
        done = True
```

After applying, retrain and monitor tensorboard closely.

---

## Debugging Tools

### Tool 1: Reward Distribution Analysis

```python
# Add to your training callback:
class RewardAnalysisCallback(BaseCallback):
    def _on_rollout_end(self):
        rewards = self.locals['rollout_buffer'].rewards
        print(f"Reward stats:")
        print(f"  Mean: {np.mean(rewards):.2f}")
        print(f"  Std: {np.std(rewards):.2f}")
        print(f"  Min: {np.min(rewards):.2f}")
        print(f"  Max: {np.max(rewards):.2f}")
        print(f"  Zeros: {(rewards == 0).sum() / len(rewards) * 100:.1f}%")
```

### Tool 2: Action Distribution Monitor

```python
class ActionMonitorCallback(BaseCallback):
    def _on_step(self):
        if self.n_calls % 1000 == 0:
            # Get recent actions
            actions = self.training_env.get_attr('recent_actions')
            action_counts = np.bincount(actions, minlength=5)
            action_pcts = action_counts / action_counts.sum() * 100

            print("Action distribution:")
            print(f"  Hold: {action_pcts[0]:.1f}%")
            print(f"  Enter Long: {action_pcts[1]:.1f}%")
            print(f"  Enter Short: {action_pcts[2]:.1f}%")
            print(f"  Exit Long: {action_pcts[3]:.1f}%")
            print(f"  Exit Short: {action_pcts[4]:.1f}%")
```

### Tool 3: Episode Analyzer

```python
def analyze_episode(env, model, deterministic=True):
    """
    Step through single episode and print decisions.
    """
    obs = env.reset()
    done = False
    episode_rewards = []
    episode_actions = []

    while not done:
        action, _states = model.predict(obs, deterministic=deterministic)

        obs, reward, done, info = env.step(action)

        episode_rewards.append(reward)
        episode_actions.append(action)

        # Print every 10 steps
        if len(episode_actions) % 10 == 0:
            print(f"Step {len(episode_actions)}: "
                  f"Action={action}, Reward={reward:.2f}, "
                  f"Position={env._position}")

    print(f"\nEpisode Summary:")
    print(f"  Total reward: {sum(episode_rewards):.2f}")
    print(f"  Avg reward: {np.mean(episode_rewards):.2f}")
    print(f"  Actions: {np.bincount(episode_actions)}")
```

Use these tools during development to understand what's happening.

---

## When to Ask for Help

If you've tried everything above and still stuck, gather this info:

```
1. Training logs (last 1000 lines)
2. Tensorboard screenshots (loss, reward, entropy)
3. Config file
4. Reward function code
5. Sample of training data (first/last 100 rows)
6. Action distribution stats
7. One episode walkthrough with analyze_episode()

Post in Freqtrade Discord #freqai channel
```

Remember: 80% of RL issues are reward function design!
