# Hyperparameter Tuning Guide

Complete reference for tuning PPO hyperparameters in FreqAI RL models.

## Table of Contents

1. [Parameter Overview](#parameter-overview)
2. [Priority-Based Tuning](#priority-based-tuning)
3. [Parameter Interactions](#parameter-interactions)
4. [Optimization Strategies](#optimization-strategies)
5. [Troubleshooting by Symptom](#troubleshooting-by-symptom)

---

## Parameter Overview

### Your Current Configuration

```python
# MtfScalperRLModel defaults
learning_rate = 3e-4        # Adam optimizer step size
n_steps = 2048              # Rollout buffer size
batch_size = 64             # Minibatch size for training
n_epochs = 10               # Gradient descent epochs per rollout
gamma = 0.99                # Discount factor
gae_lambda = 0.95           # GAE smoothing parameter
clip_range = 0.2            # PPO clipping epsilon
vf_coef = 0.5               # Value function loss coefficient
ent_coef = 0.01             # Entropy bonus coefficient
net_arch = [256, 256, 128]  # Neural network architecture
```

### Parameter Categories

**Critical (tune first)**:
- `learning_rate` - Most impactful on training
- `n_steps` - Affects sample efficiency
- `net_arch` - Model capacity

**Important (tune second)**:
- `batch_size` - Training stability
- `n_epochs` - Learning per rollout
- `gamma` - Long-term planning

**Fine-tuning (tune last)**:
- `clip_range` - Policy update constraint
- `gae_lambda` - Advantage estimation
- `ent_coef` - Exploration bonus
- `vf_coef` - Value vs policy balance

---

## Priority-Based Tuning

### Priority 1: Learning Rate

**Default**: `3e-4` (0.0003)

The single most important hyperparameter.

#### Diagnosis

**Too High** (learning rate > 5e-4 for your setup):
```
Symptoms:
- Loss explodes (NaN values)
- Policy entropy crashes to zero quickly
- Wildly inconsistent episode rewards
- Training diverges after initial progress

What's happening:
- Policy updates are too large
- Agent "forgets" previous learning
- Value function overfits to recent data
```

**Too Low** (learning rate < 1e-4):
```
Symptoms:
- Loss decreases very slowly
- Takes 100k+ steps to see improvement
- Plateau early without reaching good performance
- Training feels "stuck"

What's happening:
- Policy barely updating each step
- Can't escape local minima
- Wastes computation time
```

**Just Right** (1e-4 to 5e-4):
```
Indicators:
- Steady improvement in episode rewards
- Loss decreases smoothly (no spikes)
- Policy entropy decays gradually
- Convergence within 50k-100k steps
```

#### Tuning Strategy

```python
# Conservative approach (recommended for first run)
learning_rate = 1e-4

# Standard approach (your current)
learning_rate = 3e-4

# Aggressive approach (if training is too slow)
learning_rate = 5e-4

# Adaptive approach (advanced)
learning_rate = lambda progress: 5e-4 * (1 - progress)  # Decay over time
```

#### Empirical Guidelines

| Scenario | Recommended LR | Reasoning |
|----------|---------------|-----------|
| High-frequency (1m-5m) | 1e-4 to 3e-4 | More noisy, need stability |
| Medium-frequency (15m-1h) | 3e-4 to 5e-4 | Cleaner signals, can go faster |
| Large network (512+ nodes) | 1e-4 to 2e-4 | More parameters, easier to overfit |
| Small network (128 nodes) | 3e-4 to 7e-4 | Fewer parameters, need stronger updates |
| Sparse rewards | 1e-4 to 2e-4 | Harder problem, be conservative |
| Dense rewards | 3e-4 to 5e-4 | Strong signal, can learn faster |

### Priority 2: Network Architecture

**Default**: `[256, 256, 128]`

Determines model capacity and overfitting risk.

#### Sizing Rules

```python
# Rule of thumb: first layer = 8-12x input features

# You have ~25-30 exit-focused features
# → First layer should be 200-300 nodes

# Recommended architectures:

# Conservative (less overfitting)
net_arch = [128, 128]
# Best for: <20 features, short training (1 month data)

# Balanced (your current)
net_arch = [256, 256, 128]
# Best for: 20-40 features, medium training (3 months data)

# Large (more capacity)
net_arch = [512, 256, 128]
# Best for: 40+ features, long training (6+ months data)

# Deep (complex patterns)
net_arch = [256, 256, 256, 128]
# Best for: Multi-market strategies, regime detection needed

# Wide (parallel processing)
net_arch = [512, 512]
# Best for: Many uncorrelated features
```

#### Empirical Test

```python
def estimate_required_capacity(n_features, n_actions, avg_episode_length):
    """
    Rough estimate of first layer size.
    """
    # Minimum: enough to encode all features
    min_size = n_features * 4

    # Recommended: room for feature interactions
    rec_size = n_features * 10

    # Maximum: avoid overfitting
    max_size = n_features * 20

    print(f"Recommended first layer: {rec_size} nodes")
    print(f"Range: {min_size} - {max_size}")

    return rec_size

# For your setup:
estimate_required_capacity(n_features=30, n_actions=5, avg_episode_length=100)
# Output: Recommended first layer: 300 nodes
# Range: 120 - 600
```

#### Overfitting Check

```python
# Compare training vs validation loss
training_return = 15.2%
validation_return = 8.1%

gap = (training_return - validation_return) / training_return
# gap = 0.47 (47% degradation)

if gap > 0.3:
    print("❌ Overfitting detected!")
    print("Action: Reduce network size or add regularization")
    # Try: net_arch = [128, 128]

elif gap < 0.1:
    print("✓ Good generalization")
    print("Consider: Increasing capacity for better performance")
    # Try: net_arch = [256, 256, 256]
```

### Priority 3: Rollout Steps (n_steps)

**Default**: `2048`

How many environment steps before updating policy.

#### Impact

- **Larger n_steps** (4096-8192):
  - More stable gradients
  - Better for high variance environments
  - Slower feedback loop
  - Higher memory usage

- **Smaller n_steps** (512-1024):
  - Faster policy updates
  - More responsive to new data
  - Noisier gradients
  - Lower memory usage

#### Tuning by Timeframe

```python
# Your 5m timeframe with avg position duration ~50-100 candles

# Too small (not recommended)
n_steps = 512  # Only ~2.5 completed episodes
# Problem: Not enough data per update

# Good
n_steps = 2048  # Your current, ~10 completed episodes
# Sweet spot for 5m timeframe

# Large (if you have memory)
n_steps = 4096  # ~20 completed episodes
# Better gradient estimates, but 2x slower

# Formula:
n_steps = avg_episode_length * (10 to 20)
# For you: 100 * 15 = 1500-2000 ≈ 2048 ✓
```

#### Interaction with Batch Size

```python
# PPO requirement: n_steps must be divisible by batch_size
# Your setup: 2048 / 64 = 32 minibatches ✓

# Good combinations:
(n_steps=2048, batch_size=64)   # 32 batches
(n_steps=2048, batch_size=128)  # 16 batches
(n_steps=4096, batch_size=128)  # 32 batches

# Bad combinations:
(n_steps=2048, batch_size=100)  # 20.48 batches (fractional!)
```

---

## Parameter Interactions

### Learning Rate × Network Size

```python
# Large network needs smaller LR
if sum(net_arch) > 600:
    learning_rate = 1e-4
else:
    learning_rate = 3e-4

# Why: More parameters = easier to overfit with large updates
```

### Batch Size × n_epochs

```python
# Total gradient steps per rollout = n_epochs × (n_steps / batch_size)

# Your current:
total_updates = 10 × (2048 / 64) = 320 gradient steps per rollout

# Conservative (stable training):
batch_size = 128
n_epochs = 8
# total = 8 × (2048 / 128) = 128 updates

# Aggressive (faster learning):
batch_size = 64
n_epochs = 15
# total = 15 × (2048 / 64) = 480 updates
```

**Rule**: More updates → faster learning, but risk overfitting to current rollout.

### Gamma × Average Episode Length

```python
# Gamma determines time horizon for rewards

# For short episodes (< 50 steps):
gamma = 0.95  # Horizon ~20 steps

# For medium episodes (50-150 steps):
gamma = 0.99  # Horizon ~100 steps (your current)

# For long episodes (> 200 steps):
gamma = 0.995  # Horizon ~200 steps

# Effective horizon = 1 / (1 - gamma)
# gamma=0.99 → horizon = 100 steps = 8.3 hours at 5m
# This matches your max_position_duration = 300 candles
```

### Clip Range × Learning Rate

```python
# PPO clips policy updates to prevent large changes

# Conservative (learning_rate=1e-4):
clip_range = 0.3  # Allow larger updates (LR is already small)

# Standard (learning_rate=3e-4):
clip_range = 0.2  # Your current

# Aggressive (learning_rate=5e-4):
clip_range = 0.1  # Restrict updates (LR is already large)

# Rule: clip_range × learning_rate ≈ constant
```

---

## Optimization Strategies

### Strategy 1: Grid Search (Systematic)

```python
# Define search space
param_grid = {
    'learning_rate': [1e-4, 3e-4, 5e-4],
    'n_steps': [1024, 2048, 4096],
    'batch_size': [64, 128, 256],
    'net_arch': [[128, 128], [256, 256, 128], [512, 256]],
}

# Expected trials: 3 × 3 × 3 × 3 = 81 combinations
# Time per trial: ~4 hours
# Total time: 324 hours (13.5 days)

# Practical: Test 10-15 most promising combinations
```

### Strategy 2: Random Search (Efficient)

```python
# Sample from distributions
param_distributions = {
    'learning_rate': scipy.stats.loguniform(1e-5, 1e-3),
    'n_steps': [1024, 2048, 4096],
    'batch_size': [64, 128, 256],
    'n_epochs': scipy.stats.randint(5, 20),
    'gamma': scipy.stats.uniform(0.95, 0.05),  # 0.95 to 0.999
}

# Run 20-30 random trials
# Often finds good configs faster than grid search
```

### Strategy 3: Bayesian Optimization (Advanced)

```python
# Use scripts/hyperparameter_scanner.py (included in skill)

# Iteratively samples parameter space
# Focuses on promising regions
# Typically needs 15-25 trials to converge

# Example command:
python scripts/hyperparameter_scanner.py \
    --strategy MtfScalper_RL_Hybrid \
    --params learning_rate,n_steps,gamma \
    --ranges 1e-5:1e-3,1024:4096,0.95:0.999 \
    --trials 20 \
    --method bayesian
```

### Strategy 4: Successive Halving (Fast)

```python
# Train many configs for short time
# Eliminate bottom 50%
# Continue top 50% for longer
# Repeat until 1-3 configs remain

# Schedule:
# Round 1: 64 configs × 5k steps = 320k steps total
# Round 2: 32 configs × 10k steps = 320k steps
# Round 3: 16 configs × 20k steps = 320k steps
# Round 4: 8 configs × 40k steps = 320k steps
# Round 5: 4 configs × 80k steps = 320k steps
# Final: 2 configs × 200k steps = 400k steps

# Total: 1.98M steps (comparable to grid search)
# But finds good config early (after Round 2-3)
```

### Strategy 5: Two-Stage Tuning (Practical)

**Stage 1: Coarse Search** (2-3 days)
```python
# Test 3-5 learning rates
learning_rates = [1e-4, 3e-4, 5e-4]

# With fixed architecture
net_arch = [256, 256, 128]

# Short training (20k steps)
# Pick best LR based on episode reward trend
```

**Stage 2: Fine Tuning** (1-2 weeks)
```python
# Use best LR from Stage 1
best_lr = 3e-4

# Test architecture variations
architectures = [
    [128, 128],
    [256, 256, 128],
    [256, 256, 256],
    [512, 256, 128]
]

# Full training (100k steps)
# Evaluate on walk-forward validation
```

---

## Troubleshooting by Symptom

### Symptom: Training is Unstable

**Loss spikes, episode rewards fluctuate wildly**

```python
# Try these in order:

# 1. Reduce learning rate
learning_rate = 1e-4  # From 3e-4

# 2. Increase batch size
batch_size = 128  # From 64

# 3. Reduce entropy coefficient
ent_coef = 0.005  # From 0.01

# 4. Clip rewards
# In your environment's calculate_reward():
reward = np.clip(reward, -20, 20)

# 5. Normalize advantages
# This is PPO default, but verify it's enabled
normalize_advantage = True
```

### Symptom: Training is Too Slow

**No improvement after 50k steps**

```python
# Try these in order:

# 1. Increase learning rate
learning_rate = 5e-4  # From 3e-4

# 2. Increase entropy coefficient (more exploration)
ent_coef = 0.03  # From 0.01

# 3. More gradient updates per rollout
n_epochs = 15  # From 10

# 4. Check reward function
# Make sure rewards are non-zero!
# Print mean/std of rewards during training
```

### Symptom: Good Training, Poor Validation

**Classic overfitting**

```python
# Try these in order:

# 1. Reduce network size
net_arch = [128, 128]  # From [256, 256, 128]

# 2. Add weight decay
policy_kwargs = dict(
    net_arch=net_arch,
    optimizer_kwargs=dict(
        weight_decay=1e-4  # L2 regularization
    )
)

# 3. Reduce training epochs
n_epochs = 5  # From 10

# 4. Early stopping
# Use EvalCallback with best_model_save_path

# 5. Dropout (requires custom network)
# Advanced: modify policy to add dropout layers
```

### Symptom: Agent Only Takes One Action

**Policy collapses, e.g., always HOLD**

```python
# Try these in order:

# 1. Increase entropy bonus
ent_coef = 0.05  # From 0.01

# 2. Check reward function balance
# Are EXIT actions properly rewarded?
# Print action distribution during training

# 3. Increase clip range (allow more exploration)
clip_range = 0.3  # From 0.2

# 4. Add action-based exploration bonus
# In environment:
if action == rare_action:
    reward += 0.5  # Encourage diversity

# 5. Initialize policy with uniform distribution
# This is usually default, but verify
```

### Symptom: Value Function Not Learning

**Explained variance < 0 or near zero**

```python
# Try these in order:

# 1. Increase value function coefficient
vf_coef = 1.0  # From 0.5

# 2. Separate networks for policy and value
# (Requires custom PPO configuration)

# 3. Check reward scale
# Value function expects rewards in -100 to +100 range
# If rewards are -1 to +1, scale them up:
reward = reward * 10

# 4. Reduce gamma (shorter horizon)
gamma = 0.95  # From 0.99

# 5. More training epochs
n_epochs = 20  # From 10
```

---

## Configuration Templates

### Template 1: Stable & Conservative

**Use when**: First training run, production system, risk-averse

```python
learning_rate = 1e-4
n_steps = 2048
batch_size = 128
n_epochs = 8
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
vf_coef = 0.5
ent_coef = 0.01
net_arch = [256, 256, 128]
```

**Expected**: Slow but steady learning, low overfitting risk

### Template 2: Fast Learning

**Use when**: Experimentation, strong hardware, tight deadline

```python
learning_rate = 5e-4
n_steps = 4096
batch_size = 64
n_epochs = 15
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.15
vf_coef = 0.5
ent_coef = 0.02
net_arch = [512, 256, 128]
```

**Expected**: Rapid improvement, may be unstable, watch for overfitting

### Template 3: High Exploration

**Use when**: Sparse rewards, need to discover strategies

```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.3
vf_coef = 0.5
ent_coef = 0.05  # High entropy
net_arch = [256, 256, 128]
```

**Expected**: Diverse behaviors, slower convergence, good for discovery

### Template 4: Fine-Tuning

**Use when**: Already have decent model, want to improve

```python
# Start from your best checkpoint
learning_rate = 1e-4  # Reduced
n_steps = 2048
batch_size = 128  # Larger
n_epochs = 5  # Fewer
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.1  # Smaller updates
vf_coef = 0.5
ent_coef = 0.005  # Less exploration
net_arch = [256, 256, 128]  # Keep same
```

**Expected**: Incremental improvements, maintains existing performance

---

## Monitoring During Training

### Key Metrics to Watch

```python
# 1. Episode Reward Mean
# Should increase over time
# Flat or decreasing → problem with reward or exploration

# 2. Policy Loss
# Should decrease and stabilize
# Increasing → learning rate too high

# 3. Value Loss
# Should decrease over time
# Not decreasing → value function not learning

# 4. Entropy
# Should slowly decrease
# Crashes to zero quickly → not enough exploration
# Stays high → not learning

# 5. Explained Variance
# Should be > 0.5
# < 0 → value function predicting wrong direction
# Near 0 → value function not learning

# 6. Approx KL Divergence
# Should be < 0.02
# > 0.02 → policy changing too fast, reduce LR

# 7. Clip Fraction
# Should be 0.05 - 0.15
# > 0.3 → clip_range too small or LR too high
# < 0.01 → clip_range too large, not constraining enough
```

### TensorBoard Commands

```bash
# View training progress
tensorboard --logdir ./tensorboard/

# Compare multiple runs
tensorboard --logdir ./tensorboard/ --port 6006

# Export metrics
tensorboard --logdir ./tensorboard/ --export_csv ./metrics.csv
```

### Automated Analysis

Use included script:
```bash
python scripts/analyze_training.py \
    --tensorboard-dir ./tensorboard/ \
    --output-dir ./analysis/ \
    --generate-report
```

This generates:
- Training stability report
- Convergence analysis
- Hyperparameter impact summary
- Recommendations for next iteration

---

## Decision Flowchart

```
Start Training
      ↓
  Monitor 10k steps
      ↓
   Is loss decreasing?
    ↙          ↘
  No            Yes
   ↓             ↓
Reduce LR    Is explained variance > 0?
   ↓          ↙              ↘
Try again   No              Yes
            ↓                ↓
       Increase vf_coef   Continue training
            ↓                ↓
       Monitor 50k      Is episode reward improving?
                         ↙              ↘
                       No              Yes
                        ↓                ↓
                  Check reward      Train to convergence
                  function             ↓
                        ↓           Validate on unseen data
                  Fix & restart        ↓
                                  Gap < 30%?
                                   ↙      ↘
                                 Yes      No
                                  ↓       ↓
                              Deploy  Reduce net_arch
                                       ↓
                                   Retrain
```

---

## Final Recommendations

Based on your MtfScalper_RL_Hybrid setup:

### Immediate (try first)
```python
# Keep most current settings, small tweaks:
learning_rate = 2e-4  # Slightly more conservative
batch_size = 128      # Better gradient estimates
net_arch = [300, 256, 128]  # Match feature count (30 features)
```

### If that works well
```python
# Optimize for speed:
learning_rate = 3e-4
n_steps = 4096  # More stable gradients
net_arch = [512, 256]  # Wider first layer
```

### If experiencing instability
```python
# Maximum stability:
learning_rate = 1e-4
batch_size = 256
n_epochs = 5
clip_range = 0.1
net_arch = [256, 256]
```

Run all three configurations and compare using validation Sharpe ratio.
