# Evaluation Metrics for RL Trading Models

Comprehensive guide to evaluating RL model performance beyond simple profit.

## Table of Contents

1. [Standard Trading Metrics](#standard-trading-metrics)
2. [RL-Specific Metrics](#rl-specific-metrics)
3. [Exit Quality Metrics](#exit-quality-metrics)
4. [Risk Metrics](#risk-metrics)
5. [Comparison Framework](#comparison-framework)

---

## Standard Trading Metrics

### Basic Performance Indicators

```python
def calculate_basic_metrics(trades_df):
    """
    Standard metrics every strategy should report.
    """
    metrics = {}

    # Total return
    metrics['total_return'] = (
        trades_df['profit'].sum() /
        initial_capital
    )

    # Win rate
    metrics['win_rate'] = (
        (trades_df['profit'] > 0).sum() /
        len(trades_df)
    )

    # Average profit per trade
    metrics['avg_profit'] = trades_df['profit'].mean()

    # Average winner vs average loser
    winners = trades_df[trades_df['profit'] > 0]
    losers = trades_df[trades_df['profit'] < 0]

    metrics['avg_winner'] = winners['profit'].mean() if len(winners) > 0 else 0
    metrics['avg_loser'] = losers['profit'].mean() if len(losers) > 0 else 0

    # Profit factor
    total_wins = winners['profit'].sum()
    total_losses = abs(losers['profit'].sum())

    metrics['profit_factor'] = (
        total_wins / total_losses if total_losses > 0 else float('inf')
    )

    # Max drawdown
    cumulative_returns = (1 + trades_df['profit']).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max

    metrics['max_drawdown'] = abs(drawdown.min())

    return metrics
```

**Interpretation**:

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Total Return | > 20% | 10-20% | < 10% |
| Win Rate | > 55% | 45-55% | < 45% |
| Profit Factor | > 1.5 | 1.2-1.5 | < 1.2 |
| Max Drawdown | < 10% | 10-15% | > 15% |

### Advanced Return Metrics

```python
def calculate_risk_adjusted_returns(trades_df, risk_free_rate=0.02):
    """
    Sharpe, Sortino, Calmar ratios.
    """
    metrics = {}

    # Annualized return
    daily_returns = trades_df['profit'].resample('D').sum()
    trading_days = 252
    annual_return = daily_returns.mean() * trading_days

    # Sharpe Ratio
    returns_std = daily_returns.std() * np.sqrt(trading_days)
    metrics['sharpe_ratio'] = (
        (annual_return - risk_free_rate) / returns_std
    )

    # Sortino Ratio (only downside volatility)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(trading_days)
    metrics['sortino_ratio'] = (
        (annual_return - risk_free_rate) / downside_std
    )

    # Calmar Ratio (return / max drawdown)
    max_dd = calculate_basic_metrics(trades_df)['max_drawdown']
    metrics['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else 0

    # Omega Ratio (gains/losses above threshold)
    threshold = risk_free_rate / trading_days
    gains = daily_returns[daily_returns > threshold].sum()
    losses = abs(daily_returns[daily_returns < threshold].sum())
    metrics['omega_ratio'] = gains / losses if losses > 0 else float('inf')

    return metrics
```

**Interpretation**:

| Metric | Excellent | Good | Poor |
|--------|-----------|------|------|
| Sharpe | > 2.0 | 1.0-2.0 | < 1.0 |
| Sortino | > 2.5 | 1.5-2.5 | < 1.5 |
| Calmar | > 3.0 | 2.0-3.0 | < 2.0 |
| Omega | > 2.0 | 1.5-2.0 | < 1.5 |

---

## RL-Specific Metrics

### 1. Learning Progress Metrics

```python
def calculate_learning_metrics(training_logs):
    """
    Track how well the agent learned during training.
    """
    metrics = {}

    # Episode reward improvement
    early_rewards = training_logs['episode_reward'][:100]
    late_rewards = training_logs['episode_reward'][-100:]

    metrics['reward_improvement'] = (
        late_rewards.mean() - early_rewards.mean()
    )

    # Convergence speed (steps to reach 90% of final performance)
    target_performance = late_rewards.mean() * 0.9
    converged_idx = np.where(
        training_logs['episode_reward'].rolling(20).mean() > target_performance
    )[0]

    metrics['convergence_steps'] = (
        converged_idx[0] if len(converged_idx) > 0 else len(training_logs)
    )

    # Training stability (coefficient of variation in last 1000 episodes)
    recent_rewards = training_logs['episode_reward'][-1000:]
    metrics['training_stability'] = (
        recent_rewards.std() / abs(recent_rewards.mean())
    )

    # Value function accuracy (explained variance)
    metrics['value_accuracy'] = (
        training_logs['explained_variance'].iloc[-100:].mean()
    )

    return metrics
```

**Interpretation**:
- `reward_improvement > 5`: Good learning
- `convergence_steps < 50k`: Fast learner
- `training_stability < 0.5`: Stable training
- `value_accuracy > 0.7`: Value function learned well

### 2. Policy Quality Metrics

```python
def calculate_policy_metrics(model, eval_env, n_episodes=100):
    """
    Evaluate policy decision quality.
    """
    metrics = {}

    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    action_confidences = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        episode_actions = []

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            # Get confidence
            action_probs = model.policy.get_distribution(obs).distribution.probs
            confidence = float(action_probs[0, action].detach().numpy())

            action_counts[int(action)] += 1
            action_confidences.append(confidence)
            episode_actions.append(action)

            obs, _, done, _ = eval_env.step(action)

        episode_lengths.append(len(episode_actions))

    # Action diversity (entropy)
    total_actions = sum(action_counts.values())
    action_probs = [count / total_actions for count in action_counts.values()]
    metrics['action_entropy'] = -sum(
        p * np.log(p + 1e-10) for p in action_probs if p > 0
    )

    # Average confidence
    metrics['avg_confidence'] = np.mean(action_confidences)

    # Action balance
    metrics['action_distribution'] = action_counts

    # Episode length consistency
    metrics['avg_episode_length'] = np.mean(episode_lengths)
    metrics['episode_length_std'] = np.std(episode_lengths)

    return metrics
```

**Interpretation**:
- `action_entropy`: 0 = deterministic, 1.6 = uniform (5 actions), target 0.8-1.2
- `avg_confidence > 0.7`: Policy is confident
- Action distribution should not have any action > 60% (indicates stuck policy)

---

## Exit Quality Metrics

### 1. Exit Timing Score

```python
def calculate_exit_quality(trades_df):
    """
    Specialized metrics for exit-focused RL models.
    """
    metrics = {}

    # Profit capture ratio: profit / max_profit_during_trade
    if 'max_profit_seen' in trades_df.columns:
        trades_df['capture_ratio'] = (
            trades_df['profit'] / trades_df['max_profit_seen'].clip(lower=0.001)
        )
        metrics['avg_capture_ratio'] = trades_df['capture_ratio'].mean()
        metrics['capture_ratio_std'] = trades_df['capture_ratio'].std()

    # Exit at optimal time?
    # Compare actual exit vs optimal (holding until peak)
    if 'exit_price' in trades_df and 'future_high_20' in trades_df:
        optimal_profit = (
            trades_df['future_high_20'] - trades_df['entry_price']
        ) / trades_df['entry_price']

        metrics['exit_optimality'] = (
            trades_df['profit'].sum() / optimal_profit.sum()
        )

    # Exit timing percentile
    # Where in the candle range did we exit?
    if 'exit_price' in trades_df:
        for _, trade in trades_df.iterrows():
            entry_to_peak = trade['max_price'] - trade['entry_price']
            entry_to_exit = trade['exit_price'] - trade['entry_price']

            if entry_to_peak > 0:
                percentile = entry_to_exit / entry_to_peak
            else:
                percentile = 0

            trades_df.loc[trade.name, 'exit_percentile'] = percentile

        metrics['avg_exit_percentile'] = trades_df['exit_percentile'].mean()
        # Target: > 0.7 (exiting at 70%+ of peak profit)

    return metrics
```

**Target Benchmarks**:
- `avg_capture_ratio > 0.7`: Exiting near peak profit
- `exit_optimality > 0.6`: Better than holding until future peak
- `avg_exit_percentile > 0.65`: Timing is good

### 2. Drawdown Management

```python
def calculate_drawdown_metrics(trades_df):
    """
    How well does the agent protect profits?
    """
    metrics = {}

    # Max adverse excursion (MAE)
    if 'min_profit_during_trade' in trades_df.columns:
        metrics['avg_mae'] = trades_df['min_profit_during_trade'].mean()

        # Winners vs losers MAE
        winners = trades_df[trades_df['profit'] > 0]
        losers = trades_df[trades_df['profit'] <= 0]

        metrics['mae_winners'] = winners['min_profit_during_trade'].mean()
        metrics['mae_losers'] = losers['min_profit_during_trade'].mean()

    # Profit erosion rate
    if 'max_profit_seen' in trades_df.columns:
        trades_df['erosion'] = (
            trades_df['max_profit_seen'] - trades_df['profit']
        )
        trades_df['erosion_rate'] = (
            trades_df['erosion'] / trades_df['max_profit_seen'].clip(lower=0.001)
        )

        metrics['avg_erosion_rate'] = trades_df['erosion_rate'].mean()
        # Target: < 0.3 (losing < 30% of peak profit)

    # Exit speed after drawdown
    # How fast does agent exit when profit deteriorates?
    if 'duration_after_peak' in trades_df.columns:
        metrics['avg_exit_delay'] = trades_df['duration_after_peak'].mean()
        # Target: < 10 candles

    return metrics
```

**Benchmarks**:
- `avg_erosion_rate < 0.25`: Good profit protection
- `avg_exit_delay < 15`: Quick response to deterioration

### 3. Position Duration Analysis

```python
def calculate_duration_metrics(trades_df):
    """
    Is the agent holding positions efficiently?
    """
    metrics = {}

    # Overall duration stats
    metrics['avg_duration'] = trades_df['duration'].mean()
    metrics['median_duration'] = trades_df['duration'].median()
    metrics['max_duration'] = trades_df['duration'].max()

    # Duration vs profit correlation
    metrics['duration_profit_correlation'] = (
        trades_df['duration'].corr(trades_df['profit'])
    )
    # Ideally: winners take time, losers are quick → positive correlation

    # Winners vs losers duration
    winners = trades_df[trades_df['profit'] > 0]
    losers = trades_df[trades_df['profit'] <= 0]

    metrics['winner_duration'] = winners['duration'].mean()
    metrics['loser_duration'] = losers['duration'].mean()

    metrics['duration_ratio'] = (
        metrics['winner_duration'] / metrics['loser_duration']
        if metrics['loser_duration'] > 0 else 0
    )
    # Target: > 1.5 (winners held longer than losers)

    # Capital efficiency (profit per candle)
    trades_df['profit_per_candle'] = (
        trades_df['profit'] / trades_df['duration']
    )
    metrics['avg_profit_per_candle'] = trades_df['profit_per_candle'].mean()

    return metrics
```

**Benchmarks**:
- `duration_ratio > 1.3`: Cutting losers faster than winners
- `duration_profit_correlation > 0.2`: Positive relationship

---

## Risk Metrics

### 1. Volatility and VaR

```python
def calculate_risk_metrics(trades_df, confidence_level=0.95):
    """
    Risk assessment metrics.
    """
    metrics = {}

    # Return volatility
    metrics['returns_volatility'] = trades_df['profit'].std()

    # Value at Risk (VaR)
    metrics['var_95'] = trades_df['profit'].quantile(1 - confidence_level)
    # 95% chance of not losing more than this

    # Conditional Value at Risk (CVaR / Expected Shortfall)
    var_threshold = metrics['var_95']
    tail_losses = trades_df[trades_df['profit'] < var_threshold]['profit']
    metrics['cvar_95'] = tail_losses.mean() if len(tail_losses) > 0 else 0

    # Risk of ruin (simplified)
    # Probability of losing X% of capital
    cumulative_returns = (1 + trades_df['profit']).cumprod()
    drawdowns = (cumulative_returns - cumulative_returns.expanding().max()) / cumulative_returns.expanding().max()

    metrics['risk_of_ruin_20pct'] = (drawdowns < -0.20).sum() / len(drawdowns)

    return metrics
```

### 2. Consistency Metrics

```python
def calculate_consistency_metrics(trades_df):
    """
    How consistent is the strategy?
    """
    metrics = {}

    # Monthly return consistency
    monthly_returns = trades_df.resample('M')['profit'].sum()

    metrics['profitable_months'] = (monthly_returns > 0).sum() / len(monthly_returns)
    # Target: > 0.7 (70% of months positive)

    # Largest losing streak
    is_loser = trades_df['profit'] < 0
    losing_streaks = (is_loser != is_loser.shift()).cumsum()
    streak_lengths = is_loser.groupby(losing_streaks).sum()

    metrics['max_losing_streak'] = int(streak_lengths.max()) if len(streak_lengths) > 0 else 0

    # Largest winning streak
    is_winner = trades_df['profit'] > 0
    winning_streaks = (is_winner != is_winner.shift()).cumsum()
    streak_lengths = is_winner.groupby(winning_streaks).sum()

    metrics['max_winning_streak'] = int(streak_lengths.max()) if len(streak_lengths) > 0 else 0

    # Drawdown recovery time
    cumulative_returns = (1 + trades_df['profit']).cumprod()
    running_max = cumulative_returns.expanding().max()
    in_drawdown = cumulative_returns < running_max

    if in_drawdown.any():
        # Average time to recover from drawdown
        drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
        recovery_times = in_drawdown.groupby(drawdown_periods).sum()
        metrics['avg_recovery_time'] = recovery_times[recovery_times > 0].mean()
    else:
        metrics['avg_recovery_time'] = 0

    return metrics
```

---

## Comparison Framework

### Configuration Comparison Table

```python
def create_comparison_report(results_dict):
    """
    Compare multiple RL configurations.

    results_dict = {
        'baseline': trades_df_baseline,
        'profit_focused': trades_df_profit,
        'risk_focused': trades_df_risk,
    }
    """

    comparison = {}

    for name, trades_df in results_dict.items():
        metrics = {}

        # Basic metrics
        basic = calculate_basic_metrics(trades_df)
        metrics.update(basic)

        # Risk-adjusted returns
        risk_adj = calculate_risk_adjusted_returns(trades_df)
        metrics.update(risk_adj)

        # Exit quality
        exit_qual = calculate_exit_quality(trades_df)
        metrics.update(exit_qual)

        # Duration
        duration = calculate_duration_metrics(trades_df)
        metrics.update(duration)

        comparison[name] = metrics

    # Create DataFrame
    df = pd.DataFrame(comparison).T

    # Rank configurations
    # Higher is better for most metrics
    ranking_cols = [
        'total_return', 'sharpe_ratio', 'profit_factor',
        'avg_capture_ratio', 'duration_ratio'
    ]

    df['composite_score'] = 0
    for col in ranking_cols:
        if col in df.columns:
            # Normalize to 0-1 and add to score
            normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
            df['composite_score'] += normalized

    df = df.sort_values('composite_score', ascending=False)

    return df
```

### Report Template

```python
def generate_evaluation_report(trades_df, config_name="RL Model"):
    """
    Generate comprehensive HTML report.
    """

    report = f"""
    # RL Model Evaluation Report
    ## Configuration: {config_name}

    ### 1. Basic Performance
    """

    basic_metrics = calculate_basic_metrics(trades_df)
    for key, value in basic_metrics.items():
        report += f"- **{key}**: {value:.4f}\n"

    report += "\n### 2. Risk-Adjusted Returns\n"
    risk_adj = calculate_risk_adjusted_returns(trades_df)
    for key, value in risk_adj.items():
        report += f"- **{key}**: {value:.4f}\n"

    report += "\n### 3. Exit Quality (RL-Specific)\n"
    exit_quality = calculate_exit_quality(trades_df)
    for key, value in exit_quality.items():
        report += f"- **{key}**: {value:.4f}\n"

    report += "\n### 4. Position Management\n"
    duration = calculate_duration_metrics(trades_df)
    for key, value in duration.items():
        report += f"- **{key}**: {value:.4f}\n"

    report += "\n### 5. Risk Assessment\n"
    risk = calculate_risk_metrics(trades_df)
    for key, value in risk.items():
        report += f"- **{key}**: {value:.4f}\n"

    report += "\n### 6. Consistency\n"
    consistency = calculate_consistency_metrics(trades_df)
    for key, value in consistency.items():
        report += f"- **{key}**: {value}\n"

    # Add decision
    report += "\n\n## Recommendation\n"

    # Decision logic
    if (basic_metrics['profit_factor'] > 1.5 and
        risk_adj['sharpe_ratio'] > 1.0 and
        exit_quality.get('avg_capture_ratio', 0) > 0.65):

        report += "✅ **APPROVED for live trading**\n"
        report += "Model meets all quality thresholds.\n"

    elif (basic_metrics['profit_factor'] > 1.2 and
          risk_adj['sharpe_ratio'] > 0.5):

        report += "⚠️ **APPROVED for paper trading**\n"
        report += "Model shows promise but needs validation.\n"

    else:
        report += "❌ **NOT APPROVED**\n"
        report += "Model requires further optimization.\n"

        # Specific recommendations
        if basic_metrics['profit_factor'] < 1.2:
            report += "\n- Improve profit factor: Adjust reward weights\n"
        if risk_adj['sharpe_ratio'] < 0.5:
            report += "\n- Improve risk-adjusted returns: Increase drawdown control\n"
        if exit_quality.get('avg_capture_ratio', 0) < 0.6:
            report += "\n- Improve exit timing: Add timing quality features\n"

    return report
```

### Usage Example

```python
# After backtesting
trades_df = load_backtest_results('user_data/backtest_results/latest.json')

# Generate report
report = generate_evaluation_report(trades_df, config_name="MtfScalper_RL_v1")

# Save report
with open('evaluation_report.md', 'w') as f:
    f.write(report)

# Compare configurations
results_dict = {
    'baseline': load_backtest_results('baseline.json'),
    'profit_focused': load_backtest_results('profit_focused.json'),
    'risk_focused': load_backtest_results('risk_focused.json'),
}

comparison_df = create_comparison_report(results_dict)
comparison_df.to_csv('configuration_comparison.csv')

print("Best configuration:", comparison_df.index[0])
```

---

## Quick Decision Matrix

Use this to make go/no-go decisions:

```
| Metric | Must Have | Nice to Have |
|--------|-----------|--------------|
| Profit Factor | > 1.3 | > 1.5 |
| Sharpe Ratio | > 0.5 | > 1.0 |
| Win Rate | > 45% | > 55% |
| Max Drawdown | < 20% | < 12% |
| Capture Ratio | > 0.60 | > 0.75 |
| Duration Ratio | > 1.0 | > 1.5 |

Decision:
- 6/6 "Must Have" → ✅ Live trading
- 4-5/6 "Must Have" → ⚠️ Paper trading
- < 4/6 "Must Have" → ❌ More optimization needed
```

Remember: RL models should excel at **exit quality metrics** since that's their focus!
