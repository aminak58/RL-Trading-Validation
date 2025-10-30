#!/usr/bin/env python3
"""
Test and compare different reward function configurations.

Usage:
    python reward_backtest.py --pair BTC/USDT:USDT --timerange 20240101-20240331 --config config_rl_hybrid.json

Features:
    - Simulate different reward weights
    - Compare exit quality across configurations
    - Visualize reward distributions
    - Generate recommendation report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class RewardSimulator:
    """Simulates different reward function configurations."""

    def __init__(self):
        self.results = {}

    def calculate_profit_score(self, profit: float, config: Dict) -> float:
        """Calculate profit component of reward."""
        if profit <= -0.05:
            return -5.0
        elif profit <= -0.02:
            return -2.0
        elif profit <= -0.005:
            return -0.5
        elif profit <= 0:
            return profit * 20
        elif profit < 0.002:
            return 0.5 + profit * 100
        elif profit < 0.01:
            return 1.0 + profit * 150
        elif profit < 0.02:
            return 3.0 + profit * 100
        elif profit < 0.04:
            return 5.0 + profit * 50
        else:
            return min(15.0, 7.0 + profit * 30)

    def calculate_drawdown_score(
        self,
        current_profit: float,
        max_profit: float,
        config: Dict
    ) -> float:
        """Calculate drawdown control component."""
        if max_profit <= 0:
            return 0.0

        drawdown = (max_profit - current_profit) / max_profit

        if drawdown < 0.005:
            return 5.0
        elif drawdown < 0.01:
            return 2.0
        elif drawdown < 0.02:
            return 0.0
        else:
            return -5.0 * drawdown

    def calculate_timing_score(
        self,
        current_price: float,
        recent_high: float,
        recent_low: float,
        position_type: int,
        config: Dict
    ) -> float:
        """Calculate timing quality component."""
        price_range = recent_high - recent_low
        if price_range < 1e-8:
            return 0.0

        if position_type == 1:  # Long
            percentile = (current_price - recent_low) / price_range
            if percentile > 0.8:
                return 5.0
            elif percentile > 0.6:
                return 2.0
            else:
                return 0.0
        else:  # Short
            percentile = (current_price - recent_low) / price_range
            if percentile < 0.2:
                return 5.0
            elif percentile < 0.4:
                return 2.0
            else:
                return 0.0

    def calculate_risk_reward_score(
        self,
        current_profit: float,
        max_risk: float,
        config: Dict
    ) -> float:
        """Calculate risk/reward ratio component."""
        if max_risk <= 0:
            return 0.0

        risk_reward_ratio = current_profit / max_risk

        if risk_reward_ratio > 3.0:
            return 5.0
        elif risk_reward_ratio > 2.0:
            return 3.0
        elif risk_reward_ratio > 1.0:
            return 1.0
        else:
            return -2.0

    def calculate_total_reward(
        self,
        profit: float,
        max_profit: float,
        current_price: float,
        recent_high: float,
        recent_low: float,
        position_type: int,
        max_risk: float,
        config: Dict
    ) -> Tuple[float, Dict]:
        """Calculate total reward using given configuration."""

        weights = config.get('reward_weights', {
            'profit': 0.35,
            'drawdown': 0.25,
            'timing': 0.20,
            'risk_reward': 0.20
        })

        # Calculate components
        profit_score = self.calculate_profit_score(profit, config)
        drawdown_score = self.calculate_drawdown_score(current_profit=profit, max_profit=max_profit, config=config)
        timing_score = self.calculate_timing_score(current_price, recent_high, recent_low, position_type, config)
        risk_reward_score = self.calculate_risk_reward_score(profit, max_risk, config)

        # Weighted sum
        total = (
            weights['profit'] * profit_score +
            weights['drawdown'] * drawdown_score +
            weights['timing'] * timing_score +
            weights['risk_reward'] * risk_reward_score
        )

        components = {
            'profit_score': profit_score,
            'drawdown_score': drawdown_score,
            'timing_score': timing_score,
            'risk_reward_score': risk_reward_score
        }

        return total, components

    def simulate_trades(
        self,
        dataframe: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """Simulate trades and calculate rewards."""

        trades = []
        in_position = False
        entry_price = 0
        entry_idx = 0
        max_profit = 0
        max_risk = 0

        for i in range(100, len(dataframe) - 20):
            row = dataframe.iloc[i]

            # Simple entry logic (for simulation)
            if not in_position:
                # Entry condition: momentum positive and RSI not overbought
                if (row.get('%-momentum_5', 0) > 0 and
                    row.get('rsi', 50) < 70):

                    in_position = True
                    entry_price = row['close']
                    entry_idx = i
                    max_profit = 0
                    max_risk = 0

            elif in_position:
                # Track position metrics
                current_profit = (row['close'] - entry_price) / entry_price
                max_profit = max(max_profit, current_profit)

                # Track max adverse excursion
                position_data = dataframe.iloc[entry_idx:i+1]
                min_price = position_data['low'].min()
                adverse_excursion = (entry_price - min_price) / entry_price
                max_risk = max(max_risk, adverse_excursion)

                # Get recent high/low for timing
                recent_data = dataframe.iloc[max(0, i-20):i+1]
                recent_high = recent_data['high'].max()
                recent_low = recent_data['low'].min()

                # Calculate reward
                total_reward, components = self.calculate_total_reward(
                    profit=current_profit,
                    max_profit=max_profit,
                    current_price=row['close'],
                    recent_high=recent_high,
                    recent_low=recent_low,
                    position_type=1,  # Long
                    max_risk=max_risk,
                    config=config
                )

                # Exit decision (simplified - based on reward threshold)
                duration = i - entry_idx
                should_exit = (
                    total_reward < -3 or  # Bad reward
                    duration > 150 or  # Max duration
                    current_profit > 0.03  # Take profit
                )

                if should_exit:
                    trade_data = {
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'duration': duration,
                        'profit': current_profit,
                        'max_profit': max_profit,
                        'max_risk': max_risk,
                        'exit_reward': total_reward,
                        **components
                    }
                    trades.append(trade_data)
                    in_position = False

        return pd.DataFrame(trades)

    def compare_configurations(
        self,
        dataframe: pd.DataFrame,
        configs: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Compare multiple reward configurations."""

        print(f"Comparing {len(configs)} configurations...")

        results = {}

        for name, config in configs.items():
            print(f"  Simulating: {name}...")

            trades_df = self.simulate_trades(dataframe, config)

            if len(trades_df) > 0:
                metrics = {
                    'total_trades': len(trades_df),
                    'win_rate': (trades_df['profit'] > 0).mean() * 100,
                    'avg_profit': trades_df['profit'].mean() * 100,
                    'avg_winner': trades_df[trades_df['profit'] > 0]['profit'].mean() * 100 if (trades_df['profit'] > 0).any() else 0,
                    'avg_loser': trades_df[trades_df['profit'] < 0]['profit'].mean() * 100 if (trades_df['profit'] < 0).any() else 0,
                    'profit_factor': abs(trades_df[trades_df['profit'] > 0]['profit'].sum() / (trades_df[trades_df['profit'] < 0]['profit'].sum() + 1e-10)),
                    'avg_duration': trades_df['duration'].mean(),
                    'max_drawdown': trades_df['profit'].min() * 100,
                    'sharpe': trades_df['profit'].mean() / (trades_df['profit'].std() + 1e-10),
                    'avg_exit_reward': trades_df['exit_reward'].mean()
                }
                results[name] = metrics
                self.results[name] = trades_df

        return pd.DataFrame(results).T

    def generate_comparison_plots(self, output_dir: str = "./analysis/"):
        """Generate comparison plots."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating plots in {output_path}...")

        if len(self.results) == 0:
            print("No results to plot")
            return

        # Plot 1: Performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reward Configuration Comparison', fontsize=16)

        config_names = list(self.results.keys())

        # Win rate
        win_rates = [
            (self.results[name]['profit'] > 0).mean() * 100
            for name in config_names
        ]
        axes[0, 0].bar(config_names, win_rates)
        axes[0, 0].set_ylabel('Win Rate (%)')
        axes[0, 0].set_title('Win Rate by Configuration')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Average profit
        avg_profits = [
            self.results[name]['profit'].mean() * 100
            for name in config_names
        ]
        axes[0, 1].bar(config_names, avg_profits)
        axes[0, 1].set_ylabel('Avg Profit (%)')
        axes[0, 1].set_title('Average Profit by Configuration')
        axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Profit factor
        profit_factors = []
        for name in config_names:
            trades = self.results[name]
            wins = trades[trades['profit'] > 0]['profit'].sum()
            losses = abs(trades[trades['profit'] < 0]['profit'].sum())
            pf = wins / (losses + 1e-10)
            profit_factors.append(pf)

        axes[1, 0].bar(config_names, profit_factors)
        axes[1, 0].set_ylabel('Profit Factor')
        axes[1, 0].set_title('Profit Factor by Configuration')
        axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Breakeven')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Average duration
        avg_durations = [
            self.results[name]['duration'].mean()
            for name in config_names
        ]
        axes[1, 1].bar(config_names, avg_durations)
        axes[1, 1].set_ylabel('Duration (candles)')
        axes[1, 1].set_title('Average Position Duration')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plot_path = output_path / 'reward_comparison.png'
        plt.savefig(plot_path, dpi=100)
        print(f"Saved: {plot_path}")
        plt.close()

        # Plot 2: Reward distributions
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reward Component Distributions', fontsize=16)

        components = ['profit_score', 'drawdown_score', 'timing_score', 'risk_reward_score']
        titles = ['Profit Score', 'Drawdown Score', 'Timing Score', 'Risk/Reward Score']

        for idx, (component, title) in enumerate(zip(components, titles)):
            ax = axes[idx // 2, idx % 2]

            for name in config_names:
                if component in self.results[name].columns:
                    values = self.results[name][component]
                    ax.hist(values, alpha=0.5, label=name, bins=20)

            ax.set_xlabel(title)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{title} Distribution')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plot_path = output_path / 'reward_distributions.png'
        plt.savefig(plot_path, dpi=100)
        print(f"Saved: {plot_path}")
        plt.close()


def load_sample_data(pair: str, timerange: str) -> pd.DataFrame:
    """
    Load or generate sample data for testing.
    In production, this would load from Freqtrade data directory.
    """
    print(f"Loading data for {pair} ({timerange})...")

    # For demonstration, generate synthetic data
    # In production, use: freqtrade.data.history.load_pair_history()

    n_candles = 5000
    np.random.seed(42)

    # Generate price data
    base_price = 60000
    returns = np.random.normal(0.0001, 0.02, n_candles)
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV
    timestamps = pd.date_range(start='2024-01-01', periods=n_candles, freq='5T')

    df = pd.DataFrame({
        'date': timestamps,
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': np.random.lognormal(10, 1, n_candles)
    })

    # Add basic indicators
    df['rsi'] = 50 + np.random.randn(n_candles) * 15
    df['%-momentum_5'] = df['close'].pct_change(5)

    print(f"Loaded {len(df)} candles")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Test and compare reward function configurations"
    )
    parser.add_argument(
        '--pair',
        type=str,
        default='BTC/USDT:USDT',
        help='Trading pair'
    )
    parser.add_argument(
        '--timerange',
        type=str,
        default='20240101-20240331',
        help='Time range for backtest'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Base config file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./analysis/',
        help='Output directory'
    )

    args = parser.parse_args()

    # Load data
    dataframe = load_sample_data(args.pair, args.timerange)

    # Define configurations to test
    configs = {
        'baseline': {
            'reward_weights': {
                'profit': 0.35,
                'drawdown': 0.25,
                'timing': 0.20,
                'risk_reward': 0.20
            }
        },
        'profit_focused': {
            'reward_weights': {
                'profit': 0.50,
                'drawdown': 0.20,
                'timing': 0.15,
                'risk_reward': 0.15
            }
        },
        'risk_focused': {
            'reward_weights': {
                'profit': 0.25,
                'drawdown': 0.40,
                'timing': 0.20,
                'risk_reward': 0.15
            }
        },
        'timing_focused': {
            'reward_weights': {
                'profit': 0.30,
                'drawdown': 0.20,
                'timing': 0.35,
                'risk_reward': 0.15
            }
        }
    }

    # Run comparison
    simulator = RewardSimulator()
    comparison_df = simulator.compare_configurations(dataframe, configs)

    print("\n" + "=" * 60)
    print("Reward Configuration Comparison Results")
    print("=" * 60)
    print(comparison_df.to_string())

    # Generate outputs
    simulator.generate_comparison_plots(args.output_dir)

    # Save results
    output_path = Path(args.output_dir)
    comparison_df.to_csv(output_path / 'reward_comparison.csv')
    print(f"\nSaved comparison to {output_path / 'reward_comparison.csv'}")

    # Recommendation
    best_config = comparison_df['sharpe'].idxmax()
    print(f"\n✅ Recommended configuration: {best_config}")
    print(f"   Sharpe ratio: {comparison_df.loc[best_config, 'sharpe']:.3f}")
    print(f"   Win rate: {comparison_df.loc[best_config, 'win_rate']:.1f}%")
    print(f"   Profit factor: {comparison_df.loc[best_config, 'profit_factor']:.2f}")


if __name__ == '__main__':
    main()
