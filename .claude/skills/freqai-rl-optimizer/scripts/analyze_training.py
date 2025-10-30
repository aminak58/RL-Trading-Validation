#!/usr/bin/env python3
"""
Analyze TensorBoard logs for RL training diagnostics.

Usage:
    python analyze_training.py --tensorboard-dir ./tensorboard/ --output-dir ./analysis/

Features:
    - Parse and visualize training metrics
    - Detect training issues (unstable loss, no learning, etc.)
    - Generate diagnostic report with recommendations
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Error: tensorboard package required")
    print("Install with: pip install tensorboard")
    sys.exit(1)


class TrainingAnalyzer:
    """Analyzes RL training logs from TensorBoard."""

    def __init__(self, tensorboard_dir: str, output_dir: str = "./analysis/"):
        self.tensorboard_dir = Path(tensorboard_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {}
        self.issues = []
        self.recommendations = []

    def load_tensorboard_data(self) -> Dict[str, pd.DataFrame]:
        """Load all scalar metrics from TensorBoard logs."""
        print(f"Loading TensorBoard data from {self.tensorboard_dir}...")

        # Find event files
        event_files = list(self.tensorboard_dir.glob("**/events.out.tfevents.*"))

        if not event_files:
            print(f"Error: No TensorBoard event files found in {self.tensorboard_dir}")
            return {}

        print(f"Found {len(event_files)} event file(s)")

        # Load most recent file
        latest_file = max(event_files, key=lambda p: p.stat().st_mtime)
        print(f"Using: {latest_file.name}")

        ea = event_accumulator.EventAccumulator(str(latest_file))
        ea.Reload()

        # Extract scalar metrics
        metrics = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            df = pd.DataFrame({
                'step': [e.step for e in events],
                'value': [e.value for e in events],
                'wall_time': [e.wall_time for e in events]
            })
            metrics[tag] = df

        print(f"Loaded {len(metrics)} metrics")
        self.metrics = metrics
        return metrics

    def analyze_training_progress(self) -> Dict:
        """Analyze overall training progress."""
        analysis = {}

        # Episode reward trend
        if 'rollout/ep_rew_mean' in self.metrics:
            rewards = self.metrics['rollout/ep_rew_mean']

            # Early vs late performance
            n_samples = len(rewards)
            early_window = min(100, n_samples // 4)
            late_window = min(100, n_samples // 4)

            early_mean = rewards['value'].iloc[:early_window].mean()
            late_mean = rewards['value'].iloc[-late_window:].mean()

            improvement = late_mean - early_mean
            improvement_pct = (improvement / abs(early_mean)) * 100 if early_mean != 0 else 0

            analysis['reward_improvement'] = improvement
            analysis['reward_improvement_pct'] = improvement_pct

            # Trend direction
            if improvement_pct > 20:
                analysis['learning_status'] = "Good learning"
            elif improvement_pct > 0:
                analysis['learning_status'] = "Slow learning"
            else:
                analysis['learning_status'] = "Not learning"
                self.issues.append("Episode rewards not improving")
                self.recommendations.append(
                    "Check reward function - may be too sparse or incorrect"
                )

        # Training stability
        if 'train/loss' in self.metrics:
            loss = self.metrics['train/loss']

            # Check for NaN
            if loss['value'].isna().any():
                analysis['has_nan'] = True
                self.issues.append("Training produced NaN values")
                self.recommendations.append(
                    "Reduce learning rate by 5-10x and clip rewards"
                )
            else:
                analysis['has_nan'] = False

            # Check for explosions
            loss_std = loss['value'].std()
            loss_mean = loss['value'].mean()
            cv = loss_std / abs(loss_mean) if loss_mean != 0 else 0

            analysis['loss_cv'] = cv
            if cv > 2.0:
                analysis['stability'] = "Unstable"
                self.issues.append(f"High loss variance (CV={cv:.2f})")
                self.recommendations.append(
                    "Reduce learning rate or increase batch size"
                )
            else:
                analysis['stability'] = "Stable"

        # Convergence detection
        if 'rollout/ep_rew_mean' in self.metrics:
            rewards = self.metrics['rollout/ep_rew_mean']
            rolling_mean = rewards['value'].rolling(20).mean()

            # Detect plateau (no improvement in last 25% of training)
            quarter_point = len(rolling_mean) * 3 // 4
            recent_trend = rolling_mean.iloc[quarter_point:].diff().mean()

            if abs(recent_trend) < 0.01:
                analysis['converged'] = True
                analysis['convergence_step'] = quarter_point
            else:
                analysis['converged'] = False
                analysis['convergence_step'] = None

        return analysis

    def analyze_policy_quality(self) -> Dict:
        """Analyze policy learning quality."""
        analysis = {}

        # Entropy (exploration)
        if 'train/entropy_loss' in self.metrics:
            entropy = self.metrics['train/entropy_loss']

            initial_entropy = entropy['value'].iloc[:50].mean()
            final_entropy = entropy['value'].iloc[-50:].mean()

            analysis['initial_entropy'] = initial_entropy
            analysis['final_entropy'] = final_entropy

            # Check if crashed to zero
            if final_entropy < 0.01 and initial_entropy > 0.5:
                self.issues.append("Entropy collapsed to near-zero")
                self.recommendations.append(
                    "Increase ent_coef from 0.01 to 0.05 or higher"
                )

        # Value function accuracy
        if 'train/explained_variance' in self.metrics:
            explained_var = self.metrics['train/explained_variance']
            mean_ev = explained_var['value'].iloc[-100:].mean()

            analysis['explained_variance'] = mean_ev

            if mean_ev < 0:
                self.issues.append(f"Negative explained variance ({mean_ev:.2f})")
                self.recommendations.append(
                    "Value function predicting wrong direction - check reward function"
                )
            elif mean_ev < 0.3:
                self.issues.append(f"Low explained variance ({mean_ev:.2f})")
                self.recommendations.append(
                    "Increase vf_coef from 0.5 to 1.0 or use longer training"
                )

        # Policy gradient metrics
        if 'train/policy_gradient_loss' in self.metrics:
            pg_loss = self.metrics['train/policy_gradient_loss']
            analysis['final_pg_loss'] = pg_loss['value'].iloc[-50:].mean()

        return analysis

    def detect_issues(self) -> List[str]:
        """Detect common training issues."""
        detected_issues = []

        # Issue 1: No episode rewards recorded
        if 'rollout/ep_rew_mean' not in self.metrics:
            detected_issues.append("No episode rewards found in logs")

        # Issue 2: Very short training
        if self.metrics:
            first_metric = list(self.metrics.values())[0]
            if len(first_metric) < 100:
                detected_issues.append(
                    f"Very short training run ({len(first_metric)} steps)"
                )

        # Issue 3: Clip fraction too high
        if 'train/clip_fraction' in self.metrics:
            clip_frac = self.metrics['train/clip_fraction']
            mean_clip = clip_frac['value'].iloc[-100:].mean()

            if mean_clip > 0.3:
                detected_issues.append(
                    f"High clip fraction ({mean_clip:.2f}) - policy updates too large"
                )
                self.recommendations.append(
                    "Reduce learning rate or increase clip_range"
                )

        # Issue 4: Approx KL too high
        if 'train/approx_kl' in self.metrics:
            approx_kl = self.metrics['train/approx_kl']
            mean_kl = approx_kl['value'].iloc[-100:].mean()

            if mean_kl > 0.02:
                detected_issues.append(
                    f"High KL divergence ({mean_kl:.4f}) - policy changing too fast"
                )
                self.recommendations.append("Reduce learning rate by 2-3x")

        return detected_issues

    def generate_plots(self):
        """Generate diagnostic plots."""
        print("Generating plots...")

        # Plot 1: Training Progress (2x2 grid)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('RL Training Progress', fontsize=16)

        # Episode rewards
        if 'rollout/ep_rew_mean' in self.metrics:
            rewards = self.metrics['rollout/ep_rew_mean']
            axes[0, 0].plot(rewards['step'], rewards['value'])
            axes[0, 0].set_title('Episode Reward Mean')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)

        # Loss
        if 'train/loss' in self.metrics:
            loss = self.metrics['train/loss']
            axes[0, 1].plot(loss['step'], loss['value'])
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)

        # Entropy
        if 'train/entropy_loss' in self.metrics:
            entropy = self.metrics['train/entropy_loss']
            axes[1, 0].plot(entropy['step'], entropy['value'])
            axes[1, 0].set_title('Policy Entropy')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True, alpha=0.3)

        # Explained variance
        if 'train/explained_variance' in self.metrics:
            ev = self.metrics['train/explained_variance']
            axes[1, 1].plot(ev['step'], ev['value'])
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Explained Variance')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Explained Variance')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=100)
        print(f"Saved: {plot_path}")
        plt.close()

        # Plot 2: Policy Quality
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Policy Quality Metrics', fontsize=16)

        # Clip fraction
        if 'train/clip_fraction' in self.metrics:
            clip_frac = self.metrics['train/clip_fraction']
            axes[0].plot(clip_frac['step'], clip_frac['value'])
            axes[0].axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Target')
            axes[0].set_title('Clip Fraction')
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel('Fraction')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Approx KL
        if 'train/approx_kl' in self.metrics:
            kl = self.metrics['train/approx_kl']
            axes[1].plot(kl['step'], kl['value'])
            axes[1].axhline(y=0.02, color='r', linestyle='--', alpha=0.5, label='Threshold')
            axes[1].set_title('Approximate KL Divergence')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('KL')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / 'policy_quality.png'
        plt.savefig(plot_path, dpi=100)
        print(f"Saved: {plot_path}")
        plt.close()

    def generate_report(self):
        """Generate markdown report."""
        print("Generating report...")

        report = f"""# RL Training Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

"""

        # Training progress
        progress = self.analyze_training_progress()
        report += "### Training Progress\n\n"
        for key, value in progress.items():
            report += f"- **{key}**: {value}\n"

        # Policy quality
        policy = self.analyze_policy_quality()
        report += "\n### Policy Quality\n\n"
        for key, value in policy.items():
            if isinstance(value, float):
                report += f"- **{key}**: {value:.4f}\n"
            else:
                report += f"- **{key}**: {value}\n"

        # Issues
        all_issues = self.issues + self.detect_issues()
        if all_issues:
            report += "\n## Issues Detected\n\n"
            for i, issue in enumerate(all_issues, 1):
                report += f"{i}. ❌ {issue}\n"
        else:
            report += "\n## Issues Detected\n\n✅ No critical issues detected!\n"

        # Recommendations
        if self.recommendations:
            report += "\n## Recommendations\n\n"
            for i, rec in enumerate(self.recommendations, 1):
                report += f"{i}. {rec}\n"

        # Metrics summary
        report += "\n## Available Metrics\n\n"
        report += f"Total metrics tracked: {len(self.metrics)}\n\n"
        for metric_name in sorted(self.metrics.keys()):
            n_points = len(self.metrics[metric_name])
            report += f"- `{metric_name}` ({n_points} points)\n"

        # Save report
        report_path = self.output_dir / 'analysis_report.md'
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"Saved: {report_path}")
        return report

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("=" * 60)
        print("RL Training Analysis")
        print("=" * 60)

        # Load data
        self.load_tensorboard_data()

        if not self.metrics:
            print("Error: No metrics found. Exiting.")
            return

        # Analyze
        print("\nAnalyzing training progress...")
        progress = self.analyze_training_progress()

        print("\nAnalyzing policy quality...")
        policy = self.analyze_policy_quality()

        print("\nDetecting issues...")
        issues = self.detect_issues()

        # Generate outputs
        self.generate_plots()
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)

        # Print summary
        print(f"\nOutput directory: {self.output_dir}")
        print(f"\nFiles generated:")
        print(f"  - analysis_report.md")
        print(f"  - training_progress.png")
        print(f"  - policy_quality.png")

        if self.issues or issues:
            print(f"\n⚠️  {len(self.issues) + len(issues)} issue(s) detected")
            print("See analysis_report.md for details")
        else:
            print("\n✅ No critical issues detected")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RL training logs from TensorBoard"
    )
    parser.add_argument(
        '--tensorboard-dir',
        type=str,
        required=True,
        help='Path to tensorboard log directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./analysis/',
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate full analysis report'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = TrainingAnalyzer(args.tensorboard_dir, args.output_dir)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
