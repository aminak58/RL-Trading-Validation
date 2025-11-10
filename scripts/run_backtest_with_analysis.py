#!/usr/bin/env python3
"""
Automated Backtest + Analysis Pipeline

Runs a freqtrade backtest and automatically performs all skill-based analysis:
1. Runs backtest
2. Collects data during backtest
3. Analyzes training metrics (TensorBoard)
4. Calculates feature importance
5. Analyzes reward breakdown
6. Generates comprehensive report

Usage:
    python scripts/run_backtest_with_analysis.py \
        --timerange 20240101-20240401 \
        --strategy MtfScalper_RL_Hybrid \
        --config configs/config_rl_hybrid.json
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from user_data.data_collector import DataCollector, get_latest_session, analyze_session


def run_command(cmd: list, description: str, timeout: int = None) -> tuple[bool, str]:
    """
    Run a shell command and return success status and output.
    """
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )

        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True, result.stdout
        else:
            print(f"❌ {description} failed with code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False, str(e)


def run_backtest(config: str, timerange: str, strategy: str, freqaimodel: str = None,
                 train_enabled: bool = True, breakdown: str = None) -> bool:
    """
    Run freqtrade backtest.
    """
    cmd = [
        "freqtrade",
        "backtesting",
        "--config", config,
        "--strategy", strategy,
        "--timerange", timerange,
    ]

    if freqaimodel:
        cmd.extend(["--freqaimodel", freqaimodel])

    if train_enabled:
        cmd.append("--freqai-train-enabled")

    if breakdown:
        cmd.extend(["--breakdown", breakdown])

    success, output = run_command(
        cmd,
        f"Backtesting {strategy} on {timerange}",
        timeout=7200  # 2 hours max
    )

    return success


def analyze_training_logs(tensorboard_dir: str = "./tensorboard/") -> bool:
    """
    Analyze TensorBoard logs using analyze_training.py skill.
    """
    tensorboard_path = Path(tensorboard_dir)
    if not tensorboard_path.exists() or not list(tensorboard_path.glob("**/events.out.tfevents.*")):
        print("⚠️ No TensorBoard logs found, skipping training analysis")
        return True

    cmd = [
        "python",
        ".claude/skills/freqai-rl-optimizer/scripts/analyze_training.py",
        "--tensorboard-dir", tensorboard_dir,
        "--output-dir", "user_data/analysis_data/training_analysis/",
    ]

    success, output = run_command(
        cmd,
        "Analyzing Training Metrics",
        timeout=300
    )

    return success


def analyze_collected_data(session_id: str = None) -> bool:
    """
    Analyze collected data from DataCollector.
    """
    if session_id is None:
        session_id = get_latest_session()

    if session_id is None:
        print("⚠️ No data collection session found")
        return False

    print(f"\n{'='*60}")
    print(f"📊 Analyzing Collected Data (Session: {session_id})")
    print(f"{'='*60}\n")

    # Use data_collector's analyze_session function
    analyze_session(session_id)

    return True


def generate_comprehensive_report(output_dir: str = "user_data/analysis_data/reports/") -> bool:
    """
    Generate a comprehensive report combining all analyses.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    print(f"\n{'='*60}")
    print(f"📝 Generating Comprehensive Report")
    print(f"{'='*60}\n")

    # Get latest session
    session_id = get_latest_session()

    # Load data
    try:
        # Load summary
        summary_path = Path("user_data/analysis_data") / f"summary_{session_id}.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}

        # Create report
        report = f"""# Comprehensive Backtest Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session ID: {session_id}

---

## 📊 Executive Summary

### Data Collection
- Total Trades: {summary.get('data_counts', {}).get('trades', 0)}
- RL Episodes: {summary.get('data_counts', {}).get('rl_episodes', 0)}
- Predictions Logged: {summary.get('data_counts', {}).get('predictions', 0)}
- Reward Entries: {summary.get('data_counts', {}).get('reward_entries', 0)}

"""

        if 'trade_stats' in summary:
            stats = summary['trade_stats']
            report += f"""### Trading Performance
- **Win Rate**: {stats['win_rate']:.2%}
- **Total Trades**: {stats['total_trades']}
- **Average Profit**: {stats['avg_profit']:.2%}
- **Average Winner**: {stats['avg_winner']:.2%}
- **Average Loser**: {stats['avg_loser']:.2%}
- **Profit Factor**: {stats['profit_factor']:.2f}
- **Largest Winner**: {stats['largest_winner']:.2%}
- **Largest Loser**: {stats['largest_loser']:.2%}

"""

            # Diagnosis based on results
            report += "### 🔍 Performance Diagnosis\n\n"

            # Check for common issues
            if stats['win_rate'] > 0.5 and stats['avg_profit'] < 0:
                report += """**⚠️ CRITICAL ISSUE: High Win Rate but Negative Profit**

- **Problem**: Winning trades are too small, losing trades are too large
- **Root Cause**: Poor Risk/Reward ratio - likely exiting winners too early and holding losers too long
- **Recommendation**:
  1. Increase `timing_quality` reward weight from 0.20 to 0.30
  2. Add profit deterioration detection to exit conditions
  3. Reduce `entry_penalty_multiplier` to allow more learning

"""

            if stats['profit_factor'] < 1.0:
                report += "**❌ UNPROFITABLE**: Total losses exceed total profits\n\n"
            elif stats['profit_factor'] < 1.5:
                report += "**⚠️ MARGINALLY PROFITABLE**: Profit factor below healthy threshold\n\n"
            else:
                report += "**✅ PROFITABLE**: Good profit factor\n\n"

            if stats['avg_winner'] > 0 and stats['avg_loser'] < 0:
                rr_ratio = abs(stats['avg_winner'] / stats['avg_loser'])
                report += f"**Risk/Reward Ratio**: {rr_ratio:.2f}\n"
                if rr_ratio < 1.5:
                    report += "  - ⚠️ Poor R:R - Winners not big enough compared to losers\n"
                elif rr_ratio > 2.0:
                    report += "  - ✅ Good R:R - Winners significantly larger than losers\n"
                report += "\n"

        if 'rl_stats' in summary:
            stats = summary['rl_stats']
            report += f"""### RL Training Performance
- **Total Episodes**: {stats['total_episodes']}
- **Average Episode Reward**: {stats['avg_episode_reward']:.2f}
- **Std Dev**: {stats['std_episode_reward']:.2f}
- **Best Episode**: {stats['max_episode_reward']:.2f}
- **Worst Episode**: {stats['min_episode_reward']:.2f}

"""

            # RL diagnosis
            if stats['avg_episode_reward'] < 0:
                report += "**⚠️ NEGATIVE AVERAGE REWARD**: Model may not be learning effectively\n\n"
            elif stats['avg_episode_reward'] < 1.0:
                report += "**⚠️ LOW AVERAGE REWARD**: Model learning is weak\n\n"
            else:
                report += "**✅ POSITIVE LEARNING**: Model is learning from experience\n\n"

        if 'reward_component_stats' in summary:
            report += "### Reward Components Analysis\n\n"
            report += "| Component | Mean | Std | Min | Max |\n"
            report += "|-----------|------|-----|-----|-----|\n"

            for comp, stats in summary['reward_component_stats'].items():
                report += f"| {comp} | {stats['mean']:.2f} | {stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n"

            report += "\n"

        # Recommendations section
        report += """---

## 💡 Actionable Recommendations

### Priority 1: Immediate Actions
"""

        if 'trade_stats' in summary:
            stats = summary['trade_stats']
            if stats['win_rate'] > 0.5 and stats['avg_profit'] < 0:
                report += """
1. **Fix Risk/Reward Imbalance**
   ```python
   # In MtfScalperRLModel.py, adjust reward weights:
   reward_weights = {
       "profit": 0.45,  # Increase from 0.35
       "drawdown_control": 0.20,  # Decrease from 0.25
       "timing_quality": 0.25,  # Increase from 0.20
       "risk_reward_ratio": 0.10,  # Decrease from 0.20
   }
   ```

2. **Add Profit Protection**
   ```python
   # In strategy custom_exit(), strengthen profit protection:
   if current_profit > 0.01:  # 1% profit
       if current_profit < 0.003:  # Dropped below 0.3%
           return "profit_protection"
   ```

3. **Simplify Reward Function**
   - Test with profit-only reward first
   - Gradually add complexity
"""

        report += """
### Priority 2: Testing & Validation

1. **Walk-Forward Validation**
   ```bash
   # Test on different time periods
   python scripts/run_backtest_with_analysis.py --timerange 20240401-20240701
   ```

2. **Feature Importance Analysis**
   ```bash
   # Run feature importance to identify key predictors
   python .claude/skills/freqai-rl-optimizer/scripts/feature_importance.py \\
       --model-dir user_data/models/ \\
       --pair BTC/USDT:USDT
   ```

3. **Hyperparameter Optimization**
   ```bash
   # Scan reward weight space
   python .claude/skills/freqai-rl-optimizer/scripts/hyperparameter_scanner.py \\
       --params reward_weights.profit,reward_weights.timing_quality \\
       --ranges 0.3:0.6,0.15:0.35 \\
       --trials 10
   ```

---

## 📁 Generated Artifacts

### Data Files
- `user_data/analysis_data/trades_{session_id}.csv` - All trade details
- `user_data/analysis_data/predictions_{session_id}.csv` - RL predictions
- `user_data/analysis_data/rl_episodes_{session_id}.json` - Training episodes
- `user_data/analysis_data/reward_breakdown_{session_id}.json` - Reward details

### Analysis Reports
- `user_data/analysis_data/training_analysis/` - TensorBoard analysis
- `user_data/analysis_data/summary_{session_id}.json` - Summary statistics

### Visualization
- Open TensorBoard for training metrics: `tensorboard --logdir ./tensorboard/`

---

## 🔄 Next Steps

1. Review this report and implement Priority 1 recommendations
2. Run walk-forward validation on different time periods
3. Analyze feature importance to reduce overfitting
4. Test simplified reward functions
5. Monitor live paper trading before going live

---

**Report Location**: `{report_file}`

"""

        # Write report
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"✅ Comprehensive report generated: {report_file}")
        print(f"\n📄 View report: cat {report_file}")

        return True

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest with automated analysis pipeline"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_rl_hybrid.json',
        help='Path to freqtrade config file'
    )
    parser.add_argument(
        '--timerange',
        type=str,
        required=True,
        help='Timerange for backtest (e.g., 20240101-20240401)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='MtfScalper_RL_Hybrid',
        help='Strategy name'
    )
    parser.add_argument(
        '--freqaimodel',
        type=str,
        default='MtfScalperRLModel',
        help='FreqAI model name'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip RL training (use existing model)'
    )
    parser.add_argument(
        '--breakdown',
        type=str,
        choices=['day', 'week', 'month'],
        help='Breakdown period for backtest results'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip post-backtest analysis'
    )

    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║        AUTOMATED BACKTEST + ANALYSIS PIPELINE             ║
╚═══════════════════════════════════════════════════════════╝

Configuration:
  - Strategy: {args.strategy}
  - Timerange: {args.timerange}
  - Config: {args.config}
  - FreqAI Model: {args.freqaimodel}
  - Training: {'Disabled' if args.skip_training else 'Enabled'}

""")

    start_time = time.time()

    # Step 1: Run backtest
    success = run_backtest(
        config=args.config,
        timerange=args.timerange,
        strategy=args.strategy,
        freqaimodel=args.freqaimodel,
        train_enabled=not args.skip_training,
        breakdown=args.breakdown
    )

    if not success:
        print("\n❌ Backtest failed, aborting pipeline")
        sys.exit(1)

    if args.skip_analysis:
        print("\n⏩ Skipping analysis (--skip-analysis flag)")
        sys.exit(0)

    # Step 2: Analyze training logs
    analyze_training_logs()

    # Step 3: Analyze collected data
    analyze_collected_data()

    # Step 4: Generate comprehensive report
    generate_comprehensive_report()

    # Done
    elapsed_time = time.time() - start_time
    print(f"""

╔═══════════════════════════════════════════════════════════╗
║                    PIPELINE COMPLETE                      ║
╚═══════════════════════════════════════════════════════════╝

Total time: {elapsed_time/60:.1f} minutes

📊 Results:
  - Trade data: user_data/analysis_data/trades_*.csv
  - RL episodes: user_data/analysis_data/rl_episodes_*.json
  - Comprehensive report: user_data/analysis_data/reports/comprehensive_report_*.md

🎯 Next actions:
  1. Review comprehensive report
  2. Check TensorBoard: tensorboard --logdir ./tensorboard/
  3. Analyze feature importance
  4. Run walk-forward validation

""")


if __name__ == '__main__':
    main()
