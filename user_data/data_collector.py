#!/usr/bin/env python3
"""
Data Collection System for RL Trading Analysis

Collects and stores data needed for skill-based analysis:
- Trade-level metrics
- RL episode data
- Feature importance data
- Reward breakdown
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataCollector:
    """
    Centralized data collection for all analysis skills.

    Usage in strategy:
        from user_data.data_collector import DataCollector
        self.data_collector = DataCollector()

        # During trade
        self.data_collector.log_trade(trade_data)

        # End of backtest
        self.data_collector.save_all()
    """

    def __init__(self, output_dir: str = "user_data/analysis_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.trades = []
        self.rl_episodes = []
        self.feature_importance_data = []
        self.reward_breakdown = []
        self.predictions = []

        # Metadata
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = {}

        logger.info(f"DataCollector initialized: {self.output_dir}")

    # ═══════════════════════════════════════════════════════════
    # TRADE-LEVEL LOGGING (for feature_importance.py)
    # ═══════════════════════════════════════════════════════════

    def log_trade(self, trade_data: Dict[str, Any]):
        """
        Log complete trade information.

        Args:
            trade_data: {
                'pair': 'BTC/USDT:USDT',
                'entry_time': datetime,
                'exit_time': datetime,
                'entry_price': 50000.0,
                'exit_price': 50500.0,
                'profit_abs': 10.0,
                'profit_pct': 0.01,
                'duration_candles': 120,
                'is_short': False,
                'exit_reason': 'rl_exit',
                'features_at_entry': {'%-momentum_5': 0.01, ...},
                'features_at_exit': {'%-momentum_5': -0.005, ...},
                'rl_action': 3,
                'rl_confidence': 0.85,
            }
        """
        self.trades.append(trade_data)

    def log_prediction(self, prediction_data: Dict[str, Any]):
        """
        Log RL model prediction at each step.

        Args:
            prediction_data: {
                'timestamp': datetime,
                'pair': 'BTC/USDT:USDT',
                'features': {'%-momentum_5': 0.01, ...},
                'prediction': 3,
                'confidence': 0.85,
                'current_position': 1,  # 1=long, -1=short, 0=none
                'current_profit': 0.005,
            }
        """
        self.predictions.append(prediction_data)

    # ═══════════════════════════════════════════════════════════
    # RL EPISODE LOGGING (for reward_backtest.py)
    # ═══════════════════════════════════════════════════════════

    def log_episode_start(self, episode_data: Dict[str, Any]):
        """
        Log start of RL training episode.

        Args:
            episode_data: {
                'episode_id': 1,
                'pair': 'BTC/USDT:USDT',
                'start_step': 0,
                'initial_balance': 1000.0,
            }
        """
        self.rl_episodes.append({
            **episode_data,
            'steps': [],
            'total_reward': 0.0,
            'end_balance': None,
        })

    def log_episode_step(self, episode_id: int, step_data: Dict[str, Any]):
        """
        Log individual step within episode.

        Args:
            episode_id: Episode identifier
            step_data: {
                'step': 100,
                'action': 3,
                'reward': 2.5,
                'reward_components': {
                    'profit': 1.5,
                    'drawdown': 0.5,
                    'timing': 0.3,
                    'risk_reward': 0.2,
                },
                'state': {
                    'position': 1,
                    'entry_price': 50000,
                    'current_price': 50500,
                    'duration': 50,
                },
                'done': False,
            }
        """
        if episode_id < len(self.rl_episodes):
            episode = self.rl_episodes[episode_id]
            episode['steps'].append(step_data)
            episode['total_reward'] += step_data['reward']

    def log_episode_end(self, episode_id: int, end_data: Dict[str, Any]):
        """
        Log end of episode.

        Args:
            end_data: {
                'end_balance': 1050.0,
                'total_trades': 10,
                'win_rate': 0.6,
            }
        """
        if episode_id < len(self.rl_episodes):
            self.rl_episodes[episode_id].update(end_data)

    # ═══════════════════════════════════════════════════════════
    # REWARD BREAKDOWN (detailed analysis)
    # ═══════════════════════════════════════════════════════════

    def log_reward_calculation(self, reward_data: Dict[str, Any]):
        """
        Log detailed reward calculation for analysis.

        Args:
            reward_data: {
                'timestamp': datetime,
                'pair': 'BTC/USDT:USDT',
                'action': 3,
                'total_reward': 2.5,
                'components': {
                    'profit_score': 1.5,
                    'drawdown_score': 0.5,
                    'timing_score': 0.3,
                    'risk_reward_score': 0.2,
                },
                'weights': {
                    'profit': 0.35,
                    'drawdown': 0.25,
                    'timing': 0.20,
                    'risk_reward': 0.20,
                },
                'context': {
                    'position': 1,
                    'profit': 0.01,
                    'duration': 50,
                },
            }
        """
        self.reward_breakdown.append(reward_data)

    # ═══════════════════════════════════════════════════════════
    # CONFIGURATION TRACKING
    # ═══════════════════════════════════════════════════════════

    def set_config(self, config: Dict[str, Any]):
        """Store configuration for this run."""
        self.config = config

    # ═══════════════════════════════════════════════════════════
    # DATA EXPORT
    # ═══════════════════════════════════════════════════════════

    def save_all(self, custom_name: Optional[str] = None):
        """
        Save all collected data to disk.

        Creates files:
            - trades_YYYYMMDD_HHMMSS.json
            - rl_episodes_YYYYMMDD_HHMMSS.json
            - predictions_YYYYMMDD_HHMMSS.json
            - reward_breakdown_YYYYMMDD_HHMMSS.json
            - summary_YYYYMMDD_HHMMSS.json
        """
        suffix = custom_name or self.session_id

        # Save trades
        if self.trades:
            trades_path = self.output_dir / f"trades_{suffix}.json"
            self._save_json(trades_path, self.trades)
            logger.info(f"Saved {len(self.trades)} trades to {trades_path}")

        # Save RL episodes
        if self.rl_episodes:
            episodes_path = self.output_dir / f"rl_episodes_{suffix}.json"
            self._save_json(episodes_path, self.rl_episodes)
            logger.info(f"Saved {len(self.rl_episodes)} episodes to {episodes_path}")

        # Save predictions
        if self.predictions:
            predictions_path = self.output_dir / f"predictions_{suffix}.json"
            self._save_json(predictions_path, self.predictions)
            logger.info(f"Saved {len(self.predictions)} predictions to {predictions_path}")

        # Save reward breakdown
        if self.reward_breakdown:
            rewards_path = self.output_dir / f"reward_breakdown_{suffix}.json"
            self._save_json(rewards_path, self.reward_breakdown)
            logger.info(f"Saved {len(self.reward_breakdown)} reward entries to {rewards_path}")

        # Save summary
        summary = self._generate_summary()
        summary_path = self.output_dir / f"summary_{suffix}.json"
        self._save_json(summary_path, summary)
        logger.info(f"Saved summary to {summary_path}")

        # Also save as CSV for easier analysis
        self._save_as_csv(suffix)

    def _save_json(self, path: Path, data: Any):
        """Save data as JSON with datetime handling."""
        def default_serializer(obj):
            if isinstance(obj, (datetime, pd.Timestamp)):
                return obj.isoformat()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=default_serializer)

    def _save_as_csv(self, suffix: str):
        """Save data as CSV for easier analysis in pandas."""

        # Trades CSV
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_csv = self.output_dir / f"trades_{suffix}.csv"
            trades_df.to_csv(trades_csv, index=False)
            logger.info(f"Saved trades CSV: {trades_csv}")

        # Predictions CSV
        if self.predictions:
            # Flatten features dict
            predictions_flat = []
            for pred in self.predictions:
                flat = {
                    'timestamp': pred['timestamp'],
                    'pair': pred['pair'],
                    'prediction': pred['prediction'],
                    'confidence': pred['confidence'],
                    'current_position': pred.get('current_position', 0),
                    'current_profit': pred.get('current_profit', 0),
                }
                # Add features
                if 'features' in pred:
                    flat.update(pred['features'])
                predictions_flat.append(flat)

            predictions_df = pd.DataFrame(predictions_flat)
            predictions_csv = self.output_dir / f"predictions_{suffix}.csv"
            predictions_df.to_csv(predictions_csv, index=False)
            logger.info(f"Saved predictions CSV: {predictions_csv}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'data_counts': {
                'trades': len(self.trades),
                'rl_episodes': len(self.rl_episodes),
                'predictions': len(self.predictions),
                'reward_entries': len(self.reward_breakdown),
            },
        }

        # Trade statistics
        if self.trades:
            profits = [t['profit_pct'] for t in self.trades if 'profit_pct' in t]
            if profits:
                summary['trade_stats'] = {
                    'total_trades': len(self.trades),
                    'winning_trades': sum(1 for p in profits if p > 0),
                    'losing_trades': sum(1 for p in profits if p < 0),
                    'win_rate': sum(1 for p in profits if p > 0) / len(profits),
                    'avg_profit': np.mean(profits),
                    'avg_winner': np.mean([p for p in profits if p > 0]) if any(p > 0 for p in profits) else 0,
                    'avg_loser': np.mean([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 0,
                    'largest_winner': max(profits),
                    'largest_loser': min(profits),
                    'profit_factor': abs(sum(p for p in profits if p > 0) / sum(p for p in profits if p < 0)) if sum(p for p in profits if p < 0) != 0 else 0,
                }

        # RL episode statistics
        if self.rl_episodes:
            episode_rewards = [ep['total_reward'] for ep in self.rl_episodes if 'total_reward' in ep]
            if episode_rewards:
                summary['rl_stats'] = {
                    'total_episodes': len(self.rl_episodes),
                    'avg_episode_reward': np.mean(episode_rewards),
                    'std_episode_reward': np.std(episode_rewards),
                    'min_episode_reward': min(episode_rewards),
                    'max_episode_reward': max(episode_rewards),
                }

        # Reward component statistics
        if self.reward_breakdown:
            components = ['profit_score', 'drawdown_score', 'timing_score', 'risk_reward_score']
            component_stats = {}
            for comp in components:
                values = [r['components'].get(comp, 0) for r in self.reward_breakdown if 'components' in r]
                if values:
                    component_stats[comp] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': min(values),
                        'max': max(values),
                    }
            summary['reward_component_stats'] = component_stats

        return summary

    # ═══════════════════════════════════════════════════════════
    # DATA LOADING (for skills)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def load_trades(session_id: str, data_dir: str = "user_data/analysis_data") -> pd.DataFrame:
        """Load trades data for analysis."""
        path = Path(data_dir) / f"trades_{session_id}.csv"
        if path.exists():
            return pd.read_csv(path)
        # Try JSON
        path = Path(data_dir) / f"trades_{session_id}.json"
        if path.exists():
            with open(path, 'r') as f:
                return pd.DataFrame(json.load(f))
        raise FileNotFoundError(f"No trades data found for session {session_id}")

    @staticmethod
    def load_episodes(session_id: str, data_dir: str = "user_data/analysis_data") -> List[Dict]:
        """Load RL episodes data for analysis."""
        path = Path(data_dir) / f"rl_episodes_{session_id}.json"
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        raise FileNotFoundError(f"No episodes data found for session {session_id}")

    @staticmethod
    def load_predictions(session_id: str, data_dir: str = "user_data/analysis_data") -> pd.DataFrame:
        """Load predictions data for analysis."""
        path = Path(data_dir) / f"predictions_{session_id}.csv"
        if path.exists():
            return pd.read_csv(path)
        # Try JSON
        path = Path(data_dir) / f"predictions_{session_id}.json"
        if path.exists():
            with open(path, 'r') as f:
                return pd.DataFrame(json.load(f))
        raise FileNotFoundError(f"No predictions data found for session {session_id}")

    @staticmethod
    def list_available_sessions(data_dir: str = "user_data/analysis_data") -> List[str]:
        """List all available session IDs."""
        data_path = Path(data_dir)
        if not data_path.exists():
            return []

        # Find all summary files
        summary_files = list(data_path.glob("summary_*.json"))
        sessions = [f.stem.replace("summary_", "") for f in summary_files]
        return sorted(sessions, reverse=True)  # Most recent first


# ═══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def get_latest_session(data_dir: str = "user_data/analysis_data") -> Optional[str]:
    """Get the most recent session ID."""
    sessions = DataCollector.list_available_sessions(data_dir)
    return sessions[0] if sessions else None


def analyze_session(session_id: Optional[str] = None, data_dir: str = "user_data/analysis_data"):
    """
    Quick analysis of a session.

    Usage:
        from user_data.data_collector import analyze_session
        analyze_session()  # Analyzes latest session
    """
    if session_id is None:
        session_id = get_latest_session(data_dir)
        if session_id is None:
            print("No sessions found!")
            return

    print(f"\n{'='*60}")
    print(f"Analysis for session: {session_id}")
    print(f"{'='*60}\n")

    # Load summary
    summary_path = Path(data_dir) / f"summary_{session_id}.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)

        print("📊 Data Counts:")
        for key, value in summary.get('data_counts', {}).items():
            print(f"  - {key}: {value}")

        if 'trade_stats' in summary:
            print("\n💰 Trade Statistics:")
            stats = summary['trade_stats']
            print(f"  - Total Trades: {stats['total_trades']}")
            print(f"  - Win Rate: {stats['win_rate']:.2%}")
            print(f"  - Avg Profit: {stats['avg_profit']:.2%}")
            print(f"  - Avg Winner: {stats['avg_winner']:.2%}")
            print(f"  - Avg Loser: {stats['avg_loser']:.2%}")
            print(f"  - Profit Factor: {stats['profit_factor']:.2f}")
            print(f"  - Largest Winner: {stats['largest_winner']:.2%}")
            print(f"  - Largest Loser: {stats['largest_loser']:.2%}")

        if 'rl_stats' in summary:
            print("\n🤖 RL Statistics:")
            stats = summary['rl_stats']
            print(f"  - Total Episodes: {stats['total_episodes']}")
            print(f"  - Avg Episode Reward: {stats['avg_episode_reward']:.2f}")
            print(f"  - Std Episode Reward: {stats['std_episode_reward']:.2f}")
            print(f"  - Best Episode: {stats['max_episode_reward']:.2f}")
            print(f"  - Worst Episode: {stats['min_episode_reward']:.2f}")

        if 'reward_component_stats' in summary:
            print("\n🎯 Reward Components:")
            for comp, stats in summary['reward_component_stats'].items():
                print(f"  - {comp}: {stats['mean']:.2f} ± {stats['std']:.2f}")

    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    # Example: Analyze latest session
    analyze_session()
