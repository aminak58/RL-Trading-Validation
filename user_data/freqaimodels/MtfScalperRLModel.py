"""
MtfScalper RL Model
==================
Custom Reinforcement Learning model for intelligent exit optimization.
Uses PPO algorithm with 5-action space and multi-factor reward function.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch as th

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Base5ActionRLEnv, Actions

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

# Import data collector
import sys
import os
# Dynamic path resolution - works on any system
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from user_data.data_collector import DataCollector

logger = logging.getLogger(__name__)


class MtfScalperRLModel(ReinforcementLearner):
    """
    Custom RL model for MtfScalper exit optimization
    
    Key Features:
    - 5-action space (Hold, Long, Short, Exit Long, Exit Short)
    - Advanced reward function focused on exit quality
    - Soft constraints for entry actions
    - Position-aware state representation
    """
    
    def __init__(self, **kwargs):
        """Initialize the RL model with custom configuration"""
        super().__init__(**kwargs)

        # Model configuration
        self.model_type = "PPO"
        self.policy_type = "MlpPolicy"

        # Required RL attributes
        self.window_size = 30  # Default window size for observation
        self.max_position_duration = 300  # Max 300 candles (25 hours)

        # Training parameters - OPTIMIZED FOR CPU
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.n_steps = kwargs.get('n_steps', 4096)  # Increased for CPU efficiency
        self.batch_size = kwargs.get('batch_size', 256)  # Increased for CPU vectorization
        self.n_epochs = kwargs.get('n_epochs', 10)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_range = kwargs.get('clip_range', 0.2)
        self.vf_coef = kwargs.get('vf_coef', 0.5)
        self.ent_coef = kwargs.get('ent_coef', 0.01)
        self.net_arch = kwargs.get('net_arch', [512, 512, 256])  # Larger network for 40+ features
        self.tensorboard_log = kwargs.get('tensorboard_log', None)
        self.model_save_path = kwargs.get('model_save_path', 'user_data/models/')
        self.fee = kwargs.get('fee', 0.001)

        # CPU optimization parameters
        self.cpu_count = kwargs.get('cpu_count', 8)
        self.n_envs = kwargs.get('n_envs', 8)  # Number of parallel environments

        # Device configuration (GPU/CPU selection)
        self.device = self._determine_device(kwargs.get('device', 'auto'))
        logger.info(f"Using device: {self.device}")

        # Configure PyTorch for CPU optimization
        if self.device == "cpu":
            th.set_num_threads(self.cpu_count)
            th.set_num_interop_threads(2)

            # Additional CPU optimizations for Intel processors
            # Enable MKL (Math Kernel Library) optimizations if available
            try:
                import torch
                # Optimize for Intel CPUs
                torch.set_flush_denormal(True)  # Faster float operations
                if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
                    torch.backends.mkl.enabled = True
                    logger.info("Intel MKL optimizations enabled")
            except Exception as e:
                logger.debug(f"Advanced CPU optimizations not available: {e}")

            logger.info(f"PyTorch configured for CPU: {self.cpu_count} threads, 2 interop threads")

        # Custom reward weights
        self.reward_weights = {
            "profit": 0.35,
            "drawdown_control": 0.25,
            "timing_quality": 0.20,
            "risk_reward_ratio": 0.20
        }

        # Entry constraint parameters
        self.entry_penalty_multiplier = 15.0
        self.classic_signal_reward = 5.0  # Increased from 2.0 to encourage signal-following

        # Exit optimization parameters
        self.exit_profit_threshold = 0.02  # 2% profit for quality exit
        self.time_penalty_start = 100  # Start penalizing after 100 candles

        logger.info("MtfScalperRLModel initialized with custom reward function")

    def _determine_device(self, device_setting: str) -> str:
        """
        Determine the appropriate device for RL training based on setting and availability
        """
        import torch as th

        if device_setting == "auto":
            # Auto-detect based on availability and policy type
            if th.cuda.is_available():
                # For MLP policy, CPU is often faster unless you have a powerful GPU
                # But we'll use GPU if available to avoid warnings
                logger.info("GPU available, using CUDA for training")
                return "cuda"
            else:
                logger.info("GPU not available, using CPU for training")
                return "cpu"

        elif device_setting == "cpu":
            logger.info("CPU forced via configuration")
            return "cpu"

        elif device_setting == "cuda":
            if th.cuda.is_available():
                logger.info("GPU forced via configuration and available")
                return "cuda"
            else:
                logger.warning("GPU forced but not available, falling back to CPU")
                return "cpu"

        elif device_setting == "mps":
            # For Apple Silicon Macs
            if th.backends.mps.is_available():
                logger.info("Apple Silicon GPU available, using MPS")
                return "mps"
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"

        else:
            # Default to auto if invalid setting
            logger.warning(f"Invalid device setting '{device_setting}', using auto-detection")
            return self._determine_device("auto")
    
    class MtfScalperRLEnv(Base5ActionRLEnv):
        """
        Custom RL Environment optimized for exit decisions
        
        Modifications:
        - Advanced reward calculation
        - Soft entry constraints
        - Position duration tracking
        - Market regime awareness
        """
        
        def __init__(self, *args, **kwargs):
            """Initialize custom environment"""
            super().__init__(*args, **kwargs)

            # Track position information
            self.position_start_price = None
            self.position_start_step = None
            self.max_profit_seen = 0

            # Reward parameters (can be overridden)
            self.reward_weights = kwargs.get('reward_weights', {
                "profit": 0.35,
                "drawdown_control": 0.25,
                "timing_quality": 0.20,
                "risk_reward_ratio": 0.20
            })

            self.entry_penalty = kwargs.get('entry_penalty', 15.0)
            self.classic_signal_reward = kwargs.get('classic_signal_reward', 2.0)

            # FIXED: Add max_position_duration from config
            # Extract from config if available, otherwise use default
            config = kwargs.get('config', {})
            self.max_position_duration = config.get('freqai', {}).get('rl_config', {}).get('max_trade_duration_candles', 300)

            # Data collection
            self.data_collector = kwargs.get('data_collector', None)
            self.episode_id = 0
            self.current_episode_steps = []
            
        def calculate_reward(self, action: int) -> float:
            """
            Advanced reward function optimized for exit quality
            
            Key Components:
            1. Profit/Loss component
            2. Drawdown control
            3. Exit timing quality
            4. Risk/Reward ratio
            5. Entry constraints (soft)
            """
            
            # Get current state
            current_price = self.prices.iloc[self._current_tick]
            current_profit = self._calculate_current_profit()
            
            # ═══════════════════════════════════════════════════════════
            # ENTRY ACTION HANDLING (Soft Constraints)
            # ═══════════════════════════════════════════════════════════
            
            if action in [Actions.Long_enter, Actions.Short_enter]:
                # Check if classic entry signal exists
                classic_entry_signal = self._check_classic_entry_signal()

                if not self._is_valid(action):
                    # Invalid entry (already in position)
                    return -10.0

                if not classic_entry_signal:
                    # Stronger penalty to discourage random entries
                    # BUT: Now that RL can SEE signals, this penalty is fair
                    penalty = -5.0  # Increased from -1.0 (was too lenient)
                    return penalty
                else:
                    # CRITICAL FIX: Much higher reward for following classic signals
                    # Makes entering WITH signal more attractive than holding (0.0)
                    # Hold=0, Hold with opportunity cost=-2.0, Enter WITH signal=+15.0
                    return 15.0  # Increased from 5.0 to make entry very attractive

            # ═══════════════════════════════════════════════════════════
            # EXIT ACTION HANDLING (Main Focus)
            # ═══════════════════════════════════════════════════════════

            if action in [Actions.Long_exit, Actions.Short_exit]:
                if not self._is_valid(action):
                    # Invalid exit (not in matching position)
                    return -5.0

                # Calculate multi-factor exit reward
                exit_reward = self._calculate_exit_quality_reward(current_profit)
                return exit_reward

            # ═══════════════════════════════════════════════════════════
            # HOLD ACTION
            # ═══════════════════════════════════════════════════════════

            if action == Actions.Neutral:
                if self._position != 0:
                    # In position - evaluate holding cost
                    holding_reward = self._calculate_holding_reward(current_profit)
                    return holding_reward
                else:
                    # Not in position - check for missed opportunity
                    if classic_entry_signal:
                        # Penalty for holding when there's a tradeable signal (opportunity cost)
                        return -2.0
                    else:
                        # Neutral reward for waiting when no signal
                        return 0.0  # Changed from 0.01 to not reward inaction

            # Small reward for any valid action to encourage exploration
            reward = 0.01

            # Log reward calculation if data collector available
            if self.data_collector:
                try:
                    current_price = self.prices.iloc[self._current_tick]
                    current_profit = self._calculate_current_profit()

                    self.data_collector.log_reward_calculation({
                        'timestamp': self._current_tick,
                        'pair': getattr(self, 'pair', 'unknown'),
                        'action': action,
                        'total_reward': reward,
                        'components': {
                            'profit_score': 0,
                            'drawdown_score': 0,
                            'timing_score': 0,
                            'risk_reward_score': 0,
                        },
                        'weights': self.reward_weights,
                        'context': {
                            'position': self._position,
                            'profit': current_profit,
                            'duration': self._current_tick - self.position_start_step if self.position_start_step else 0,
                            'price': float(current_price),
                        },
                    })
                except Exception as e:
                    logger.debug(f"Error logging reward: {e}")

            return reward
        
        def _calculate_exit_quality_reward(self, current_profit: float) -> float:
            """
            Calculate multi-dimensional exit quality score
            """
            
            # Component 1: Profit Score (35%)
            profit_score = self._calculate_profit_score(current_profit)
            
            # Component 2: Drawdown Control (25%)
            drawdown_score = self._calculate_drawdown_score()
            
            # Component 3: Timing Quality (20%)
            timing_score = self._calculate_timing_score()
            
            # Component 4: Risk/Reward Ratio (20%)
            risk_reward_score = self._calculate_risk_reward_score(current_profit)
            
            # Weighted combination
            total_reward = (
                self.reward_weights["profit"] * profit_score +
                self.reward_weights["drawdown_control"] * drawdown_score +
                self.reward_weights["timing_quality"] * timing_score +
                self.reward_weights["risk_reward_ratio"] * risk_reward_score
            )

            # Log detailed reward breakdown if data collector available
            if self.data_collector:
                try:
                    current_price = self.prices.iloc[self._current_tick]

                    self.data_collector.log_reward_calculation({
                        'timestamp': self._current_tick,
                        'pair': getattr(self, 'pair', 'unknown'),
                        'action': 'exit',  # This is called during exit
                        'total_reward': total_reward,
                        'components': {
                            'profit_score': profit_score,
                            'drawdown_score': drawdown_score,
                            'timing_score': timing_score,
                            'risk_reward_score': risk_reward_score,
                        },
                        'weights': self.reward_weights,
                        'context': {
                            'position': self._position,
                            'profit': current_profit,
                            'duration': self._current_tick - self.position_start_step if self.position_start_step else 0,
                            'price': float(current_price),
                            'max_profit_seen': self.max_profit_seen,
                        },
                    })
                except Exception as e:
                    logger.debug(f"Error logging exit reward: {e}")

            return total_reward
        
        def _calculate_profit_score(self, profit: float) -> float:
            """
            Enhanced profit scoring with more balanced rewards for better RL learning
            """
            if profit <= -0.05:  # -5% or worse
                return -5.0
            elif profit <= -0.02:  # -2% to -5%
                return -2.0
            elif profit <= -0.005:  # -0.5% to -2%
                return -0.5
            elif profit <= 0:  # Small loss to breakeven
                return profit * 20  # Reduced penalty
            elif profit < 0.002:  # 0 to 0.2% (very small profit)
                return 0.5 + profit * 100  # Small but positive reward
            elif profit < 0.01:  # 0.2% to 1% (small profit)
                return 1.0 + profit * 150  # Good reward
            elif profit < 0.02:  # 1% to 2% (good profit)
                return 3.0 + profit * 100  # Strong reward
            elif profit < 0.04:  # 2% to 4% (excellent profit)
                return 5.0 + profit * 50  # Very strong reward
            else:  # 4%+ (outstanding profit)
                return min(15.0, 7.0 + profit * 30)  # Capped but generous reward
        
        def _calculate_drawdown_score(self) -> float:
            """
            Score based on drawdown control
            Penalizes positions that experienced large drawdowns
            """
            if self.position_start_step is None:
                return 0.0
            
            # Calculate maximum drawdown during position
            position_prices = self.prices.iloc[self.position_start_step:self._current_tick + 1]
            
            if self._position == 1:  # Long
                max_price = position_prices.max()
                current_price = self.prices.iloc[self._current_tick]
                drawdown = (max_price - current_price) / max_price if max_price > 0 else 0
            else:  # Short
                min_price = position_prices.min()
                current_price = self.prices.iloc[self._current_tick]
                drawdown = (current_price - min_price) / min_price if min_price > 0 else 0
            
            # Score calculation (lower drawdown = better score)
            if drawdown < 0.005:  # Less than 0.5%
                return 5.0
            elif drawdown < 0.01:  # Less than 1%
                return 2.0
            elif drawdown < 0.02:  # Less than 2%
                return 0.0
            else:  # More than 2%
                return -5.0 * drawdown
        
        def _calculate_timing_score(self) -> float:
            """
            Score based on exit timing
            Rewards exits near local extremes
            """
            if self._current_tick < 20:
                return 0.0
            
            # Look at recent price action
            lookback = 20
            recent_prices = self.prices.iloc[max(0, self._current_tick - lookback):self._current_tick + 1]
            current_price = self.prices.iloc[self._current_tick]
            
            if self._position == 1:  # Long position
                # Good timing if exiting near recent high
                price_percentile = (current_price - recent_prices.min()) / (recent_prices.max() - recent_prices.min() + 1e-10)
                if price_percentile > 0.8:  # Top 20% of recent range
                    return 5.0
                elif price_percentile > 0.6:
                    return 2.0
                else:
                    return 0.0
            
            elif self._position == -1:  # Short position
                # Good timing if exiting near recent low
                price_percentile = (current_price - recent_prices.min()) / (recent_prices.max() - recent_prices.min() + 1e-10)
                if price_percentile < 0.2:  # Bottom 20% of recent range
                    return 5.0
                elif price_percentile < 0.4:
                    return 2.0
                else:
                    return 0.0
            
            return 0.0
        
        def _calculate_risk_reward_score(self, current_profit: float) -> float:
            """
            Score based on risk/reward ratio achieved
            """
            if self.position_start_step is None:
                return 0.0
            
            # Calculate maximum adverse excursion (risk taken)
            position_prices = self.prices.iloc[self.position_start_step:self._current_tick + 1]
            
            if self._position == 1:  # Long
                min_price = position_prices.min()
                max_risk = (self.position_start_price - min_price) / self.position_start_price if self.position_start_price > 0 else 0
            else:  # Short
                max_price = position_prices.max()
                max_risk = (max_price - self.position_start_price) / self.position_start_price if self.position_start_price > 0 else 0
            
            # Calculate risk/reward ratio with epsilon protection
            if max_risk > 0:
                # FIXED: Add epsilon protection to prevent division by zero/NaN
                epsilon = 1e-10
                risk_reward_ratio = current_profit / (max_risk + epsilon)
                
                if risk_reward_ratio > 3.0:  # Excellent R:R
                    return 5.0
                elif risk_reward_ratio > 2.0:  # Good R:R
                    return 3.0
                elif risk_reward_ratio > 1.0:  # Acceptable R:R
                    return 1.0
                else:  # Poor R:R
                    return -2.0
            
            return 0.0
        
        def _calculate_holding_reward(self, current_profit: float) -> float:
            """
            Calculate reward for holding position
            Includes time decay to encourage timely exits
            """
            if self.position_start_step is None:
                return 0.0
            
            # Position duration
            position_duration = self._current_tick - self.position_start_step
            
            # Reduced time penalty (more lenient for learning)
            if position_duration > 500:  # Increased max duration
                time_penalty = -3.0  # Reduced penalty
            elif position_duration > 200:  # Extended grace period
                time_penalty = -0.005 * (position_duration - 200)  # Smaller penalty
            else:
                time_penalty = 0.0
            
            # Profit tracking penalty (penalize if profit is deteriorating)
            if current_profit > self.max_profit_seen:
                self.max_profit_seen = current_profit
                profit_penalty = 0.0
            else:
                # Penalize for letting profit erode
                profit_erosion = self.max_profit_seen - current_profit
                profit_penalty = -10.0 * profit_erosion if profit_erosion > 0.01 else 0.0
            
            return time_penalty + profit_penalty
        
        def _check_classic_entry_signal(self) -> bool:
            """
            Check if classic MtfScalper entry signal exists at current step

            CRITICAL FIX: Now checks RL features (%-classic_*_signal) first
            These features are available in the RL environment during training
            The old approach (checking enter_long/enter_short) didn't work because
            those columns don't exist in the RL environment's dataframe
            """
            if self._current_tick < 1:
                return False

            current_row = self.df.iloc[self._current_tick]

            # PRIMARY CHECK: Use RL features (available during training)
            # These are created in feature_engineering_standard()
            if "%-classic_long_signal" in current_row and current_row["%-classic_long_signal"] == 1:
                return True
            if "%-classic_short_signal" in current_row and current_row["%-classic_short_signal"] == 1:
                return True
            if "%-has_signal" in current_row and current_row["%-has_signal"] == 1:
                return True

            # FALLBACK CHECK: Direct strategy signals (only in backtest mode)
            if "enter_long" in current_row and current_row["enter_long"] == 1:
                return True
            if "enter_short" in current_row and current_row["enter_short"] == 1:
                return True

            # FALLBACK CHECK: Strategy entry tags (Freqtrade format)
            if "enter_tag" in current_row:
                enter_tag = current_row["enter_tag"]
                if pd.notna(enter_tag) and enter_tag in ["enter_long", "enter_short"]:
                    return True

            # No signal detected
            return False
        
        def _calculate_current_profit(self) -> float:
            """Calculate current profit percentage"""
            if self._position == 0 or self.position_start_price is None:
                return 0.0
            
            current_price = self.prices.iloc[self._current_tick]
            
            if self._position == 1:  # Long
                return (current_price - self.position_start_price) / self.position_start_price
            else:  # Short
                return (self.position_start_price - current_price) / self.position_start_price
        
        def step(self, action: int) -> Tuple:
            """
            Override step to track position information and log data
            """
            # Get current state before action
            current_price = self.prices.iloc[self._current_tick]
            current_profit = self._calculate_current_profit()

            # Track position entry
            if self._position == 0 and action in [Actions.Long_enter, Actions.Short_enter]:
                self.position_start_price = self.prices.iloc[self._current_tick]
                self.position_start_step = self._current_tick
                self.max_profit_seen = 0.0

            # Track position exit
            elif self._position != 0 and action in [Actions.Long_exit, Actions.Short_exit]:
                self.position_start_price = None
                self.position_start_step = None
                self.max_profit_seen = 0.0

            # Call parent step
            obs, reward, done, truncated, info = super().step(action)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # CRITICAL FIX: Simulate custom_exit() logic during training
            # This prevents reality gap between training and production
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if self._position != 0:
                current_profit = self._calculate_current_profit()
                position_duration = self._current_tick - self.position_start_step if self.position_start_step else 0

                # Time-based forced exit (simulates strategy's max_position_duration)
                if position_duration > self.max_position_duration:
                    info['custom_exit_triggered'] = True
                    info['custom_exit_reason'] = 'time_limit'
                    reward += -5.0  # Penalty for hitting time limit
                    done = True
                    logger.debug(f"Time-based exit triggered at step {self._current_tick} (duration: {position_duration})")

                # Emergency profit protection (simulates strategy's emergency_exit_profit)
                elif current_profit < -0.03:  # emergency_exit_profit = -3%
                    info['custom_exit_triggered'] = True
                    info['custom_exit_reason'] = 'emergency_stop'
                    reward += -10.0  # Heavy penalty for emergency stop
                    done = True
                    logger.debug(f"Emergency exit triggered at step {self._current_tick} (profit: {current_profit:.2%})")

                # Breakeven protection (simulates strategy's breakeven_protection)
                elif self.max_profit_seen > 0.015 and current_profit < 0.005:  # Was profitable, now near breakeven
                    info['custom_exit_triggered'] = True
                    info['custom_exit_reason'] = 'breakeven_protection'
                    reward += -2.0  # Moderate penalty for profit erosion
                    done = True
                    logger.debug(f"Breakeven protection triggered at step {self._current_tick}")

            # FIXED: Reduced logging frequency to prevent GB-sized log files
            # Only log important events: entries, exits, and every 100th step
            if self.data_collector:
                try:
                    should_log = (
                        action in [Actions.Long_enter, Actions.Short_enter, Actions.Long_exit, Actions.Short_exit]  # Entry/exit actions
                        or done  # Episode end
                        or self._current_tick % 100 == 0  # Every 100 candles
                    )

                    if should_log:
                        step_data = {
                            'step': self._current_tick,
                            'action': int(action),
                            'reward': float(reward),
                            'state': {
                                'position': int(self._position),
                                'entry_price': float(self.position_start_price) if self.position_start_price else 0.0,
                                'current_price': float(current_price),
                                'duration': int(self._current_tick - self.position_start_step) if self.position_start_step else 0,
                                'profit': float(current_profit),
                            },
                            'done': bool(done),
                        }
                        self.current_episode_steps.append(step_data)

                    # If episode ended, log it
                    if done:
                        self.data_collector.log_episode_end(self.episode_id, {
                            'end_balance': float(self._total_profit),
                            'total_trades': len([s for s in self.current_episode_steps if s['action'] in [1, 2]]),
                            'total_steps': len(self.current_episode_steps),
                        })
                except Exception as e:
                    logger.debug(f"Error logging step data: {e}")

            return obs, reward, done, truncated, info
        
        def reset(self, seed=None, options=None):
            """Reset environment and custom tracking variables"""
            self.position_start_price = None
            self.position_start_step = None
            self.max_profit_seen = 0.0

            # Log episode start if data collector available
            if self.data_collector:
                try:
                    # Save previous episode steps if any
                    if self.current_episode_steps:
                        for step_data in self.current_episode_steps:
                            self.data_collector.log_episode_step(self.episode_id, step_data)

                    # Start new episode
                    self.episode_id += 1
                    self.current_episode_steps = []

                    self.data_collector.log_episode_start({
                        'episode_id': self.episode_id,
                        'pair': getattr(self, 'pair', 'unknown'),
                        'start_step': 0,
                        'initial_balance': float(getattr(self, '_total_profit', 1000.0)),
                    })
                except Exception as e:
                    logger.debug(f"Error logging episode start: {e}")

            return super().reset()
    
    def fit(self, data_dictionary: Dict[str, DataFrame], pair: str = "") -> Any:
        """
        Train the RL model with custom environment and parameters
        """
        
        logger.info(f"Starting training for {pair}")
        
        # Set random seed for reproducibility
        seed = data_dictionary.get("seed", 42)
        set_random_seed(seed)
        
        # Get training parameters
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]

        # CRITICAL: Store actual training features after VarianceThreshold filtering
        self.actual_training_features = list(train_df.columns)
        logger.info(f"Training data shape: {train_df.shape}")
        logger.info(f"Stored {len(self.actual_training_features)} features for prediction consistency")

        # Prepare datasets with vectorized environments for CPU optimization
        # Try vectorized first, fallback to single env if fails
        try:
            logger.info(f"Creating vectorized training environment with {self.n_envs} parallel environments")
            self.train_env = self._create_vec_env(train_df, pair, n_envs=self.n_envs, is_train=True)
        except Exception as e:
            logger.warning(f"Failed to create vectorized environment: {e}")
            logger.info("Falling back to single environment (non-vectorized)")
            self.train_env = self._create_env(train_df, pair, is_train=True)

        # Eval environment can be single (no need for parallelization during eval)
        self.eval_env = self._create_env(test_df, pair, is_train=False)
        
        # Create model
        model = self._create_model()
        
        # Setup callbacks
        callbacks = self._create_callbacks()
        
        # Train model
        total_timesteps = len(train_df) * self.rl_config.get("train_cycles", 30)
        
        logger.info(f"Training PPO model for {total_timesteps} timesteps")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        logger.info(f"Training completed for {pair}")
        
        return model
    
    def _create_vec_env(self, df: DataFrame, pair: str, n_envs: int = 8, is_train: bool = True) -> Any:
        """
        Create vectorized environments for parallel rollout collection (CPU optimization)

        Args:
            df: Training/test dataframe
            pair: Trading pair name
            n_envs: Number of parallel environments (default 8 for CPU)
            is_train: Whether this is for training or prediction

        Returns:
            SubprocVecEnv with n_envs parallel environments
        """
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

        def make_env(rank: int):
            """Create environment factory for each process"""
            def _init():
                env = self._create_env(df, pair, is_train=is_train)
                # Set different seed for each environment
                if hasattr(env, 'seed'):
                    env.seed(self.random_state + rank)
                return env
            return _init

        # Use SubprocVecEnv for true CPU parallelism
        # Each environment runs in its own process
        logger.info(f"Creating {n_envs} parallel environments for {'training' if is_train else 'prediction'}")
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

        return vec_env

    def _create_env(self, df: DataFrame, pair: str, config: dict = None, df_raw: DataFrame = None, is_train: bool = True) -> Any:
        """
        Create custom environment instance with proper data handling
        """

        # Ensure we have the required 'close' column for prices
        close_col = None
        if "close" in df.columns:
            close_col = "close"
        elif "%-close" in df.columns:
            close_col = "%-close"
        else:
            # Try to find close column in original data
            for col in df.columns:
                if "close" in col.lower():
                    close_col = col
                    break

            if close_col is None:
                raise ValueError("No 'close' column found in dataframe for RL environment")

        # Make sure we have a 'close' column for the RL environment
        if close_col != "close":
            df = df.copy()
            df["close"] = df[close_col]

        # Handle NaN values in the data
        df_clean = df.copy()

        # Check for completely empty rows
        if df_clean.isna().all(axis=1).any():
            logger.warning(f"Found completely empty rows in data for {pair}, removing them")
            df_clean = df_clean[~df_clean.isna().all(axis=1)]

        # Check if we have any data left after cleaning
        if len(df_clean) == 0:
            raise ValueError(f"All training data was dropped due to NaN values for {pair}")

        # Forward-fill remaining NaN values for price data
        price_columns = ["open", "high", "low", "close", "volume"]
        for col in price_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].ffill().bfill()

        # FIXED: Improved NaN handling for feature columns
        # Use forward-fill → back-fill → zero as last resort
        feature_columns = [col for col in df_clean.columns if col.startswith("%")]
        if feature_columns:
            # Forward-fill preserves trend
            df_clean[feature_columns] = df_clean[feature_columns].ffill()
            # Back-fill handles start-of-series NaNs
            df_clean[feature_columns] = df_clean[feature_columns].bfill()
            # Fill any remaining NaNs with 0 (rare edge case)
            df_clean[feature_columns] = df_clean[feature_columns].fillna(0)
            logger.debug(f"NaN handling: ffill → bfill → 0 for {len(feature_columns)} feature columns")

        # Final check for critical NaN values
        if df_clean["close"].isna().any():
            logger.error(f"Critical: NaN values found in close price column for {pair}")
            df_clean = df_clean.dropna(subset=["close"])

            if len(df_clean) == 0:
                raise ValueError(f"No valid price data available for {pair} after cleaning")

        logger.info(f"RL environment created for {pair} with {len(df_clean)} data points")

        # Use provided config and df_raw, or create minimal ones
        if config is None:
            config = {
                'timeframe': '5m',
                'stake_currency': 'USDT',
                'stake_amount': 'unlimited',
                'max_open_trades': 3,
                'fee': 0.001,  # Fixed: Use hardcoded fee instead of self.fee for multiprocessing compatibility
                'freqai': {
                    'rl_config': {
                        'train_cycles': 30,
                        'add_state_info': False,
                        'max_trade_duration_candles': 300,
                        'max_training_drawdown_pct': 0.15,
                        'cpu_count': 8,
                        'model_type': 'PPO',
                        'policy_type': 'MlpPolicy',
                        'net_arch': [512, 256, 128],  # Upgraded for 40+ features
                        'model_reward_parameters': {
                            'rr': 1,
                            'profit_aim': 0.02,
                            'max_profit': 0.10,
                            'max_loss': -0.05
                        }
                    }
                }
            }

        if df_raw is None:
            df_raw = df_clean.copy()

        # Build prices DataFrame using FreqAI standard raw price columns
        raw_price_columns = {
            "open": "%-raw_open",
            "high": "%-raw_high",
            "low": "%-raw_low",
            "close": "%-raw_close",
            "volume": "%-raw_volume"
        }

        price_data = {}
        for target_col, source_col in raw_price_columns.items():
            if source_col in df_clean.columns:
                price_data[target_col] = df_clean[source_col]

        if len(price_data) == 0:
            raise ValueError("No raw price columns found in dataframe for RL environment")

        prices_df = pd.DataFrame(price_data)

        # Initialize data collector for RL episodes (training only)
        data_collector = None
        if is_train:
            try:
                data_collector = DataCollector(output_dir="user_data/analysis_data/rl_training")
                data_collector.set_config({
                    'model': 'MtfScalperRLModel',
                    'pair': pair,
                    'is_training': True,
                    'reward_weights': self.reward_weights,
                })
            except Exception as e:
                logger.warning(f"Failed to initialize data collector: {e}")

        env_config = {
            "config": config,
            "df_raw": df_raw,
            "df": df_clean,
            "prices": prices_df,
            "reward_kwargs": {
                "reward_weights": self.reward_weights,
                "entry_penalty": self.entry_penalty_multiplier,
                "classic_signal_reward": self.classic_signal_reward,
                "rr": 1.0,
                "profit_aim": 0.02,
                "max_profit": 0.10,
                "max_loss": -0.05
            },
            "window_size": self.window_size,
            "fee": self.fee,
            "pair": pair,
        }

        env = self.MtfScalperRLEnv(**env_config)

        # Set data_collector after environment creation (bypasses BaseEnvironment validation)
        if data_collector:
            env.data_collector = data_collector

        if is_train:
            # Wrap in monitor for training
            env = Monitor(env)

        return env
    
    def _create_model(self) -> PPO:
        """
        Create PPO model with optimized hyperparameters
        """

        policy_kwargs = dict(
            net_arch=self.net_arch,
            activation_fn=th.nn.ReLU,
            optimizer_kwargs=dict(
                weight_decay=1e-5
            )
        )

        model = PPO(
            policy=self.policy_type,
            env=self.train_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            vf_coef=self.vf_coef,
            ent_coef=self.ent_coef,
            max_grad_norm=1.0,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.tensorboard_log,
            device=self.device,
            verbose=1
        )

        return model
    
    def _create_callbacks(self) -> list:
        """
        Create training callbacks for monitoring
        """
        
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=self.model_save_path,
            log_path=self.model_save_path,
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        callbacks.append(eval_callback)
        
        # Custom logging callback
        class LoggingCallback(BaseCallback):
            def __init__(self, verbose=0):
                super().__init__(verbose)
                self.episode_rewards = []
                self.episode_lengths = []
                
            def _on_step(self) -> bool:
                # Log episode statistics
                if self.locals.get("dones", [False])[0]:
                    episode_reward = self.locals.get("episode_rewards", [0])[0]
                    episode_length = self.locals.get("episode_lengths", [0])[0]
                    
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    
                    if len(self.episode_rewards) % 10 == 0:
                        mean_reward = np.mean(self.episode_rewards[-10:])
                        mean_length = np.mean(self.episode_lengths[-10:])
                        logger.info(f"Last 10 episodes - Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.1f}")
                
                return True
        
        callbacks.append(LoggingCallback())
        
        return callbacks
    
    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen) -> Tuple[DataFrame, DataFrame]:
        """
        Make predictions using the trained model
        Fixed version that handles labels_mean KeyError for FreqAI 2025.10
        """

        # Get the trained model
        model = self.model

        # Check if model exists (should have been trained in fit())
        if model is None:
            logger.error("Model is None - training may have failed or not been called")
            raise ValueError("Model not trained. Please run training first.")

        # Prepare features
        filtered_df, _ = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            training_filter=False
        )

        # CRITICAL: Use only the features that were used during training
        # This ensures prediction shape matches training shape
        if hasattr(self, 'actual_training_features') and self.actual_training_features:
            logger.info(f"Filtering prediction features to match training: {len(self.actual_training_features)} features")
            # Keep only features that exist in both
            available_features = [f for f in self.actual_training_features if f in filtered_df.columns]
            if len(available_features) != len(self.actual_training_features):
                logger.warning(f"Only {len(available_features)}/{len(self.actual_training_features)} training features available in prediction")
            filtered_df = filtered_df[available_features]
            logger.info(f"Prediction data shape after filtering: {filtered_df.shape}")

        # Create environment for prediction
        pred_env = self._create_env(filtered_df, dk.pair, is_train=False)

        # Generate predictions with batch processing for CPU optimization
        actions = []
        confidences = []
        batch_size = 256  # Optimal batch size for CPU

        obs, _ = pred_env.reset()
        obs_buffer = []
        indices_buffer = []

        # Collect observations in batches
        for i in range(len(filtered_df)):
            obs_buffer.append(obs.copy())
            indices_buffer.append(i)

            # Process batch when full or at end
            if len(obs_buffer) >= batch_size or i == len(filtered_df) - 1:
                # Batch prediction
                for obs_single in obs_buffer:
                    action, _states = model.predict(obs_single, deterministic=True)

                    # FIXED: Use actual model probabilities for confidence
                    # Get action probabilities from policy network
                    try:
                        with th.no_grad():
                            obs_tensor = th.tensor(obs_single, dtype=th.float32).unsqueeze(0).to(model.device)
                            # Get distribution from policy
                            distribution = model.policy.get_distribution(obs_tensor)
                            action_probs = distribution.distribution.probs
                            # Confidence is the probability of the selected action
                            confidence = float(action_probs[0, int(action)].item())
                    except Exception as e:
                        # Fallback to simple heuristic if probability extraction fails
                        logger.debug(f"Failed to extract action probabilities: {e}")
                        confidence = 0.8 if action != 0 else 0.5

                    actions.append(int(action))
                    confidences.append(confidence)

                    obs, _, done, _, _ = pred_env.step(action)
                    if done:
                        obs, _ = pred_env.reset()

                # Clear buffer
                obs_buffer = []
                indices_buffer = []
            else:
                # Step environment without prediction
                obs, _, done, _, _ = pred_env.step(0)  # Neutral action
                if done:
                    obs, _ = pred_env.reset()

        # Create prediction dataframe
        pred_df = DataFrame({
            "&-action": actions,
            "&-action_confidence": confidences
        }, index=filtered_df.index)

        # COMPREHENSIVE FIX: Handle ALL prediction columns dynamically
        if not hasattr(dk, 'data'):
            dk.data = {}
        if "labels_mean" not in dk.data:
            dk.data["labels_mean"] = {}
            dk.data["labels_std"] = {}

        # Process all prediction columns (not just hardcoded ones)
        import numpy as np
        prediction_columns = [col for col in pred_df.columns if col.startswith("&-")]

        for column in prediction_columns:
            if column not in dk.data["labels_mean"]:
                column_data = pred_df[column].values

                # Calculate appropriate mean and std based on column type
                if column == "&-action":
                    # For action columns, use statistical measures of action distribution
                    mean_val = float(np.mean(column_data)) if len(column_data) > 0 else 2.0
                    std_val = float(np.std(column_data)) if len(set(column_data)) > 1 else 1.0
                elif column.endswith("_confidence"):
                    # For confidence columns, use confidence distribution statistics
                    mean_val = float(np.mean(column_data)) if len(column_data) > 0 else 0.7
                    std_val = float(np.std(column_data)) if len(set(column_data)) > 1 else 0.1
                else:
                    # For other prediction columns, use generic statistics
                    mean_val = float(np.mean(column_data)) if len(column_data) > 0 else 0.0
                    std_val = float(np.std(column_data)) if len(set(column_data)) > 1 else 1.0

                dk.data["labels_mean"][column] = mean_val
                dk.data["labels_std"][column] = std_val

                logger.info(f"Added {column} to labels_mean: {mean_val:.3f}, std: {std_val:.3f}")

        # Fill any missing values
        pred_df = pred_df.fillna(0)

        return pred_df, dk.do_predict
