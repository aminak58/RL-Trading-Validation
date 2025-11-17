# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
"""
MtfScalper RL Hybrid Strategy
=============================
Combines classic Multi-Timeframe entry logic with RL-powered exit optimization.

Architecture:
- Entry: Classic MtfScalper logic (3 timeframe alignment)
- Exit: RL Agent with 5-action space
- Safety: Emergency exit conditions + position time limits
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import reduce
import logging
import atexit
import signal
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import (
    IStrategy,
    informative,
    IntParameter,
    DecimalParameter,
    CategoricalParameter
)
from freqtrade.persistence import Trade

# Import data collector
import sys
import os
# Dynamic path resolution - works on any system
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from user_data.data_collector import DataCollector

logger = logging.getLogger(__name__)


class MtfScalper_RL_Hybrid(IStrategy):
    """
    Hybrid Strategy: Classic Entry + RL Exit
    
    Phase 1 Implementation:
    - Inherits entry logic from MtfScalper
    - Adds RL-based exit decision making
    - Includes safety mechanisms and position tracking
    """
    
    INTERFACE_VERSION = 3
    
    # ═══════════════════════════════════════════════════════════
    # BASE CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 240  # Will be auto-increased by FreqAI for MTF
    can_short = True
    
    # Baseline risk config
    minimal_roi = {
        "0": 0.04,    # 4% immediate
        "30": 0.02,   # 2% after 30 minutes
        "60": 0.01,   # 1% after 60 minutes
    }
    stoploss = -0.10
    
    # ═══════════════════════════════════════════════════════════
    # CLASSIC PARAMETERS (from MtfScalper)
    # ═══════════════════════════════════════════════════════════
    
    # Technical indicators
    atr_length: int = 14
    atr_multiplier: float = 1.5
    ema_fast_len: int = 9
    ema_slow_len: int = 21
    ema_trend_len: int = 200
    adx_len: int = 14
    adx_threshold: int = 25
    
    # Hyperoptable parameters (Classic)
    buy_rsi = IntParameter(low=40, high=70, default=55, space="buy", optimize=True)
    sell_rsi = IntParameter(low=40, high=70, default=55, space="sell", optimize=True)
    adx_thr_buy = IntParameter(low=20, high=35, default=25, space="buy", optimize=True)
    adx_thr_sell = IntParameter(low=20, high=35, default=25, space="sell", optimize=True)
    atr_threshold = IntParameter(low=1, high=10, default=5, space="buy", optimize=True)
    
    # ═══════════════════════════════════════════════════════════
    # RL PARAMETERS (New for Hybrid)
    # ═══════════════════════════════════════════════════════════
    
    # FreqAI configuration
    freqai_enabled = True
    
    # RL Exit thresholds (lowered to work with undertrained model)
    rl_exit_confidence = DecimalParameter(0.2, 0.5, default=0.3, space="sell", optimize=True)
    max_position_duration = IntParameter(12, 96, default=48, space="sell", optimize=True)  # in 5m candles
    
    # Safety parameters
    emergency_exit_profit = DecimalParameter(-0.05, -0.02, default=-0.03, space="sell")
    breakeven_trigger = DecimalParameter(0.01, 0.03, default=0.02, space="sell")
    
    # Risk management
    risk_per_trade: float = 0.02

    # Data collection
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Initialize data collector
        self.data_collector = DataCollector()
        self.data_collector.set_config({
            'strategy': 'MtfScalper_RL_Hybrid',
            'timeframe': self.timeframe,
            'risk_per_trade': self.risk_per_trade,
            'reward_weights': {
                "profit": 0.35,
                "drawdown_control": 0.25,
                "timing_quality": 0.20,
                "risk_reward_ratio": 0.20
            }
        })

        # Register cleanup handler for backtest mode (bot_end() not called in backtest)
        atexit.register(self._save_datacollector)

        # Setup signal handlers for graceful shutdown (Ctrl+C, kill, etc.)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("Data collector initialized with atexit and signal handlers")

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully (SIGTERM, SIGINT)"""
        logger.warning(f"⚠️  Received signal {signum}, saving data before exit...")
        self._save_datacollector()
        logger.info("Graceful shutdown complete")
        sys.exit(0)

    def _save_datacollector(self):
        """Save DataCollector data on exit - works in backtest mode"""
        try:
            if hasattr(self, 'data_collector') and self.data_collector:
                trades_count = len(self.data_collector.trades)
                signals_count = len(getattr(self.data_collector, 'signal_propagation', []))
                decisions_count = len(getattr(self.data_collector, 'model_decisions', []))

                logger.info(f"💾 Saving DataCollector on exit: "
                           f"Trades={trades_count}, "
                           f"Signals={signals_count}, "
                           f"Decisions={decisions_count}")

                self.data_collector.save_all()
                logger.info("✅ DataCollector saved successfully")
        except Exception as e:
            logger.error(f"❌ Failed to save DataCollector: {e}")

    # ═══════════════════════════════════════════════════════════
    # FREQAI CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    
    def freqai_config(self) -> Dict[str, Any]:
        """Returns FreqAI configuration for RL model"""
        return {
            "enabled": False,  # DIAGNOSTIC: Temporarily disabled to test classic strategy baseline
            "purge_old_models": False,
            "train_period_days": 30,
            "backtest_period_days": 7,
            "identifier": "MtfScalperRL_v1",
            "data_kitchen_thread_count": 2,
            
            "feature_parameters": {
                "include_timeframes": ["5m", "15m", "1h"],
                "include_corr_pairlist": [],  # CRITICAL: Must match config JSON exactly!
                "label_period_candles": 20,
                "include_shifted_candles": 3,
                "indicator_periods_candles": [10, 20, 50],
                "DI_threshold": 0.9,
                "weight_factor": 0.9,
                "principal_component_analysis": False,
                "use_SVM_to_remove_outliers": False,
                "use_DBSCAN_to_remove_outliers": False,
            },
            
            "data_split_parameters": {
                "test_size": 0.15,
                "random_state": 42,
                "shuffle": False,
            },
            
            "model_training_parameters": {
                "model_type": "PPO",
                "policy_type": "MlpPolicy",
                "learning_rate": 0.0003,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "vf_coef": 0.5,
                "ent_coef": 0.01,
                "target_kl": 0.015,
                "verbose": 1,
                "tensorboard_log": "./tensorboard/",
            },
            
            "rl_config": {
                "train_cycles": 30,
                "max_trade_duration_candles": 300,  # 25 hours max
                "model_type": "PPO",
                "policy_type": "MlpPolicy",
                "net_arch": [512, 256, 128],  # Upgraded for 40+ features
                "model_reward_parameters": {
                    "rr": 1,
                    "profit_aim": 0.025,  # 2.5% target
                }
            }
        }
    
    # ═══════════════════════════════════════════════════════════
    # INFORMATIVE TIMEFRAMES (from MtfScalper)
    # ═══════════════════════════════════════════════════════════
    
    @informative("15m")
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_len)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_len)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_len)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe
    
    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_len)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_len)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_len)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe
    
    # ═══════════════════════════════════════════════════════════
    # FEATURE ENGINEERING FOR RL
    # ═══════════════════════════════════════════════════════════
    
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                      metadata: Dict, **kwargs) -> DataFrame:
        """
        Advanced feature engineering specifically optimized for RL exit decisions.
        Focuses on exit-relevant features rather than entry signals.
        """
        
        dataframe = self.feature_engineering_expand_basic(dataframe, metadata=metadata, **kwargs)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Position-Aware Features (Critical for Exit)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Price momentum for exit timing
        dataframe["%-momentum_5"] = dataframe["close"].pct_change(5)
        dataframe["%-momentum_10"] = dataframe["close"].pct_change(10)
        dataframe["%-momentum_20"] = dataframe["close"].pct_change(20)
        
        # Acceleration (second derivative of price)
        dataframe["%-acceleration"] = dataframe["%-momentum_5"].diff()
        
        # Distance from recent high/low (exit at extremes)
        dataframe["%-dist_from_high_20"] = (dataframe["high"].rolling(20).max() - dataframe["close"]) / dataframe["close"]
        dataframe["%-dist_from_low_20"] = (dataframe["close"] - dataframe["low"].rolling(20).min()) / dataframe["close"]
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Market Microstructure (Exit Quality)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Volume patterns (liquidity for exit)
        dataframe["%-volume_ratio_5"] = dataframe["volume"] / dataframe["volume"].rolling(5).mean()
        dataframe["%-volume_ratio_20"] = dataframe["volume"] / dataframe["volume"].rolling(20).mean()
        
        # Bid-ask spread proxy
        dataframe["%-spread_proxy"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        dataframe["%-spread_ma_ratio"] = dataframe["%-spread_proxy"] / dataframe["%-spread_proxy"].rolling(20).mean()
        
        # Trade velocity (rapid changes indicate reversal)
        dataframe["%-trade_velocity"] = dataframe["volume"].diff() / (dataframe["volume"].shift(1) + 1e-10)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Divergence Indicators (Reversal Signals)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # RSI Divergence
        price_higher = dataframe["close"] > dataframe["close"].shift(10)
        rsi_lower = dataframe["rsi"] < dataframe["rsi"].shift(10)
        dataframe["%-bearish_divergence"] = (price_higher & rsi_lower).astype(int)
        
        price_lower = dataframe["close"] < dataframe["close"].shift(10)
        rsi_higher = dataframe["rsi"] > dataframe["rsi"].shift(10)
        dataframe["%-bullish_divergence"] = (price_lower & rsi_higher).astype(int)
        
        # Volume-Price Divergence
        volume_decreasing = dataframe["volume"] < dataframe["volume"].rolling(10).mean()
        price_increasing = dataframe["close"] > dataframe["close"].shift(5)
        dataframe["%-volume_price_divergence"] = (volume_decreasing & price_increasing).astype(int)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Support/Resistance Levels
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Pivot points
        dataframe["%-pivot"] = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        dataframe["%-r1"] = 2 * dataframe["%-pivot"] - dataframe["low"]
        dataframe["%-s1"] = 2 * dataframe["%-pivot"] - dataframe["high"]
        
        # Distance to pivot levels
        dataframe["%-dist_to_r1"] = (dataframe["%-r1"] - dataframe["close"]) / dataframe["close"]
        dataframe["%-dist_to_s1"] = (dataframe["close"] - dataframe["%-s1"]) / dataframe["close"]
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Multi-Timeframe Exit Signals
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # 15m momentum for exit (safe access - preserves original logic)
        if "rsi_15m" in dataframe.columns:
            dataframe["%-rsi_15m_exit_long"] = (dataframe["rsi_15m"] > 70).astype(int)
            dataframe["%-rsi_15m_exit_short"] = (dataframe["rsi_15m"] < 30).astype(int)
        else:
            dataframe["%-rsi_15m_exit_long"] = 0
            dataframe["%-rsi_15m_exit_short"] = 0

        # 1h trend change detection (safe access - preserves original logic)
        if "ema_fast_1h" in dataframe.columns and "ema_slow_1h" in dataframe.columns:
            dataframe["%-ema_cross_bearish_1h"] = (
                (dataframe["ema_fast_1h"].shift(1) > dataframe["ema_slow_1h"].shift(1)) &
                (dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"])
            ).astype(int)

            dataframe["%-ema_cross_bullish_1h"] = (
                (dataframe["ema_fast_1h"].shift(1) < dataframe["ema_slow_1h"].shift(1)) &
                (dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"])
            ).astype(int)
        else:
            dataframe["%-ema_cross_bearish_1h"] = 0
            dataframe["%-ema_cross_bullish_1h"] = 0
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Risk/Reward Features
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # ATR-based profit potential
        dataframe["%-profit_potential"] = dataframe["atr"] * 2 / dataframe["close"]
        
        # Risk score (higher = more risky to hold)
        dataframe["%-risk_score"] = (
            dataframe["%-spread_proxy"] * 0.3 +
            (1 / (dataframe["%-volume_ratio_5"] + 0.1)) * 0.3 +
            dataframe["atr"] / dataframe["close"] * 0.4
        )

        # ═══════════════════════════════════════════════════════════
        # ADVANCED EXIT-SPECIFIC FEATURES (NEW)
        # ═══════════════════════════════════════════════════════════

        # 1. Profit erosion indicator - when profit starts deteriorating
        dataframe["%-profit_erosion"] = (
            dataframe["high"].rolling(20).max() - dataframe["close"]
        ) / dataframe["close"]

        # 2. Volume exhaustion - decreasing volume indicates trend ending
        dataframe["%-volume_exhaustion"] = (
            dataframe["volume"].rolling(5).std() /
            (dataframe["volume"].rolling(20).std() + 1e-10)
        )

        # 3. Volatility regime - helps adapt exit strategy to market conditions
        if "atr" in dataframe.columns:
            dataframe["%-volatility_regime"] = (
                dataframe["atr"] / (dataframe["atr"].rolling(50).mean() + 1e-10)
            )
        else:
            dataframe["%-volatility_regime"] = 1.0

        # 4. Time-in-position proxy - how long the current trend has been active
        if "ema_fast" in dataframe.columns and "ema_slow" in dataframe.columns:
            dataframe["%-trend_age"] = (
                (dataframe["ema_fast"] > dataframe["ema_slow"])
            ).rolling(50).sum()
        else:
            dataframe["%-trend_age"] = 0

        # ═══════════════════════════════════════════════════════════
        # CRITICAL FIX: CLASSIC ENTRY SIGNALS AS RL FEATURES
        # ═══════════════════════════════════════════════════════════
        # Problem: feature_engineering_standard() runs BEFORE populate_entry_trend()
        # Solution: Calculate classic signals HERE using same logic as populate_entry_trend()
        # This ensures:
        # 1. Signals have variance (real calculation, not copy from non-existent columns)
        # 2. RL model can see classic entry signals during training
        # 3. Independent of execution order

        # Classic LONG signal (same logic as populate_entry_trend)
        classic_long_conditions = []
        if all(col in dataframe.columns for col in ['rsi', 'ema_fast', 'ema_slow', 'adx']):
            classic_long_conditions.append(dataframe['rsi'] < self.buy_rsi.value)
            classic_long_conditions.append(dataframe['ema_fast'] > dataframe['ema_slow'])
            classic_long_conditions.append(dataframe['adx'] > self.adx_thr_buy.value)
            classic_long_conditions.append(dataframe['volume'] > 0)

            if classic_long_conditions:
                dataframe['%-classic_long_signal'] = (
                    reduce(lambda x, y: x & y, classic_long_conditions)
                ).astype(float)
            else:
                dataframe['%-classic_long_signal'] = 0.0
        else:
            dataframe['%-classic_long_signal'] = 0.0

        # Classic SHORT signal (same logic as populate_entry_trend)
        classic_short_conditions = []
        if all(col in dataframe.columns for col in ['rsi', 'ema_fast', 'ema_slow', 'adx']):
            classic_short_conditions.append(dataframe['rsi'] > (100 - self.buy_rsi.value))
            classic_short_conditions.append(dataframe['ema_fast'] < dataframe['ema_slow'])
            classic_short_conditions.append(dataframe['adx'] > self.adx_thr_buy.value)
            classic_short_conditions.append(dataframe['volume'] > 0)

            if classic_short_conditions:
                dataframe['%-classic_short_signal'] = (
                    reduce(lambda x, y: x & y, classic_short_conditions)
                ).astype(float)
            else:
                dataframe['%-classic_short_signal'] = 0.0
        else:
            dataframe['%-classic_short_signal'] = 0.0

        # Combined signal indicator
        dataframe['%-has_signal'] = (
            (dataframe['%-classic_long_signal'] == 1) |
            (dataframe['%-classic_short_signal'] == 1)
        ).astype(float)

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        Standard feature engineering required for RL models.
        Includes raw price data that RL environment needs for price access.

        NOTE: Classic signal features are now calculated in feature_engineering_expand_all()
        to avoid execution order issues (expand_all runs after populate_indicators).
        """

        # CRITICAL: Raw price data for RL environment (FreqAI standard requirement)
        # These are ONLY created here, not duplicated elsewhere
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_volume"] = dataframe["volume"]

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        Basic feature expansion including all original MtfScalper indicators
        """
        
        # Copy all classic indicators from populate_indicators
        dataframe = self.populate_indicators_base(dataframe, metadata)
        
        # Ensure all required FreqAI features are present
        # Safe datetime features with error handling
        try:
            if hasattr(dataframe.index, 'dayofweek'):
                dataframe["%-day_of_week"] = dataframe.index.dayofweek
            else:
                dataframe["%-day_of_week"] = 0
        except:
            dataframe["%-day_of_week"] = 0

        try:
            if hasattr(dataframe.index, 'hour'):
                dataframe["%-hour_of_day"] = dataframe.index.hour
            else:
                dataframe["%-hour_of_day"] = 0
        except:
            dataframe["%-hour_of_day"] = 0

        # NOTE: Raw OHLCV (%-raw_*) are created in feature_engineering_standard()
        # Here we only add derived price-based features
        dataframe["%-price_change"] = dataframe["close"].pct_change()
        dataframe["%-high_low_ratio"] = dataframe["high"] / dataframe["low"]
        dataframe["%-close_open_ratio"] = dataframe["close"] / dataframe["open"]

        # Clean up any NaN values
        dataframe = dataframe.fillna(0)

        return dataframe
    
    def populate_indicators_base(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Base indicators from MtfScalper (unchanged)"""
        
        # Base TF indicators
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_len)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_len)
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=self.ema_trend_len)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_len)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_length)
        
        # Additional indicators from original
        dataframe["ma"] = ta.SMA(dataframe, timeperiod=10)
        macd_data = ta.MACD(dataframe)
        dataframe["macd"] = macd_data["macd"]
        dataframe["roc"] = ta.ROC(dataframe, timeperiod=2)
        dataframe["momentum"] = ta.MOM(dataframe, timeperiod=4)

        # Bollinger Bands with proper error handling
        bb_result = ta.BBANDS(dataframe, timeperiod=20)
        dataframe["bb_upper"], dataframe["bb_middle"], dataframe["bb_lower"] = bb_result

        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["stoch"] = ta.STOCH(dataframe)["slowk"]
        dataframe["obv"] = ta.OBV(dataframe)
        
        return dataframe
    
    # ═══════════════════════════════════════════════════════════
    # MAIN INDICATOR POPULATION
    # ═══════════════════════════════════════════════════════════
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Main indicator population - combines classic and RL features
        """

        # Initialize FreqAI if available
        if hasattr(self, 'freqai') and self.freqai:
            self.freqai.start(dataframe, metadata, self)

        # Classic indicators
        dataframe = self.populate_indicators_base(dataframe, metadata)

        # Add RL action column placeholder
        dataframe["&-action"] = 0

        # Periodic auto-save to prevent data loss during long training/backtest
        # This runs during indicator calculation, which happens in both training and backtest
        if len(dataframe) > 0 and len(dataframe) % 1000 == 0:
            try:
                # Save checkpoint with candle count identifier
                checkpoint_id = f"auto_{len(dataframe)}_{metadata.get('pair', 'unknown')}"
                self.data_collector.save_all(custom_name=checkpoint_id)
                logger.info(f"📦 Auto-checkpoint saved at {len(dataframe)} candles for {metadata.get('pair')}")
            except Exception as e:
                logger.debug(f"Auto-save skipped: {e}")

        return dataframe
    
    # ═══════════════════════════════════════════════════════════
    # ENTRY LOGIC (Classic MtfScalper)
    # ═══════════════════════════════════════════════════════════
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry logic: Pure classic MtfScalper multi-timeframe alignment
        No RL involvement in entry decisions for Phase 1
        """
        
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Multi-Timeframe Trend Alignment
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Base timeframe (5m) conditions
        main_trend_up = (
            (dataframe["ema_fast"] > dataframe["ema_slow"]) & 
            (dataframe["close"] > dataframe["ema_trend"])
        )
        main_trend_down = (
            (dataframe["ema_fast"] < dataframe["ema_slow"]) & 
            (dataframe["close"] < dataframe["ema_trend"])
        )
        main_strong_trend_buy = dataframe["adx"] > self.adx_thr_buy.value
        main_strong_trend_sell = dataframe["adx"] > self.adx_thr_sell.value
        
        # 15m confirmation
        confirm_trend_up = dataframe["ema_fast_15m"] > dataframe["ema_slow_15m"]
        confirm_trend_down = dataframe["ema_fast_15m"] < dataframe["ema_slow_15m"]
        confirm_strong_trend_buy = dataframe["adx_15m"] > self.adx_thr_buy.value
        confirm_strong_trend_sell = dataframe["adx_15m"] > self.adx_thr_sell.value
        
        # 1h filter
        filter_trend_up = dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]
        filter_trend_down = dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"]
        filter_strong_trend_buy = dataframe["adx_1h"] > self.adx_thr_buy.value
        filter_strong_trend_sell = dataframe["adx_1h"] > self.adx_thr_sell.value
        
        # Perfect alignment across all timeframes
        aligned_bullish = main_trend_up & confirm_trend_up & filter_trend_up
        aligned_bearish = main_trend_down & confirm_trend_down & filter_trend_down
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Volatility Filter
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        atr_pct = (dataframe["atr"] / dataframe["close"]) * 100
        volatility_filter = atr_pct < self.atr_threshold.value
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Final Entry Conditions
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        buy_condition = (
            aligned_bullish &
            main_strong_trend_buy &
            confirm_strong_trend_buy &
            filter_strong_trend_buy &
            (dataframe["rsi"] > self.buy_rsi.value) &
            (dataframe["close"] > dataframe["open"]) &
            volatility_filter
        )
        
        sell_condition = (
            aligned_bearish &
            main_strong_trend_sell &
            confirm_strong_trend_sell &
            filter_strong_trend_sell &
            (dataframe["rsi"] < self.sell_rsi.value) &
            (dataframe["close"] < dataframe["open"]) &
            volatility_filter
        )
        
        dataframe.loc[buy_condition, "enter_long"] = 1
        dataframe.loc[sell_condition, "enter_short"] = 1

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DATA COLLECTION: Signal Generation Logging
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Count signals for pipeline tracking
        total_candles = len(dataframe)
        long_signals = int(buy_condition.sum())
        short_signals = int(sell_condition.sum())
        total_signals = long_signals + short_signals

        # Log signal generation for debugging
        if hasattr(self, 'data_collector') and self.data_collector is not None:
            # Get current timestamp
            current_time = dataframe.iloc[-1]['date'] if 'date' in dataframe.columns else pd.Timestamp.now()

            # Log signal generation data
            signal_data = {
                'timestamp': current_time,
                'pair': metadata['pair'],
                'stage': 'classic_signal_generation',
                'total_candles': total_candles,
                'long_signals': long_signals,
                'short_signals': short_signals,
                'total_signals': total_signals,
                'signal_percentage': (total_signals / total_candles) * 100 if total_candles > 0 else 0,
                'success': total_signals > 0,
                'failure_reason': 'no_signals_generated' if total_signals == 0 else None
            }
            self.data_collector.log_signal_propagation(signal_data)

            # Log detailed signal analysis for pipeline debugging
            if total_signals > 0:
                # Analyze why signals were generated
                avg_adx = float(dataframe.loc[buy_condition | sell_condition, 'adx'].mean())
                avg_rsi = float(dataframe.loc[buy_condition | sell_condition, 'rsi'].mean())
                avg_atr_pct = float((dataframe.loc[buy_condition | sell_condition, 'atr'] /
                                   dataframe.loc[buy_condition | sell_condition, 'close']).mean() * 100)

                analysis_data = {
                    'timestamp': current_time,
                    'pair': metadata['pair'],
                    'selected_action': 0,  # Analysis only, no action
                    'confidence': 1.0,
                    'model_output': [0, 0, 0, 0, 0],  # Placeholder
                    'state_info': {
                        'signal_analysis': {
                            'avg_adx': avg_adx,
                            'avg_rsi': avg_rsi,
                            'avg_atr_pct': avg_atr_pct,
                            'adx_threshold': self.adx_thr_buy.value,
                            'rsi_buy_threshold': self.buy_rsi.value,
                            'rsi_sell_threshold': self.sell_rsi.value,
                            'atr_threshold': self.atr_threshold.value,
                            'timeframes_aligned': True,  # Signals only generated if aligned
                            'volatility_filtered': True   # Signals only generated if volatility ok
                        }
                    }
                }
                self.data_collector.log_model_decision(analysis_data)

        # Log entry signals for debugging (keep existing logs)
        if buy_condition.any():
            logger.info(f"🚀 CLASSIC ENTRY SIGNAL: {long_signals} Long signals for {metadata['pair']}")
        if sell_condition.any():
            logger.info(f"🚀 CLASSIC ENTRY SIGNAL: {short_signals} Short signals for {metadata['pair']}")

        if total_signals == 0:
            logger.warning(f"❌ NO CLASSIC SIGNALS: {metadata['pair']} - Entry conditions too restrictive")

        return dataframe
    
    # ═══════════════════════════════════════════════════════════
    # EXIT LOGIC (RL-Powered)
    # ═══════════════════════════════════════════════════════════
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit logic: RL-based decision making with safety mechanisms
        
        Action Space:
        0: Hold/Neutral
        1: Enter Long (ignored in exit)
        2: Enter Short (ignored in exit)  
        3: Exit Long
        4: Exit Short
        """
        
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        
        # Only process if we have FreqAI predictions
        if self.freqai_enabled and "&-action" in dataframe.columns:

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # RL Exit Signals
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            # Get RL predictions (action recommendations)
            rl_actions = dataframe["&-action"]

            # Count RL actions for pipeline debugging
            rl_entry_actions = ((rl_actions == 1) | (rl_actions == 2)).sum()
            rl_exit_actions = ((rl_actions == 3) | (rl_actions == 4)).sum()
            rl_hold_actions = (rl_actions == 0).sum()

            # Exit long positions when RL suggests (action 3)
            rl_exit_long = (rl_actions == 3)

            # Exit short positions when RL suggests (action 4)
            rl_exit_short = (rl_actions == 4)

            # Apply confidence threshold if available
            if "&-action_confidence" in dataframe.columns:
                confidence = dataframe["&-action_confidence"]
                rl_exit_long = rl_exit_long & (confidence > self.rl_exit_confidence.value)
                rl_exit_short = rl_exit_short & (confidence > self.rl_exit_confidence.value)

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # DATA COLLECTION: RL Processing Logging
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            if hasattr(self, 'data_collector') and self.data_collector is not None:
                # Get current timestamp
                current_time = dataframe.iloc[-1]['date'] if 'date' in dataframe.columns else pd.Timestamp.now()

                # Count classic signals from entry logic for comparison
                classic_long_signals = (dataframe["enter_long"] == 1).sum()
                classic_short_signals = (dataframe["enter_short"] == 1).sum()
                classic_signals = classic_long_signals + classic_short_signals

                # Log RL processing data
                rl_processing_data = {
                    'timestamp': current_time,
                    'pair': metadata['pair'],
                    'signal_flow': {
                        'classic_signals': int(classic_signals),
                        'rl_entry_actions': int(rl_entry_actions),
                        'rl_exit_actions': int(rl_exit_actions),
                        'rl_hold_actions': int(rl_hold_actions),
                        'total_rl_actions': len(rl_actions),
                        'rl_available': True,
                        'success': rl_entry_actions > 0 or rl_exit_actions > 0,
                        'failure_reason': 'rl_model_no_actions' if rl_entry_actions == 0 and rl_exit_actions == 0 else None,
                        'pipeline_stage': 'rl_processing'
                    }
                }
                self.data_collector.log_signal_propagation(rl_processing_data)

                # CRITICAL: Detect pipeline breakdown - classic signals but no RL entry actions
                if classic_signals > 0 and rl_entry_actions == 0:
                    pipeline_breakdown_data = {
                        'timestamp': current_time,
                        'pair': metadata['pair'],
                        'pipeline_stage': 'rl_entry_conversion',
                        'success': False,
                        'failure_reason': 'rl_model_not_converting_classic_signals_to_entry_actions',
                        'classic_signals': int(classic_signals),
                        'rl_entry_actions': int(rl_entry_actions),
                        'conversion_rate': 0.0,
                        'severity': 'critical'
                    }
                    self.data_collector.log_pipeline_breakdown(pipeline_breakdown_data)
                    logger.error(f"🚨 PIPELINE BREAKDOWN: {metadata['pair']} - {classic_signals} classic signals but 0 RL entry actions!")

                # Log detailed RL model decisions
                if rl_entry_actions > 0 or rl_exit_actions > 0:
                    # Analyze RL decision patterns
                    avg_confidence = float(dataframe["&-action_confidence"].mean()) if "&-action_confidence" in dataframe.columns else 0.0
                    action_distribution = {
                        'hold': int(rl_hold_actions),
                        'enter_long': int((rl_actions == 1).sum()),
                        'enter_short': int((rl_actions == 2).sum()),
                        'exit_long': int((rl_actions == 3).sum()),
                        'exit_short': int((rl_actions == 4).sum())
                    }

                    model_decision_data = {
                        'timestamp': current_time,
                        'pair': metadata['pair'],
                        'model_decision': {
                            'avg_confidence': avg_confidence,
                            'confidence_threshold': self.rl_exit_confidence.value if hasattr(self, 'rl_exit_confidence') else 0.5,
                            'action_distribution': action_distribution,
                            'total_predictions': len(rl_actions),
                            'active_positions': action_distribution['enter_long'] + action_distribution['enter_short'],
                            'exit_signals': action_distribution['exit_long'] + action_distribution['exit_short']
                        }
                    }
                    self.data_collector.log_model_decision(model_decision_data)

                # Log signal propagation from classic to RL
                signal_propagation_data = {
                    'timestamp': current_time,
                    'pair': metadata['pair'],
                    'signal_flow': {
                        'classic_signals_generated': int(classic_signals),
                        'rl_predictions_available': len(rl_actions),
                        'rl_entry_actions_taken': int(rl_entry_actions),
                        'rl_exit_actions_taken': int(rl_exit_actions),
                        'propagation_success': rl_entry_actions > 0,
                        'propagation_rate': (rl_entry_actions / classic_signals) if classic_signals > 0 else 0.0,
                        'pipeline_stage': 'classic_to_rl_conversion'
                    }
                }
                self.data_collector.log_signal_propagation(signal_propagation_data)

            # Log RL processing summary
            logger.info(f"🤖 RL PROCESSING: {metadata['pair']}")
            logger.info(f"   Entry actions: {rl_entry_actions}, Exit actions: {rl_exit_actions}, Hold: {rl_hold_actions}")

            if classic_signals > 0 and rl_entry_actions == 0:
                logger.error(f"🚨 CRITICAL: Classic signals ({classic_signals}) not converted to RL entry actions!")
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Emergency Exit Conditions (Safety)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # Extreme RSI conditions (with proper type checking)
            emergency_exit_long = (
                (dataframe["rsi"] > 85) |  # Extreme overbought
                (dataframe["close"] < dataframe["bb_lower"].fillna(float('inf'))) |  # Below Bollinger Band
                (dataframe["%-volume_ratio_5"] < 0.3) if "%-volume_ratio_5" in dataframe.columns else False
            )

            emergency_exit_short = (
                (dataframe["rsi"] < 15) |  # Extreme oversold
                (dataframe["close"] > dataframe["bb_upper"].fillna(float('-inf'))) |  # Above Bollinger Band
                (dataframe["%-volume_ratio_5"] < 0.3) if "%-volume_ratio_5" in dataframe.columns else False
            )
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Combine Signals
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            dataframe.loc[rl_exit_long | emergency_exit_long, "exit_long"] = 1
            dataframe.loc[rl_exit_short | emergency_exit_short, "exit_short"] = 1
            
        else:
            # Fallback to classic exit conditions if RL not available
            logger.warning(f"RL predictions not available for {metadata['pair']}, using classic exits")
            
            # Classic emergency exits
            dataframe.loc[dataframe["rsi"] > 80, "exit_long"] = 1
            dataframe.loc[dataframe["rsi"] < 20, "exit_short"] = 1
        
        return dataframe
    
    # ═══════════════════════════════════════════════════════════
    # CUSTOM EXIT LOGIC
    # ═══════════════════════════════════════════════════════════
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, 
                   current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        Custom exit logic for position management
        Implements time-based stops and dynamic profit protection
        """
        
        # Calculate position duration
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60  # in minutes
        trade_duration_candles = int(trade_duration / 5)  # convert to 5m candles
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Time-based Exit
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if trade_duration_candles > self.max_position_duration.value:
            return f"time_exit_{trade_duration_candles}_candles"
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Dynamic Profit Protection
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if current_profit > self.breakeven_trigger.value:
            # Move to breakeven + small profit
            if current_profit < 0.005:  # If profit drops below 0.5%
                return "breakeven_protection"
        
        # Emergency stop
        if current_profit < self.emergency_exit_profit.value:
            exit_reason = "emergency_stop"
            # Log to data collector
            self.data_collector.log_trade({
                'pair': pair,
                'exit_time': current_time,
                'exit_price': current_rate,
                'profit_abs': trade.calc_profit_abs(current_rate),
                'profit_pct': current_profit,
                'duration_candles': trade_duration_candles,
                'is_short': trade.is_short,
                'exit_reason': exit_reason,
                'rl_action': 3 if not trade.is_short else 4,  # Emergency exit
                'rl_confidence': 1.0,  # High confidence for safety
            })
            return exit_reason

        return None
    
    # ═══════════════════════════════════════════════════════════
    # POSITION SIZING & LEVERAGE (from MtfScalper)
    # ═══════════════════════════════════════════════════════════
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """Risk-based position sizing"""
        
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            if dataframe is None or dataframe.empty:
                return proposed_stake
                
            last = dataframe.iloc[-1]
            atr_val = float(last.get("atr", 0.0))
            
            if atr_val <= 0 or current_rate <= 0:
                return proposed_stake
            
            stop_distance = self.atr_multiplier * atr_val
            if stop_distance <= 0:
                return proposed_stake
            
            move_pct = stop_distance / current_rate
            if move_pct <= 0:
                return proposed_stake
            
            available_equity = self.wallets.get_total_stake_amount()
            risk_amount = self.risk_per_trade * available_equity
            
            if risk_amount <= 0:
                return proposed_stake
            
            desired_stake = risk_amount / max(1e-12, leverage * move_pct)
            
            # Safety caps
            max_equity_cap = 0.05 * available_equity
            stake_cap = min(float(max_stake), max_equity_cap)
            
            if min_stake is not None:
                desired_stake = max(desired_stake, float(min_stake))
            desired_stake = min(desired_stake, stake_cap)
            
            return float(min(desired_stake, proposed_stake * 2.0))
            
        except Exception as e:
            logger.error(f"Error in custom_stake_amount: {e}")
            return proposed_stake
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """Target leverage 3x"""
        return float(min(3.0, max_leverage))
    
    # ═══════════════════════════════════════════════════════════
    # PROTECTIONS (from MtfScalper)
    # ═══════════════════════════════════════════════════════════
    
    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 12,
                "max_allowed_drawdown": 0.1,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 4,
                "only_per_pair": False,
                "only_per_side": False,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 2,
                "required_profit": 0.02,
            },
        ]
    
    # ═══════════════════════════════════════════════════════════
    # DATA COLLECTION HOOKS
    # ═══════════════════════════════════════════════════════════

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                          side: str, **kwargs) -> bool:
        """
        Override to log trade entry data.
        """
        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return True

        # Get features at entry
        last_candle = dataframe.iloc[-1]
        features_at_entry = {}

        # Extract all %-prefixed features (RL features)
        for col in dataframe.columns:
            if col.startswith('%-') or col.startswith('&-'):
                try:
                    features_at_entry[col] = float(last_candle[col])
                except:
                    features_at_entry[col] = str(last_candle[col])

        # Also include key technical indicators
        for indicator in ['rsi', 'adx', 'atr', 'ema_fast', 'ema_slow', 'macd']:
            if indicator in last_candle:
                try:
                    features_at_entry[indicator] = float(last_candle[indicator])
                except:
                    pass

        # Log prediction at entry
        self.data_collector.log_prediction({
            'timestamp': current_time,
            'pair': pair,
            'features': features_at_entry,
            'prediction': int(last_candle.get('&-action', 0)),
            'confidence': float(last_candle.get('&-action_confidence', 0.0)),
            'current_position': 0,  # About to enter
            'current_profit': 0.0,
            'entry_signal': entry_tag,
            'side': side,
        })

        return True  # Allow trade

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Override to log trade exit data.
        """
        # Get current dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return True

        # Get features at exit
        last_candle = dataframe.iloc[-1]
        features_at_exit = {}

        # Extract all %-prefixed features
        for col in dataframe.columns:
            if col.startswith('%-') or col.startswith('&-'):
                try:
                    features_at_exit[col] = float(last_candle[col])
                except:
                    features_at_exit[col] = str(last_candle[col])

        # Technical indicators
        for indicator in ['rsi', 'adx', 'atr', 'ema_fast', 'ema_slow', 'macd']:
            if indicator in last_candle:
                try:
                    features_at_exit[indicator] = float(last_candle[indicator])
                except:
                    pass

        # Calculate trade duration in candles
        duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        duration_candles = int(duration_minutes / 5)  # 5m timeframe

        # Get features at entry (if stored)
        features_at_entry = {}  # Would need to store this at entry

        # Log complete trade
        trade_data = {
            'pair': pair,
            'entry_time': trade.open_date_utc.isoformat(),
            'exit_time': current_time.isoformat(),
            'entry_price': float(trade.open_rate),
            'exit_price': float(rate),
            'profit_abs': float(trade.close_profit_abs) if trade.close_profit_abs else 0.0,
            'profit_pct': float(trade.close_profit) if trade.close_profit else 0.0,
            'duration_candles': duration_candles,
            'duration_minutes': duration_minutes,
            'is_short': trade.is_short,
            'exit_reason': exit_reason,
            'features_at_entry': features_at_entry,
            'features_at_exit': features_at_exit,
            'rl_action': int(last_candle.get('&-action', 0)),
            'rl_confidence': float(last_candle.get('&-action_confidence', 0.0)),
            'stake_amount': float(trade.stake_amount),
            'leverage': float(trade.leverage) if hasattr(trade, 'leverage') else 1.0,
        }

        self.data_collector.log_trade(trade_data)

        # Log prediction at exit
        self.data_collector.log_prediction({
            'timestamp': current_time,
            'pair': pair,
            'features': features_at_exit,
            'prediction': int(last_candle.get('&-action', 0)),
            'confidence': float(last_candle.get('&-action_confidence', 0.0)),
            'current_position': 1 if trade.is_short else -1,
            'current_profit': float(trade.close_profit) if trade.close_profit else 0.0,
            'exit_reason': exit_reason,
        })

        return True  # Allow exit

    # ═══════════════════════════════════════════════════════════
    # ENHANCED DATA COLLECTION (New for RL Debugging)
    # ═══════════════════════════════════════════════════════════

    def log_signal_generation(self, dataframe: DataFrame, metadata: dict):
        """
        Log classic signal generation for debugging pipeline.

        Critical: This tracks where 3.45% classic signals are generated
        """
        timestamp = datetime.now()

        # Count signals
        long_signals = dataframe["enter_long"].sum()
        short_signals = dataframe["enter_short"].sum()
        total_signals = long_signals + short_signals

        if total_signals > 0:
            # Log signal propagation start
            self.data_collector.log_signal_propagation({
                'timestamp': timestamp,
                'pair': metadata['pair'],
                'classic_long_signal': long_signals,
                'classic_short_signal': short_signals,
                'total_candles': len(dataframe),
                'signal_rate': total_signals / len(dataframe),
                'pipeline_stage': 'signal_generation',
                'success': True,
                'context': {
                    'current_price': dataframe['close'].iloc[-1],
                    'volume': dataframe['volume'].iloc[-1] if 'volume' in dataframe else None,
                    'spread': None,  # Add if available
                }
            })

    def log_rl_processing(self, dataframe: DataFrame, metadata: dict):
        """
        Log RL model processing and decision making.

        Critical: This tracks RL model predictions vs classic signals
        """
        if '&-action' in dataframe.columns:
            timestamp = datetime.now()

            # Count RL actions
            action_counts = dataframe['&-action'].value_counts().to_dict()

            # Analyze action distribution
            hold_actions = action_counts.get(0, 0)
            enter_long = action_counts.get(1, 0)
            enter_short = action_counts.get(2, 0)
            exit_long = action_counts.get(3, 0)
            exit_short = action_counts.get(4, 0)

            # Get latest RL action for detailed logging
            latest_action = dataframe['&-action'].iloc[-1]
            latest_confidence = dataframe.get('&-confidence', pd.Series([0.5] * len(dataframe))).iloc[-1]

            # Log model decision
            self.data_collector.log_model_decision({
                'timestamp': timestamp,
                'pair': metadata['pair'],
                'model_output': [
                    action_counts.get(0, 0) / len(dataframe),  # Hold
                    action_counts.get(1, 0) / len(dataframe),  # Enter Long
                    action_counts.get(2, 0) / len(dataframe),  # Enter Short
                    action_counts.get(3, 0) / len(dataframe),  # Exit Long
                    action_counts.get(4, 0) / len(dataframe),  # Exit Short
                ],
                'selected_action': latest_action,
                'confidence': latest_confidence,
                'state_info': {
                    'total_candles': len(dataframe),
                    'action_distribution': action_counts,
                    'entry_actions': enter_long + enter_short,
                    'exit_actions': exit_long + exit_short,
                    'hold_ratio': hold_actions / len(dataframe),
                }
            })

            # Check for signal-to-action mismatch
            classic_signals = (dataframe["enter_long"].sum() + dataframe["enter_short"].sum())
            rl_entry_actions = enter_long + enter_short

            if classic_signals > 0 and rl_entry_actions == 0:
                # This is the critical bug! Classic signals but no RL entry actions
                self.data_collector.log_pipeline_breakdown({
                    'timestamp': timestamp,
                    'pair': metadata['pair'],
                    'pipeline_stage': 'rl_processing',
                    'success': False,
                    'failure_reason': 'rl_model_not_converting_signals_to_actions',
                    'signal_strength': classic_signals / len(dataframe),
                    'threshold_required': 0.1,  # At least some conversion expected
                    'context': {
                        'classic_signals': classic_signals,
                        'rl_entry_actions': rl_entry_actions,
                        'total_candles': len(dataframe),
                        'hold_actions': hold_actions,
                        'latest_action': latest_action,
                        'latest_confidence': latest_confidence
                    }
                })

    def log_trade_execution(self, trade: Trade, entry_time: datetime,
                           entry_price: float, side: str):
        """
        Log successful trade execution.

        Critical: This confirms end-to-end pipeline success
        """
        self.data_collector.log_trade({
            'pair': trade.pair,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'is_short': side == 'short',
            'exit_reason': None,  # Will be filled on exit
            'features_at_entry': {},  # Will be populated if available
            'rl_action': 1 if side == 'long' else 2,
            'rl_confidence': 1.0,  # Trade executed
        })

        # Log successful signal propagation
        self.data_collector.log_signal_propagation({
            'timestamp': entry_time,
            'pair': trade.pair,
            'classic_long_signal': 1 if side == 'long' else 0,
            'classic_short_signal': 1 if side == 'short' else 0,
            'rl_raw_action': 1 if side == 'long' else 2,
            'rl_confidence': 1.0,
            'position_state': 1 if side == 'long' else -1,
            'trade_executed': True,
            'pipeline_stage': 'trade_execution',
            'success': True,
            'execution_delay_ms': 0,  # Immediate in backtest
        })

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """Called at the start of each bot loop - use for periodic saves and monitoring."""

        # Log DataCollector stats every 15 minutes for progress tracking
        if current_time.minute % 15 == 0:
            trades_count = len(self.data_collector.trades)
            signals_count = len(getattr(self.data_collector, 'signal_propagation', []))
            decisions_count = len(getattr(self.data_collector, 'model_decisions', []))

            logger.info(f"📊 DataCollector stats at {current_time:%Y-%m-%d %H:%M}: "
                       f"Trades={trades_count}, "
                       f"Signals={signals_count}, "
                       f"Decisions={decisions_count}")

        # Auto-save checkpoint every 6 hours for crash protection
        if current_time.minute == 0 and current_time.hour % 6 == 0:
            try:
                checkpoint_name = f"checkpoint_{current_time:%Y%m%d_%H%M}"
                logger.info(f"💾 Creating checkpoint: {checkpoint_name}")
                self.data_collector.save_all(custom_name=checkpoint_name)
                logger.info(f"✅ Checkpoint saved successfully")
            except Exception as e:
                logger.warning(f"⚠️ Checkpoint save failed: {e}")

    def bot_end(self, **kwargs) -> None:
        """Called when bot ends - save collected data."""
        try:
            logger.info("Saving collected data...")
            self.data_collector.save_all()
            logger.info("Data collection complete!")
        except Exception as e:
            logger.error(f"Error saving collected data: {e}")

    # ═══════════════════════════════════════════════════════════
    # INFORMATIVE PAIRS (for RL correlation)
    # ═══════════════════════════════════════════════════════════

    def informative_pairs(self):
        """Define additional pairs for correlation features"""
        # CRITICAL: Must return empty list to match config's include_corr_pairlist: []
        # Adding pairs here would create feature mismatch between training and prediction
        return []
