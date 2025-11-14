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
import logging
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
    
    # RL Exit thresholds
    rl_exit_confidence = DecimalParameter(0.5, 0.9, default=0.7, space="sell", optimize=True)
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
        logger.info("Data collector initialized")

    # ═══════════════════════════════════════════════════════════
    # FREQAI CONFIGURATION
    # ═══════════════════════════════════════════════════════════
    
    def freqai_config(self) -> Dict[str, Any]:
        """Returns FreqAI configuration for RL model"""
        return {
            "enabled": True,
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

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        Standard feature engineering required for RL models.
        Includes raw price data that RL environment needs for price access.
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
        dataframe["bb_upper"], _, dataframe["bb_lower"] = ta.BBANDS(dataframe, timeperiod=20)
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
        
        # Log entry signals for debugging
        if buy_condition.any():
            logger.info(f"Long entry signal for {metadata['pair']}")
        if sell_condition.any():
            logger.info(f"Short entry signal for {metadata['pair']}")
        
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
            # Emergency Exit Conditions (Safety)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # Extreme RSI conditions
            emergency_exit_long = (
                (dataframe["rsi"] > 85) |  # Extreme overbought
                (dataframe["close"] < dataframe["bb_lower"]) |  # Below Bollinger Band
                (dataframe["%-volume_ratio_5"] < 0.3) if "%-volume_ratio_5" in dataframe.columns else False
            )
            
            emergency_exit_short = (
                (dataframe["rsi"] < 15) |  # Extreme oversold
                (dataframe["close"] > dataframe["bb_upper"]) |  # Above Bollinger Band
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
            return "emergency_stop"
        
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

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        """Called at the start of each bot loop."""
        # Can be used for periodic checks
        pass

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
