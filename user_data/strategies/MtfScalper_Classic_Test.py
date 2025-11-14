# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
"""
MtfScalper Classic Test Strategy
================================
PURE CLASSIC VERSION - No RL components

Purpose: Test if the original strategy generates ANY entry signals at all.
This extracts ONLY the entry logic from MtfScalper_RL_Hybrid to test signal generation.

Architecture:
- Entry: IDENTICAL to MtfScalper_RL_Hybrid (exact copy)
- Exit: Simple ROI/StopLoss (no RL)
- Goal: Prove whether the strategy generates signals or not
"""

from datetime import datetime
from typing import Optional
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

logger = logging.getLogger(__name__)

# Try to import qtpylib for bollinger bands
try:
    import qtpylib
except ImportError:
    qtpylib = None
    logger.warning("qtpylib not available, bollinger bands will not work")


class MtfScalper_Classic_Test(IStrategy):
    """
    PURE CLASSIC TEST - Exact copy of MtfScalper_RL_Hybrid entry logic
    NO RL components whatsoever

    Purpose: Test if the strategy generates ANY signals
    """

    INTERFACE_VERSION = 3

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # HYPERPARAMETERS - EXACT COPY FROM ORIGINAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Timeframes (REQUIRED by IStrategy)
    timeframe = '5m'
    base_timeframe = '5m'

    # ADX thresholds (exact copy)
    adx_thr_buy = IntParameter(20, 40, default=25, space='buy')
    adx_thr_sell = IntParameter(20, 40, default=25, space='sell')

    # ATR volatility threshold (exact copy)
    atr_threshold = DecimalParameter(1.0, 10.0, default=5.0, decimals=1, space='buy')

    # RSI levels (exact copy)
    buy_rsi = IntParameter(30, 70, default=55, space='buy')
    sell_rsi = IntParameter(30, 70, default=45, space='sell')

    # Minimal ROI - Very simple to see if entries work
    minimal_roi = {
        "0": 0.04,    # 4% profit target
        "30": 0.02,   # 2% after 30 minutes
        "60": 0.01    # 1% after 60 minutes
    }

    # Stoploss
    stoploss = -0.10  # -10% stop loss

    # Trailing stop (disabled for test)
    trailing_stop = False

    # Order settings - Back to limit orders for proper testing
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
    }

    # Order time in force
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # Position settings
    max_open_trades = 3

    # Startup candle count
    startup_candle_count: int = 240

    # Process only new candles
    process_only_new_candles = True

    # Use exit signal (we won't use it, but keep for compatibility)
    use_exit_signal = True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INFORMATIVE TIMEFRAMES - EXACT COPY FROM ORIGINAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @informative('15m')
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """15m indicators - exact copy"""
        return self.calculate_all_timeframe_indicators(dataframe)

    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """1h indicators - exact copy"""
        return self.calculate_all_timeframe_indicators(dataframe)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INDICATOR CALCULATION - EXACT COPY FROM ORIGINAL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def calculate_all_timeframe_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate all indicators for any timeframe - exact copy"""

        # Moving averages
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=200)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['+di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['-di'] = ta.MINUS_DI(dataframe, timeperiod=14)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Bollinger Bands
        if qtpylib is not None:
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
            dataframe['bb_lowerband'] = bollinger['lower']
            dataframe['bb_middleband'] = bollinger['mid']
            dataframe['bb_upperband'] = bollinger['upper']

        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        return dataframe

    def populate_indicators_base(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Base timeframe indicators - exact copy from original"""

        # Moving averages
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_trend'] = ta.EMA(dataframe, timeperiod=200)

        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['+di'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['-di'] = ta.MINUS_DI(dataframe, timeperiod=14)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Bollinger Bands
        if qtpylib is not None:
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
            dataframe['bb_lowerband'] = bollinger['lower']
            dataframe['bb_middleband'] = bollinger['mid']
            dataframe['bb_upperband'] = bollinger['upper']

        # Stochastic
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']

        # Additional indicators from original
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=14)
        dataframe["stoch"] = ta.STOCH(dataframe)["slowk"]
        dataframe["obv"] = ta.OBV(dataframe)

        return dataframe

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # MAIN INDICATOR POPULATION - NO FREQAI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Main indicator population - NO FreqAI"""

        # Classic indicators only
        dataframe = self.populate_indicators_base(dataframe, metadata)

        return dataframe

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ENTRY LOGIC - EXACT COPY FROM MtfScalper_RL_Hybrid
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        ENTRY LOGIC - EXACT COPY FROM MtfScalper_RL_Hybrid

        This is the EXACT same logic that should be generating signals
        but apparently doesn't. We're testing it in isolation.
        """

        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Multi-Timeframe Trend Alignment - EXACT COPY
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
        # Volatility Filter - EXACT COPY
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        atr_pct = (dataframe["atr"] / dataframe["close"]) * 100
        volatility_filter = atr_pct < self.atr_threshold.value

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Final Entry Conditions - EXACT COPY
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

        # Set entry signals
        dataframe.loc[buy_condition, "enter_long"] = 1
        dataframe.loc[sell_condition, "enter_short"] = 1

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # DEBUG LOGGING - Count signals
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        total_candles = len(dataframe)
        long_signals = dataframe["enter_long"].sum()
        short_signals = dataframe["enter_short"].sum()
        total_signals = long_signals + short_signals

        signal_percentage = (total_signals / total_candles) * 100 if total_candles > 0 else 0

        logger.info(f"🔍 SIGNAL ANALYSIS for {metadata['pair']}:")
        logger.info(f"   Total candles: {total_candles}")
        logger.info(f"   Long signals: {long_signals} ({(long_signals/total_candles)*100:.4f}%)")
        logger.info(f"   Short signals: {short_signals} ({(short_signals/total_candles)*100:.4f}%)")
        logger.info(f"   Total signals: {total_signals} ({signal_percentage:.6f}%)")

        if total_signals == 0:
            logger.warning(f"❌ NO SIGNALS GENERATED for {metadata['pair']}!")
            logger.warning(f"   This proves the strategy entry logic is too restrictive")

        return dataframe

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EXIT LOGIC - SIMPLE (NO RL)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Simple exit logic - NO RL

        Just using ROI and stoploss for this test
        """

        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # Simple exit conditions
        long_exit = (
            (dataframe["rsi"] > 70)  # Overbought
        )

        short_exit = (
            (dataframe["rsi"] < 30)  # Oversold
        )

        # Add Bollinger Bands exit if available
        if qtpylib is not None:
            long_exit = long_exit | (dataframe["close"] < dataframe["bb_lowerband"])  # Price hit lower BB
            short_exit = short_exit | (dataframe["close"] > dataframe["bb_upperband"])  # Price hit upper BB

        dataframe.loc[long_exit, "exit_long"] = 1
        dataframe.loc[short_exit, "exit_short"] = 1

        return dataframe

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CUSTOM STOPLOSS (optional)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Simple custom stoploss - NO RL
        """
        return self.stoploss

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRADE CONFIRMATION
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_side: str, **kwargs) -> bool:
        """
        Confirm trade entry - always true for this test
        """
        logger.info(f"✅ ENTRY CONFIRMED: {pair} {entry_side} @ {rate}")
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """
        Confirm trade exit - always true for this test
        """
        logger.info(f"✅ EXIT CONFIRMED: {pair} {trade.is_short} @ {rate} (Reason: {exit_reason})")
        return True


# Required import for bollinger bands
try:
    import qtpylib
except ImportError:
    logger.warning("qtpylib not available, bollinger bands will not work")