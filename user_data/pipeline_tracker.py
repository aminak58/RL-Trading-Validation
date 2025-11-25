#!/usr/bin/env python3
"""
Comprehensive Pipeline Tracker for RL Trading System
====================================================

Tracks the complete flow from classic signals to final trades:
1. Classic signal generation
2. FreqAI feature engineering
3. RL model prediction
4. RL action decision
5. Strategy entry/exit decision
6. Trade execution

Usage:
    from user_data.pipeline_tracker import PipelineTracker

    tracker = PipelineTracker()
    tracker.track_classic_signal(...)
    tracker.track_rl_prediction(...)
    tracker.track_action(...)
    tracker.generate_report()
"""

import logging
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline stages"""
    CLASSIC_SIGNAL = "classic_signal"
    FREQAI_FEATURES = "freqai_features"
    RL_PREDICTION = "rl_prediction"
    RL_ACTION = "rl_action"
    STRATEGY_DECISION = "strategy_decision"
    TRADE_EXECUTION = "trade_execution"


class SignalType(Enum):
    """Signal types"""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class RLAction(Enum):
    """RL Actions"""
    HOLD = 0
    LONG_ENTER = 1
    SHORT_ENTER = 2
    LONG_EXIT = 3
    SHORT_EXIT = 4


@dataclass
class PipelineEvent:
    """Single event in the pipeline"""
    timestamp: str
    candle_date: str
    pair: str
    stage: str

    # Classic signal info
    classic_long_signal: Optional[float] = None
    classic_short_signal: Optional[float] = None

    # FreqAI info
    freqai_enabled: Optional[bool] = None
    feature_count: Optional[int] = None
    features_available: Optional[bool] = None

    # RL prediction info
    rl_prediction: Optional[float] = None
    rl_confidence: Optional[float] = None
    rl_action: Optional[int] = None
    rl_action_name: Optional[str] = None

    # Strategy decision
    entry_signal: Optional[float] = None
    exit_signal: Optional[float] = None

    # Trade execution
    trade_executed: Optional[bool] = None

    # Failure tracking
    blocked_at_stage: Optional[str] = None
    block_reason: Optional[str] = None

    # Additional context
    metadata: Optional[Dict[str, Any]] = None


class PipelineTracker:
    """
    Comprehensive tracker for the entire RL trading pipeline

    Tracks every step from classic signal generation to trade execution,
    identifying exactly where and why signals fail to convert to trades.
    """

    def __init__(self, output_dir: str = "user_data/pipeline_tracking"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Event storage
        self.events: List[PipelineEvent] = []

        # Statistics
        self.stats = {
            "classic_signals": 0,
            "classic_long": 0,
            "classic_short": 0,
            "rl_predictions": 0,
            "rl_entry_actions": 0,
            "rl_hold_actions": 0,
            "strategy_entries": 0,
            "trades_executed": 0,
            "blocks": {}  # stage -> count
        }

        logger.info(f"Pipeline Tracker initialized: {self.output_dir}")

    def track_classic_signal(
        self,
        candle_date: str,
        pair: str,
        long_signal: float,
        short_signal: float,
        metadata: Optional[Dict] = None
    ):
        """Track classic signal generation"""
        event = PipelineEvent(
            timestamp=datetime.now().isoformat(),
            candle_date=candle_date,
            pair=pair,
            stage=PipelineStage.CLASSIC_SIGNAL.value,
            classic_long_signal=long_signal,
            classic_short_signal=short_signal,
            metadata=metadata
        )

        self.events.append(event)

        if long_signal > 0 or short_signal > 0:
            self.stats["classic_signals"] += 1
            if long_signal > 0:
                self.stats["classic_long"] += 1
            if short_signal > 0:
                self.stats["classic_short"] += 1

        return event

    def track_freqai_features(
        self,
        candle_date: str,
        pair: str,
        enabled: bool,
        feature_count: int,
        features_available: bool,
        block_reason: Optional[str] = None
    ):
        """Track FreqAI feature engineering"""
        event = PipelineEvent(
            timestamp=datetime.now().isoformat(),
            candle_date=candle_date,
            pair=pair,
            stage=PipelineStage.FREQAI_FEATURES.value,
            freqai_enabled=enabled,
            feature_count=feature_count,
            features_available=features_available
        )

        if block_reason:
            event.blocked_at_stage = PipelineStage.FREQAI_FEATURES.value
            event.block_reason = block_reason
            self._track_block(PipelineStage.FREQAI_FEATURES.value, block_reason)

        self.events.append(event)
        return event

    def track_rl_prediction(
        self,
        candle_date: str,
        pair: str,
        prediction: float,
        confidence: float,
        action: int,
        action_name: str,
        block_reason: Optional[str] = None
    ):
        """Track RL model prediction"""
        event = PipelineEvent(
            timestamp=datetime.now().isoformat(),
            candle_date=candle_date,
            pair=pair,
            stage=PipelineStage.RL_PREDICTION.value,
            rl_prediction=prediction,
            rl_confidence=confidence,
            rl_action=action,
            rl_action_name=action_name
        )

        self.stats["rl_predictions"] += 1

        if action in [RLAction.LONG_ENTER.value, RLAction.SHORT_ENTER.value]:
            self.stats["rl_entry_actions"] += 1
        elif action == RLAction.HOLD.value:
            self.stats["rl_hold_actions"] += 1

        if block_reason:
            event.blocked_at_stage = PipelineStage.RL_PREDICTION.value
            event.block_reason = block_reason
            self._track_block(PipelineStage.RL_PREDICTION.value, block_reason)

        self.events.append(event)
        return event

    def track_strategy_decision(
        self,
        candle_date: str,
        pair: str,
        entry_signal: float,
        exit_signal: float,
        block_reason: Optional[str] = None
    ):
        """Track strategy's final entry/exit decision"""
        event = PipelineEvent(
            timestamp=datetime.now().isoformat(),
            candle_date=candle_date,
            pair=pair,
            stage=PipelineStage.STRATEGY_DECISION.value,
            entry_signal=entry_signal,
            exit_signal=exit_signal
        )

        if entry_signal > 0:
            self.stats["strategy_entries"] += 1

        if block_reason:
            event.blocked_at_stage = PipelineStage.STRATEGY_DECISION.value
            event.block_reason = block_reason
            self._track_block(PipelineStage.STRATEGY_DECISION.value, block_reason)

        self.events.append(event)
        return event

    def track_trade_execution(
        self,
        candle_date: str,
        pair: str,
        executed: bool,
        block_reason: Optional[str] = None
    ):
        """Track actual trade execution"""
        event = PipelineEvent(
            timestamp=datetime.now().isoformat(),
            candle_date=candle_date,
            pair=pair,
            stage=PipelineStage.TRADE_EXECUTION.value,
            trade_executed=executed
        )

        if executed:
            self.stats["trades_executed"] += 1
        elif block_reason:
            event.blocked_at_stage = PipelineStage.TRADE_EXECUTION.value
            event.block_reason = block_reason
            self._track_block(PipelineStage.TRADE_EXECUTION.value, block_reason)

        self.events.append(event)
        return event

    def _track_block(self, stage: str, reason: str):
        """Track pipeline blocks"""
        key = f"{stage}:{reason}"
        self.stats["blocks"][key] = self.stats["blocks"].get(key, 0) + 1

    def get_stats(self) -> Dict:
        """Get current statistics"""
        return self.stats.copy()

    def generate_report(self, filename: Optional[str] = None) -> str:
        """Generate comprehensive pipeline report"""
        if filename is None:
            filename = f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        report_path = self.output_dir / f"{filename}.md"

        # Calculate conversion rates
        signal_to_prediction = (
            (self.stats["rl_predictions"] / self.stats["classic_signals"] * 100)
            if self.stats["classic_signals"] > 0 else 0
        )

        prediction_to_action = (
            (self.stats["rl_entry_actions"] / self.stats["rl_predictions"] * 100)
            if self.stats["rl_predictions"] > 0 else 0
        )

        action_to_strategy = (
            (self.stats["strategy_entries"] / self.stats["rl_entry_actions"] * 100)
            if self.stats["rl_entry_actions"] > 0 else 0
        )

        strategy_to_trade = (
            (self.stats["trades_executed"] / self.stats["strategy_entries"] * 100)
            if self.stats["strategy_entries"] > 0 else 0
        )

        overall_conversion = (
            (self.stats["trades_executed"] / self.stats["classic_signals"] * 100)
            if self.stats["classic_signals"] > 0 else 0
        )

        # Generate markdown report
        report = f"""# Pipeline Tracking Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics

| Metric | Count |
|--------|-------|
| Classic Signals Generated | {self.stats['classic_signals']} |
| - Long Signals | {self.stats['classic_long']} |
| - Short Signals | {self.stats['classic_short']} |
| RL Predictions Made | {self.stats['rl_predictions']} |
| RL Entry Actions | {self.stats['rl_entry_actions']} |
| RL Hold Actions | {self.stats['rl_hold_actions']} |
| Strategy Entry Decisions | {self.stats['strategy_entries']} |
| **Trades Executed** | **{self.stats['trades_executed']}** |

## Conversion Funnel

```
Classic Signals:      {self.stats['classic_signals']}
                      ↓ {signal_to_prediction:.1f}%
RL Predictions:       {self.stats['rl_predictions']}
                      ↓ {prediction_to_action:.1f}%
RL Entry Actions:     {self.stats['rl_entry_actions']}
                      ↓ {action_to_strategy:.1f}%
Strategy Entries:     {self.stats['strategy_entries']}
                      ↓ {strategy_to_trade:.1f}%
Trades Executed:      {self.stats['trades_executed']}
```

**Overall Conversion Rate**: {overall_conversion:.2f}%

## Pipeline Blocks

"""

        if self.stats["blocks"]:
            report += "| Stage | Reason | Count |\n"
            report += "|-------|--------|-------|\n"
            for block_key, count in sorted(
                self.stats["blocks"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                stage, reason = block_key.split(":", 1)
                report += f"| {stage} | {reason} | {count} |\n"
        else:
            report += "*No blocks detected*\n"

        report += f"\n## Detailed Events\n\nTotal events tracked: {len(self.events)}\n"
        report += f"\nSee `{filename}.csv` for full event log.\n"

        # Save report
        with open(report_path, 'w') as f:
            f.write(report)

        # Save events as CSV
        self._save_events_csv(filename)

        # Save events as JSON
        self._save_events_json(filename)

        logger.info(f"Pipeline report generated: {report_path}")
        return str(report_path)

    def _save_events_csv(self, filename: str):
        """Save events as CSV"""
        if not self.events:
            return

        csv_path = self.output_dir / f"{filename}.csv"

        # Convert events to dict
        events_dict = [asdict(event) for event in self.events]

        # Create DataFrame
        df = pd.DataFrame(events_dict)

        # Save CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Events CSV saved: {csv_path}")

    def _save_events_json(self, filename: str):
        """Save events as JSON"""
        if not self.events:
            return

        json_path = self.output_dir / f"{filename}.json"

        # Convert events to dict
        events_dict = [asdict(event) for event in self.events]

        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(events_dict, f, indent=2)

        logger.info(f"Events JSON saved: {json_path}")

    def clear(self):
        """Clear all tracked events and stats"""
        self.events.clear()
        self.stats = {
            "classic_signals": 0,
            "classic_long": 0,
            "classic_short": 0,
            "rl_predictions": 0,
            "rl_entry_actions": 0,
            "rl_hold_actions": 0,
            "strategy_entries": 0,
            "trades_executed": 0,
            "blocks": {}
        }
        logger.info("Pipeline tracker cleared")


# Global singleton instance
_tracker_instance = None


def get_tracker() -> PipelineTracker:
    """Get global pipeline tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PipelineTracker()
    return _tracker_instance
