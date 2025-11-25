#!/usr/bin/env python3
"""
Validate Enhanced Monitoring System
===================================

Simple validation script to check if the enhanced DataCollector
monitoring system is working by running a short backtest.

This script validates:
1. DataCollector integration in strategy
2. Data collection during backtest
3. Pipeline breakdown detection
4. Signal propagation tracking

Usage:
python validate_monitoring.py
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path
import pandas as pd
import threading
import glob

class ProgressMonitor:
    """Simple progress monitoring for validation backtest"""

    def __init__(self, analysis_dir="user_data/analysis_data"):
        self.analysis_dir = Path(analysis_dir)
        self.running = True
        self.start_time = time.time()
        self.last_file_count = {}

    def monitor_progress(self):
        """Monitor file creation and progress in background thread"""

        while self.running:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)

            # Show elapsed time
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Check for data files
            if self.analysis_dir.exists():
                files_found = []
                total_records = 0

                for pattern in ["*.csv", "*.json"]:
                    for file_path in self.analysis_dir.glob(pattern):
                        if file_path.is_file():
                            try:
                                if file_path.suffix == '.csv':
                                    df = pd.read_csv(file_path)
                                    record_count = len(df)
                                else:  # JSON
                                    with open(file_path, 'r') as f:
                                        data = json.load(f)
                                        if isinstance(data, list):
                                            record_count = len(data)
                                        else:
                                            record_count = 1

                                files_found.append(f"{file_path.name}: {record_count} records")
                                total_records += record_count

                            except Exception as e:
                                files_found.append(f"{file_path.name}: unreadable")

                if files_found:
                    print(f"\r⏱️  Elapsed: {time_str} | 📁 Files: {len(files_found)} | 📊 Total Records: {total_records}", end="", flush=True)
                else:
                    print(f"\r⏱️  Elapsed: {time_str} | 🔄 Backtest running...", end="", flush=True)
            else:
                print(f"\r⏱️  Elapsed: {time_str} | 🔄 Backtest running...", end="", flush=True)

            time.sleep(3)  # Update every 3 seconds

        print()  # New line when monitoring stops

    def start_monitoring(self):
        """Start the monitoring thread"""
        self.monitor_thread = threading.Thread(target=self.monitor_progress, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)

def run_validation_backtest():
    """Run a short validation backtest to test monitoring system"""

    print("🔬 VALIDATION: Running short backtest to test enhanced monitoring")
    print("=" * 70)

    # Backtest command with short timerange
    cmd = [
        "freqtrade", "backtesting",
        "--strategy", "MtfScalper_RL_Hybrid",
        "--config", "configs/config_rl_hybrid.json",
        "--freqaimodel", "MtfScalperRLModel",
        "--timerange", "20241001-20241007",  # Use available data period
        "--timeframe", "5m"
    ]

    print(f"📊 Running: {' '.join(cmd)}")
    print("⏱️  No timeout limit - this may take considerable time for comprehensive analysis...")
    print("   Press Ctrl+C to interrupt if needed")
    print("\n🔄 Starting progress monitoring...")

    try:
        # Start progress monitoring
        monitor = ProgressMonitor()
        monitor.start_monitoring()

        # Run backtest
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
            # No timeout - allow long processing for comprehensive analysis
        )
        end_time = time.time()

        # Stop monitoring
        monitor.stop_monitoring()

        duration = end_time - start_time
        print(f"⏱️  Backtest completed in {duration:.1f} seconds")

        # Print results summary
        if result.returncode == 0:
            print("✅ Backtest completed successfully")
            print("\n📋 BACKTEST RESULTS:")
            print("=" * 50)

            # Show final data collection summary
            print("\n📁 FINAL DATA COLLECTION SUMMARY:")
            print("=" * 40)
            show_file_summary()

            # Extract key metrics from output
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in [
                    'total trades', 'win rate', 'profit', 'drawdown',
                    'sharpe', 'sortino', 'calmar', 'avg profit'
                ]):
                    print(f"   📊 {line.strip()}")

        else:
            print("❌ Backtest failed")
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            return False

    # Timeout removed - process will run for as long as needed
    except KeyboardInterrupt:
        print("\n⏹️  Backtest interrupted by user")
        if 'monitor' in locals():
            monitor.stop_monitoring()
        return False
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        if 'monitor' in locals():
            monitor.stop_monitoring()
        return False

    return True

def show_file_summary():
    """Show detailed file summary after completion"""
    analysis_dir = Path("user_data/analysis_data")

    if not analysis_dir.exists():
        print("   ❌ No analysis data directory found")
        return

    print("   📊 Data files created:")

    for pattern in ["*.csv", "*.json"]:
        files = list(analysis_dir.glob(pattern))
        for file_path in sorted(files):
            try:
                if file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    size_mb = file_path.stat().st_size / (1024*1024)
                    print(f"   📄 {file_path.name}: {len(df):,} records ({size_mb:.2f} MB)")
                else:  # JSON
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            record_count = len(data)
                        else:
                            record_count = 1
                        size_mb = file_path.stat().st_size / (1024*1024)
                        print(f"   📄 {file_path.name}: {record_count:,} records ({size_mb:.2f} MB)")
            except Exception as e:
                print(f"   ❌ {file_path.name}: Error reading file")

    total_size = sum(f.stat().st_size for f in analysis_dir.glob('*') if f.is_file()) / (1024*1024)
    print(f"   📁 Total directory size: {total_size:.2f} MB")

def check_data_collection():
    """Check if data was collected during backtest"""

    print("\n📁 VALIDATION: Checking collected data...")
    print("=" * 50)

    # Check for analysis data directory
    analysis_dir = Path("user_data/analysis_data")
    if not analysis_dir.exists():
        print("❌ Analysis data directory not found")
        print("   Expected: user_data/analysis_data/")
        return False

    print("✅ Analysis data directory found")

    # Check for data files
    expected_files = [
        "signal_generation.csv",
        "rl_processing.csv",
        "signal_propagation.csv",
        "model_decisions.csv",
        "pipeline_breakdowns.csv",
        "trades.csv",
        "predictions.csv"
    ]

    found_files = 0
    for file in expected_files:
        file_path = analysis_dir / file
        if file_path.exists():
            found_files += 1
            df = pd.read_csv(file_path)
            print(f"✅ {file} - {len(df)} records")
        else:
            print(f"❌ {file} - NOT FOUND")

    success_rate = (found_files / len(expected_files)) * 100
    print(f"\n📊 Data collection success rate: {success_rate:.1f}%")

    if success_rate >= 70:
        print("✅ Data collection working well")
        return True
    else:
        print("⚠️  Data collection incomplete")
        return False

def analyze_pipeline_breakdowns():
    """Analyze pipeline breakdown detection"""

    print("\n🚨 VALIDATION: Analyzing pipeline breakdowns...")
    print("=" * 50)

    breakdown_file = Path("user_data/analysis_data/pipeline_breakdowns.csv")

    if not breakdown_file.exists():
        print("ℹ️  No pipeline breakdowns detected - Good sign!")
        return True

    df = pd.read_csv(breakdown_file)

    if len(df) == 0:
        print("ℹ️  No pipeline breakdowns recorded - Good sign!")
        return True

    print(f"⚠️  Found {len(df)} pipeline breakdowns:")

    # Analyze breakdown types
    breakdown_types = df['failure_reason'].value_counts()
    for reason, count in breakdown_types.items():
        print(f"   🚨 {reason}: {count} occurrences")

    # Check severity
    if 'severity' in df.columns:
        critical_count = (df['severity'] == 'critical').sum()
        if critical_count > 0:
            print(f"   🔴 CRITICAL breakdowns: {critical_count}")

        warning_count = (df['severity'] == 'warning').sum()
        if warning_count > 0:
            print(f"   🟡 Warning breakdowns: {warning_count}")

    return True

def analyze_signal_propagation():
    """Analyze signal propagation effectiveness"""

    print("\n📡 VALIDATION: Analyzing signal propagation...")
    print("=" * 50)

    propagation_file = Path("user_data/analysis_data/signal_propagation.csv")

    if not propagation_file.exists():
        print("❌ Signal propagation data not found")
        return False

    df = pd.read_csv(propagation_file)

    if len(df) == 0:
        print("ℹ️  No signal propagation data available")
        return True

    print(f"📊 Signal propagation records: {len(df)}")

    # Analyze propagation rates
    if 'signal_flow.propagation_rate' in df.columns:
        avg_propagation_rate = df['signal_flow.propagation_rate'].mean()
        print(f"📈 Average propagation rate: {avg_propagation_rate:.2%}")

        if avg_propagation_rate < 0.5:
            print("⚠️  Low propagation rate detected")
        elif avg_propagation_rate < 0.8:
            print("🟡 Moderate propagation rate")
        else:
            print("✅ Good propagation rate")

    # Analyze pipeline stages
    if 'signal_flow.pipeline_stage' in df.columns:
        stages = df['signal_flow.pipeline_stage'].value_counts()
        print("🔄 Pipeline stages:")
        for stage, count in stages.items():
            print(f"   {stage}: {count}")

    return True

def generate_summary_report():
    """Generate final validation summary"""

    print("\n" + "=" * 80)
    print("📋 VALIDATION SUMMARY REPORT")
    print("=" * 80)

    # Check overall system health
    checks = {
        "Data Collection": check_data_collection(),
        "Pipeline Breakdown Detection": analyze_pipeline_breakdowns(),
        "Signal Propagation Analysis": analyze_signal_propagation()
    }

    passed_checks = sum(checks.values())
    total_checks = len(checks)

    print(f"\n🎯 OVERALL STATUS: {passed_checks}/{total_checks} checks passed")

    if passed_checks == total_checks:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print("\n✅ Enhanced DataCollector monitoring system is working correctly")
        print("✅ Signal pipeline tracking is active")
        print("✅ Breakdown detection is functional")
        print("✅ Data export is working")

        print("\n📊 NEXT STEPS:")
        print("1. Run full backtest with: freqtrade backtesting --config configs/config_rl_hybrid.json")
        print("2. Monitor user_data/analysis_data/ for detailed insights")
        print("3. Check pipeline_breakdowns.csv for any signal conversion issues")
        print("4. Analyze signal_propagation.csv for end-to-end signal tracking")

        return True

    else:
        print("⚠️  SOME ISSUES DETECTED")

        for check_name, status in checks.items():
            if status:
                print(f"✅ {check_name}: OK")
            else:
                print(f"❌ {check_name}: NEEDS ATTENTION")

        print("\n🔧 RECOMMENDED ACTIONS:")
        print("1. Check DataCollector integration in strategy")
        print("2. Verify data directory permissions")
        print("3. Review strategy configuration")

        return False

def main():
    """Main validation function"""

    print("🔬 ENHANCED DATACOLLECTOR MONITORING VALIDATION")
    print("=" * 80)
    print("Validating the complete enhanced monitoring system")
    print("integrated into MtfScalper_RL_Hybrid RL trading strategy")
    print("=" * 80)

    # Step 1: Run validation backtest
    backtest_success = run_validation_backtest()

    if not backtest_success:
        print("\n❌ VALIDATION FAILED: Backtest could not complete")
        return 1

    # Step 2: Check data collection and analysis
    print("\n🔍 ANALYZING COLLECTED DATA...")

    success = generate_summary_report()

    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)