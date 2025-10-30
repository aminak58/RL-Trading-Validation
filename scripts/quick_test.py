#!/usr/bin/env python3
"""
Quick Test Script for MtfScalper RL Hybrid Strategy
====================================================
This script validates the strategy setup and runs basic tests.
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_environment():
    """Check if all required packages are installed"""
    print_header("🔍 بررسی محیط")
    
    required_packages = {
        'freqtrade': 'freqtrade',
        'torch': 'torch',
        'stable_baselines3': 'stable-baselines3',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'talib': 'TA-Lib'
    }
    
    missing = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} نصب شده")
        except ImportError:
            print(f"❌ {package_name} نصب نشده")
            missing.append(package_name)
    
    if missing:
        print(f"\n⚠️ پکیج‌های زیر را نصب کنید:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def validate_strategy_files():
    """Check if all strategy files exist"""
    print_header("📁 بررسی فایل‌ها")
    
    required_files = {
        'MtfScalper_RL_Hybrid.py': 'استراتژی اصلی',
        'MtfScalperRLModel.py': 'مدل RL',
        'config_rl_hybrid.json': 'فایل کانفیگ',
        'feature_analysis.ipynb': 'نوت‌بوک تحلیل',
        'setup_guide.md': 'راهنمای نصب'
    }
    
    all_exist = True
    for filename, description in required_files.items():
        if Path(filename).exists():
            print(f"✅ {filename} - {description}")
        else:
            print(f"❌ {filename} - {description} یافت نشد")
            all_exist = False
    
    return all_exist

def test_strategy_syntax():
    """Test if strategy can be imported without errors"""
    print_header("🧪 تست Syntax استراتژی")
    
    try:
        # Test importing the strategy
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "MtfScalper_RL_Hybrid",
            "MtfScalper_RL_Hybrid.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check if class exists
        if hasattr(module, 'MtfScalper_RL_Hybrid'):
            print("✅ استراتژی بدون خطا import شد")
            
            # Check key methods
            strategy = module.MtfScalper_RL_Hybrid
            required_methods = [
                'populate_indicators',
                'populate_entry_trend',
                'populate_exit_trend',
                'feature_engineering_expand_all'
            ]
            
            for method in required_methods:
                if hasattr(strategy, method):
                    print(f"  ✓ متد {method} موجود است")
                else:
                    print(f"  ✗ متد {method} یافت نشد")
            
            return True
        else:
            print("❌ کلاس MtfScalper_RL_Hybrid یافت نشد")
            return False
            
    except Exception as e:
        print(f"❌ خطا در import استراتژی: {e}")
        return False

def test_rl_model_syntax():
    """Test if RL model can be imported"""
    print_header("🤖 تست مدل RL")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "MtfScalperRLModel",
            "MtfScalperRLModel.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'MtfScalperRLModel'):
            print("✅ مدل RL بدون خطا import شد")
            
            # Check custom environment
            if hasattr(module.MtfScalperRLModel, 'MtfScalperRLEnv'):
                print("  ✓ محیط سفارشی RL تعریف شده")
            
            return True
        else:
            print("❌ کلاس MtfScalperRLModel یافت نشد")
            return False
            
    except Exception as e:
        print(f"❌ خطا در import مدل RL: {e}")
        return False

def validate_config():
    """Validate configuration file"""
    print_header("⚙️ بررسی کانفیگ")
    
    try:
        with open('config_rl_hybrid.json', 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ['exchange', 'freqai', 'stake_currency']
        for section in required_sections:
            if section in config:
                print(f"✅ بخش {section} موجود است")
            else:
                print(f"❌ بخش {section} یافت نشد")
                return False
        
        # Check FreqAI configuration
        if 'freqai' in config:
            freqai_config = config['freqai']
            if freqai_config.get('enabled'):
                print("✅ FreqAI فعال است")
                
                # Check RL config
                if 'rl_config' in freqai_config:
                    rl_config = freqai_config['rl_config']
                    print(f"  • مدل: {rl_config.get('model_type', 'N/A')}")
                    print(f"  • چرخه‌های آموزش: {rl_config.get('train_cycles', 'N/A')}")
                    print(f"  • حداکثر مدت معامله: {rl_config.get('max_trade_duration_candles', 'N/A')} کندل")
            else:
                print("⚠️ FreqAI غیرفعال است")
        
        return True
        
    except Exception as e:
        print(f"❌ خطا در خواندن کانفیگ: {e}")
        return False

def generate_sample_data():
    """Generate sample data for testing"""
    print_header("📊 تولید داده‌های نمونه")
    
    try:
        # Generate sample OHLCV data
        dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='5T')
        
        data = {
            'date': dates,
            'open': np.random.uniform(40000, 42000, len(dates)),
            'high': np.random.uniform(41000, 43000, len(dates)),
            'low': np.random.uniform(39000, 41000, len(dates)),
            'close': np.random.uniform(40000, 42000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        }
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        # Ensure high > low
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        print(f"✅ تولید {len(df)} کندل داده نمونه")
        print(f"  • بازه زمانی: {df.index[0]} تا {df.index[-1]}")
        print(f"  • میانگین قیمت: ${df['close'].mean():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ خطا در تولید داده: {e}")
        return None

def test_indicator_calculation(df):
    """Test indicator calculation on sample data"""
    print_header("📈 تست محاسبه اندیکاتورها")
    
    try:
        import talib.abstract as ta
        
        # Calculate basic indicators
        df['ema_9'] = ta.EMA(df, timeperiod=9)
        df['ema_21'] = ta.EMA(df, timeperiod=21)
        df['rsi'] = ta.RSI(df, timeperiod=14)
        df['atr'] = ta.ATR(df, timeperiod=14)
        
        print("✅ اندیکاتورهای پایه محاسبه شدند:")
        print(f"  • EMA 9: {df['ema_9'].iloc[-1]:.2f}")
        print(f"  • EMA 21: {df['ema_21'].iloc[-1]:.2f}")
        print(f"  • RSI: {df['rsi'].iloc[-1]:.2f}")
        print(f"  • ATR: {df['atr'].iloc[-1]:.2f}")
        
        # Test exit features
        df['momentum_5'] = df['close'].pct_change(5)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
        
        print("\n✅ ویژگی‌های خروج محاسبه شدند:")
        print(f"  • Momentum 5: {df['momentum_5'].iloc[-1]:.4f}")
        print(f"  • Volume Ratio: {df['volume_ratio'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ خطا در محاسبه اندیکاتورها: {e}")
        return False

def create_summary_report():
    """Create a summary report of all tests"""
    print_header("📋 گزارش نهایی")
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'environment_check': check_environment(),
        'files_exist': validate_strategy_files(),
        'strategy_syntax': test_strategy_syntax(),
        'rl_model_syntax': test_rl_model_syntax(),
        'config_valid': validate_config()
    }
    
    # Generate and test sample data
    df = generate_sample_data()
    if df is not None:
        results['indicator_test'] = test_indicator_calculation(df)
    else:
        results['indicator_test'] = False
    
    # Calculate overall status
    all_passed = all(results.values() if k != 'timestamp' else True for k in results)
    
    print("\n" + "="*60)
    print("  خلاصه نتایج:")
    print("="*60)
    
    for key, value in results.items():
        if key == 'timestamp':
            continue
        status = "✅" if value else "❌"
        print(f"{status} {key.replace('_', ' ').title()}")
    
    print("\n" + "="*60)
    if all_passed:
        print("  🎉 تمام تست‌ها با موفقیت انجام شد!")
        print("  استراتژی آماده استفاده است.")
    else:
        print("  ⚠️ برخی تست‌ها ناموفق بودند.")
        print("  لطفاً مشکلات را برطرف کنید.")
    print("="*60)
    
    # Save report
    with open('test_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n📄 گزارش در فایل test_report.json ذخیره شد")
    
    return all_passed

def main():
    """Main function"""
    print("\n🚀 شروع تست MtfScalper RL Hybrid Strategy")
    print("Version: 1.0.0")
    print("Date:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # Run all tests and create report
    success = create_summary_report()
    
    if success:
        print("\n✨ مراحل بعدی:")
        print("1. دانلود داده‌ها: freqtrade download-data --pairs BTC/USDT --days 540")
        print("2. آموزش مدل: freqtrade trade --config config_rl_hybrid.json")
        print("3. بک‌تست: freqtrade backtesting --config config_rl_hybrid.json")
    else:
        print("\n❗ لطفاً ابتدا مشکلات را حل کنید")
        print("راهنما: cat setup_guide.md")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
