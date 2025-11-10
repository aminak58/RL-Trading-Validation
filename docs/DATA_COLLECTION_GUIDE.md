# Data Collection & Analysis Guide

این راهنمای جامع نحوه استفاده از سیستم data collection و analysis را برای بهبود RL trading strategy توضیح می‌دهد.

---

## 📋 فهرست مطالب

1. [نصب و راه‌اندازی](#setup)
2. [اجرای Backtest با Data Collection](#running-backtest)
3. [تحلیل داده‌های جمع‌آوری شده](#analyzing-data)
4. [استفاده از Skills برای تحلیل عمیق](#using-skills)
5. [تشخیص و رفع مشکلات](#troubleshooting)
6. [بهبود مدل بر اساس نتایج](#model-improvement)

---

## 🚀 نصب و راه‌اندازی {#setup}

### پیش‌نیازها

کدهای جدید به پکیج‌های زیر نیاز دارند:

```bash
pip install tensorboard matplotlib pandas numpy
```

### بررسی نصب

بررسی کنید که همه فایل‌های جدید موجود هستند:

```bash
ls -la user_data/data_collector.py
ls -la scripts/run_backtest_with_analysis.py
ls -la .claude/skills/freqai-rl-optimizer/scripts/
```

---

## 🎯 اجرای Backtest با Data Collection {#running-backtest}

### روش 1: استفاده از اسکریپت خودکار (پیشنهادی)

این روش تمام مراحل را به صورت خودکار انجام می‌دهد:

```bash
python scripts/run_backtest_with_analysis.py \
    --timerange 20240101-20240401 \
    --strategy MtfScalper_RL_Hybrid \
    --config configs/config_rl_hybrid.json
```

**این اسکریپت به ترتیب:**
1. ✅ Backtest را اجرا می‌کند
2. ✅ داده‌ها را جمع‌آوری می‌کند (trades, predictions, episodes, rewards)
3. ✅ TensorBoard logs را تحلیل می‌کند
4. ✅ گزارش جامع تولید می‌کند
5. ✅ پیشنهادات بهبود ارائه می‌دهد

### روش 2: اجرای دستی

اگر می‌خواهید خودتان کنترل کامل داشته باشید:

```bash
# 1. اجرای backtest
freqtrade backtesting \
    --config configs/config_rl_hybrid.json \
    --strategy MtfScalper_RL_Hybrid \
    --freqaimodel MtfScalperRLModel \
    --timerange 20240101-20240401 \
    --freqai-train-enabled

# 2. تحلیل داده‌های جمع‌آوری شده
python -c "from user_data.data_collector import analyze_session; analyze_session()"

# 3. تحلیل TensorBoard logs
python .claude/skills/freqai-rl-optimizer/scripts/analyze_training.py \
    --tensorboard-dir ./tensorboard/ \
    --output-dir ./analysis/
```

### گزینه‌های مفید

```bash
# بدون training (استفاده از مدل موجود)
python scripts/run_backtest_with_analysis.py \
    --timerange 20240401-20240701 \
    --skip-training

# با breakdown روزانه
python scripts/run_backtest_with_analysis.py \
    --timerange 20240101-20240401 \
    --breakdown day

# فقط backtest (بدون analysis)
python scripts/run_backtest_with_analysis.py \
    --timerange 20240101-20240401 \
    --skip-analysis
```

---

## 📊 تحلیل داده‌های جمع‌آوری شده {#analyzing-data}

### ساختار داده‌ها

بعد از اجرای backtest، داده‌ها در `user_data/analysis_data/` ذخیره می‌شوند:

```
user_data/analysis_data/
├── trades_20241110_153045.csv          # اطلاعات کامل هر معامله
├── trades_20241110_153045.json         # همان داده به فرمت JSON
├── predictions_20241110_153045.csv     # پیش‌بینی‌های RL در هر candle
├── predictions_20241110_153045.json
├── rl_episodes_20241110_153045.json    # اطلاعات episode های training
├── reward_breakdown_20241110_153045.json # جزئیات محاسبه reward
└── summary_20241110_153045.json        # خلاصه آمار
```

### مشاهده سریع نتایج

```bash
# نمایش خلاصه آخرین session
python -c "from user_data.data_collector import analyze_session; analyze_session()"

# لیست تمام session ها
python -c "from user_data.data_collector import DataCollector; print(DataCollector.list_available_sessions())"

# تحلیل یک session خاص
python -c "from user_data.data_collector import analyze_session; analyze_session('20241110_153045')"
```

### تحلیل در Python/Jupyter

```python
import pandas as pd
from user_data.data_collector import DataCollector

# Load داده‌ها
session_id = "20241110_153045"  # آخرین session شما
trades = DataCollector.load_trades(session_id)
predictions = DataCollector.load_predictions(session_id)

# تحلیل trades
print(f"Total trades: {len(trades)}")
print(f"Win rate: {(trades['profit_pct'] > 0).mean():.2%}")
print(f"Avg profit: {trades['profit_pct'].mean():.2%}")

# تحلیل winning vs losing trades
winners = trades[trades['profit_pct'] > 0]
losers = trades[trades['profit_pct'] < 0]

print(f"\nAvg winner: {winners['profit_pct'].mean():.2%}")
print(f"Avg loser: {losers['profit_pct'].mean():.2%}")
print(f"Risk/Reward: {abs(winners['profit_pct'].mean() / losers['profit_pct'].mean()):.2f}")

# بررسی exit reasons
print("\nExit reasons:")
print(trades['exit_reason'].value_counts())

# Duration analysis
print(f"\nAvg duration (winners): {winners['duration_candles'].mean():.0f} candles")
print(f"Avg duration (losers): {losers['duration_candles'].mean():.0f} candles")
```

---

## 🔬 استفاده از Skills برای تحلیل عمیق {#using-skills}

### 1. Training Analysis (TensorBoard)

```bash
# تحلیل metrics آموزش
python .claude/skills/freqai-rl-optimizer/scripts/analyze_training.py \
    --tensorboard-dir ./tensorboard/ \
    --output-dir ./analysis/training/

# نمایش TensorBoard
tensorboard --logdir ./tensorboard/
```

**چه چیزهایی را بررسی کنید:**
- ✅ Episode rewards در حال افزایش است؟
- ✅ Loss در حال کاهش است؟
- ✅ Entropy به صفر crash نکرده؟
- ✅ Explained variance > 0.5 است؟

### 2. Feature Importance Analysis

```bash
python .claude/skills/freqai-rl-optimizer/scripts/feature_importance.py \
    --model-dir user_data/models/MtfScalperRL_v2/ \
    --pair BTC/USDT:USDT
```

**استفاده از نتایج:**
- فیچرهایی با importance < 0.01 را حذف کنید
- روی top 20 فیچر تمرکز کنید
- Feature engineering جدید بر اساس فیچرهای مهم

### 3. Reward Breakdown Analysis

```bash
python .claude/skills/freqai-rl-optimizer/scripts/reward_backtest.py \
    --session-id 20241110_153045 \
    --output-dir ./analysis/rewards/
```

**سوالات کلیدی:**
- کدام reward component بیشترین تأثیر را دارد؟
- آیا reward components با هم conflict دارند؟
- آیا مقادیر reward در range معقول هستند؟

### 4. Hyperparameter Optimization

```bash
# اسکن فضای reward weights
python .claude/skills/freqai-rl-optimizer/scripts/hyperparameter_scanner.py \
    --strategy MtfScalper_RL_Hybrid \
    --params reward_weights.profit,reward_weights.drawdown \
    --ranges 0.3:0.5,0.2:0.4 \
    --trials 10
```

---

## 🔍 تشخیص و رفع مشکلات {#troubleshooting}

### مشکل 1: Win Rate بالا اما Profit منفی

**علامت:**
```
Win Rate: 55%
Profit: -67$
```

**تشخیص:**
- Winners خیلی کوچک
- Losers خیلی بزرگ
- Risk/Reward ratio بد

**راه حل:**

1. **بررسی Average Winner vs Loser:**
```python
trades = DataCollector.load_trades(session_id)
print(f"Avg winner: {trades[trades['profit_pct'] > 0]['profit_pct'].mean():.2%}")
print(f"Avg loser: {trades[trades['profit_pct'] < 0]['profit_pct'].mean():.2%}")
```

2. **تنظیم Reward Weights:**
```python
# در MtfScalperRLModel.py
reward_weights = {
    "profit": 0.45,          # افزایش از 0.35
    "drawdown_control": 0.20, # کاهش از 0.25
    "timing_quality": 0.25,   # افزایش از 0.20
    "risk_reward_ratio": 0.10, # کاهش از 0.20
}
```

3. **اضافه کردن Profit Protection:**
```python
# در strategy custom_exit()
if current_profit > 0.01:  # 1% سود
    if current_profit < 0.003:  # افت به 0.3%
        return "profit_protection"
```

### مشکل 2: RL Model یاد نمی‌گیرد

**علامت:**
```
Avg Episode Reward: -2.5
Episode rewards not improving
```

**تشخیص:**
- Reward function خیلی پیچیده
- Entry penalties خیلی بالا
- Feature mismatch

**راه حل:**

1. **ساده‌سازی Reward:**
```python
# فقط روی profit تمرکز کنید
reward_weights = {
    "profit": 1.0,
    "drawdown_control": 0.0,
    "timing_quality": 0.0,
    "risk_reward_ratio": 0.0,
}
```

2. **کاهش Entry Penalty:**
```python
# در MtfScalperRLModel
entry_penalty_multiplier = 5.0  # کاهش از 15.0
classic_signal_reward = 5.0     # افزایش از 2.0
```

3. **بررسی Features:**
```bash
python .claude/skills/freqai-rl-optimizer/scripts/feature_importance.py \
    --model-dir user_data/models/
```

### مشکل 3: Overfitting

**علامت:**
- Training performance عالی
- Validation/Test performance بد

**راه حل:**

1. **Walk-Forward Validation:**
```bash
# تست روی دوره‌های مختلف
for month in 01 02 03 04 05 06; do
    python scripts/run_backtest_with_analysis.py \
        --timerange 202404${month}-202405${month}
done
```

2. **Feature Reduction:**
```python
# حذف فیچرهای با importance پایین
# نگهداری فقط top 20 فیچر
```

3. **Regularization:**
```python
# در model creation
policy_kwargs = dict(
    net_arch=[128, 128],  # کوچکتر از [256, 256, 128]
    activation_fn=th.nn.ReLU,
    optimizer_kwargs=dict(
        weight_decay=1e-4  # افزایش از 1e-5
    )
)
```

---

## 🎯 بهبود مدل بر اساس نتایج {#model-improvement}

### فرآیند بهبود تکراری

```
1. Run Backtest
   ↓
2. Analyze Results
   ↓
3. Identify Issues
   ↓
4. Modify Model/Strategy
   ↓
5. Test Changes
   ↓
6. Compare Results
   ↓
7. Repeat
```

### Checklist بهبود

#### مرحله 1: تحلیل اولیه (1 ساعت)
- [ ] Run backtest با data collection
- [ ] بررسی summary report
- [ ] شناسایی مشکل اصلی (Win rate, Profit factor, R:R)
- [ ] بررسی TensorBoard metrics

#### مرحله 2: تشخیص ریشه مشکل (1-2 ساعت)
- [ ] تحلیل trade data (winners vs losers)
- [ ] بررسی exit reasons
- [ ] تحلیل reward components
- [ ] بررسی feature importance

#### مرحله 3: تغییرات (2-3 ساعت)
- [ ] تنظیم reward weights
- [ ] ساده‌سازی features
- [ ] اصلاح exit logic
- [ ] تست تک‌تک تغییرات

#### مرحله 4: Validation (2-4 ساعت)
- [ ] Walk-forward validation
- [ ] مقایسه با baseline
- [ ] تست در شرایط مختلف بازار
- [ ] بررسی stability

### Example: بهبود براساس نتایج شما

بر اساس نتایج شما (Win Rate 55%, Profit -67$):

```python
# ❌ تنظیمات فعلی
reward_weights = {
    "profit": 0.35,
    "drawdown_control": 0.25,
    "timing_quality": 0.20,
    "risk_reward_ratio": 0.20
}

# ✅ تنظیمات پیشنهادی
reward_weights = {
    "profit": 0.50,  # تمرکز بیشتر روی سود
    "drawdown_control": 0.20,
    "timing_quality": 0.30,  # خروج بهتر
    "risk_reward_ratio": 0.00,  # حذف موقت
}

# Entry/Exit adjustments
entry_penalty_multiplier = 10.0  # کاهش از 15.0
exit_profit_threshold = 0.01  # افزایش از 0.02
```

**تست این تغییرات:**
```bash
# 1. اعمال تغییرات در کد
# 2. اجرای backtest جدید
python scripts/run_backtest_with_analysis.py --timerange 20240101-20240401

# 3. مقایسه نتایج
python -c "
from user_data.data_collector import DataCollector
import json

# Load old and new results
sessions = DataCollector.list_available_sessions()
old_session = sessions[1]  # قبل از تغییرات
new_session = sessions[0]  # بعد از تغییرات

# Compare
for session_id in [old_session, new_session]:
    with open(f'user_data/analysis_data/summary_{session_id}.json') as f:
        data = json.load(f)
        stats = data['trade_stats']
        print(f'{session_id}: Win Rate={stats[\"win_rate\"]:.2%}, Profit Factor={stats[\"profit_factor\"]:.2f}')
"
```

---

## 📚 منابع اضافی

### Documentation
- [Freqtrade Docs](https://www.freqtrade.io/en/stable/)
- [FreqAI Docs](https://www.freqtrade.io/en/stable/freqai/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

### Skills
- `analyze_training.py` - Training metrics analysis
- `feature_importance.py` - SHAP-based feature analysis
- `reward_backtest.py` - Reward function testing
- `hyperparameter_scanner.py` - Grid search optimization

### اسکریپت‌های کمکی
```bash
# Quick analysis
python -c "from user_data.data_collector import analyze_session; analyze_session()"

# List sessions
python -c "from user_data.data_collector import DataCollector; print('\n'.join(DataCollector.list_available_sessions()))"

# TensorBoard
tensorboard --logdir ./tensorboard/ --port 6006
```

---

## 🎓 نکات نهایی

### Do's ✅
- همیشه قبل از تغییرات بزرگ، baseline backtest بگیرید
- تغییرات را به صورت تک‌تک تست کنید
- از walk-forward validation استفاده کنید
- داده‌های collected را منظم بررسی کنید
- نتایج را document کنید

### Don'ts ❌
- چند تغییر را همزمان اعمال نکنید
- فقط به یک metric نگاه نکنید (Win Rate, Profit, etc.)
- بدون validation تغییرات را deploy نکنید
- از overfitting غافل نشوید
- بدون data collection تصمیم نگیرید

---

**موفق باشید! 🚀**

برای سوالات یا مشکلات، issue در GitHub ایجاد کنید.
