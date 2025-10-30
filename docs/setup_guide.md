# راهنمای راه‌اندازی MtfScalper RL Hybrid Strategy

## 📋 Phase 1 تکمیل شد! 

### فایل‌های ایجاد شده:
1. **MtfScalper_RL_Hybrid.py** - استراتژی هیبریدی با ورود کلاسیک و خروج RL
2. **MtfScalperRLModel.py** - مدل RL سفارشی با reward function بهینه
3. **config_rl_hybrid.json** - کانفیگ Freqtrade برای futures trading
4. **feature_analysis.ipynb** - نوت‌بوک تحلیل features
5. **setup_guide.md** - این راهنما

## 🚀 راه‌اندازی سریع

### 1. نصب وابستگی‌ها
```bash
# فعال‌سازی محیط مجازی
source .venv/bin/activate

# نصب پکیج‌های مورد نیاز
pip install torch stable-baselines3 sb3-contrib tensorboard
```

### 2. کپی فایل‌ها به مسیر صحیح
```bash
# کپی استراتژی
cp MtfScalper_RL_Hybrid.py user_data/strategies/

# کپی مدل RL
cp MtfScalperRLModel.py user_data/freqaimodels/

# کپی کانفیگ
cp config_rl_hybrid.json user_data/
```

### 3. دانلود داده‌ها
```bash
# دانلود 18 ماه داده برای BTC/USDT
freqtrade download-data \
    --pairs BTC/USDT \
    --exchange binance \
    --timeframe 5m 15m 1h \
    --days 540 \
    --trading-mode futures \
    --data-format json
```

### 4. آموزش اولیه مدل RL
```bash
# آموزش مدل (Phase 2)
freqtrade trade \
    --config config_rl_hybrid.json \
    --strategy MtfScalper_RL_Hybrid \
    --freqaimodel MtfScalperRLModel \
    --db-url sqlite:///tradesv3_rl.sqlite
```

## 📊 بک‌تست

### بک‌تست استراتژی کلاسیک (بدون RL)
```bash
freqtrade backtesting \
    --config config_rl_hybrid.json \
    --strategy MtfScalper \
    --timeframe 5m \
    --timerange 20240101-20241030
```

### بک‌تست استراتژی هیبریدی (با RL)
```bash
freqtrade backtesting \
    --config config_rl_hybrid.json \
    --strategy MtfScalper_RL_Hybrid \
    --freqaimodel MtfScalperRLModel \
    --timeframe 5m \
    --timerange 20240101-20241030
```

## 🔧 تنظیمات کلیدی

### پارامترهای قابل تنظیم در استراتژی:

```python
# RL Exit Parameters
rl_exit_confidence = 0.7  # حداقل confidence برای خروج RL
max_position_duration = 48  # حداکثر مدت نگهداری (48 * 5min = 4 hours)
emergency_exit_profit = -0.03  # خروج اضطراری در -3% ضرر
breakeven_trigger = 0.02  # انتقال به breakeven در +2% سود
```

### تنظیمات Reward Function:

```python
reward_weights = {
    "profit": 0.35,         # وزن سود/زیان
    "drawdown_control": 0.25,  # کنترل drawdown
    "timing_quality": 0.20,    # کیفیت زمان‌بندی
    "risk_reward_ratio": 0.20  # نسبت ریسک به پاداش
}
```

## 📈 مانیتورینگ و آنالیز

### 1. TensorBoard برای مانیتورینگ آموزش
```bash
tensorboard --logdir ./tensorboard/
# باز کردن در مرورگر: http://localhost:6006
```

### 2. تحلیل Features با Notebook
```bash
jupyter notebook feature_analysis.ipynb
```

### 3. لاگ‌های استراتژی
```bash
tail -f user_data/logs/freqtrade.log | grep MtfScalper
```

## 🎯 چک‌لیست Phase 2

### هفته 1: آموزش و بهینه‌سازی
- [ ] آموزش اولیه مدل با 30 چرخه
- [ ] تحلیل feature importance
- [ ] تنظیم reward weights بر اساس نتایج
- [ ] آموزش مجدد با پارامترهای بهینه

### هفته 2: بک‌تست و اعتبارسنجی
- [ ] بک‌تست 3 سال گذشته
- [ ] Walk-forward analysis
- [ ] مقایسه با استراتژی کلاسیک
- [ ] بررسی overfitting

### معیارهای موفقیت:
- ✅ Sharpe Ratio > 1.5
- ✅ Profit Factor > 1.8
- ✅ Max Drawdown < 12%
- ✅ Win Rate > 55%

## 🛠️ Troubleshooting

### مشکل: "FreqAI model not found"
```bash
# بررسی مسیر مدل
ls user_data/freqaimodels/
# باید MtfScalperRLModel.py را ببینید
```

### مشکل: "Not enough data"
```bash
# دانلود داده‌های بیشتر
freqtrade download-data --days 600
```

### مشکل: "CUDA out of memory"
```python
# در MtfScalperRLModel.py:
self.use_cuda = False  # غیرفعال کردن GPU
```

## 📞 پشتیبانی و سوالات

برای سوالات فنی:
1. بررسی لاگ‌ها: `user_data/logs/freqtrade.log`
2. اجرای تست‌های واحد: `pytest user_data/strategies/test_*.py`
3. مستندات FreqAI: https://www.freqtrade.io/en/stable/freqai/

## 🔄 مراحل بعدی

### Phase 3: Production Ready
1. اضافه کردن market regime detection
2. پیاده‌سازی position sizing با RL
3. ساخت dashboard مانیتورینگ
4. تست در paper trading

### Phase 4: Advanced Features
1. Ensemble models (PPO + SAC + TD3)
2. Meta-learning برای adaptability
3. News sentiment integration
4. On-chain data features

---

**توجه:** این استراتژی در حال توسعه است. قبل از استفاده با سرمایه واقعی، حتماً:
- حداقل 2 هفته paper trading انجام دهید
- نتایج بک‌تست را به دقت بررسی کنید
- با مبالغ کوچک شروع کنید

✅ **Phase 1 با موفقیت تکمیل شد!**
