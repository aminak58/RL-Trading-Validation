# Freqtrade RL Trading Strategy

## 🎯 استراتژی معاملاتی هوش مصنوعی ترکیبی

این ریپازیتوری شامل استراتژی معاملاتی `MtfScalper_RL_Hybrid` است که:
- **ورود کلاسیک**: سیگنال‌های تکنیکال چند زمانی (5m, 15m, 1h)
- **خروج RL**: تصمیمات هوشمند با Reinforcement Learning
- **فutures trading**: معاملات اهرمی با 3x leverage

## 📦 محتویات ریپازیتوری

```
RL-Trading-Validation/
├── README.md                     (این فایل)
├── requirements.txt              (وابستگی‌های نسخه سازگار)
├── user_data/
│   ├── strategies/
│   │   └── MtfScalper_RL_Hybrid.py    (استراتژی اصلی)
│   ├── freqaimodels/
│   │   └── MtfScalperRLModel.py        (مدل RL)
│   ├── configs/
│   │   └── config_rl_hybrid.json       (کانفیگ اصلی)
│   ├── data/                    (داده‌های آماده 540 روزه)
│   │   └── binance/
│   │       ├── BTC_USDT_USDT-5m-futures.json
│   │       ├── ETH_USDT_USDT-5m-futures.json
│   │       ├── SOL_USDT_USDT-5m-futures.json
│   │       └── DOGE_USDT_USDT-5m-futures.json
│   └── notebooks/
│       └── feature_analysis.ipynb     (تحلیل ویژگی‌ها)
├── user_data/models/            (مدل‌های آموزش دیده با Git LFS)
└── colab_setup.ipynb           (راه‌اندازی کولب)
```

## 🚀 اجرا در Google Colab

### مرحله ۱: کلون ریپازیتوری
```python
!git clone https://github.com/yourusername/RL-Trading-Validation.git
%cd RL-Trading-Validation
```

### مرحله ۲: نصب وابستگی‌ها
```python
!pip install -r requirements.txt
```

### مرحله ۳: آموزش مدل RL
```python
!freqtrade backtesting --config user_data/configs/config_rl_hybrid.json --strategy MtfScalper_RL_Hybrid --freqaimodel MtfScalperRLModel --timeframe 5m --timerange 20240101-20240201
```

### مرحله ۴: اجرای بک‌تست
```python
!freqtrade backtesting --config user_data/configs/config_rl_hybrid.json --strategy MtfScalper_RL_Hybrid --freqaimodel MtfScalperRLModel --timeframe 5m --timerange 20240301-20240401 --breakdown day
```

## ✨ ویژگی‌های خاص

### 🧠 مدل هوشمند RL
- **Reward Function سفارشی**: ترکیب سود، ریسک، و زمانبندی
- **5 Action Space**: Hold, Long, Short, Exit Long, Exit Short
- **PPO Algorithm**: آموزش پایدار و بهینه

### 📊 ویژگی‌های تکنیکال
- **Multi-Timeframe**: تحلیل همزمان 5m, 15m, 1h
- **50+ Features**: RSI, MACD, ADX, EMA, Volume, و...
- **Safe Entry Constraints**: محدودیت‌های ورود امن برای RL

### 🔧 API Compatibility
- **نسخه‌های پین شده**: هیچ تداخلی API
- **Tested Environment**: کاملاً تست شده
- **Colab Ready**: آماده برای اجرا در کولب

## 📈 نتایج مورد انتظار

- **Episode Rewards**: بهبود از 0.468 به 1.39+
- **Training Time**: ~20-30 دقیقه در کولب
- **Backtest Coverage**: 540 روز داده
- **Pairs**: BTC, ETH, SOL, DOGE futures

## ⚠️ نکات مهم

1. **کولب بدون نیاز به اینترنت**: همه داده‌ها از قبل در ریپازیتوری
2. **IP محدودیت حل شده**: هیچ دانلودی از بایننس نیاز نیست
3. **مدل‌های آماده**: با Git LFS در دسترس
4. **نسخه سازگار**: مشکلات API حل شده

## 🛠️ توسعه و شخصی‌سازی

### اضافه کردن جفت‌ارز جدید:
1. داده‌ها را به `user_data/data/binance/` اضافه کنید
2. کانفیگ را آپدیت کنید
3. دوباره آموزش دهید

### تغییر پارامترهای RL:
1. `MtfScalperRLModel.py` را ویرایش کنید
2. Reward function را سفارشی کنید
3. مجدداً آموزش دهید

## 📞 پشتیبانی

این پروژه برای اجرای کامل در Google Colab طراحی شده. اگر مشکلی داشتید:
1. مطمئن شوید GPU در کولب فعال است
2. نسخه‌های requirements.txt نصب شده باشند
3. داده‌ها در مسیر صحیح قرار داشته باشند

## 📄 مجوز

این پروژه برای تحقیق و آموزش طراحی شده. مسئولیت استفاده تجدیدی بر عهده کاربر است.

---

**طراحی شده برای محدودیت‌های IP بایننس در کولب** ✅