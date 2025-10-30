# Freqtrade RL Trading - Docker Container

این داکر کانتینر شامل تمام ابزارهای لازم برای اجرای استراتژی معاملاتی RL ترکیبی شماست، از جمله داده‌های معاملاتی و تمام وابستگی‌های مورد نیاز برای اجرا در Google Colab.

## 🎯 ویژگی‌ها

- ✅ **محیط کامل و ایزوله**: شامل Freqtrade، PyTorch، Stable-Baselines3
- ✅ **داده‌های آماده**: شامل 540 روز داده futures برای BTC، ETH، SOL، DOGE
- ✅ **API سازگار**: رفع مشکلات عدم تطابق Gym/SB3
- ✅ **مناسب برای Colab**: بدون وابستگی خارجی
- ✅ **نسخه‌های پین شده**: ثبات و سازگاری تضمین شده

## 📦 فایل‌های اصلی

- `Dockerfile` - تعریف کانتینر داکر
- `docker-compose.yml` - تنظیمات داکر کامپوز
- `build_and_run.sh` - اسکریپت ساخت و اجرا
- `requirements.txt` - وابستگی‌های پایتون
- `user_data/` - استراتژی‌ها، مدل‌ها، داده‌ها

## 🚀 نحوه استفاده

### 1. ساخت کانتینر
```bash
./build_and_run.sh build
```

### 2. دانلود داده‌ها
```bash
./build_and_run.sh download
```

### 3. آموزش مدل RL
```bash
./build_and_run.sh train
```

### 4. اجرای بک‌تست
```bash
./build_and_run.sh backtest
```

### 5. اجرا در Google Colab

برای اجرا در Colab:

```python
# در Colab اجرا کنید
!git clone <your-repo>
%cd RL-Trading-Validation

# ساخت و اجرای کانتینر
!docker build -t freqtrade-rl .
!docker run -it --rm freqtrade-rl train
```

## 🔧 دستورات مستقیم داکر

```bash
# ساخت کانتینر
docker build -t freqtrade-rl .

# اجرای دستوری
docker run --rm freqtrade-rl train
docker run --rm freqtrade-rl backtest

# شل تعاملی
docker run -it --rm freqtrade-rl shell

# جوپیتر نوت‌بوک
docker run -p 8888:8888 --rm freqtrade-rl notebook
```

## 📊 ساختار داده‌ها

```
user_data/
├── data/
│   └── binance/
│       ├── BTC_USDT_USDT-5m.json
│       ├── ETH_USDT_USDT-5m.json
│       └── ... (540 روز داده)
├── models/
│   └── MtfScalperRL_v1/  (مدل‌های آموزش دیده)
├── strategies/
│   └── MtfScalper_RL_Hybrid.py
└── freqaimodels/
    └── MtfScalperRLModel.py
```

## 🐛 حل مشکلات API

این کانتینر مشکلات زیر را حل می‌کند:

1. **ValueError: predict() tuple vs numpy array** - با نسخه‌های سازگار SB3 و Gymnasium
2. **AttributeError: 'RangeIndex' object has no attribute 'dayofweek'** - با pandas و numpy پین شده
3. **Background process tracking issues** - با محیط ایزوله داکر

## 🔍 اطلاعات نسخه‌ها

- **Python**: 3.10-slim
- **Freqtrade**: آخرین نسخه پایدار
- **PyTorch**: 2.0.0+
- **Stable-Baselines3**: 2.2.0+
- **Gymnasium**: 0.28.0+
- **Pandas**: 2.0.3+
- **NumPy**: 1.26.0+

## 💾 پاکسازی

```bash
# توقف کانتینرها
./build_and_run.sh stop

# پاکسازی کامل
./build_and_run.sh clean
```

## 🌐 دسترسی به نتایج

- **Jupyter Notebook**: http://localhost:8888
- **مدل‌های آموزش دیده**: در `user_data/models/`
- **نتایج بک‌تست**: در لاگ‌های خروجی

## 📞 پشتیبانی

این کانتینر تمام مشکلات محیطی و API که تجربه کردید را حل می‌کند. برای هرگونه مشکل، لاگ‌های داکر را بررسی کنید:

```bash
docker-compose logs -f
```