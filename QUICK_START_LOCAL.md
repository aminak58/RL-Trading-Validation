# شروع سریع - اجرای محلی روی All-in-One 2017

## 🎯 خلاصه توصیه‌ها

### سیستم شما:
- **CPU**: Intel i5 نسل 7
- **RAM**: 32GB ✅
- **GPU**: 4GB VRAM
- **Storage**: 512GB SSD

### مدل‌های پیشنهادی:

#### 1️⃣ برای RL Trading (پروژه فعلی) ✅

**وضعیت**: پروژه شما کاملاً روی سیستم شما اجرا می‌شود!

```bash
# تست سریع
python test_local_setup.py
```

**تنظیمات بهینه**:
- ✅ استفاده از **CPU** به جای GPU (سریع‌تر برای PPO)
- ✅ کانفیگ بهینه: `configs/config_local_optimized.json`
- ✅ معماری شبکه: [256, 256, 128] - مناسب

#### 2️⃣ برای LLM (اختیاری - تحلیل احساسات/اخبار)

| مدل | حجم | VRAM | سرعت | توصیه |
|-----|------|------|------|-------|
| **Gemma 2B (Q4)** | 1.6GB | 2GB | 15-20 t/s | ⭐⭐⭐⭐⭐ |
| **Phi-3 Mini** | 2.3GB | 2.5GB | 12-18 t/s | ⭐⭐⭐⭐ |
| **Llama 3.2 3B** | 2GB | 2GB | 10-15 t/s | ⭐⭐⭐ |
| **TinyLlama 1.1B** | <1GB | 1GB | 25+ t/s | ⭐⭐⭐ |

---

## 🚀 راه‌اندازی در 3 مرحله

### مرحله 1: تست سیستم

```bash
# اجرای تست
python test_local_setup.py
```

این اسکریپت بررسی می‌کند:
- ✓ Python و PyTorch
- ✓ Stable Baselines3
- ✓ Freqtrade
- ✓ منابع سیستم
- ✓ تست آموزش RL

### مرحله 2: استفاده از کانفیگ بهینه

```bash
# کپی کانفیگ بهینه
cp configs/config_local_optimized.json configs/config.json

# یا استفاده مستقیم
freqtrade backtesting \
    --config configs/config_local_optimized.json \
    --strategy MtfScalper_RL_Hybrid \
    --freqaimodel MtfScalperRLModel
```

### مرحله 3: نصب LLM (اختیاری)

```bash
# نصب Ollama
curl -fsSL https://ollama.com/install.sh | sh

# دانلود Gemma 2B
ollama pull gemma2:2b

# تست
ollama run gemma2:2b
```

---

## 📊 تغییرات کلیدی در کانفیگ بهینه

```json
{
  "rl_config": {
    "device": "cpu",           // ← تغییر از "auto" به "cpu"
    "cpu_count": 4,            // ← تغییر از 8 به 4
    "train_cycles": 25,        // ← کاهش از 30 به 25
    "n_steps": 1024,           // ← کاهش از 2048 به 1024
    "batch_size": 64           // ← بهینه برای CPU
  }
}
```

**چرا این تغییرات؟**
- CPU سریع‌تر از GPU 4GB برای شبکه‌های کوچک PPO
- i5-7th gen دارای 4 کور فیزیکی
- کاهش حافظه مورد نیاز
- سرعت آموزش بهتر

---

## 💡 استفاده همزمان RL + LLM

```python
# مثال: ترکیب RL Trading با تحلیل احساسات

# 1. RL Training روی CPU
import torch
model = PPO("MlpPolicy", env, device="cpu")

# 2. LLM Inference روی GPU (همزمان)
# در ترمینال دیگر:
# ollama run gemma2:2b

# 3. Embedding برای Feature Engineering
from sentence_transformers import SentenceTransformer
sentiment = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB
```

این setup بهترین استفاده از منابع را می‌کند:
- **CPU**: RL Training
- **GPU**: LLM Inference
- **RAM**: Data Processing

---

## 🔧 عیب‌یابی

### مشکل: آموزش RL خیلی کند است

**راه‌حل**:
```bash
# 1. مطمئن شوید از CPU استفاده می‌کنید
grep '"device"' configs/config_local_optimized.json
# باید "cpu" باشد نه "auto" یا "cuda"

# 2. کاهش تعداد cycles
# train_cycles: 25 → 20

# 3. کاهش اندازه شبکه
# net_arch: [256, 256, 128] → [128, 128, 64]
```

### مشکل: LLM OOM (Out of Memory) روی GPU

**راه‌حل**:
```bash
# 1. استفاده از مدل کوچک‌تر
ollama run tinyllama  # به جای gemma2:2b

# 2. یا اجرا روی CPU
CUDA_VISIBLE_DEVICES="" ollama run gemma2:2b
```

### مشکل: RAM کافی نیست

**راه‌حل**:
```bash
# کاهش تعداد pair‌ها در whitelist
# pair_whitelist: ["BTC/USDT:USDT", "ETH/USDT:USDT"]  # فقط 2 جفت
```

---

## 📈 عملکرد مورد انتظار

### RL Training:
- **سرعت**: ~15-20 دقیقه برای 25 cycles
- **RAM**: 4-6GB استفاده
- **CPU**: 80-100% استفاده

### LLM Inference:
- **Gemma 2B**: 15-20 tokens/sec
- **Phi-3 Mini**: 12-18 tokens/sec
- **GPU**: 2-3GB VRAM استفاده

---

## 📚 فایل‌های مهم

1. **LOCAL_EXECUTION_GUIDE.md** - راهنمای کامل فارسی
2. **configs/config_local_optimized.json** - کانفیگ بهینه
3. **test_local_setup.py** - اسکریپت تست سیستم
4. **این فایل** - شروع سریع

---

## ✅ چک‌لیست آمادگی

- [ ] Python 3.8+ نصب شده
- [ ] PyTorch نصب شده
- [ ] Stable Baselines3 نصب شده
- [ ] Freqtrade نصب شده
- [ ] test_local_setup.py اجرا شده و موفق بوده
- [ ] config_local_optimized.json بررسی شده
- [ ] (اختیاری) Ollama نصب شده

---

## 🎯 نتیجه‌گیری

### ✅ مدل‌های مناسب برای سیستم شما:

**برای RL Trading:**
- ✅ PPO با [256, 256, 128] روی CPU
- ✅ PPO با [128, 128, 64] روی CPU (سریع‌تر)
- ❌ مدل‌های بزرگ‌تر (نیاز به GPU قوی‌تر)

**برای LLM/NLP:**
- ✅ Gemma 2B (Q4) - بهترین
- ✅ Phi-3 Mini (Q4)
- ✅ Llama 3.2 3B (Q4)
- ✅ TinyLlama 1.1B
- ❌ مدل‌های 7B+ (نیاز به 8GB+ VRAM)

**برای Embeddings:**
- ✅ all-MiniLM-L6-v2 (80MB)
- ✅ paraphrase-MiniLM-L3-v2 (60MB)
- ✅ distilbert-base-uncased (260MB)

---

## 📞 منابع اضافی

- **Ollama**: https://ollama.com
- **Stable Baselines3**: https://stable-baselines3.readthedocs.io
- **Freqtrade**: https://www.freqtrade.io
- **Sentence Transformers**: https://www.sbert.net

---

**موفق باشید! 🚀**
