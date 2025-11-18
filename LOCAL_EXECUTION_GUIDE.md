# راهنمای اجرای محلی - سیستم All-in-One 2017

## مشخصات سیستم
- **CPU**: Intel i5 نسل 7
- **RAM**: 32GB
- **GPU**: 4GB VRAM
- **Storage**: 512GB SSD
- **Display**: 5K

---

## 🎯 بهینه‌سازی برای RL Trading (پروژه فعلی)

### 1. تنظیمات فعلی پروژه شما

پروژه شما از **PPO** با شبکه عصبی متوسط استفاده می‌کند:
```python
net_arch = [256, 256, 128]  # ~180K parameters
```

**خبر خوب**: این معماری روی 4GB GPU اجرا می‌شود! ✅

### 2. بهینه‌سازی‌های پیشنهادی

#### آپشن 1: استفاده از CPU (سریع‌تر برای PPO کوچک)
```json
"rl_config": {
    "device": "cpu",  // تغییر از "auto" به "cpu"
    "cpu_count": 4,   // تغییر از 8 به 4 (برای i5-7th gen)
}
```

**چرا CPU؟**
- PPO با شبکه کوچک روی CPU معمولاً سریع‌تر است
- 32GB RAM کافی است
- GPU 4GB برای مدل‌های NLP/LLM استفاده کنید

#### آپشن 2: کاهش اندازه شبکه (اگر مشکل دارید)
```python
# در MtfScalperRLModel.py
self.net_arch = [128, 128, 64]  # ~45K parameters
```

---

## 🤖 مدل‌های پیشنهادی برای LLM/NLP (استفاده از 4GB GPU)

اگر می‌خواهید از LLM برای تحلیل اخبار یا تحلیل احساسات بازار استفاده کنید:

### مدل‌های بهینه برای 4GB VRAM:

#### 1. **Gemma 2B (Q4_K_M)** - پیشنهاد اول ⭐
```bash
# نصب Ollama
curl -fsSL https://ollama.com/install.sh | sh

# دانلود و اجرا
ollama run gemma2:2b
```
- **حجم**: ~1.6GB
- **سرعت**: 15-20 tokens/sec
- **مناسب برای**: تحلیل احساسات، خلاصه‌سازی اخبار

#### 2. **Phi-3 Mini (3.8B - Q4)**
```bash
ollama run phi3:mini
```
- **حجم**: ~2.3GB
- **سرعت**: 12-18 tokens/sec
- **مناسب برای**: استخراج ویژگی از متن، Q&A

#### 3. **Llama 3.2 3B (Q4)**
```bash
ollama run llama3.2:3b
```
- **حجم**: ~2GB
- **سرعت**: 10-15 tokens/sec
- **مناسب برای**: تحلیل عمومی، chatbot

#### 4. **TinyLlama 1.1B**
```bash
ollama run tinyllama
```
- **حجم**: <1GB
- **سرعت**: 25+ tokens/sec
- **مناسب برای**: وظایف ساده و سریع

---

## 📊 مدل‌های Embedding برای تحلیل احساسات

برای تحلیل احساسات توییتر/اخبار بازار:

### مدل‌های کوچک و کارآمد:

1. **all-MiniLM-L6-v2** (حجم: ~80MB)
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

2. **paraphrase-MiniLM-L3-v2** (حجم: ~60MB)
```python
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
```

3. **distilbert-base-uncased** (حجم: ~260MB)
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('distilbert-base-uncased')
```

---

## 🔧 نصب و راه‌اندازی

### برای RL Trading (پروژه فعلی):

```bash
# 1. آپدیت config
cd /home/user/RL-Trading-Validation
nano configs/config_rl_hybrid.json
# تغییر "device": "cpu" و "cpu_count": 4

# 2. تست اجرا
freqtrade backtesting \
    --config configs/config_rl_hybrid.json \
    --strategy MtfScalper_RL_Hybrid \
    --freqaimodel MtfScalperRLModel \
    --timeframe 5m \
    --timerange 20240101-20240201
```

### برای LLM (اختیاری):

```bash
# نصب Ollama
curl -fsSL https://ollama.com/install.sh | sh

# دانلود مدل‌ها
ollama pull gemma2:2b
ollama pull phi3:mini

# تست
ollama run gemma2:2b
```

---

## 💡 پیشنهاد: ترکیب RL + LLM

می‌توانید از LLM برای افزودن سیگنال‌های احساسی به استراتژی استفاده کنید:

```python
# مثال: تحلیل احساسات اخبار
from sentence_transformers import SentenceTransformer
sentiment_model = SentenceTransformer('all-MiniLM-L6-v2')

def analyze_news(text):
    embedding = sentiment_model.encode(text)
    # استفاده از embedding به عنوان feature در RL
    return embedding
```

---

## 📈 مقایسه عملکرد

| مدل | حجم | VRAM | RAM | سرعت | مناسب برای |
|-----|-----|------|-----|------|-----------|
| **PPO (فعلی)** | ~1GB | 500MB | 2GB | سریع | RL Trading ✅ |
| **Gemma 2B** | 1.6GB | 2GB | 4GB | متوسط | تحلیل احساسات ✅ |
| **Phi-3 Mini** | 2.3GB | 2.5GB | 6GB | متوسط | استخراج ویژگی ✅ |
| **Llama 3.2 3B** | 2GB | 2GB | 8GB | کند | تحلیل عمومی ✅ |
| **TinyLlama** | <1GB | 1GB | 2GB | خیلی سریع | وظایف ساده ✅ |

---

## ⚠️ نکات مهم

### برای RL Training:
1. **CPU بهتر از GPU است** برای PPO با شبکه کوچک
2. **32GB RAM کافی است** برای همه چیز
3. **استفاده همزمان**: RL روی CPU + LLM روی GPU ✅

### برای LLM:
1. **فقط Q4 quantization** استفاده کنید
2. **حداکثر 3B parameters** برای 4GB GPU
3. **Ollama** ساده‌ترین راه است

### محدودیت‌ها:
- ❌ مدل‌های 7B+ (نیاز به 8GB+ VRAM)
- ❌ Fine-tuning مدل‌های بزرگ (نیاز به 16GB+ VRAM)
- ✅ Inference همه مدل‌های 3B- با Q4
- ✅ RL Training با شبکه‌های متوسط

---

## 🚀 توصیه نهایی برای پروژه شما

### Setup پیشنهادی:

1. **RL Training روی CPU**
   ```json
   "device": "cpu",
   "cpu_count": 4
   ```

2. **LLM برای تحلیل احساسات روی GPU**
   ```bash
   ollama run gemma2:2b
   ```

3. **Embedding Models برای Feature Engineering**
   ```python
   SentenceTransformer('all-MiniLM-L6-v2')
   ```

این setup بهترین استفاده از منابع شما را می‌کند:
- CPU: RL Training
- GPU: LLM Inference
- RAM: Data Processing

---

## 📞 منابع اضافی

- [Ollama Documentation](https://ollama.com/docs)
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Freqtrade FreqAI Docs](https://www.freqtrade.io/en/stable/freqai/)
- [Sentence Transformers](https://www.sbert.net/)

---

**ساخته شده برای سیستم All-in-One 2017** ✅
