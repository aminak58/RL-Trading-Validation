# راهنمای نصب و استفاده از Claude Code Extension

## 🔌 نصب افزونه

### در Cursor/VS Code:
1. باز کردن Extensions Panel:
   - کلید میانبر: `Ctrl+Shift+X` (یا `Cmd+Shift+X` در Mac)
   - یا از منو: `View` → `Extensions`

2. جستجو و نصب:
   - جستجوی "Claude Code" یا "Anthropic Claude"
   - کلیک روی `Install` برای افزونه رسمی Anthropic

3. احراز هویت:
   - پس از نصب، نیاز به کلید API Anthropic دارید
   - دریافت از: https://console.anthropic.com/
   - وارد کردن در Settings → Extensions → Claude Code

## ⚙️ پیکربندی پروژه

### فایل `.clauderc` (اختیاری اما توصیه می‌شود)

این فایل در ریشه پروژه قرار می‌گیرد و مسیرهای مهم را مشخص می‌کند:

```json
{
  "include": [
    "user_data/strategies/**/*.py",
    "user_data/freqaimodels/**/*.py",
    "configs/**/*.json",
    "scripts/**/*.py",
    "*.py"
  ],
  "exclude": [
    "**/__pycache__/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/models/**",
    "**/backtest_results/**",
    "user_data/data/**"
  ],
  "context": {
    "description": "Freqtrade RL Trading Strategy with Hybrid Entry/Exit",
    "key_files": [
      "user_data/strategies/MtfScalper_RL_Hybrid.py",
      "user_data/freqaimodels/MtfScalperRLModel.py",
      "configs/config_rl_hybrid.json"
    ],
    "language": "python",
    "framework": "freqtrade"
  }
}
```

## 🚀 استفاده از Claude Code

### روش‌های دسترسی:

1. **Command Palette:**
   - `Ctrl+Shift+P` → جستجوی "Claude: Chat"
   - `Ctrl+Shift+P` → جستجوی "Claude: Explain Code"

2. **کلیدهای میانبر:**
   - `Ctrl+L` - باز کردن چت
   - `Ctrl+K` - Edit with Claude (پس از انتخاب کد)

3. **منوی راست‌کلیک:**
   - انتخاب کد → `Ask Claude`
   - انتخاب کد → `Explain with Claude`

4. **Inline Suggestions:**
   - Claude Code به صورت خودکار پیشنهادات کد ارائه می‌دهد
   - `Tab` برای پذیرش پیشنهاد

## 💡 استفاده برای این پروژه

### سوالات مفید برای Claude Code:

1. **درک استراتژی:**
   - "چطور استراتژی MtfScalper_RL_Hybrid کار می‌کند؟"
   - "کدام indicators برای ورود استفاده می‌شوند؟"

2. **بهینه‌سازی مدل RL:**
   - "چطور می‌توانم reward function را بهبود بدهم؟"
   - "آیا action space مناسبی تعریف شده است؟"

3. **دیباگ:**
   - "این خطا به چه دلیلی رخ می‌دهد؟"
   - "چرا مدل به درستی آموزش نمی‌بیند؟"

4. **افزودن قابلیت:**
   - "چطور می‌توانم feature جدیدی اضافه کنم؟"
   - "چطور می‌توانم از timeframe جدیدی استفاده کنم؟"

## 📋 مقایسه با Cursor Chat

| ویژگی | Claude Code | Cursor Chat (Auto) |
|-------|------------|-------------------|
| منبع | Anthropic | Cursor |
| API Key | نیاز دارد | نیاز ندارد |
| هزینه | ممکن است هزینه داشته باشد | رایگان در Cursor |
| کیفیت کد | عالی | عالی |
| پشتیبانی پروژه | بله | بله |

## 🔍 نکات مهم

1. **کلید API:**
   - از کلید API خود محافظت کنید
   - هرگز آن را در Git commit نکنید
   - از متغیرهای محیطی استفاده کنید

2. **محدودیت‌ها:**
   - ممکن است محدودیت تعداد درخواست داشته باشد
   - برای پروژه‌های بزرگ، از `.clauderc` برای فیلتر کردن استفاده کنید

3. **بهینه‌سازی:**
   - فایل‌های غیرضروری را در `.clauderc` exclude کنید
   - فقط فایل‌های مرتبط را include کنید

## 📝 مثال استفاده

### بهبود Reward Function:

1. باز کردن فایل `MtfScalperRLModel.py`
2. انتخاب بخش `compute_reward`
3. راست‌کلیک → `Ask Claude`
4. پرسیدن: "چطور می‌توانم این reward function را بهبود بدهم تا drawdown کمتر شود؟"

### اضافه کردن Feature جدید:

1. باز کردن فایل `MtfScalper_RL_Hybrid.py`
2. انتخاب تابع `populate_indicators`
3. استفاده از `Ctrl+K` برای Edit with Claude
4. درخواست: "یک feature جدید برای volume profile اضافه کن"

## 🔗 لینک‌های مفید

- مستندات رسمی: https://docs.anthropic.com/claude/docs/claude-code
- دریافت API Key: https://console.anthropic.com/
- Community: https://discord.gg/anthropic

---

**نکته:** اگر در Cursor هستید، می‌توانید هم از Cursor Chat (Auto) و هم از Claude Code استفاده کنید!

















