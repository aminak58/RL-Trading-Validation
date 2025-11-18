#!/usr/bin/env python3
"""
تست راه‌اندازی محلی برای سیستم All-in-One 2017
i5-7th gen / 32GB RAM / 4GB GPU
"""

import sys
import subprocess
import platform

def print_header(text):
    """چاپ هدر زیبا"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_python():
    """بررسی نسخه Python"""
    print_header("بررسی Python")
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ نیاز به Python 3.8+")
        return False
    print("✅ نسخه Python مناسب است")
    return True

def check_pytorch():
    """بررسی PyTorch و CUDA"""
    print_header("بررسی PyTorch")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")

        # بررسی CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠ CUDA not available - will use CPU")

        # بررسی MPS (برای Mac)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✓ Apple Silicon MPS available")

        print("✅ PyTorch نصب شده است")
        return True
    except ImportError:
        print("❌ PyTorch نصب نشده")
        return False

def check_stable_baselines3():
    """بررسی Stable Baselines3"""
    print_header("بررسی Stable Baselines3")
    try:
        import stable_baselines3 as sb3
        print(f"✓ Stable Baselines3 {sb3.__version__}")

        # تست ساخت یک مدل ساده
        from stable_baselines3 import PPO
        import gym

        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0, device="cpu")
        print("✓ PPO model test successful")

        print("✅ Stable Baselines3 کار می‌کند")
        return True
    except ImportError:
        print("❌ Stable Baselines3 نصب نشده")
        return False
    except Exception as e:
        print(f"⚠ خطا در تست: {e}")
        return False

def check_freqtrade():
    """بررسی Freqtrade"""
    print_header("بررسی Freqtrade")
    try:
        import freqtrade
        print(f"✓ Freqtrade نصب شده")

        # بررسی FreqAI
        try:
            from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
            print("✓ FreqAI available")
        except:
            print("⚠ FreqAI may not be available")

        print("✅ Freqtrade نصب شده است")
        return True
    except ImportError:
        print("❌ Freqtrade نصب نشده")
        return False

def check_system_resources():
    """بررسی منابع سیستم"""
    print_header("بررسی منابع سیستم")

    # CPU
    import psutil
    cpu_count = psutil.cpu_count()
    print(f"✓ CPU Cores: {cpu_count}")

    # RAM
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    print(f"✓ Total RAM: {ram_gb:.1f} GB")
    print(f"✓ Available RAM: {ram.available / (1024**3):.1f} GB")

    if ram_gb < 16:
        print("⚠ کمتر از 16GB RAM - ممکن است محدودیت داشته باشید")
    else:
        print("✅ RAM کافی است")

    # Disk
    disk = psutil.disk_usage('/')
    disk_gb = disk.free / (1024**3)
    print(f"✓ Free Disk: {disk_gb:.1f} GB")

    return True

def test_small_rl_training():
    """تست آموزش RL کوچک"""
    print_header("تست آموزش RL")

    try:
        import gym
        from stable_baselines3 import PPO
        import time

        print("ساخت محیط CartPole...")
        env = gym.make("CartPole-v1")

        # تست CPU
        print("\nتست آموزش روی CPU...")
        start_time = time.time()
        model_cpu = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device="cpu",
            n_steps=128,
            batch_size=32
        )
        model_cpu.learn(total_timesteps=1000)
        cpu_time = time.time() - start_time
        print(f"✓ آموزش CPU: {cpu_time:.2f} ثانیه")

        # تست GPU (اگر موجود باشد)
        try:
            import torch
            if torch.cuda.is_available():
                print("\nتست آموزش روی GPU...")
                start_time = time.time()
                model_gpu = PPO(
                    "MlpPolicy",
                    env,
                    verbose=0,
                    device="cuda",
                    n_steps=128,
                    batch_size=32
                )
                model_gpu.learn(total_timesteps=1000)
                gpu_time = time.time() - start_time
                print(f"✓ آموزش GPU: {gpu_time:.2f} ثانیه")

                if cpu_time < gpu_time:
                    print("\n💡 توصیه: برای PPO کوچک، CPU سریع‌تر است!")
                else:
                    print(f"\n💡 GPU {cpu_time/gpu_time:.1f}x سریع‌تر است")
        except:
            pass

        print("\n✅ تست RL موفقیت‌آمیز بود")
        return True

    except Exception as e:
        print(f"❌ خطا در تست RL: {e}")
        return False

def check_ollama():
    """بررسی Ollama برای LLM"""
    print_header("بررسی Ollama (اختیاری)")

    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ Ollama: {result.stdout.strip()}")

            # لیست مدل‌های نصب شده
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if "gemma2:2b" in result.stdout:
                print("✓ Gemma 2B نصب شده")
            else:
                print("⚠ Gemma 2B نصب نشده - برای نصب:")
                print("  ollama pull gemma2:2b")

            print("✅ Ollama آماده است")
            return True
        else:
            print("⚠ Ollama نصب نشده")
            print("برای نصب:")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
            return False
    except FileNotFoundError:
        print("⚠ Ollama نصب نشده")
        print("برای نصب:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        return False
    except Exception as e:
        print(f"⚠ خطا در بررسی Ollama: {e}")
        return False

def print_recommendations():
    """چاپ توصیه‌های نهایی"""
    print_header("توصیه‌های نهایی")

    print("""
📊 برای RL Trading (پروژه فعلی):
   ✓ استفاده از CPU برای PPO
   ✓ cpu_count: 4 در config
   ✓ device: "cpu"

🤖 برای LLM (اختیاری):
   ✓ Gemma 2B (1.6GB) - بهترین انتخاب
   ✓ Phi-3 Mini (2.3GB) - جایگزین خوب
   ✓ TinyLlama (900MB) - سریع‌ترین

💾 استفاده بهینه از منابع:
   CPU: RL Training
   GPU: LLM Inference
   RAM: Data Processing

📁 فایل‌های مهم:
   ✓ LOCAL_EXECUTION_GUIDE.md - راهنمای کامل
   ✓ configs/config_local_optimized.json - کانفیگ بهینه
    """)

def main():
    """تابع اصلی"""
    print_header("تست راه‌اندازی محلی - All-in-One 2017")
    print("i5-7th gen / 32GB RAM / 4GB GPU")

    results = []

    # بررسی‌های اصلی
    results.append(("Python", check_python()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("Stable Baselines3", check_stable_baselines3()))
    results.append(("Freqtrade", check_freqtrade()))
    results.append(("System Resources", check_system_resources()))

    # تست‌های اختیاری
    results.append(("RL Training Test", test_small_rl_training()))
    results.append(("Ollama", check_ollama()))

    # خلاصه نتایج
    print_header("خلاصه نتایج")

    passed = sum(1 for _, status in results if status)
    total = len(results)

    for name, status in results:
        status_text = "✅ OK" if status else "❌ FAIL"
        print(f"{name:25s} {status_text}")

    print(f"\nنتیجه: {passed}/{total} تست موفق")

    if passed >= 5:  # حداقل 5 تست باید موفق باشد
        print("\n🎉 سیستم شما آماده است!")
        print_recommendations()
    else:
        print("\n⚠ برخی مشکلات وجود دارد - لطفاً requirements.txt را نصب کنید:")
        print("  pip install -r requirements.txt")

    return passed >= 5

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nمتوقف شد توسط کاربر")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ خطای غیرمنتظره: {e}")
        sys.exit(1)
