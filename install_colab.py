#!/usr/bin/env python3
"""
Colab Installation Script for Freqtrade RL Trading
Handles version compatibility issues automatically
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"🔧 {description}")
    print(f"   Running: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success!")
            return True
        else:
            print(f"   ❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

def install_dependencies():
    """Install all dependencies with error handling"""

    print("🚀 Installing Freqtrade RL Trading Dependencies")
    print("=" * 50)

    # Step 1: Basic dependencies
    print("\n📦 Step 1: Installing basic dependencies...")
    basic_deps = [
        "freqtrade",
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ]

    for dep in basic_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

    # Step 2: PyTorch (most problematic)
    print("\n🔥 Step 2: Installing PyTorch...")

    # Try PyTorch with CUDA first (for Colab GPU)
    pytorch_commands = [
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "pip install torch torchvision torchaudio",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    ]

    pytorch_installed = False
    for cmd in pytorch_commands:
        if run_command(cmd, "Installing PyTorch"):
            pytorch_installed = True
            break
        print("   ⚠️ Trying next method...")

    if not pytorch_installed:
        print("   ❌ PyTorch installation failed. Please install manually.")

    # Step 3: RL libraries
    print("\n🧠 Step 3: Installing RL libraries...")
    rl_deps = [
        "stable-baselines3>=2.2.0",
        "gymnasium>=0.28.0",
    ]

    for dep in rl_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

    # Step 4: Data processing
    print("\n📊 Step 4: Installing data processing...")
    data_deps = [
        "datasieve==0.1.9",
        "joblib>=1.3.0",
    ]

    for dep in data_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

    # Step 5: Visualization
    print("\n📈 Step 5: Installing visualization...")
    viz_deps = [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "jupyter",
    ]

    for dep in viz_deps:
        run_command(f"pip install {dep}", f"Installing {dep}")

    # Step 6: TA-Lib (often problematic)
    print("\n📈 Step 6: Installing TA-Lib...")
    ta_lib_commands = [
        "pip install ta-lib>=0.4.25",
        "pip install TA-Lib",
    ]

    ta_lib_installed = False
    for cmd in ta_lib_commands:
        if run_command(cmd, "Installing TA-Lib"):
            ta_lib_installed = True
            break
        print("   ⚠️ Trying alternative...")

    if not ta_lib_installed:
        print("   ⚠️ TA-Lib installation failed. Technical indicators may not work.")
        print("   💡 You can try: !apt-get install -y ta-lib")

    print("\n✅ Installation process completed!")
    print("\n🧪 Testing installation...")

    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("gymnasium", "Gymnasium"),
        ("freqtrade", "Freqtrade"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
    ]

    failed_imports = []
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError as e:
            print(f"   ❌ {name}: {e}")
            failed_imports.append(name)

    if failed_imports:
        print(f"\n⚠️ Some imports failed: {', '.join(failed_imports)}")
        print("   You may need to install these manually.")
    else:
        print("\n🎉 All imports successful! Ready for RL trading!")

    return len(failed_imports) == 0

if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        print("🌟 Running in Google Colab")
        # Check GPU availability
        if torch.cuda.is_available():
            print("   🚀 GPU available!")
        else:
            print("   💻 Using CPU (GPU may be faster)")
    except ImportError:
        print("💻 Not running in Colab")

    # Run installation
    success = install_dependencies()

    if success:
        print("\n🎯 Ready to run Freqtrade RL Trading!")
        print("   Next: Run your backtesting commands")
    else:
        print("\n⚠️ Some issues occurred. Check the errors above.")

    print("\n📖 For manual installation, use:")
    print("   !pip install freqtrade torch torchvision stable-baselines3 gymnasium")