#!/usr/bin/env python3
"""
Vast AI RL Trading Framework Launcher

This script ONLY handles infrastructure:
1. Creates Vast AI instance
2. Sets up environment
3. Copies project to instance
4. Runs the framework
5. Copies results back
6. Terminates instance

All trading parameters remain in the framework config files.
"""

import json
import logging
import argparse
import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vast_ai_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VastAILauncher:
    """Main class for managing Vast AI infrastructure"""

    def __init__(self, config_path: str = "vast_ai_config.json"):
        self.config_path = Path(config_path)
        self.ssh_key_path = Path("C:/Users/Kiyan-System/.ssh/vast_fix")
        self.work_dir = Path.cwd()
        self.results_dir = self.work_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Load infrastructure configuration
        self.config = self._load_config()

        # Instance tracking
        self.instance_id = None
        self.instance_ip = None

    def _load_config(self) -> Dict:
        """Load infrastructure configuration from file"""
        default_config = {
            "vastai": {
                "min_gpu_count": 1,
                "min_gpu_ram_gb": 16,
                "min_disk_gb": 50,
                "max_price_per_hour": 0.5,
                "preferred_gpu_types": ["RTX 3090", "RTX 4090", "A4000", "A5000"],
                "image": "nvidia/cuda:12.1.1-devel-ubuntu22.04"
            },
            "execution": {
                "max_runtime_hours": 24,
                "auto_terminate": True,
                "copy_results": True,
                "cleanup_temp": True
            }
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults for infrastructure settings only
                for key in default_config:
                    if key not in user_config:
                        user_config[key] = default_config[key]
                    elif key == "vastai":
                        for subkey in default_config[key]:
                            if subkey not in user_config[key]:
                                user_config[key][subkey] = default_config[key][subkey]
                return user_config
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")

        # Save default config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        logger.info(f"Created default config at {self.config_path}")
        return default_config

    def check_vastai_cli(self) -> bool:
        """Check if Vast AI CLI is installed and configured"""
        try:
            result = subprocess.run(["vastai", "show", "user"],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info("Vast AI CLI is properly configured")
                return True
            else:
                logger.error("Vast AI CLI error: %s", result.stderr)
                return False
        except subprocess.TimeoutExpired:
            logger.error("Vast AI CLI timeout")
            return False
        except FileNotFoundError:
            logger.error("Vast AI CLI not found. Please install it first.")
            return False

    def find_suitable_instance(self) -> Optional[Dict]:
        """Find suitable Vast AI instances based on infrastructure configuration"""
        logger.info("Searching for suitable Vast AI instances...")

        # Build query string for vastai search offers
        query_parts = [
            f"num_gpus>={self.config['vastai']['min_gpu_count']}",
            f"gpu_ram>={self.config['vastai']['min_gpu_ram_gb']}",
            f"disk_space>={self.config['vastai']['min_disk_gb']}",
            f"dph<={self.config['vastai']['max_price_per_hour']}",
            "rentable=True",
            "verified=True",
            "geolocation notin [US,PR]"  # Exclude US regions for Binance access
        ]

        # Add GPU preferences if specified
        if self.config['vastai']['preferred_gpu_types']:
            gpu_names = ', '.join([f'"{gpu}"' for gpu in self.config['vastai']['preferred_gpu_types']])
            query_parts.append(f"gpu_name in [{gpu_names}]")

        # Add region preferences
        if self.config['vastai'].get('preferred_regions'):
            regions = ', '.join([f'"{region}"' for region in self.config['vastai']['preferred_regions']])
            query_parts.append(f"geolocation in [{regions}]")

        # Ensure excluded regions
        if self.config['vastai'].get('excluded_regions'):
            excluded = ', '.join([f'"{region}"' for region in self.config['vastai']['excluded_regions']])
            query_parts.append(f"geolocation notin [{excluded}]")

        query = ' '.join(query_parts)

        cmd = [
            "vastai", "search", "offers", query,
            "--order", "reliability-",
            "--raw"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                logger.error("Error searching instances: %s", result.stderr)
                return None

            instances = json.loads(result.stdout)
            if not instances:
                logger.error("No suitable instances found")
                return None

            # Filter for reliable instances first (reliability > 0.9)
            reliable_instances = [inst for inst in instances if inst.get('reliability', 0) > 0.9]

            # Filter out US regions if specified
            if self.config['vastai'].get('excluded_regions'):
                excluded_regions = self.config['vastai']['excluded_regions']
                reliable_instances = [
                    inst for inst in reliable_instances
                    if inst.get('geolocation', '').upper() not in [r.upper() for r in excluded_regions]
                ]

            # Prefer preferred regions
            if self.config['vastai'].get('preferred_regions'):
                preferred_regions = [r.upper() for r in self.config['vastai']['preferred_regions']]
                preferred_instances = [
                    inst for inst in reliable_instances
                    if inst.get('geolocation', '').upper() in preferred_regions
                ]
                target_instances = preferred_instances if preferred_instances else reliable_instances
            else:
                target_instances = reliable_instances

            # If no reliable instances in preferred regions, use all reliable non-US instances
            if not target_instances:
                target_instances = reliable_instances

            # Sort by price first (lowest price), then by reliability (highest)
            target_instances.sort(key=lambda x: (x.get('dph', 999), -x.get('reliability', 0)))
            selected = target_instances[0]

            reliability = selected.get('reliability', 0)
            dph = selected.get('dph', 0)
            gpu_name = selected.get('gpu_name', 'Unknown')
            geolocation = selected.get('geolocation', 'Unknown')

            logger.info(f"Selected instance: {gpu_name}")
            logger.info(f"🌍 Location: {geolocation}")
            logger.info(f"💰 Price: ${dph:.3f}/hr")
            logger.info(f"📊 Reliability: {reliability:.2f}")

            # Check if location is suitable for Binance
            us_regions = ['US', 'PR', 'USA', 'PUERTO RICO']
            if geolocation.upper() in us_regions:
                logger.warning("⚠️  WARNING: US location detected - Binance access may be restricted")
            else:
                logger.info("✅ Non-US location - Good for Binance access")

            # Show alternatives if price is high
            if dph > 0.3:
                cheaper_options = [inst for inst in target_instances[1:3] if inst.get('dph', 999) < dph]
                if cheaper_options:
                    logger.info(f"💡 Cheaper alternatives available:")
                    for i, alt in enumerate(cheaper_options[:2], 1):
                        alt_price = alt.get('dph', 0)
                        alt_gpu = alt.get('gpu_name', 'Unknown')
                        alt_rel = alt.get('reliability', 0)
                        alt_geo = alt.get('geolocation', 'Unknown')
                        logger.info(f"   {i}. {alt_gpu} ({alt_geo}) - ${alt_price:.3f}/hr (reliability: {alt_rel:.2f})")

            return selected

        except Exception as e:
            logger.error("Error finding instance: %s", e)
            return None

    def create_instance(self, instance: Dict) -> bool:
        """Create Vast AI instance"""
        logger.info("Creating Vast AI instance...")

        cmd = [
            "vastai", "create", "instance", str(instance['id']),
            "--image", self.config['vastai']['image'],
            "--disk", str(self.config['vastai']['min_disk_gb']),
            "--onstart-cmd", "apt-get update && apt-get install -y python3 python3-pip git curl",
            "--ssh"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                # Parse instance ID from response
                output = result.stdout.strip()
                try:
                    # Try to parse JSON first
                    response = json.loads(output)
                    if response.get('success'):
                        self.instance_id = str(response.get('new_contract'))
                        logger.info(f"Instance {self.instance_id} created successfully")
                        return True
                    else:
                        logger.error("Instance creation failed: %s", response)
                        return False
                except json.JSONDecodeError:
                    # If not JSON, try to extract from text response
                    if "Started" in output and "new_contract" in output:
                        # Extract contract ID from the text response
                        import re
                        match = re.search(r"'new_contract': (\d+)", output)
                        if match:
                            self.instance_id = match.group(1)
                            logger.info(f"Instance {self.instance_id} created successfully")
                            return True
                    logger.error("Failed to parse instance creation response: %s", output)
                    return False
            else:
                logger.error("Error creating instance: %s", result.stderr)
                return False

        except Exception as e:
            logger.error("Error creating instance: %s", e)
            return False

    def wait_for_instance_ready(self, timeout: int = 900) -> bool:
        """Wait for instance to be ready for SSH connection"""
        logger.info("Waiting for instance to be ready...")
        logger.info("⏱️  This may take 2-5 minutes for instance initialization")

        start_time = time.time()
        wait_count = 0

        # Initial wait - let the instance start booting
        logger.info("🔄 Waiting for initial instance startup...")
        time.sleep(120)  # Wait 2 minutes initially

        while time.time() - start_time < timeout:
            wait_count += 1
            elapsed = int(time.time() - start_time)
            logger.info(f"🔍 Checking instance status (attempt {wait_count}, {elapsed}s elapsed)...")

            try:
                result = subprocess.run(["vastai", "show", "instance", self.instance_id],
                                      capture_output=True, text=True, timeout=30)

                if result.returncode == 0:
                    try:
                        instance_data = json.loads(result.stdout)
                        if instance_data:
                            status = instance_data.get('actual_status', '')
                            logger.info(f"📊 Current status: {status}")

                            if status == 'running':
                                self.instance_ip = instance_data.get('public_ipaddr')
                                if self.instance_ip:
                                    logger.info(f"✅ Instance is running at {self.instance_ip}")

                                    # Test SSH connectivity
                                    logger.info("🔌 Testing SSH connectivity...")
                                    ssh_test = subprocess.run([
                                        "ssh", "-i", str(self.ssh_key_path),
                                        "-o", "StrictHostKeyChecking=no",
                                        "-o", "ConnectTimeout=10",
                                        "-o", "BatchMode=yes",
                                        f"root@{self.instance_ip}",
                                        "echo 'SSH Connection Successful'"
                                    ], capture_output=True, text=True, timeout=30)

                                    if ssh_test.returncode == 0:
                                        logger.info("✅ SSH connection successful")
                                        return True
                                    else:
                                        logger.info(f"⏳ SSH not ready yet, waiting...")
                                else:
                                    logger.warning("⚠️  Instance running but no IP assigned yet")

                            elif status in ['starting', 'loading', 'initializing', 'scheduling']:
                                logger.info(f"⏳ Instance is {status}, waiting...")
                            elif status == 'failed':
                                logger.error(f"❌ Instance failed to start")
                                return False
                            else:
                                logger.warning(f"⚠️  Unexpected status: {status}")

                        else:
                            logger.warning("⚠️  No instance data received")

                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️  Failed to parse instance status: {e}")
                        logger.debug(f"Raw response: {result.stdout[:200]}...")

                else:
                    logger.warning(f"⚠️  Failed to get instance status: {result.stderr}")

            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Status check timeout")
            except Exception as e:
                logger.error(f"❌ Error checking instance status: {e}")

            # Progressive wait times
            if wait_count <= 4:
                time.sleep(30)  # First 2 minutes: check every 30s
            elif wait_count <= 8:
                time.sleep(45)  # Next 3 minutes: check every 45s
            else:
                time.sleep(60)  # After that: check every minute

        logger.error(f"❌ Instance not ready within {timeout//60} minutes")
        return False

    def setup_environment(self) -> bool:
        """Setup the environment on remote instance"""
        logger.info("🔧 Setting up environment on remote instance...")
        logger.info(f"🌐 Connecting to {self.instance_ip}")

        # Step 1: Create workspace directory
        logger.info("📁 Creating workspace directory...")
        try:
            result = subprocess.run([
                "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                f"root@{self.instance_ip}",
                "mkdir -p /workspace && echo 'Workspace created successfully'"
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                logger.error(f"❌ Failed to create workspace: {result.stderr}")
                return False
            logger.info("✅ Workspace directory created")
        except Exception as e:
            logger.error(f"❌ Error creating workspace: {e}")
            return False

        # Step 2: Copy project files
        logger.info("📋 Copying project files to instance...")
        try:
            result = subprocess.run([
                "scp", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                "-r", ".", f"root@{self.instance_ip}:/workspace/"
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(f"❌ Failed to copy files: {result.stderr}")
                return False
            logger.info("✅ Project files copied successfully")
        except Exception as e:
            logger.error(f"❌ Error copying files: {e}")
            return False

        # Step 3: Verify project structure
        logger.info("🔍 Verifying project structure...")
        try:
            result = subprocess.run([
                "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                f"root@{self.instance_ip}",
                "ls -la /workspace/scripts/ /workspace/user_data/strategies/ /workspace/user_data/freqaimodels/"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                logger.info("✅ Project structure verified")
                logger.debug(f"Remote files: {result.stdout}")
            else:
                logger.warning(f"⚠️  Project structure verification failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"⚠️  Error verifying structure: {e}")

        # Step 4: Install system dependencies
        logger.info("📦 Installing system dependencies...")
        try:
            result = subprocess.run([
                "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                f"root@{self.instance_ip}",
                """
                apt-get update && \
                apt-get install -y python3-pip python3-venv git curl wget && \
                echo 'System dependencies installed'
                """
            ], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                logger.error(f"❌ Failed to install system dependencies: {result.stderr}")
                return False
            logger.info("✅ System dependencies installed")
        except Exception as e:
            logger.error(f"❌ Error installing system dependencies: {e}")
            return False

        # Step 5: Install PyTorch
        logger.info("🔥 Installing PyTorch (this may take several minutes)...")
        try:
            result = subprocess.run([
                "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                f"root@{self.instance_ip}",
                """
                cd /workspace && \
                pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
                python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" && \
                echo 'PyTorch installation completed'
                """
            ], capture_output=True, text=True, timeout=900)  # 15 minutes timeout

            if result.returncode != 0:
                logger.error(f"❌ Failed to install PyTorch: {result.stderr}")
                return False
            logger.info("✅ PyTorch installed successfully")
            if result.stdout:
                logger.debug(f"PyTorch output: {result.stdout}")
        except Exception as e:
            logger.error(f"❌ Error installing PyTorch: {e}")
            return False

        # Step 6: Install trading and RL dependencies
        logger.info("💹 Installing trading and RL dependencies...")
        try:
            result = subprocess.run([
                "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                f"root@{self.instance_ip}",
                """
                cd /workspace && \
                echo "Installing FreqTrade and dependencies..." && \
                pip3 install freqtrade && \
                echo "Installing data science packages..." && \
                pip3 install pandas numpy matplotlib seaborn plotly && \
                echo "Installing machine learning packages..." && \
                pip3 install scikit-learn scipy && \
                echo "Installing RL packages..." && \
                pip3 install stable-baselines3[extra] gymnasium shimmy && \
                echo "Installing tensorboard and logging..." && \
                pip3 install tensorboard tensorboardX wandb && \
                echo "Installing additional utilities..." && \
                pip3 install tqdm colorama rich requests && \
                echo "Testing installations..." && \
                python3 -c "
try:
    import freqtrade
    print('✅ FreqTrade installed')
except Exception as e:
    print(f'❌ FreqTrade error: {e}')

try:
    import pandas as pd
    import numpy as np
    print('✅ Pandas and NumPy installed')
except Exception as e:
    print(f'❌ Pandas/NumPy error: {e}')

try:
    import torch
    print(f'✅ PyTorch {torch.__version__} installed')
except Exception as e:
    print(f'❌ PyTorch error: {e}')

try:
    import stable_baselines3
    print('✅ Stable Baselines3 installed')
except Exception as e:
    print(f'❌ Stable Baselines3 error: {e}')

try:
    import tensorboard
    print('✅ TensorBoard installed')
except Exception as e:
    print(f'❌ TensorBoard error: {e}')

print('🎉 All dependencies verification completed')
" && \
                echo 'Trading and RL dependencies installation completed'
                """
            ], capture_output=True, text=True, timeout=1200)  # 20 minutes timeout

            if result.returncode != 0:
                logger.error(f"❌ Failed to install trading dependencies: {result.stderr}")
                return False
            logger.info("✅ Trading and RL dependencies installed successfully")
            if result.stdout:
                logger.info(f"📋 Installation output:\n{result.stdout}")
        except Exception as e:
            logger.error(f"❌ Error installing trading dependencies: {e}")
            return False

        # Step 7: Final comprehensive verification
        logger.info("🎯 Running final comprehensive verification...")
        try:
            result = subprocess.run([
                "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                f"root@{self.instance_ip}",
                """
                cd /workspace && \
                python3 -c "
print('🔍 COMPREHENSIVE ENVIRONMENT VERIFICATION')
print('='*50)

# Check core dependencies
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'   CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
except Exception as e:
    print(f'❌ PyTorch error: {e}')

try:
    import freqtrade
    print(f'✅ FreqTrade: {freqtrade.__version__}')
except Exception as e:
    print(f'❌ FreqTrade error: {e}')

try:
    import stable_baselines3 as sb3
    print(f'✅ Stable Baselines3: {sb3.__version__}')
except Exception as e:
    print(f'❌ Stable Baselines3 error: {e}')

try:
    import pandas as pd
    print(f'✅ Pandas: {pd.__version__}')
except Exception as e:
    print(f'❌ Pandas error: {e}')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except Exception as e:
    print(f'❌ NumPy error: {e}')

try:
    import matplotlib.pyplot as plt
    print('✅ Matplotlib installed')
except Exception as e:
    print(f'❌ Matplotlib error: {e}')

try:
    import sklearn
    print(f'✅ Scikit-learn: {sklearn.__version__}')
except Exception as e:
    print(f'❌ Scikit-learn error: {e}')

try:
    import tensorboard
    print(f'✅ TensorBoard: {tensorboard.__version__}')
except Exception as e:
    print(f'❌ TensorBoard error: {e}')

# Check IP geolocation for Binance access
try:
    import requests
    import json
    response = requests.get('https://ipapi.co/json/', timeout=10)
    if response.status_code == 200:
        ip_info = response.json()
        country = ip_info.get('country', 'Unknown')
        ip = ip_info.get('ip', 'Unknown')
        print(f'🌐 IP Address: {ip}')
        print(f'🌍 Location: {country}')

        if country in ['US', 'PR']:
            print(f'⚠️  WARNING: US IP detected - Binance US restrictions may apply')
        else:
            print(f'✅ Non-US IP - Good for Binance access')
except Exception as e:
    print(f'⚠️  Could not check IP geolocation: {e}')

# Check project structure
import os
print()
print('📁 PROJECT STRUCTURE VERIFICATION')
print('='*50)

project_files = [
    'scripts/run_backtest_with_analysis.py',
    'user_data/data_collector.py',
    'user_data/strategies/MtfScalper_RL_Hybrid.py',
    'user_data/freqaimodels/MtfScalperRLModel.py'
]

for file_path in project_files:
    if os.path.exists(file_path):
        print(f'✅ {file_path}')
    else:
        print(f'❌ {file_path} - MISSING!')

# Test data collector import
try:
    from user_data.data_collector import DataCollector
    print('✅ DataCollector import successful')
except Exception as e:
    print(f'❌ DataCollector import error: {e}')

# Test strategy import
try:
    sys.path.append('/workspace/user_data/strategies')
    import MtfScalper_RL_Hybrid
    print('✅ Strategy import successful')
except Exception as e:
    print(f'❌ Strategy import error: {e}')

# Test model import
try:
    sys.path.append('/workspace/user_data/freqaimodels')
    import MtfScalperRLModel
    print('✅ Model import successful')
except Exception as e:
    print(f'❌ Model import error: {e}')

print()
print('🎉 ENVIRONMENT READY FOR RL TRADING FRAMEWORK')
print('='*50)
"
                """
            ], capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✅ Comprehensive environment verification completed successfully")
                logger.info(f"📊 Verification results:\n{result.stdout}")

                # Check for any ❌ marks in output
                if "❌" in result.stdout:
                    logger.warning("⚠️  Some components failed verification. Check the output above.")
                else:
                    logger.info("🎉 All components verified successfully!")

            else:
                logger.error(f"❌ Environment verification failed: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error during verification: {e}")
            return False

        logger.info("🎉 Environment setup completed successfully!")
        return True

    def run_framework(self) -> bool:
        """Run the RL trading framework on remote instance"""
        logger.info("Running RL trading framework...")

        # Note: Trading parameters are NOT modified here
        # The framework uses its own configuration files

        cmd = f"ssh -i {self.ssh_key_path} -o StrictHostKeyChecking=no root@{self.instance_ip} " + \
              f"'cd /workspace && python scripts/run_backtest_with_analysis.py'"

        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=int(self.config['execution']['max_runtime_hours'] * 3600))

            # Log output regardless of success/failure
            logger.info("Framework output:\n%s", result.stdout)
            if result.stderr:
                logger.warning("Framework warnings:\n%s", result.stderr)

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            logger.error("Framework execution timed out")
            return False
        except Exception as e:
            logger.error("Error running framework: %s", e)
            return False

    def copy_results(self) -> bool:
        """Copy results from remote instance to local"""
        logger.info("📥 Copying results from remote instance...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_results_dir = self.results_dir / f"run_{timestamp}"
        local_results_dir.mkdir(exist_ok=True)

        logger.info(f"📁 Local results directory: {local_results_dir}")

        # Define what to copy (in order of importance)
        copy_targets = [
            ("/workspace/results/", "backtest_results"),
            ("/workspace/user_data/analysis_data/", "collected_data"),
            ("/workspace/user_data/models/", "trained_models"),
            ("/workspace/user_data/backtest_results/", "backtest_results"),
            ("/workspace/user_data/hyperopt_results/", "hyperopt_results"),
            ("/workspace/logs/", "execution_logs"),
            ("/workspace/tensorboard_logs/", "tensorboard_logs")
        ]

        copied_anything = False

        for remote_path, local_subdir in copy_targets:
            logger.info(f"🔍 Checking {remote_path}...")

            # First check if the directory exists on remote
            check_cmd = [
                "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                f"root@{self.instance_ip}",
                f"test -d {remote_path} && echo 'EXISTS' || echo 'NOT_FOUND'"
            ]

            try:
                result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
                if "EXISTS" in result.stdout:
                    logger.info(f"✅ Found {remote_path}, copying...")

                    # Create local subdirectory
                    local_target_dir = local_results_dir / local_subdir
                    local_target_dir.mkdir(exist_ok=True)

                    # Copy the contents
                    copy_cmd = [
                        "scp", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                        "-r", f"root@{self.instance_ip}:{remote_path}*", str(local_target_dir)
                    ]

                    result = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=300)

                    if result.returncode == 0:
                        # Check if anything was actually copied
                        if any(local_target_dir.iterdir()):
                            logger.info(f"✅ Successfully copied {local_subdir}")
                            copied_anything = True

                            # List what was copied
                            try:
                                list_cmd = [
                                    "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                                    f"root@{self.instance_ip}",
                                    f"find {remote_path} -type f | head -10"
                                ]
                                list_result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=30)
                                if list_result.returncode == 0 and list_result.stdout.strip():
                                    logger.info(f"📋 Sample files copied:\n{list_result.stdout}")
                            except:
                                pass
                        else:
                            logger.info(f"⚠️  {local_subdir} directory exists but is empty")
                    else:
                        logger.warning(f"⚠️  Failed to copy {local_subdir}: {result.stderr}")
                else:
                    logger.info(f"⚠️  {remote_path} not found on remote")
            except Exception as e:
                logger.warning(f"⚠️  Error checking {remote_path}: {e}")

        # Try to copy any additional files that might be useful
        try:
            logger.info("🔍 Looking for additional result files...")
            additional_files_cmd = [
                "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                f"root@{self.instance_ip}",
                """
                find /workspace -name "*.json" -o -name "*.csv" -o -name "*.html" -o -name "*.log" | \
                grep -E "(backtest|analysis|result|model)" | head -20
                """
            ]

            result = subprocess.run(additional_files_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout.strip():
                logger.info(f"📋 Additional files found:\n{result.stdout}")

                # Copy these additional files
                additional_files = result.stdout.strip().split('\n')
                for file_path in additional_files:
                    if file_path.strip():
                        try:
                            copy_cmd = [
                                "scp", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                                f"root@{self.instance_ip}:{file_path}", str(local_results_dir)
                            ]
                            subprocess.run(copy_cmd, capture_output=True, text=True, timeout=60)
                            copied_anything = True
                        except:
                            pass
        except:
            pass

        # Create a summary of what was copied
        try:
            summary_cmd = [
                "ssh", "-i", str(self.ssh_key_path), "-o", "StrictHostKeyChecking=no",
                f"root@{self.instance_ip}",
                """
                echo "=== REMOTE WORKSPACE SUMMARY ===" && \
                du -sh /workspace/* 2>/dev/null | head -10 && \
                echo "=== RECENT FILES ===" && \
                find /workspace -type f -mtime -1 | head -10
                """
            ]

            result = subprocess.run(summary_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                summary_file = local_results_dir / "remote_summary.txt"
                with open(summary_file, 'w') as f:
                    f.write(result.stdout)
                logger.info("📋 Remote workspace summary saved")
        except:
            pass

        if copied_anything:
            logger.info(f"🎉 Results successfully copied to {local_results_dir}")

            # Show what we got
            try:
                total_size = subprocess.run([
                    "du", "-sh", str(local_results_dir)
                ], capture_output=True, text=True, timeout=30)
                if total_size.returncode == 0:
                    logger.info(f"📊 Total size copied: {total_size.stdout.strip()}")

                # List top-level directories
                logger.info("📁 Copied directories:")
                for item in local_results_dir.iterdir():
                    if item.is_dir():
                        size_result = subprocess.run([
                            "du", "-sh", str(item)
                        ], capture_output=True, text=True, timeout=30)
                        size = size_result.stdout.strip().split('\t')[0] if size_result.returncode == 0 else "N/A"
                        logger.info(f"  📂 {item.name}/ ({size})")
            except:
                pass

            return True
        else:
            logger.warning("⚠️  No results were found to copy")
            return False

    def terminate_instance(self, force: bool = False) -> bool:
        """Terminate the Vast AI instance"""
        if not self.instance_id:
            return True

        # Ask for user confirmation unless forced
        if not force:
            print(f"\n{'='*60}")
            print(f"🔴 INSTANCE TERMINATION CONFIRMATION")
            print(f"{'='*60}")
            print(f"Instance ID: {self.instance_id}")
            print(f"Instance IP: {self.instance_ip}")
            print(f"\n⚠️  This will permanently terminate the instance.")
            print(f"⚠️  All data on the instance will be lost.")
            print(f"⚠️  This action cannot be undone.")

            # Get current instance cost info
            try:
                result = subprocess.run(["vastai", "show", "instance", self.instance_id],
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    instance_data = json.loads(result.stdout)
                    if instance_data:
                        dph = instance_data.get('dph', 0)
                        gpu_name = instance_data.get('gpu_name', 'Unknown')
                        print(f"\n💰 Current cost: ${dph:.3f}/hour")
                        print(f"💻 GPU: {gpu_name}")
            except:
                pass

            print(f"\n{'='*60}")

            while True:
                response = input("Do you want to terminate this instance? (yes/no): ").lower().strip()
                if response in ['yes', 'y']:
                    break
                elif response in ['no', 'n']:
                    logger.info(f"Instance {self.instance_id} will remain running")
                    print(f"💡 You can terminate it later with: vastai destroy instance {self.instance_id}")
                    return False
                else:
                    print("Please enter 'yes' or 'no'")

        logger.info(f"Terminating instance {self.instance_id}...")

        cmd = ["vastai", "destroy", "instance", self.instance_id]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                logger.info("Instance terminated successfully")
                print("✅ Instance terminated successfully")
                return True
            else:
                logger.error("Error terminating instance: %s", result.stderr)
                print(f"❌ Error terminating instance: {result.stderr}")
                return False
        except Exception as e:
            logger.error("Error terminating instance: %s", e)
            print(f"❌ Error terminating instance: {e}")
            return False

    def run_complete_workflow(self) -> bool:
        """Run the complete infrastructure workflow"""
        try:
            # Check prerequisites
            if not self.check_vastai_cli():
                return False

            # Find and create instance
            instance = self.find_suitable_instance()
            if not instance:
                return False

            if not self.create_instance(instance):
                return False

            # Wait for instance to be ready
            if not self.wait_for_instance_ready():
                self.terminate_instance(force=True)
                return False

            # Setup and run framework
            if not self.setup_environment():
                self.terminate_instance(force=True)
                return False

            framework_success = self.run_framework()

            # Copy results if configured
            if self.config['execution']['copy_results']:
                self.copy_results()

            # Handle instance termination
            if self.config['execution']['auto_terminate']:
                if self.instance_id:
                    print(f"\n{'='*60}")
                    print("🎯 FRAMEWORK EXECUTION COMPLETED")
                    print(f"{'='*60}")
                    print(f"Instance ID: {self.instance_id}")
                    print(f"Instance IP: {self.instance_ip}")

                    # Show cost summary
                    try:
                        result = subprocess.run(["vastai", "show", "instance", self.instance_id],
                                              capture_output=True, text=True, timeout=30)
                        if result.returncode == 0:
                            instance_data = json.loads(result.stdout)
                            if instance_data:
                                dph = instance_data.get('dph', 0)
                                gpu_name = instance_data.get('gpu_name', 'Unknown')
                                geolocation = instance_data.get('geolocation', 'Unknown')
                                print(f"💻 GPU: {gpu_name}")
                                print(f"🌍 Location: {geolocation}")
                                print(f"💰 Current cost: ${dph:.3f}/hour")

                                # Check Binance access
                                us_regions = ['US', 'PR', 'USA', 'PUERTO RICO']
                                if geolocation.upper() in us_regions:
                                    print(f"⚠️  WARNING: US location - Binance access may be restricted")
                                else:
                                    print(f"✅ Non-US location - Good for Binance access")
                    except:
                        pass

                # Ask for termination confirmation
                self.terminate_instance(force=False)

            return framework_success

        except KeyboardInterrupt:
            logger.info("Workflow interrupted by user")
            if self.instance_id:
                print(f"\n⚠️  Workflow interrupted! Instance {self.instance_id} is still running.")
                print(f"💡 You can terminate it manually with: vastai destroy instance {self.instance_id}")
            return False
        except Exception as e:
            logger.error("Workflow error: %s", e)
            if self.instance_id:
                print(f"\n❌ Error occurred! Instance {self.instance_id} is still running.")
                print(f"💡 You can terminate it manually with: vastai destroy instance {self.instance_id}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Vast AI RL Trading Framework Launcher")
    parser.add_argument("--config", default="vast_ai_config.json",
                       help="Infrastructure configuration file path")
    parser.add_argument("--gpu-count", type=int, help="Minimum GPU count")
    parser.add_argument("--min-gpu-ram", type=int, help="Minimum GPU VRAM in GB")
    parser.add_argument("--max-price", type=float, help="Maximum price per hour")
    parser.add_argument("--no-terminate", action="store_true",
                       help="Don't auto-terminate instance")
    parser.add_argument("--skip-copy", action="store_true",
                       help="Skip copying results")
    parser.add_argument("--runtime-hours", type=int,
                       help="Maximum runtime hours")

    args = parser.parse_args()

    # Create launcher
    launcher = VastAILauncher(args.config)

    # Override infrastructure config with command line arguments (NO trading parameters)
    if args.gpu_count:
        launcher.config['vastai']['min_gpu_count'] = args.gpu_count
    if args.min_gpu_ram:
        launcher.config['vastai']['min_gpu_ram_gb'] = args.min_gpu_ram
    if args.max_price:
        launcher.config['vastai']['max_price_per_hour'] = args.max_price
    if args.no_terminate:
        launcher.config['execution']['auto_terminate'] = False
    if args.skip_copy:
        launcher.config['execution']['copy_results'] = False
    if args.runtime_hours:
        launcher.config['execution']['max_runtime_hours'] = args.runtime_hours

    # Run workflow
    logger.info("Starting Vast AI infrastructure setup for RL Trading Framework")
    logger.info("Note: Trading parameters are handled by the framework configuration files")
    success = launcher.run_complete_workflow()

    if success:
        logger.info("Infrastructure workflow completed successfully")
        sys.exit(0)
    else:
        logger.error("Infrastructure workflow failed")
        sys.exit(1)


if __name__ == "__main__":
    main()