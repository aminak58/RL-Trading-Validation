# Freqtrade RL Trading Strategy Docker Container
# Complete environment for reinforcement learning trading strategy
# Compatible with Google Colab and local execution

FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    gcc \
    g++ \
    make \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /freqtrade

# Install TA-Lib (required for technical indicators)
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Freqtrade
RUN pip install --no-cache-dir freqtrade

# Create user_data directory structure
RUN mkdir -p user_data/{strategies,freqaimodels,configs,data,models,notebooks}

# Copy project files
COPY user_data/strategies/MtfScalper_RL_Hybrid.py user_data/strategies/
COPY user_data/freqaimodels/MtfScalperRLModel.py user_data/freqaimodels/
COPY configs/config_rl_hybrid.json user_data/configs/
COPY user_data/notebooks/ user_data/notebooks/

# Copy trading data (if available)
# This will be mounted or copied separately

# Set proper permissions
RUN chmod +x user_data/strategies/*.py user_data/freqaimodels/*.py

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function to check if data exists\n\
check_data() {\n\
    if [ ! -d "user_data/data/binance" ]; then\n\
        echo "Trading data not found. Downloading sample data..."\n\
        mkdir -p user_data/data/binance\n\
        # This would download sample data if needed\n\
    fi\n\
}\n\
\n\
# Main execution\n\
case "$1" in\n\
    "download")\n\
        echo "Downloading trading data..."\n\
        freqtrade download-data --config user_data/configs/config_rl_hybrid.json --pairs BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT DOGE/USDT:USDT --timeframes 5m 15m 1h --days 365 --trading-mode futures\n\
        ;;\n\
    "train")\n\
        echo "Starting RL training..."\n\
        check_data\n\
        freqtrade backtesting --config user_data/configs/config_rl_hybrid.json --strategy MtfScalper_RL_Hybrid --freqaimodel MtfScalperRLModel --timeframe 5m --timerange 20240101-20240201\n\
        ;;\n\
    "backtest")\n\
        echo "Running backtest..."\n\
        check_data\n\
        freqtrade backtesting --config user_data/configs/config_rl_hybrid.json --strategy MtfScalper_RL_Hybrid --freqaimodel MtfScalperRLModel --timerange 20240301-20240401\n\
        ;;\n\
    "notebook")\n\
        echo "Starting Jupyter notebook..."\n\
        pip install jupyter\n\
        cd user_data/notebooks\n\
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root\n\
        ;;\n\
    "shell")\n\
        echo "Starting interactive shell..."\n\
        /bin/bash\n\
        ;;\n\
    *)\n\
        echo "Usage: $0 {download|train|backtest|notebook|shell}"\n\
        echo "  download - Download trading data"\n\
        echo "  train    - Train RL model"\n\
        echo "  backtest - Run backtest"\n\
        echo "  notebook - Start Jupyter notebook"\n\
        echo "  shell    - Start interactive shell"\n\
        exit 1\n\
        ;;\n\
esac\n\
' > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

# Expose Jupyter port
EXPOSE 8888

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["shell"]