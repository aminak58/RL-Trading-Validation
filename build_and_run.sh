#!/bin/bash

# Build and Run Script for Freqtrade RL Trading Container
# This script builds the Docker container and provides easy commands for execution

set -e

echo "🚀 Freqtrade RL Trading Container Setup"
echo "======================================"

# Build the Docker image
echo "📦 Building Docker image..."
docker-compose build

# Function to run container with different commands
run_container() {
    local command=$1
    echo "🔄 Running container with command: $command"
    docker-compose run --rm freqtrade-rl $command
}

# Main menu
case "$1" in
    "build")
        echo "📦 Building only..."
        docker-compose build
        ;;
    "download")
        echo "📥 Downloading trading data..."
        run_container download
        ;;
    "train")
        echo "🧠 Training RL model..."
        run_container train
        ;;
    "backtest")
        echo "📊 Running backtest..."
        run_container backtest
        ;;
    "notebook")
        echo "📓 Starting Jupyter notebook..."
        echo "🌐 Access at: http://localhost:8888"
        docker-compose up -d freqtrade-rl
        sleep 3
        docker-compose exec freqtrade-rl /usr/local/bin/entrypoint.sh notebook
        ;;
    "shell")
        echo "🖥️ Starting interactive shell..."
        run_container shell
        ;;
    "logs")
        echo "📝 Showing logs..."
        docker-compose logs -f
        ;;
    "stop")
        echo "🛑 Stopping container..."
        docker-compose down
        ;;
    "clean")
        echo "🧹 Cleaning up..."
        docker-compose down -v
        docker system prune -f
        ;;
    *)
        echo "Usage: $0 {build|download|train|backtest|notebook|shell|logs|stop|clean}"
        echo ""
        echo "Commands:"
        echo "  build    - Build Docker image only"
        echo "  download - Download trading data"
        echo "  train    - Train RL model"
        echo "  backtest - Run backtest"
        echo "  notebook - Start Jupyter notebook (http://localhost:8888)"
        echo "  shell    - Start interactive shell"
        echo "  logs     - Show container logs"
        echo "  stop     - Stop container"
        echo "  clean    - Clean up containers and images"
        echo ""
        echo "Example workflow:"
        echo "  $0 build"
        echo "  $0 download"
        echo "  $0 train"
        echo "  $0 backtest"
        exit 1
        ;;
esac

echo "✅ Command completed successfully!"