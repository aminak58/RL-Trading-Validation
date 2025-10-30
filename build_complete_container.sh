#!/bin/bash

# Build Complete RL Trading Container
# This script creates a Docker container with the entire project

set -e

echo "🏗️ Building Complete RL Trading Container"
echo "=========================================="

# Configuration
CONTAINER_NAME="freqtrade-rl-complete"
IMAGE_NAME="freqtrade-rl-complete"
TAG="v1.0"
DOCKERFILE="Dockerfile.complete"

# Check if we're in the right directory
if [ ! -f "$DOCKERFILE" ]; then
    echo "❌ Error: $DOCKERFILE not found in current directory"
    echo "   Please run this script from the project root directory"
    exit 1
fi

echo "📍 Current directory: $(pwd)"
echo "📋 Building with Dockerfile: $DOCKERFILE"
echo "🏷️  Image name: $IMAGE_NAME:$TAG"
echo ""

# Step 1: Analyze project size
echo "📊 Analyzing project size..."
PROJECT_SIZE=$(du -sh . | cut -f1)
DATA_SIZE=$(du -sh user_data/data/ 2>/dev/null | cut -f1 || echo "0")
CODE_SIZE=$(du -sh user_data/ --exclude=data | cut -f1 || echo "0")

echo "   📁 Total project: $PROJECT_SIZE"
echo "   💾 Trading data: $DATA_SIZE"
echo "   📄 Code & configs: $CODE_SIZE"
echo ""

# Step 2: Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "   Please start Docker and try again"
    exit 1
fi

# Step 3: Build the container
echo "🔨 Building Docker container..."
echo "   This may take 10-20 minutes depending on your internet speed"
echo ""

# Build with progress indication
docker build \
    -f $DOCKERFILE \
    -t $IMAGE_NAME:$TAG \
    -t $IMAGE_NAME:latest \
    . 2>&1 | while IFS= read -r line; do
        echo "   $line"
    done

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container built successfully!"
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi

# Step 4: Analyze built container
echo ""
echo "📊 Container Information:"
docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

# Step 5: Test the container
echo ""
echo "🧪 Testing container..."
docker run --rm $IMAGE_NAME:$TAG freqtrade-rl.sh status

# Step 6: Create container info file
echo ""
echo "📋 Creating container information..."
cat > container_info.txt << EOF
Freqtrade RL Trading Container Information
==========================================

Build Date: $(date)
Image Name: $IMAGE_NAME:$TAG
Project Size: $PROJECT_SIZE
Data Size: $DATA_SIZE

Container Contents:
- Complete RL-Trading-Validation project
- All trading data (540 days for BTC, ETH, SOL, DOGE)
- All dependencies (Freqtrade, PyTorch, SB3, etc.)
- Strategy files and configurations

Usage Examples:
1. Interactive shell:
   docker run -it --rm $IMAGE_NAME:$TAG

2. Train RL model:
   docker run --rm $IMAGE_NAME:$TAG freqtrade-rl.sh train

3. Run backtest:
   docker run --rm $IMAGE_NAME:$TAG freqtrade-rl.sh backtest

4. Start analysis notebook:
   docker run -p 8888:8888 --rm $IMAGE_NAME:$TAG freqtrade-rl.sh analyze

5. Check container status:
   docker run --rm $IMAGE_NAME:$TAG freqtrade-rl.sh status

Next Steps:
1. Tag for Docker Hub: docker tag $IMAGE_NAME:$TAG yourusername/$IMAGE_NAME:$TAG
2. Push to Docker Hub: docker push yourusername/$IMAGE_NAME:$TAG
3. Use in Colab with: !docker pull yourusername/$IMAGE_NAME:$TAG
EOF

echo "✅ Container information saved to container_info.txt"
echo ""
echo "🎉 Complete container is ready!"
echo ""
echo "📋 Quick commands:"
echo "   🖥️  Interactive: docker run -it --rm $IMAGE_NAME:$TAG"
echo "   🧠 Train model: docker run --rm $IMAGE_NAME:$TAG freqtrade-rl.sh train"
echo "   📊 Backtest:    docker run --rm $IMAGE_NAME:$TAG freqtrade-rl.sh backtest"
echo "   📋 Status:      docker run --rm $IMAGE_NAME:$TAG freqtrade-rl.sh status"
echo ""
echo "📤 For Docker Hub upload:"
echo "   docker tag $IMAGE_NAME:$TAG yourusername/$IMAGE_NAME:$TAG"
echo "   docker push yourusername/$IMAGE_NAME:$TAG"