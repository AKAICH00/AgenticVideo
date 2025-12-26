#!/bin/bash
###############################################################################
# Build and Push Docker Images for Video-Printer
#
# Usage:
#   ./scripts/build-push.sh                    # Default: ghcr.io/akaich00, v3.0
#   REGISTRY=myregistry.com VERSION=v3.1 ./scripts/build-push.sh
#
# Prerequisites:
#   - Docker running locally
#   - Logged into container registry (docker login)
###############################################################################
set -e

REGISTRY="${REGISTRY:-ghcr.io/akaich00}"
VERSION="${VERSION:-v3.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "============================================================"
echo "Building Video-Printer Images"
echo "Registry: $REGISTRY"
echo "Version:  $VERSION"
echo "============================================================"

# Build agents image (Python services)
echo ""
echo "[1/2] Building video-printer-agents:$VERSION..."
docker build \
    -f Dockerfile.agents \
    -t "$REGISTRY/video-printer-agents:$VERSION" \
    -t "$REGISTRY/video-printer-agents:latest" \
    .

# Build remotion image (Node SSR)
echo ""
echo "[2/2] Building video-printer-remotion:$VERSION..."
docker build \
    -f remotion/Dockerfile \
    -t "$REGISTRY/video-printer-remotion:$VERSION" \
    -t "$REGISTRY/video-printer-remotion:latest" \
    ./remotion

echo ""
echo "============================================================"
echo "Pushing Images to $REGISTRY"
echo "============================================================"

echo ""
echo "Pushing video-printer-agents..."
docker push "$REGISTRY/video-printer-agents:$VERSION"
docker push "$REGISTRY/video-printer-agents:latest"

echo ""
echo "Pushing video-printer-remotion..."
docker push "$REGISTRY/video-printer-remotion:$VERSION"
docker push "$REGISTRY/video-printer-remotion:latest"

echo ""
echo "============================================================"
echo "SUCCESS! Images pushed:"
echo "  - $REGISTRY/video-printer-agents:$VERSION"
echo "  - $REGISTRY/video-printer-remotion:$VERSION"
echo "============================================================"
