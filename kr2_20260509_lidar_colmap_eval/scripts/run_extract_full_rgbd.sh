#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-kr2_20260509_lidar_colmap_eval/configs/scene_20260509.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] config file not found: $CONFIG_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

if [[ ! -d "$BAG_DIR" ]]; then
  echo "[ERROR] bag directory not found: $BAG_DIR"
  exit 1
fi

mkdir -p "$EXTRACTED_DIR"

echo "scene_name=$SCENE_NAME"
echo "bag_dir=$BAG_DIR"
echo "rgb_topic=$RGB_TOPIC"
echo "depth_topic=$DEPTH_TOPIC"
echo "output_dir=$EXTRACTED_DIR"
echo "pair_tolerance_ms=$PAIR_TOLERANCE_MS"

python3 scripts/extract_rosbag_rgbd.py \
  --bag "$BAG_DIR" \
  --rgb-topic "$RGB_TOPIC" \
  --depth-topic "$DEPTH_TOPIC" \
  --output-dir "$EXTRACTED_DIR" \
  --target-count 0 \
  --tolerance-ms "$PAIR_TOLERANCE_MS"

echo "Done."
echo "RGB:   $EXTRACTED_DIR/rgb"
echo "Depth: $EXTRACTED_DIR/depth"
echo "Manifest: $EXTRACTED_DIR/manifest.json"
