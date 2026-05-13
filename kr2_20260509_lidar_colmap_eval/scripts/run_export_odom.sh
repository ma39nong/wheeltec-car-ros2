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

RGB_DIR="$EXTRACTED_DIR/rgb"
OUTPUT_DIR="$WORK_ROOT/outputs/odom"

if [[ ! -d "$RGB_DIR" ]]; then
  echo "[ERROR] RGB directory not found: $RGB_DIR"
  echo "Run scripts/run_extract_full_rgbd.sh first."
  exit 1
fi

python3 "$WORK_ROOT/scripts/export_odom_loop_candidates.py" \
  --bag "$BAG_DIR" \
  --topic "$ODOM_TOPIC" \
  --rgb-dir "$RGB_DIR" \
  --output-dir "$OUTPUT_DIR"

python3 "$WORK_ROOT/scripts/export_odom_loop_candidates.py" \
  --bag "$BAG_DIR" \
  --topic "$ODOM_COMBINED_TOPIC" \
  --rgb-dir "$RGB_DIR" \
  --output-dir "$OUTPUT_DIR"
