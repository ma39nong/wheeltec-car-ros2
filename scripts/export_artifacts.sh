#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/configs/experiment.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] config file not found: $CONFIG_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/results/exports/export_$TS"
ZIP_PATH="$ROOT_DIR/results/exports/wheeltec_colmap_export_$TS.zip"

mkdir -p "$OUT_DIR"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -e "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -r "$src" "$dst"
    echo "[OK] copied: $src"
  else
    echo "[WARN] missing: $src"
  fi
}

copy_if_exists "$SPARSE_DIR/0" "$OUT_DIR/workspace/colmap/sparse/0"
copy_if_exists "$SPARSE_TXT_DIR" "$OUT_DIR/workspace/colmap/sparse_txt"
copy_if_exists "$TRAJECTORY_PATH" "$OUT_DIR/results/trajectory_plots/trajectory.tum"
copy_if_exists "$SCREENSHOT_DIR" "$OUT_DIR/results/screenshots"
copy_if_exists "$METRICS_CSV" "$OUT_DIR/results/reports/experiment_metrics.csv"
copy_if_exists "configs/experiment.env" "$OUT_DIR/configs/experiment.env"
copy_if_exists "scripts/run_colmap_stable.sh" "$OUT_DIR/scripts/run_colmap_stable.sh"
copy_if_exists "scripts/colmap_images_to_tum.py" "$OUT_DIR/scripts/colmap_images_to_tum.py"
copy_if_exists "scripts/summarize_sparse_model.py" "$OUT_DIR/scripts/summarize_sparse_model.py"
copy_if_exists "README.md" "$OUT_DIR/README.md"

cat > "$OUT_DIR/MANIFEST.txt" << EOF
Export time: $TS
Workspace: $ROOT_DIR
Scene: $SCENE_NAME

Included (if available):
- workspace/colmap/sparse/0
- workspace/colmap/sparse_txt
- results/trajectory_plots/trajectory.tum
- results/screenshots
- results/reports/experiment_metrics.csv
- configs/experiment.env
- scripts/run_colmap_stable.sh
- scripts/colmap_images_to_tum.py
- scripts/summarize_sparse_model.py
EOF

(
  cd "$ROOT_DIR/results/exports"
  zip -r "$ZIP_PATH" "export_$TS" >/dev/null
)

echo "Export directory: $OUT_DIR"
echo "Zip package: $ZIP_PATH"
