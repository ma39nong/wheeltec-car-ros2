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

LOG_DIR="${LOG_DIR:-results/logs}"
RESOURCE_DIR="${RESOURCE_DIR:-results/resource_logs}"
RESOURCE_SAMPLE_INTERVAL_SEC="${RESOURCE_SAMPLE_INTERVAL_SEC:-1}"
CHANGED_VARIABLE="${CHANGED_VARIABLE:-none}"
VARIABLE_VALUE="${VARIABLE_VALUE:-baseline}"
RUN_NOTES="${RUN_NOTES:-}"

if ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] colmap not found in PATH"
  exit 1
fi

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "[ERROR] image directory not found: $IMAGE_DIR"
  exit 1
fi

image_count=$(find "$IMAGE_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ')
if [[ "$image_count" -lt 20 ]]; then
  echo "[WARN] only $image_count images found; reconstruction may be unstable"
fi

mkdir -p "$(dirname "$DB_PATH")" "$SPARSE_DIR" "$SPARSE_TXT_DIR" "$LOG_DIR" "$RESOURCE_DIR"
rm -f "$DB_PATH"
rm -rf "$SPARSE_DIR"/* "$SPARSE_TXT_DIR"/*

run_id="$(date +%Y%m%d_%H%M%S)"
start_ts="$(date +%s)"
log_path="$LOG_DIR/run_colmap_${run_id}.log"
resource_run_dir="$RESOURCE_DIR/run_${run_id}"
mkdir -p "$resource_run_dir"

exec > >(tee "$log_path") 2>&1

monitor_pid=""
cleanup() {
  local exit_code=$?
  if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
  if [[ -d "$resource_run_dir" ]]; then
    python3 scripts/summarize_resource_logs.py --resource-dir "$resource_run_dir" || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT

bash scripts/monitor_resources.sh "$$" "$resource_run_dir" "$RESOURCE_SAMPLE_INTERVAL_SEC" >/dev/null 2>&1 &
monitor_pid=$!

echo "run_id=$run_id"
echo "scene_name=$SCENE_NAME"
echo "image_dir=$IMAGE_DIR"
echo "image_count=$image_count"
echo "resource_log_dir=$resource_run_dir"

echo "[1/4] Feature extraction"
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$IMAGE_DIR" \
  --ImageReader.single_camera "$SINGLE_CAMERA" \
  --SiftExtraction.use_gpu "$SIFT_USE_GPU" \
  --SiftExtraction.max_num_features "$SIFT_MAX_NUM_FEATURES" \
  --SiftExtraction.estimate_affine_shape "$SIFT_ESTIMATE_AFFINE_SHAPE" \
  --SiftExtraction.domain_size_pooling "$SIFT_DOMAIN_SIZE_POOLING"

echo "[2/4] Sequential matching"
colmap sequential_matcher \
  --database_path "$DB_PATH" \
  --SiftMatching.use_gpu "$MATCH_USE_GPU" \
  --SiftMatching.guided_matching "$GUIDED_MATCHING" \
  --SequentialMatching.overlap "$SEQUENTIAL_OVERLAP" \
  --SequentialMatching.loop_detection "$LOOP_DETECTION"

echo "[3/4] Sparse mapping"
mapper_init_args=()
if [[ -n "${MAPPER_INIT_IMAGE_ID1:-}" ]] && [[ -n "${MAPPER_INIT_IMAGE_ID2:-}" ]]; then
  mapper_init_args+=(--Mapper.init_image_id1 "$MAPPER_INIT_IMAGE_ID1")
  mapper_init_args+=(--Mapper.init_image_id2 "$MAPPER_INIT_IMAGE_ID2")
  echo "forced_init_image_pair=$MAPPER_INIT_IMAGE_ID1,$MAPPER_INIT_IMAGE_ID2"
fi

colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$IMAGE_DIR" \
  --output_path "$SPARSE_DIR" \
  --Mapper.init_min_num_inliers "$MAPPER_INIT_MIN_NUM_INLIERS" \
  --Mapper.abs_pose_min_num_inliers "$MAPPER_ABS_POSE_MIN_NUM_INLIERS" \
  --Mapper.min_model_size "$MAPPER_MIN_MODEL_SIZE" \
  --Mapper.multiple_models "$MAPPER_MULTIPLE_MODELS" \
  "${mapper_init_args[@]}"

if [[ ! -d "$SPARSE_DIR/0" ]]; then
  echo "[ERROR] sparse model not found at $SPARSE_DIR/0"
  exit 1
fi

echo "[4/4] Export sparse model to TXT"
colmap model_converter \
  --input_path "$SPARSE_DIR/0" \
  --output_path "$SPARSE_TXT_DIR" \
  --output_type TXT

end_ts="$(date +%s)"
duration_sec=$((end_ts - start_ts))

echo "duration_sec=$duration_sec"
echo "sparse_model=$SPARSE_DIR/0"
echo "sparse_txt=$SPARSE_TXT_DIR"
echo "log_path=$log_path"
echo "resource_log_dir=$resource_run_dir"

python3 scripts/summarize_sparse_model.py \
  --images-txt "$SPARSE_TXT_DIR/images.txt" \
  --points3d-txt "$SPARSE_TXT_DIR/points3D.txt" \
  --total-images "$image_count" \
  --duration-sec "$duration_sec" \
  --scene-name "$SCENE_NAME" \
  --changed-variable "$CHANGED_VARIABLE" \
  --variable-value "$VARIABLE_VALUE" \
  --notes "$RUN_NOTES" \
  --output "$METRICS_CSV"

echo "Done. Next steps:"
echo "1) python3 scripts/colmap_images_to_tum.py --input $SPARSE_TXT_DIR/images.txt --output $TRAJECTORY_PATH"
echo "2) Save one screenshot under $SCREENSHOT_DIR"
echo "3) bash scripts/export_artifacts.sh"
