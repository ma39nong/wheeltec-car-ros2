#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${1:-kr2_learned_descriptors_phase2/configs/colmap_lightglue.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[ERROR] config file not found: $CONFIG_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

if ! command -v colmap >/dev/null 2>&1; then
  echo "[ERROR] colmap not found in PATH"
  exit 1
fi

is_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

safe_tag() {
  printf '%s' "$1" | tr -cs '[:alnum:]_-' '_'
}

archive_previous_run() {
  if [[ ! -d "$RUN_DIR" ]]; then
    return 0
  fi
  if ! find "$RUN_DIR" -mindepth 1 -print -quit | grep -q .; then
    return 0
  fi

  local archive_root="${RUN_DIR}_archives"
  local timestamp
  local tag
  timestamp="$(date +%Y%m%d_%H%M%S)"
  tag="$(safe_tag "${timestamp}_strides${PAIR_STRIDES:-unknown}_kp${MAX_KEYPOINTS:-unknown}_ransac${RANSAC_REPROJ_THRESHOLD:-unknown}")"
  mkdir -p "$archive_root"
  cp -a "$RUN_DIR" "$archive_root/$tag"
  echo "archived_previous_run=$archive_root/$tag"
}

if is_enabled "${ARCHIVE_PREVIOUS_RUNS:-1}"; then
  archive_previous_run
fi

mkdir -p "$SPARSE_DIR" "$SPARSE_TXT_DIR" "$(dirname "$TRAJECTORY_PATH")" "$(dirname "$METRICS_CSV")" "$LOG_DIR"
rm -rf "$SPARSE_DIR"/* "$SPARSE_TXT_DIR"/*

run_id="$(date +%Y%m%d_%H%M%S)"
start_ts="$(date +%s)"
log_path="$LOG_DIR/run_lightglue_mapper_${run_id}.log"

exec > >(tee "$log_path") 2>&1

echo "run_id=$run_id"
echo "scene_name=$SCENE_NAME"
echo "database=$DB_PATH"
echo "image_dir=$IMAGE_DIR"

echo "[1/5] Build COLMAP database from learned features and matches"
python3 kr2_learned_descriptors_phase2/scripts/build_colmap_database_from_lightglue.py \
  --config "$CONFIG_FILE"

echo "[2/5] Import LightGlue matches and run COLMAP geometry verification"
colmap matches_importer \
  --database_path "$DB_PATH" \
  --match_list_path "$MATCH_LIST_PATH" \
  --match_type raw \
  --SiftMatching.use_gpu 0 \
  --SiftMatching.min_num_inliers "$MIN_INLIERS_FOR_DATABASE" \
  --SiftMatching.max_error "$GEOMETRY_MAX_ERROR" \
  --SiftMatching.guided_matching "$GEOMETRY_GUIDED_MATCHING"

echo "[3/5] Sparse mapping"
mapper_args=(
  mapper
  --database_path "$DB_PATH"
  --image_path "$IMAGE_DIR"
  --output_path "$SPARSE_DIR"
  --Mapper.init_min_num_inliers "$MAPPER_INIT_MIN_NUM_INLIERS"
  --Mapper.abs_pose_min_num_inliers "$MAPPER_ABS_POSE_MIN_NUM_INLIERS"
  --Mapper.min_model_size "$MAPPER_MIN_MODEL_SIZE"
  --Mapper.multiple_models "$MAPPER_MULTIPLE_MODELS"
)

if [[ "${MAPPER_FIX_INTRINSICS:-0}" == "1" ]]; then
  mapper_args+=(
    --Mapper.ba_refine_focal_length 0
    --Mapper.ba_refine_principal_point 0
    --Mapper.ba_refine_extra_params 0
  )
fi

colmap "${mapper_args[@]}"

if [[ ! -d "$SPARSE_DIR/0" ]]; then
  echo "[ERROR] sparse model not found at $SPARSE_DIR/0"
  exit 1
fi

echo "[4/5] Export sparse model to TXT"
colmap model_converter \
  --input_path "$SPARSE_DIR/0" \
  --output_path "$SPARSE_TXT_DIR" \
  --output_type TXT

echo "[5/5] Export TUM trajectory and metrics"
python3 scripts/colmap_images_to_tum.py \
  --input "$SPARSE_TXT_DIR/images.txt" \
  --output "$TRAJECTORY_PATH"

end_ts="$(date +%s)"
duration_sec=$((end_ts - start_ts))
total_images=$(find "$IMAGE_DIR" -maxdepth 1 -type f | wc -l | tr -d ' ')

python3 scripts/summarize_sparse_model.py \
  --images-txt "$SPARSE_TXT_DIR/images.txt" \
  --points3d-txt "$SPARSE_TXT_DIR/points3D.txt" \
  --total-images "$total_images" \
  --duration-sec "$duration_sec" \
  --scene-name "$SCENE_NAME" \
  --changed-variable "frontend" \
  --variable-value "superpoint_lightglue" \
  --notes "COLMAP mapper with imported LightGlue raw matches and fixed baseline intrinsics" \
  --output "$METRICS_CSV"

echo "duration_sec=$duration_sec"
echo "sparse_model=$SPARSE_DIR/0"
echo "sparse_txt=$SPARSE_TXT_DIR"
echo "trajectory=$TRAJECTORY_PATH"
echo "metrics=$METRICS_CSV"
echo "log_path=$log_path"
