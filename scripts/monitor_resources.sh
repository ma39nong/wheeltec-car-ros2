#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <root_pid> <output_dir> [interval_sec]" >&2
  exit 1
fi

ROOT_PID="$1"
OUTPUT_DIR="$2"
INTERVAL_SEC="${3:-1}"

mkdir -p "$OUTPUT_DIR"

SYSTEM_CSV="$OUTPUT_DIR/system_usage.csv"
PROCESS_CSV="$OUTPUT_DIR/process_usage.csv"
GPU_CSV="$OUTPUT_DIR/gpu_usage.csv"
META_TXT="$OUTPUT_DIR/monitor_meta.txt"

printf 'timestamp,epoch_sec,cpu_percent,mem_total_kb,mem_available_kb,mem_used_kb,swap_total_kb,swap_free_kb,load1,load5,load15\n' > "$SYSTEM_CSV"
printf 'timestamp,epoch_sec,colmap_proc_count,colmap_cpu_percent,colmap_rss_kb,colmap_vsz_kb,colmap_threads\n' > "$PROCESS_CSV"
printf 'timestamp,index,name,utilization_gpu,memory_used_mb,memory_total_mb,temperature_c\n' > "$GPU_CSV"

echo "root_pid=$ROOT_PID" > "$META_TXT"
echo "interval_sec=$INTERVAL_SEC" >> "$META_TXT"
echo "start_time=$(date '+%F %T')" >> "$META_TXT"

collect_descendants() {
  local root="$1"
  local all_children=()
  local queue=("$root")
  local current children child

  while [[ ${#queue[@]} -gt 0 ]]; do
    current="${queue[0]}"
    queue=("${queue[@]:1}")
    mapfile -t children < <(pgrep -P "$current" || true)
    for child in "${children[@]}"; do
      all_children+=("$child")
      queue+=("$child")
    done
  done

  if [[ ${#all_children[@]} -gt 0 ]]; then
    printf '%s\n' "${all_children[@]}"
  fi
}

read_cpu_totals() {
  awk '/^cpu / {total=0; for (i=2; i<=NF; ++i) total+=$i; idle=$5+$6; print total, idle; exit}' /proc/stat
}

prev_cpu=($(read_cpu_totals))
prev_total="${prev_cpu[0]}"
prev_idle="${prev_cpu[1]}"

while kill -0 "$ROOT_PID" 2>/dev/null; do
  timestamp="$(date '+%F %T')"
  epoch_sec="$(date '+%s')"

  current_cpu=($(read_cpu_totals))
  total="${current_cpu[0]}"
  idle="${current_cpu[1]}"
  total_delta=$((total - prev_total))
  idle_delta=$((idle - prev_idle))
  cpu_percent="0.00"
  if [[ "$total_delta" -gt 0 ]]; then
    cpu_percent="$(awk -v td="$total_delta" -v id="$idle_delta" 'BEGIN {printf "%.2f", (td - id) * 100 / td}')"
  fi
  prev_total="$total"
  prev_idle="$idle"

  mem_total_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
  mem_available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  mem_used_kb=$((mem_total_kb - mem_available_kb))
  swap_total_kb="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)"
  swap_free_kb="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)"
  read -r load1 load5 load15 _ < /proc/loadavg
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$timestamp" "$epoch_sec" "$cpu_percent" "$mem_total_kb" "$mem_available_kb" "$mem_used_kb" \
    "$swap_total_kb" "$swap_free_kb" "$load1" "$load5" "$load15" >> "$SYSTEM_CSV"

  mapfile -t descendants < <(collect_descendants "$ROOT_PID")
  colmap_pids=()
  for pid in "${descendants[@]}"; do
    if [[ -z "$pid" ]]; then
      continue
    fi
    # A descendant can disappear between pgrep and ps. Never let that race
    # terminate the monitor loop under `set -euo pipefail`.
    comm="$(ps -p "$pid" -o comm= 2>/dev/null | awk '{print $1}' || true)"
    if [[ "$comm" == "colmap" ]]; then
      colmap_pids+=("$pid")
    fi
  done

  if [[ ${#colmap_pids[@]} -gt 0 ]]; then
    pid_list="$(IFS=,; echo "${colmap_pids[*]}")"
    read -r proc_count proc_cpu proc_rss proc_vsz proc_threads < <(
      ps -p "$pid_list" -o pcpu=,rss=,vsz=,nlwp= 2>/dev/null | \
        awk 'BEGIN {count=0; cpu=0; rss=0; vsz=0; thr=0}
             NF >= 4 {count+=1; cpu+=$1; rss+=$2; vsz+=$3; thr+=$4}
             END {printf "%d %.2f %d %d %d\n", count, cpu, rss, vsz, thr}'
    )
  else
    proc_count=0
    proc_cpu="0.00"
    proc_rss=0
    proc_vsz=0
    proc_threads=0
  fi

  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$timestamp" "$epoch_sec" "$proc_count" "$proc_cpu" "$proc_rss" "$proc_vsz" "$proc_threads" >> "$PROCESS_CSV"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null | \
      awk -F', *' -v ts="$timestamp" 'NF >= 6 {printf "%s,%s,%s,%s,%s,%s,%s\n", ts, $1, $2, $3, $4, $5, $6}' >> "$GPU_CSV" || true
  fi

  sleep "$INTERVAL_SEC"
done

echo "end_time=$(date '+%F %T')" >> "$META_TXT"
