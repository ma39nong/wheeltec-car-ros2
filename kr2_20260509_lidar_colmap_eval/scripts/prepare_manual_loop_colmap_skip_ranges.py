#!/usr/bin/env python3
"""Prepare a manual first-loop segment with skipped source-frame ranges."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.glob("*.png") if p.is_file())


def parse_skip_ranges(values: list[str]) -> list[tuple[int, int]]:
    ranges = []
    for value in values:
        start_text, end_text = value.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        if start > end:
            raise ValueError(f"Invalid skip range: {value}")
        ranges.append((start, end))
    return ranges


def is_skipped(index: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= index <= end for start, end in ranges)


def copy_pairs(rgb_items: list[Path], depth_items: list[Path], indices: list[int], output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    for out_idx, src_idx in enumerate(indices):
        rgb_src = rgb_items[src_idx]
        depth_src = depth_items[src_idx]
        timestamp = rgb_src.stem.split("_")[-1]
        stem = f"{out_idx:06d}_{timestamp}"
        shutil.copy2(rgb_src, rgb_dir / f"{stem}.png")
        shutil.copy2(depth_src, depth_dir / f"{stem}.png")


def env_text(work_root: str, label: str, image_dir: Path, stride: int, overlap: int) -> str:
    run_dir = f"{work_root}/runs/sift_colmap_{label}_overlap{overlap}"
    return f"""SCENE_NAME=scene_20260509_sift_colmap_{label}_overlap{overlap}
IMAGE_DIR={image_dir}

DB_PATH={run_dir}/workspace/colmap/database.db
SPARSE_DIR={run_dir}/workspace/colmap/sparse
SPARSE_TXT_DIR={run_dir}/workspace/colmap/sparse_txt
TRAJECTORY_PATH={run_dir}/results/trajectory_plots/trajectory.tum
METRICS_CSV={run_dir}/results/reports/experiment_metrics.csv
SCREENSHOT_DIR={run_dir}/results/screenshots
LOG_DIR={run_dir}/results/logs
RESOURCE_DIR={run_dir}/results/resource_logs

SINGLE_CAMERA=1
SIFT_USE_GPU=0
MATCH_USE_GPU=0
SIFT_MAX_NUM_FEATURES=8192
SIFT_ESTIMATE_AFFINE_SHAPE=1
SIFT_DOMAIN_SIZE_POOLING=1
GUIDED_MATCHING=1
SEQUENTIAL_OVERLAP={overlap}
LOOP_DETECTION=0

MAPPER_INIT_MIN_NUM_INLIERS=20
MAPPER_ABS_POSE_MIN_NUM_INLIERS=15
MAPPER_MIN_MODEL_SIZE=3
MAPPER_MULTIPLE_MODELS=0

CHANGED_VARIABLE=manual_first_loop_filter
VARIABLE_VALUE={label}_stride{stride}_overlap{overlap}
RUN_NOTES="2026-05-09 manual first-loop segment with low-motion ranges removed before stride={stride} keyframe selection"
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", default="kr2_20260509_lidar_colmap_eval")
    parser.add_argument("--input-dir", default="kr2_20260509_lidar_colmap_eval/data/extracted/full_rgbd", type=Path)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--end-index", required=True, type=int, help="Inclusive end index")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--skip-range", action="append", default=[], help="Original source range START:END_INCLUSIVE")
    args = parser.parse_args()

    rgb_items = list_images(args.input_dir / "rgb")
    depth_items = list_images(args.input_dir / "depth")
    if len(rgb_items) != len(depth_items):
        raise ValueError(f"RGB/depth count mismatch: {len(rgb_items)} vs {len(depth_items)}")
    if not (0 <= args.start_index <= args.end_index < len(rgb_items)):
        raise ValueError(f"Require 0 <= start <= end < {len(rgb_items)}")

    skip_ranges = parse_skip_ranges(args.skip_range)
    segment_indices = list(range(args.start_index, args.end_index + 1))
    kept_indices = [idx for idx in segment_indices if not is_skipped(idx, skip_ranges)]
    keyframe_indices = kept_indices[:: args.stride]

    skip_label = "_skip" + "-".join(f"{a}-{b}" for a, b in skip_ranges) if skip_ranges else ""
    label = f"manual_loop_{args.start_index:06d}_{args.end_index:06d}_stride{args.stride}{skip_label}"
    segment_dir = Path(args.work_root) / "data" / "visual_loop" / f"rgbd_{args.start_index:06d}_{args.end_index:06d}_manual{skip_label}"
    keyframe_dir = Path(args.work_root) / "data" / "keyframes" / label
    config_path = Path(args.work_root) / "configs" / f"sift_colmap_{label}_overlap{args.overlap}.env"

    copy_pairs(rgb_items, depth_items, kept_indices, segment_dir)
    copy_pairs(rgb_items, depth_items, keyframe_indices, keyframe_dir)

    manifest = {
        "input_dir": str(args.input_dir),
        "segment_dir": str(segment_dir),
        "keyframe_dir": str(keyframe_dir),
        "start_index": args.start_index,
        "end_index_inclusive": args.end_index,
        "skip_ranges_inclusive": skip_ranges,
        "original_segment_count": len(segment_indices),
        "kept_segment_count": len(kept_indices),
        "stride": args.stride,
        "keyframe_count": len(keyframe_indices),
        "first_source_rgb": rgb_items[kept_indices[0]].name,
        "last_source_rgb": rgb_items[kept_indices[-1]].name,
        "colmap_config": str(config_path),
    }
    (segment_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (keyframe_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(env_text(args.work_root, label, keyframe_dir / "rgb", args.stride, args.overlap), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
