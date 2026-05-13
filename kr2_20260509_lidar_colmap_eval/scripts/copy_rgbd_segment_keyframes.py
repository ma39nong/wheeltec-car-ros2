#!/usr/bin/env python3
"""Copy a contiguous RGB-D segment and a stride-based keyframe subset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.glob("*.png") if p.is_file())


def copy_pairs(rgb_items: list[Path], depth_items: list[Path], indices: list[int], output_dir: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--first-loop-dir", required=True, type=Path)
    parser.add_argument("--keyframe-dir", required=True, type=Path)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--end-index", required=True, type=int, help="Exclusive end index")
    parser.add_argument("--stride", required=True, type=int)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    rgb_items = list_images(args.input_dir / "rgb")
    depth_items = list_images(args.input_dir / "depth")
    if len(rgb_items) != len(depth_items):
        raise ValueError(f"RGB/depth count mismatch: {len(rgb_items)} vs {len(depth_items)}")
    if not (0 <= args.start_index < args.end_index <= len(rgb_items)):
        raise ValueError(f"Require 0 <= start < end <= {len(rgb_items)}")

    segment_indices = list(range(args.start_index, args.end_index))
    keyframe_indices = segment_indices[:: args.stride]

    copy_pairs(rgb_items, depth_items, segment_indices, args.first_loop_dir)
    copy_pairs(rgb_items, depth_items, keyframe_indices, args.keyframe_dir)

    manifest = {
        "input_dir": str(args.input_dir),
        "first_loop_dir": str(args.first_loop_dir),
        "keyframe_dir": str(args.keyframe_dir),
        "start_index": args.start_index,
        "end_index_exclusive": args.end_index,
        "segment_count": len(segment_indices),
        "stride": args.stride,
        "keyframe_count": len(keyframe_indices),
        "first_source_rgb": rgb_items[segment_indices[0]].name,
        "last_source_rgb": rgb_items[segment_indices[-1]].name,
    }
    args.keyframe_dir.parent.mkdir(parents=True, exist_ok=True)
    (args.first_loop_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.keyframe_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
