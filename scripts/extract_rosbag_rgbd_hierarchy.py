#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
from cv_bridge import CvBridge

from extract_rosbag_rgbd import convert_depth
from extract_rosbag_rgbd import convert_rgb
from extract_rosbag_rgbd import load_messages
from extract_rosbag_rgbd import pair_by_nearest


def export_pairs(pairs, output_dir: Path):
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    bridge = CvBridge()
    for index, (_, rgb_sync_ts, rgb_msg, _, _, depth_msg, _) in enumerate(pairs):
        rgb_image = convert_rgb(rgb_msg, bridge)
        depth_image = convert_depth(depth_msg, bridge)
        stem = f"{index:06d}_{rgb_sync_ts}"
        cv2.imwrite(str(rgb_dir / f"{stem}.png"), rgb_image)
        cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_image)


def main():
    parser = argparse.ArgumentParser(
        description="Extract a contiguous RGB-D segment from a ROS2 bag and derive a stride-based child subset"
    )
    parser.add_argument("--bag", required=True, help="Path to ros2 bag directory")
    parser.add_argument("--rgb-topic", required=True, help="RGB image topic")
    parser.add_argument("--depth-topic", required=True, help="Depth image topic")
    parser.add_argument("--output-root", required=True, help="Root output directory for parent/child subsets")
    parser.add_argument("--parent-name", default="half_segment", help="Parent segment directory name")
    parser.add_argument("--child-name", default="keyframes_stride3", help="Child subset directory name under parent")
    parser.add_argument("--start-ratio", type=float, default=0.0, help="Inclusive start ratio in [0,1)")
    parser.add_argument("--end-ratio", type=float, default=0.5, help="Exclusive end ratio in (0,1]")
    parser.add_argument("--start-index", type=int, default=None, help="Optional inclusive start index in aligned pairs")
    parser.add_argument("--end-index", type=int, default=None, help="Optional exclusive end index in aligned pairs")
    parser.add_argument("--child-stride", type=int, default=3, help="Keep one frame every N frames within the parent segment")
    parser.add_argument("--tolerance-ms", type=float, default=50.0, help="Max timestamp difference for RGB/depth pairing")
    args = parser.parse_args()

    if not (0.0 <= args.start_ratio < args.end_ratio <= 1.0):
        raise ValueError("Require 0.0 <= start-ratio < end-ratio <= 1.0")
    if args.child_stride < 1:
        raise ValueError("child-stride must be >= 1")

    bag_path = Path(args.bag)
    output_root = Path(args.output_root)
    parent_dir = output_root / args.parent_name
    child_dir = parent_dir / args.child_name

    rgb_items = load_messages(bag_path, args.rgb_topic)
    depth_items = load_messages(bag_path, args.depth_topic)
    aligned_pairs = pair_by_nearest(rgb_items, depth_items, int(args.tolerance_ms * 1e6))

    if not aligned_pairs:
        raise RuntimeError("No aligned RGB/depth pairs found")

    total_pairs = len(aligned_pairs)
    if args.start_index is not None or args.end_index is not None:
        start_index = 0 if args.start_index is None else args.start_index
        end_index = total_pairs if args.end_index is None else args.end_index
    else:
        start_index = int(total_pairs * args.start_ratio)
        end_index = int(total_pairs * args.end_ratio)

    if not (0 <= start_index < end_index <= total_pairs):
        raise ValueError(f"Require 0 <= start-index < end-index <= {total_pairs}, got {start_index}..{end_index}")

    parent_pairs = aligned_pairs[start_index:end_index]
    child_pairs = parent_pairs[:: args.child_stride]

    if not parent_pairs:
        raise RuntimeError("Parent segment is empty")
    if not child_pairs:
        raise RuntimeError("Child subset is empty")

    export_pairs(parent_pairs, parent_dir)
    export_pairs(child_pairs, child_dir)

    manifest = {
        "bag": str(bag_path),
        "rgb_topic": args.rgb_topic,
        "depth_topic": args.depth_topic,
        "aligned_pairs_total": total_pairs,
        "parent_name": args.parent_name,
        "child_name": args.child_name,
        "start_ratio": args.start_ratio,
        "end_ratio": args.end_ratio,
        "start_index_arg": args.start_index,
        "end_index_arg": args.end_index,
        "parent_start_index": start_index,
        "parent_end_index": end_index,
        "parent_count": len(parent_pairs),
        "child_stride": args.child_stride,
        "child_count": len(child_pairs),
        "pair_tolerance_ms": args.tolerance_ms,
        "note": "Child subset is created from the parent contiguous segment by keeping every Nth frame.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
