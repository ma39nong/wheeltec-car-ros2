#!/usr/bin/env python3
import argparse
import json
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


def message_header_stamp_ns(msg):
    if not hasattr(msg, "header"):
        return None
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def load_messages(bag_path: Path, topic_name: str):
    storage_options = StorageOptions(uri=str(bag_path), storage_id="sqlite3")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic_name not in topics:
        available = ", ".join(sorted(topics))
        raise RuntimeError(f"Topic not found: {topic_name}. Available topics: {available}")

    msg_type = get_message(topics[topic_name])
    result = []

    while reader.has_next():
        current_topic, data, timestamp = reader.read_next()
        if current_topic != topic_name:
            continue
        msg = deserialize_message(data, msg_type)
        header_ts = message_header_stamp_ns(msg)
        result.append((timestamp, header_ts, msg))

    return result


def pair_by_nearest(rgb_items, depth_items, tolerance_ns):
    depth_sync_times = [item[1] if item[1] is not None else item[0] for item in depth_items]
    pairs = []
    used_depth_indices = set()

    for rgb_bag_ts, rgb_header_ts, rgb_msg in rgb_items:
        rgb_sync_ts = rgb_header_ts if rgb_header_ts is not None else rgb_bag_ts
        pos = bisect_left(depth_sync_times, rgb_sync_ts)
        candidates = []
        if pos < len(depth_items):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)

        best_idx = None
        best_delta = None
        for idx in candidates:
            if idx in used_depth_indices:
                continue
            delta = abs(depth_sync_times[idx] - rgb_sync_ts)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = idx

        if best_idx is None or best_delta is None or best_delta > tolerance_ns:
            continue

        used_depth_indices.add(best_idx)
        pairs.append(
            (
                rgb_bag_ts,
                rgb_sync_ts,
                rgb_msg,
                depth_items[best_idx][0],
                depth_sync_times[best_idx],
                depth_items[best_idx][2],
                best_delta,
            )
        )

    return pairs


def uniform_sample(items, target_count):
    if target_count <= 0 or target_count >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=target_count, dtype=int)
    return [items[idx] for idx in indices]


def convert_rgb(msg, bridge: CvBridge):
    image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if msg.encoding.lower() == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding.lower() == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def convert_depth(msg, bridge: CvBridge):
    depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    elif depth.dtype != np.uint16:
        depth = depth.astype(np.uint16)
    return depth


def main():
    parser = argparse.ArgumentParser(description="Extract aligned RGB and depth images from a ROS 2 bag")
    parser.add_argument("--bag", required=True, help="Path to ros2 bag directory")
    parser.add_argument("--rgb-topic", required=True, help="RGB image topic")
    parser.add_argument("--depth-topic", required=True, help="Depth image topic")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument("--target-count", type=int, default=0, help="Desired aligned frame count; 0 means use all aligned pairs")
    parser.add_argument("--tolerance-ms", type=float, default=50.0, help="Max timestamp difference for RGB/depth pairing")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    output_dir = Path(args.output_dir)
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    bridge = CvBridge()
    rgb_items = load_messages(bag_path, args.rgb_topic)
    depth_items = load_messages(bag_path, args.depth_topic)
    pairs = pair_by_nearest(rgb_items, depth_items, int(args.tolerance_ms * 1e6))

    if not pairs:
        raise RuntimeError("No aligned RGB/depth pairs found")

    requested_count = args.target_count if args.target_count > 0 else len(pairs)
    export_count = min(requested_count, len(pairs))
    sampled_pairs = uniform_sample(pairs, export_count)

    for index, (rgb_bag_ts, rgb_sync_ts, rgb_msg, depth_bag_ts, depth_sync_ts, depth_msg, delta_ns) in enumerate(sampled_pairs):
        rgb_image = convert_rgb(rgb_msg, bridge)
        depth_image = convert_depth(depth_msg, bridge)
        stem = f"{index:06d}_{rgb_sync_ts}"
        cv2.imwrite(str(rgb_dir / f"{stem}.png"), rgb_image)
        cv2.imwrite(str(depth_dir / f"{stem}.png"), depth_image)

    manifest = {
        "bag": str(bag_path),
        "rgb_topic": args.rgb_topic,
        "depth_topic": args.depth_topic,
        "rgb_messages": len(rgb_items),
        "depth_messages": len(depth_items),
        "aligned_pairs": len(pairs),
        "requested_count": requested_count,
        "exported_count": export_count,
        "pair_tolerance_ms": args.tolerance_ms,
        "note": "requested_count is capped at aligned_pairs when the bag contains fewer synchronized frames.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
