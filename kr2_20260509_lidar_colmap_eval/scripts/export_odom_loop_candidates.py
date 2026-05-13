#!/usr/bin/env python3
"""Export odometry from a ROS2 bag and suggest first-loop end candidates."""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_left
from pathlib import Path

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


def stamp_ns(msg, fallback: int) -> int:
    if hasattr(msg, "header"):
        stamp = msg.header.stamp
        value = stamp.sec * 10**9 + stamp.nanosec
        if value > 0:
            return value
    return fallback


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def load_odom(bag_dir: Path, topic_name: str) -> list[dict[str, float | int]]:
    storage_options = StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic_name not in topics:
        available = ", ".join(sorted(topics))
        raise RuntimeError(f"Topic not found: {topic_name}. Available topics: {available}")
    msg_type = get_message(topics[topic_name])

    rows: list[dict[str, float | int]] = []
    while reader.has_next():
        topic, data, bag_ts = reader.read_next()
        if topic != topic_name:
            continue
        msg = deserialize_message(data, msg_type)
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        ts = stamp_ns(msg, bag_ts)
        rows.append(
            {
                "timestamp_ns": ts,
                "bag_timestamp_ns": bag_ts,
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z),
                "qx": float(ori.x),
                "qy": float(ori.y),
                "qz": float(ori.z),
                "qw": float(ori.w),
                "yaw": yaw_from_quaternion(float(ori.x), float(ori.y), float(ori.z), float(ori.w)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def cumulative_path_lengths(rows: list[dict[str, float | int]]) -> list[float]:
    lengths = [0.0]
    for prev, cur in zip(rows, rows[1:]):
        dx = float(cur["x"]) - float(prev["x"])
        dy = float(cur["y"]) - float(prev["y"])
        lengths.append(lengths[-1] + math.hypot(dx, dy))
    return lengths


def nearest_index(timestamps: list[int], target: int) -> int:
    pos = bisect_left(timestamps, target)
    candidates = []
    if pos < len(timestamps):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    return min(candidates, key=lambda idx: abs(timestamps[idx] - target))


def list_rgb_timestamps(rgb_dir: Path) -> list[int]:
    timestamps = []
    for path in sorted(rgb_dir.glob("*.png")):
        try:
            timestamps.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    if not timestamps:
        raise FileNotFoundError(f"No timestamped RGB png files found in {rgb_dir}")
    return timestamps


def suggest_loop_candidates(
    rows: list[dict[str, float | int]],
    rgb_timestamps: list[int],
    min_elapsed_sec: float,
    min_path_m: float,
    return_radius_m: float,
    max_candidates: int,
) -> list[dict[str, float | int]]:
    start = rows[0]
    start_x = float(start["x"])
    start_y = float(start["y"])
    start_ts = int(start["timestamp_ns"])
    lengths = cumulative_path_lengths(rows)
    odom_timestamps = [int(row["timestamp_ns"]) for row in rows]

    candidates = []
    for idx, row in enumerate(rows):
        elapsed = (int(row["timestamp_ns"]) - start_ts) / 1e9
        if elapsed < min_elapsed_sec or lengths[idx] < min_path_m:
            continue
        distance_to_start = math.hypot(float(row["x"]) - start_x, float(row["y"]) - start_y)
        if distance_to_start > return_radius_m:
            continue
        rgb_idx = nearest_index(rgb_timestamps, int(row["timestamp_ns"]))
        candidates.append(
            {
                "odom_index": idx,
                "timestamp_ns": int(row["timestamp_ns"]),
                "elapsed_sec": round(elapsed, 3),
                "path_length_m": round(lengths[idx], 3),
                "distance_to_start_m": round(distance_to_start, 3),
                "nearest_rgb_index": rgb_idx,
                "nearest_rgb_timestamp_ns": rgb_timestamps[rgb_idx],
                "rgb_time_delta_ms": round(abs(rgb_timestamps[rgb_idx] - int(row["timestamp_ns"])) / 1e6, 3),
            }
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--topic", default="/odom_combined")
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-elapsed-sec", type=float, default=40.0)
    parser.add_argument("--min-path-m", type=float, default=4.0)
    parser.add_argument("--return-radius-m", type=float, default=0.8)
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args()

    rows = load_odom(args.bag, args.topic)
    if not rows:
        raise RuntimeError(f"No odometry messages found on {args.topic}")

    rgb_timestamps = list_rgb_timestamps(args.rgb_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    odom_path = args.output_dir / f"{args.topic.strip('/').replace('/', '_')}.csv"
    candidates_path = args.output_dir / f"{args.topic.strip('/').replace('/', '_')}_loop_candidates.csv"

    write_csv(odom_path, rows)
    candidates = suggest_loop_candidates(
        rows,
        rgb_timestamps,
        args.min_elapsed_sec,
        args.min_path_m,
        args.return_radius_m,
        args.max_candidates,
    )
    if candidates:
        write_csv(candidates_path, candidates)
    else:
        candidates_path.write_text("no candidates found\n", encoding="utf-8")

    lengths = cumulative_path_lengths(rows)
    print(f"topic: {args.topic}")
    print(f"odom_messages: {len(rows)}")
    print(f"path_length_m: {lengths[-1]:.3f}")
    print(f"wrote: {odom_path}")
    print(f"candidates: {len(candidates)}")
    print(f"wrote: {candidates_path}")


if __name__ == "__main__":
    main()
