#!/usr/bin/env python3
"""Export a 2D map-frame robot trajectory from ROS2 TF messages."""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_left
from pathlib import Path

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


def stamp_ns(transform, fallback: int) -> int:
    stamp = transform.header.stamp
    value = stamp.sec * 10**9 + stamp.nanosec
    return value if value > 0 else fallback


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def read_tf_pairs(
    bag_dir: Path,
    map_frame: str,
    odom_frame: str,
    base_frame: str,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if "/tf" not in topics:
        raise RuntimeError("/tf topic not found in bag")
    msg_type = get_message(topics["/tf"])

    map_to_odom = []
    odom_to_base = []
    while reader.has_next():
        topic, data, bag_ts = reader.read_next()
        if topic != "/tf":
            continue
        msg = deserialize_message(data, msg_type)
        for transform in msg.transforms:
            parent = transform.header.frame_id
            child = transform.child_frame_id
            trans = transform.transform.translation
            rot = transform.transform.rotation
            row = {
                "timestamp_ns": stamp_ns(transform, bag_ts),
                "x": float(trans.x),
                "y": float(trans.y),
                "yaw": yaw_from_quaternion(float(rot.x), float(rot.y), float(rot.z), float(rot.w)),
            }
            if parent == map_frame and child == odom_frame:
                map_to_odom.append(row)
            elif parent == odom_frame and child == base_frame:
                odom_to_base.append(row)

    map_to_odom.sort(key=lambda row: int(row["timestamp_ns"]))
    odom_to_base.sort(key=lambda row: int(row["timestamp_ns"]))
    if not map_to_odom:
        raise RuntimeError(f"No TF found: {map_frame} -> {odom_frame}")
    if not odom_to_base:
        raise RuntimeError(f"No TF found: {odom_frame} -> {base_frame}")
    return map_to_odom, odom_to_base


def nearest(rows: list[dict[str, float | int]], timestamp_ns: int) -> dict[str, float | int]:
    timestamps = [int(row["timestamp_ns"]) for row in rows]
    idx = bisect_left(timestamps, timestamp_ns)
    candidates = []
    if idx < len(rows):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    return rows[min(candidates, key=lambda item: abs(timestamps[item] - timestamp_ns))]


def compose_2d(parent_child: dict[str, float | int], child_base: dict[str, float | int]) -> tuple[float, float, float]:
    px = float(parent_child["x"])
    py = float(parent_child["y"])
    pyaw = float(parent_child["yaw"])
    cx = float(child_base["x"])
    cy = float(child_base["y"])
    cyaw = float(child_base["yaw"])
    cos_yaw = math.cos(pyaw)
    sin_yaw = math.sin(pyaw)
    x = px + cos_yaw * cx - sin_yaw * cy
    y = py + sin_yaw * cx + cos_yaw * cy
    yaw = math.atan2(math.sin(pyaw + cyaw), math.cos(pyaw + cyaw))
    return x, y, yaw


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def path_length(rows: list[dict[str, float | int]]) -> float:
    total = 0.0
    for prev, cur in zip(rows, rows[1:]):
        total += math.hypot(float(cur["x"]) - float(prev["x"]), float(cur["y"]) - float(prev["y"]))
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom_combined")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--max-dt-sec", type=float, default=0.25)
    args = parser.parse_args()

    map_to_odom, odom_to_base = read_tf_pairs(args.bag, args.map_frame, args.odom_frame, args.base_frame)
    rows = []
    max_dt_ns = int(args.max_dt_sec * 1e9)
    for base_row in odom_to_base:
        ts = int(base_row["timestamp_ns"])
        map_row = nearest(map_to_odom, ts)
        dt_ns = abs(int(map_row["timestamp_ns"]) - ts)
        if dt_ns > max_dt_ns:
            continue
        x, y, yaw = compose_2d(map_row, base_row)
        rows.append(
            {
                "timestamp_ns": ts,
                "map_tf_timestamp_ns": int(map_row["timestamp_ns"]),
                "tf_time_delta_ms": dt_ns / 1e6,
                "x": x,
                "y": y,
                "z": 0.0,
                "yaw": yaw,
            }
        )

    if not rows:
        raise RuntimeError("No composed map-frame trajectory rows")
    write_csv(args.output, rows)
    loop = math.hypot(float(rows[-1]["x"]) - float(rows[0]["x"]), float(rows[-1]["y"]) - float(rows[0]["y"]))
    print(f"map_to_odom_tf: {len(map_to_odom)}")
    print(f"odom_to_base_tf: {len(odom_to_base)}")
    print(f"written_rows: {len(rows)}")
    print(f"path_length_m: {path_length(rows):.6f}")
    print(f"loop_closure_m: {loop:.6f}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

