#!/usr/bin/env python3
"""Project odometry poses into the map frame using map->odom TF."""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_left
from pathlib import Path

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def read_odom_csv(path: Path) -> list[dict[str, float | int]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "yaw": float(row.get("yaw", 0.0)),
                }
            )
    if not rows:
        raise RuntimeError(f"No odom rows found: {path}")
    rows.sort(key=lambda item: int(item["timestamp_ns"]))
    return rows


def stamp_ns(transform, fallback: int) -> int:
    stamp = transform.header.stamp
    value = stamp.sec * 10**9 + stamp.nanosec
    return value if value > 0 else fallback


def read_map_to_odom_tf(bag_dir: Path, map_frame: str, odom_frame: str) -> list[dict[str, float | int]]:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if "/tf" not in topics:
        raise RuntimeError("/tf topic not found in bag")
    msg_type = get_message(topics["/tf"])

    rows = []
    while reader.has_next():
        topic, data, bag_ts = reader.read_next()
        if topic != "/tf":
            continue
        msg = deserialize_message(data, msg_type)
        for transform in msg.transforms:
            if transform.header.frame_id != map_frame or transform.child_frame_id != odom_frame:
                continue
            trans = transform.transform.translation
            rot = transform.transform.rotation
            rows.append(
                {
                    "timestamp_ns": stamp_ns(transform, bag_ts),
                    "x": float(trans.x),
                    "y": float(trans.y),
                    "yaw": yaw_from_quaternion(float(rot.x), float(rot.y), float(rot.z), float(rot.w)),
                }
            )
    if not rows:
        raise RuntimeError(f"No TF found: {map_frame} -> {odom_frame}")
    rows.sort(key=lambda item: int(item["timestamp_ns"]))
    return rows


def interpolate_tf(rows: list[dict[str, float | int]], timestamp_ns: int, max_dt_ns: int) -> dict[str, float | int] | None:
    timestamps = [int(row["timestamp_ns"]) for row in rows]
    idx = bisect_left(timestamps, timestamp_ns)
    if 0 < idx < len(rows):
        before = rows[idx - 1]
        after = rows[idx]
        left_dt = timestamp_ns - int(before["timestamp_ns"])
        right_dt = int(after["timestamp_ns"]) - timestamp_ns
        if left_dt <= max_dt_ns and right_dt <= max_dt_ns:
            denom = int(after["timestamp_ns"]) - int(before["timestamp_ns"])
            alpha = 0.0 if denom <= 0 else (timestamp_ns - int(before["timestamp_ns"])) / denom
            yaw0 = float(before["yaw"])
            yaw1 = float(after["yaw"])
            yaw_delta = math.atan2(math.sin(yaw1 - yaw0), math.cos(yaw1 - yaw0))
            return {
                "timestamp_ns": timestamp_ns,
                "source_timestamp_ns": timestamp_ns,
                "x": (1.0 - alpha) * float(before["x"]) + alpha * float(after["x"]),
                "y": (1.0 - alpha) * float(before["y"]) + alpha * float(after["y"]),
                "yaw": math.atan2(math.sin(yaw0 + alpha * yaw_delta), math.cos(yaw0 + alpha * yaw_delta)),
                "dt_ns": max(left_dt, right_dt),
            }

    candidates = []
    if idx < len(rows):
        candidates.append(rows[idx])
    if idx > 0:
        candidates.append(rows[idx - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda row: abs(int(row["timestamp_ns"]) - timestamp_ns))
    dt_ns = abs(int(best["timestamp_ns"]) - timestamp_ns)
    if dt_ns > max_dt_ns:
        return None
    out = dict(best)
    out["dt_ns"] = dt_ns
    return out


def compose_2d(map_to_odom: dict[str, float | int], odom_pose: dict[str, float | int]) -> tuple[float, float, float]:
    mx = float(map_to_odom["x"])
    my = float(map_to_odom["y"])
    myaw = float(map_to_odom["yaw"])
    ox = float(odom_pose["x"])
    oy = float(odom_pose["y"])
    oyaw = float(odom_pose["yaw"])
    cos_yaw = math.cos(myaw)
    sin_yaw = math.sin(myaw)
    x = mx + cos_yaw * ox - sin_yaw * oy
    y = my + sin_yaw * ox + cos_yaw * oy
    yaw = math.atan2(math.sin(myaw + oyaw), math.cos(myaw + oyaw))
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
    parser.add_argument("--odom", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--odom-frame", default="odom_combined")
    parser.add_argument("--max-tf-dt-sec", type=float, default=1.0)
    args = parser.parse_args()

    odom_rows = read_odom_csv(args.odom)
    tf_rows = read_map_to_odom_tf(args.bag, args.map_frame, args.odom_frame)
    max_dt_ns = int(args.max_tf_dt_sec * 1e9)

    output_rows = []
    for odom_pose in odom_rows:
        ts = int(odom_pose["timestamp_ns"])
        tf_pose = interpolate_tf(tf_rows, ts, max_dt_ns)
        if tf_pose is None:
            continue
        x, y, yaw = compose_2d(tf_pose, odom_pose)
        output_rows.append(
            {
                "timestamp_ns": ts,
                "map_tf_time_delta_ms": float(tf_pose["dt_ns"]) / 1e6,
                "x": x,
                "y": y,
                "z": 0.0,
                "yaw": yaw,
            }
        )

    if not output_rows:
        raise RuntimeError("No projected map-frame odom rows")
    write_csv(args.output, output_rows)
    loop = math.hypot(
        float(output_rows[-1]["x"]) - float(output_rows[0]["x"]),
        float(output_rows[-1]["y"]) - float(output_rows[0]["y"]),
    )
    print(f"odom_rows: {len(odom_rows)}")
    print(f"map_to_odom_tf: {len(tf_rows)}")
    print(f"written_rows: {len(output_rows)}")
    print(f"path_length_m: {path_length(output_rows):.6f}")
    print(f"loop_closure_m: {loop:.6f}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

