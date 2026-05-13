#!/usr/bin/env python3
"""Align a COLMAP TUM trajectory to odom in 2D and write error metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


AXES = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}


def read_tum(path: Path) -> list[tuple[float, np.ndarray]]:
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append((float(parts[0]), np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)))
    if not rows:
        raise RuntimeError(f"No TUM poses found: {path}")
    rows.sort(key=lambda item: item[0])
    return rows


def read_odom(path: Path) -> list[tuple[float, np.ndarray]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = float(row["timestamp_ns"]) / 1e9
            rows.append((timestamp, np.array([float(row["x"]), float(row["y"])], dtype=float)))
    if not rows:
        raise RuntimeError(f"No odom poses found: {path}")
    rows.sort(key=lambda item: item[0])
    return rows


def associate(
    visual: list[tuple[float, np.ndarray]],
    odom: list[tuple[float, np.ndarray]],
    max_dt: float,
    interpolate_reference: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    odom_ts = np.array([item[0] for item in odom], dtype=float)
    odom_xy = np.array([item[1] for item in odom], dtype=float)
    matched_visual = []
    matched_odom = []
    matched_time = []
    for timestamp, point3d in visual:
        idx = int(np.searchsorted(odom_ts, timestamp))
        if interpolate_reference and 0 < idx < len(odom_ts):
            before = idx - 1
            after = idx
            left_dt = timestamp - odom_ts[before]
            right_dt = odom_ts[after] - timestamp
            if left_dt <= max_dt and right_dt <= max_dt:
                denom = odom_ts[after] - odom_ts[before]
                alpha = 0.0 if denom <= 0 else (timestamp - odom_ts[before]) / denom
                matched_visual.append(point3d)
                matched_odom.append((1.0 - alpha) * odom_xy[before] + alpha * odom_xy[after])
                matched_time.append(timestamp)
                continue
        candidates = []
        if 0 <= idx < len(odom_ts):
            candidates.append(idx)
        if 0 <= idx - 1 < len(odom_ts):
            candidates.append(idx - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda item: abs(odom_ts[item] - timestamp))
        if abs(odom_ts[best] - timestamp) <= max_dt:
            matched_visual.append(point3d)
            matched_odom.append(odom_xy[best])
            matched_time.append(timestamp)
    if len(matched_visual) < 3:
        raise RuntimeError(f"Only {len(matched_visual)} associations found; need at least 3")
    return np.array(matched_time), np.array(matched_visual), np.array(matched_odom)


def estimate_similarity_2d(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_centered = source - src_mean
    tgt_centered = target - tgt_mean

    covariance = (tgt_centered.T @ src_centered) / len(source)
    u_mat, singular_values, vt_mat = np.linalg.svd(covariance)
    rotation = u_mat @ vt_mat
    if np.linalg.det(rotation) < 0:
        u_mat[:, -1] *= -1
        rotation = u_mat @ vt_mat

    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.sum(singular_values) / variance) if variance > 0 else 1.0
    translation = tgt_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def apply_similarity(source: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (source @ rotation.T) + translation


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def write_aligned_csv(path: Path, times: np.ndarray, aligned: np.ndarray, reference: np.ndarray, errors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "aligned_x", "aligned_y", "odom_x", "odom_y", "error_m"])
        for timestamp, point, ref, err in zip(times, aligned, reference, errors):
            writer.writerow([f"{timestamp:.9f}", f"{point[0]:.8f}", f"{point[1]:.8f}", f"{ref[0]:.8f}", f"{ref[1]:.8f}", f"{err:.8f}"])


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_transform_json(
    path: Path,
    axis_name: str,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "axis_name": axis_name,
        "axis_ids": list(AXES[axis_name]),
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tum", required=True, type=Path)
    parser.add_argument("--odom", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-dt", type=float, default=0.08)
    parser.add_argument(
        "--interpolate-reference",
        action="store_true",
        help="Linearly interpolate the 2D reference trajectory to visual timestamps when possible.",
    )
    args = parser.parse_args()

    times, visual3d, odom2d = associate(
        read_tum(args.tum),
        read_odom(args.odom),
        args.max_dt,
        args.interpolate_reference,
    )

    best = None
    for axis_name, axis_ids in AXES.items():
        source2d = visual3d[:, axis_ids]
        scale, rotation, translation = estimate_similarity_2d(source2d, odom2d)
        aligned = apply_similarity(source2d, scale, rotation, translation)
        errors = np.linalg.norm(aligned - odom2d, axis=1)
        rmse = math.sqrt(float(np.mean(errors * errors)))
        candidate = (rmse, axis_name, scale, rotation, translation, aligned, errors)
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    rmse, axis_name, scale, rotation, translation, aligned, errors = best
    loop_visual = float(np.linalg.norm(aligned[-1] - aligned[0]))
    loop_odom = float(np.linalg.norm(odom2d[-1] - odom2d[0]))
    loop_delta = abs(loop_visual - loop_odom)

    output_dir = args.output_dir / args.name
    write_aligned_csv(output_dir / "aligned_trajectory.csv", times, aligned, odom2d, errors)
    write_transform_json(output_dir / "similarity_transform.json", axis_name, scale, rotation, translation)
    report_lines = [
        f"name: {args.name}",
        f"tum: {args.tum}",
        f"odom: {args.odom}",
        f"associations: {len(times)}",
        f"max_dt_sec: {args.max_dt}",
        f"interpolate_reference: {args.interpolate_reference}",
        f"best_visual_axes: {axis_name}",
        f"scale: {scale:.8f}",
        f"rmse_m: {rmse:.6f}",
        f"mean_error_m: {float(errors.mean()):.6f}",
        f"median_error_m: {float(np.median(errors)):.6f}",
        f"max_error_m: {float(errors.max()):.6f}",
        f"odom_path_length_m: {path_length(odom2d):.6f}",
        f"aligned_visual_path_length_m: {path_length(aligned):.6f}",
        f"odom_loop_closure_m: {loop_odom:.6f}",
        f"aligned_visual_loop_closure_m: {loop_visual:.6f}",
        f"loop_closure_delta_m: {loop_delta:.6f}",
        f"aligned_csv: {output_dir / 'aligned_trajectory.csv'}",
        f"similarity_transform: {output_dir / 'similarity_transform.json'}",
    ]
    write_report(output_dir / "alignment_report.txt", report_lines)
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
