#!/usr/bin/env python3
"""Plot COLMAP projected sparse points and trajectories over a LiDAR map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_map_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def parse_origin(value: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in value.strip().strip("[]").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected origin [x, y, yaw], got: {value}")
    return parts[0], parts[1], parts[2]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("Expected name=path")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_aligned_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    aligned = []
    odom = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            aligned.append([float(row["aligned_x"]), float(row["aligned_y"])])
            odom.append([float(row["odom_x"]), float(row["odom_y"])])
    if not aligned:
        raise RuntimeError(f"No rows found: {path}")
    return np.array(aligned, dtype=float), np.array(odom, dtype=float)


def load_transform(path: Path) -> tuple[list[int], float, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        [int(item) for item in payload["axis_ids"]],
        float(payload["scale"]),
        np.array(payload["rotation"], dtype=float),
        np.array(payload["translation"], dtype=float),
    )


def read_points3d(path: Path, max_error: float | None) -> tuple[np.ndarray, np.ndarray]:
    points = []
    colors = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
        error = float(parts[7])
        if max_error is not None and error > max_error:
            continue
        points.append([x, y, z])
        colors.append([r / 255.0, g / 255.0, b / 255.0])
    if not points:
        raise RuntimeError(f"No points loaded from: {path}")
    return np.array(points, dtype=float), np.array(colors, dtype=float)


def apply_transform(points3d: np.ndarray, axis_ids: list[int], scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    source2d = points3d[:, axis_ids]
    return scale * (source2d @ rotation.T) + translation


def add_direction_arrows(ax, points: np.ndarray, color: str, count: int) -> None:
    if count <= 0 or len(points) < 3:
        return
    indices = np.linspace(1, len(points) - 2, min(count, len(points) - 2), dtype=int)
    for idx in indices:
        start = points[idx]
        end = points[idx + 1]
        if np.linalg.norm(end - start) <= 1e-6:
            continue
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0, "shrinkA": 0, "shrinkB": 0},
        )


def bounds_for(point_sets: list[np.ndarray], margin: float) -> tuple[float, float, float, float]:
    merged = np.vstack(point_sets)
    min_x, min_y = merged.min(axis=0)
    max_x, max_y = merged.max(axis=0)
    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-yaml", required=True, type=Path)
    parser.add_argument("--trajectory", action="append", default=[], help="name=aligned_trajectory.csv")
    parser.add_argument("--points", action="append", default=[], help="name=points3D.txt")
    parser.add_argument("--transform", action="append", default=[], help="name=similarity_transform.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="COLMAP sparse points and trajectories aligned to LiDAR map")
    parser.add_argument("--max-point-error", type=float, default=3.0)
    parser.add_argument("--point-size", type=float, default=0.25)
    parser.add_argument("--point-alpha", type=float, default=0.25)
    parser.add_argument("--auto-crop", action="store_true")
    parser.add_argument("--crop-margin", type=float, default=1.2)
    args = parser.parse_args()

    trajectory_paths = dict(parse_named_path(item) for item in args.trajectory)
    point_paths = dict(parse_named_path(item) for item in args.points)
    transform_paths = dict(parse_named_path(item) for item in args.transform)
    if not trajectory_paths or not point_paths or not transform_paths:
        raise ValueError("--trajectory, --points, and --transform are required")

    map_info = read_map_yaml(args.map_yaml)
    image_path = args.map_yaml.parent / map_info["image"]
    resolution = float(map_info["resolution"])
    origin_x, origin_y, _ = parse_origin(map_info["origin"])
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read map: {image_path}")
    height, width = image.shape
    extent = [origin_x, origin_x + width * resolution, origin_y, origin_y + height * resolution]

    colors = {"sift": "tab:blue", "lightglue": "tab:orange", "superpoint": "tab:orange"}
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(np.flipud(image), cmap="gray", extent=extent, origin="lower", alpha=0.88)

    plotted = []
    odom_plotted = False
    for name, traj_path in trajectory_paths.items():
        aligned, odom = load_aligned_csv(traj_path)
        if not odom_plotted:
            ax.plot(odom[:, 0], odom[:, 1], color="tab:green", linewidth=3.0, label="LiDAR/TF reference")
            add_direction_arrows(ax, odom, "tab:green", 5)
            ax.scatter(odom[0, 0], odom[0, 1], s=70, marker="o", color="tab:green", edgecolor="white", zorder=7)
            ax.scatter(odom[-1, 0], odom[-1, 1], s=90, marker="x", color="tab:green", linewidth=2.0, zorder=7)
            plotted.append(odom)
            odom_plotted = True
        color = next((value for key, value in colors.items() if key in name.lower()), "black")
        if name in point_paths and name in transform_paths:
            axis_ids, scale, rotation, translation = load_transform(transform_paths[name])
            points3d, point_colors = read_points3d(point_paths[name], args.max_point_error)
            projected = apply_transform(points3d, axis_ids, scale, rotation, translation)
            ax.scatter(projected[:, 0], projected[:, 1], s=args.point_size, c=color, alpha=args.point_alpha, linewidths=0, label=f"{name} sparse points")
            plotted.append(projected)
        ax.plot(aligned[:, 0], aligned[:, 1], color=color, linewidth=2.4, label=f"{name} trajectory")
        add_direction_arrows(ax, aligned, color, 8)
        ax.scatter(aligned[0, 0], aligned[0, 1], s=70, marker="o", color=color, edgecolor="white", zorder=8)
        ax.scatter(aligned[-1, 0], aligned[-1, 1], s=90, marker="x", color=color, linewidth=2.0, zorder=8)
        plotted.append(aligned)

    if args.auto_crop and plotted:
        min_x, max_x, min_y, max_y = bounds_for(plotted, args.crop_margin)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", framealpha=0.92, markerscale=6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

