#!/usr/bin/env python3
"""Plot aligned COLMAP trajectories over a ROS occupancy-grid map."""

from __future__ import annotations

import argparse
import csv
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
    cleaned = value.strip().strip("[]")
    parts = [float(item.strip()) for item in cleaned.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected origin [x, y, yaw], got: {value}")
    return parts[0], parts[1], parts[2]


def load_aligned_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    aligned = []
    odom = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            aligned.append([float(row["aligned_x"]), float(row["aligned_y"])])
            odom.append([float(row["odom_x"]), float(row["odom_y"])])
    if not aligned:
        raise RuntimeError(f"No trajectory rows found: {path}")
    return np.array(aligned, dtype=float), np.array(odom, dtype=float)


def parse_traj_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--trajectory must use name=path format")
    name, path = value.split("=", 1)
    return name, Path(path)


def add_direction_arrows(ax, points: np.ndarray, color: str, count: int) -> None:
    if count <= 0 or len(points) < 3:
        return
    indices = np.linspace(1, len(points) - 2, min(count, len(points) - 2), dtype=int)
    for idx in indices:
        start = points[idx]
        end = points[idx + 1]
        delta = end - start
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-6:
            continue
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.2,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )


def bounds_for_points(point_sets: list[np.ndarray], margin: float) -> tuple[float, float, float, float]:
    merged = np.vstack(point_sets)
    min_x, min_y = merged.min(axis=0)
    max_x, max_y = merged.max(axis=0)
    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-yaml", required=True, type=Path)
    parser.add_argument("--trajectory", action="append", default=[], help="name=aligned_trajectory.csv")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="COLMAP trajectories aligned to LiDAR map")
    parser.add_argument("--auto-crop", action="store_true", help="Crop around plotted trajectories.")
    parser.add_argument("--crop-margin", type=float, default=1.5)
    parser.add_argument("--arrow-count", type=int, default=8)
    parser.add_argument("--no-markers", action="store_true")
    args = parser.parse_args()

    if not args.trajectory:
        raise ValueError("At least one --trajectory name=path is required")

    map_info = read_map_yaml(args.map_yaml)
    image_path = args.map_yaml.parent / map_info["image"]
    resolution = float(map_info["resolution"])
    origin_x, origin_y, _ = parse_origin(map_info["origin"])

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read map image: {image_path}")

    height, width = image.shape
    extent = [
        origin_x,
        origin_x + width * resolution,
        origin_y,
        origin_y + height * resolution,
    ]

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(np.flipud(image), cmap="gray", extent=extent, origin="lower", alpha=0.88)

    odom_plotted = False
    plotted_points = []
    colors = {
        "sift": "tab:blue",
        "lightglue": "tab:orange",
        "superpoint": "tab:orange",
    }
    for item in args.trajectory:
        name, path = parse_traj_arg(item)
        aligned, odom = load_aligned_csv(path)
        if not odom_plotted:
            ax.plot(odom[:, 0], odom[:, 1], color="tab:green", linewidth=3.0, label="LiDAR/TF reference")
            add_direction_arrows(ax, odom, "tab:green", max(3, args.arrow_count // 2))
            plotted_points.append(odom)
            odom_plotted = True
        color = next((value for key, value in colors.items() if key.lower() in name.lower()), None)
        ax.plot(aligned[:, 0], aligned[:, 1], linewidth=2.3, label=name, color=color)
        add_direction_arrows(ax, aligned, color or "black", args.arrow_count)
        plotted_points.append(aligned)
        if not args.no_markers:
            ax.scatter(aligned[0, 0], aligned[0, 1], s=70, marker="o", color=color, edgecolor="white", zorder=5)
            ax.scatter(aligned[-1, 0], aligned[-1, 1], s=90, marker="x", color=color, linewidth=2.2, zorder=5)

    if not args.no_markers and plotted_points:
        ref = plotted_points[0]
        ax.scatter(ref[0, 0], ref[0, 1], s=85, marker="o", color="tab:green", edgecolor="white", zorder=6)
        ax.scatter(ref[-1, 0], ref[-1, 1], s=105, marker="x", color="tab:green", linewidth=2.4, zorder=6)
        ax.text(ref[0, 0], ref[0, 1] + 0.18, "start", color="black", fontsize=9, weight="bold")
        ax.text(ref[-1, 0], ref[-1, 1] + 0.18, "end", color="black", fontsize=9, weight="bold")

    if args.auto_crop and plotted_points:
        min_x, max_x, min_y, max_y = bounds_for_points(plotted_points, args.crop_margin)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.set_title(args.title)
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.22)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
