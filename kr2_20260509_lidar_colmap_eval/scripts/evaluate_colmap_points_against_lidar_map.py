#!/usr/bin/env python3
"""Evaluate projected COLMAP sparse points against LiDAR occupancy walls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
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


def load_transform(path: Path) -> tuple[list[int], float, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        [int(item) for item in payload["axis_ids"]],
        float(payload["scale"]),
        np.array(payload["rotation"], dtype=float),
        np.array(payload["translation"], dtype=float),
    )


def read_points3d(path: Path, max_error: float | None) -> np.ndarray:
    points = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        error = float(parts[7])
        if max_error is not None and error > max_error:
            continue
        points.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not points:
        raise RuntimeError(f"No points loaded from: {path}")
    return np.array(points, dtype=float)


def apply_transform(
    points3d: np.ndarray,
    axis_ids: list[int],
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    source2d = points3d[:, axis_ids]
    return scale * (source2d @ rotation.T) + translation


def world_to_image_pixels(
    points: np.ndarray,
    origin_x: float,
    origin_y: float,
    resolution: float,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    cols = np.rint((points[:, 0] - origin_x) / resolution).astype(int)
    rows_from_bottom = np.rint((points[:, 1] - origin_y) / resolution).astype(int)
    rows = height - 1 - rows_from_bottom
    inside = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    return np.column_stack([rows, cols]), inside


def summarize(name: str, points: np.ndarray, distances_m: np.ndarray, inside: np.ndarray) -> dict[str, float | int | str]:
    valid_distances = distances_m[inside]
    if len(valid_distances) == 0:
        raise RuntimeError(f"No projected points fall inside the map for {name}")
    return {
        "name": name,
        "points_loaded": int(len(points)),
        "points_inside_map": int(inside.sum()),
        "inside_map_percent": float(100.0 * inside.mean()),
        "mean_wall_distance_m": float(np.mean(valid_distances)),
        "median_wall_distance_m": float(np.median(valid_distances)),
        "p90_wall_distance_m": float(np.percentile(valid_distances, 90)),
        "p95_wall_distance_m": float(np.percentile(valid_distances, 95)),
        "max_wall_distance_m": float(np.max(valid_distances)),
        "within_0.10m_percent": float(100.0 * np.mean(valid_distances <= 0.10)),
        "within_0.20m_percent": float(100.0 * np.mean(valid_distances <= 0.20)),
        "within_0.30m_percent": float(100.0 * np.mean(valid_distances <= 0.30)),
        "within_0.50m_percent": float(100.0 * np.mean(valid_distances <= 0.50)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-yaml", required=True, type=Path)
    parser.add_argument("--points", action="append", default=[], help="name=points3D.txt")
    parser.add_argument("--transform", action="append", default=[], help="name=similarity_transform.json")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--max-point-error", type=float, default=3.0)
    parser.add_argument("--occupied-max-gray", type=int, default=100)
    args = parser.parse_args()

    point_paths = dict(parse_named_path(item) for item in args.points)
    transform_paths = dict(parse_named_path(item) for item in args.transform)
    missing_transforms = sorted(set(point_paths) - set(transform_paths))
    if missing_transforms:
        raise ValueError(f"Missing --transform entries for: {', '.join(missing_transforms)}")

    map_info = read_map_yaml(args.map_yaml)
    image_path = args.map_yaml.parent / map_info["image"]
    resolution = float(map_info["resolution"])
    origin_x, origin_y, _ = parse_origin(map_info["origin"])
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read map: {image_path}")

    occupied = image <= args.occupied_max_gray
    if not np.any(occupied):
        raise RuntimeError(f"No occupied pixels found with occupied max gray={args.occupied_max_gray}")
    free_space = np.where(occupied, 0, 255).astype(np.uint8)
    distance_pixels = cv2.distanceTransform(free_space, cv2.DIST_L2, 5)
    distance_m = distance_pixels * resolution

    rows = []
    height, width = image.shape
    for name, points_path in point_paths.items():
        points3d = read_points3d(points_path, args.max_point_error)
        axis_ids, scale, rotation, translation = load_transform(transform_paths[name])
        projected = apply_transform(points3d, axis_ids, scale, rotation, translation)
        pixels, inside = world_to_image_pixels(projected, origin_x, origin_y, resolution, height, width)
        point_distances = np.full(len(projected), np.nan, dtype=float)
        point_distances[inside] = distance_m[pixels[inside, 0], pixels[inside, 1]]
        rows.append(summarize(name, projected, point_distances, inside))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"{row['name']}: mean={row['mean_wall_distance_m']:.3f}m "
            f"median={row['median_wall_distance_m']:.3f}m "
            f"p90={row['p90_wall_distance_m']:.3f}m "
            f"inside={row['points_inside_map']}/{row['points_loaded']}"
        )
    print(f"wrote: {args.output_csv}")
    if args.output_json:
        print(f"wrote: {args.output_json}")


if __name__ == "__main__":
    main()
