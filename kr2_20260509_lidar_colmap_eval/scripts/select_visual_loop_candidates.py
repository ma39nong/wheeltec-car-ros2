#!/usr/bin/env python3
"""Rank RGB-D loop segments by odometry closure and visual start/end similarity."""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np


def angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def read_odom(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "timestamp_ns": float(row["timestamp_ns"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "yaw": float(row["yaw"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No odom rows found: {path}")
    return rows


def list_rgb(rgb_dir: Path) -> list[tuple[int, int, Path]]:
    items: list[tuple[int, int, Path]] = []
    for path in sorted(rgb_dir.glob("*.png")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        items.append((int(parts[0]), int(parts[-1]), path))
    if not items:
        raise RuntimeError(f"No timestamped RGB files found: {rgb_dir}")
    return items


def cumulative_lengths(xy: np.ndarray) -> np.ndarray:
    lengths = np.zeros(len(xy), dtype=np.float64)
    if len(xy) > 1:
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        lengths[1:] = np.cumsum(steps)
    return lengths


def associate_odom(rgb_items: list[tuple[int, int, Path]], odom: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    odom_ts = [int(row["timestamp_ns"]) for row in odom]
    odom_xy = np.array([[row["x"], row["y"]] for row in odom], dtype=np.float64)
    odom_yaw = np.array([row["yaw"] for row in odom], dtype=np.float64)
    out_xy = []
    out_yaw = []
    for _, timestamp, _ in rgb_items:
        pos = bisect_left(odom_ts, timestamp)
        choices = []
        if pos < len(odom_ts):
            choices.append(pos)
        if pos > 0:
            choices.append(pos - 1)
        best = min(choices, key=lambda idx: abs(odom_ts[idx] - timestamp))
        out_xy.append(odom_xy[best])
        out_yaw.append(odom_yaw[best])
    return np.array(out_xy), np.array(out_yaw)


def gray_thumbnail(path: Path, size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def image_scores(path_a: Path, path_b: Path, size: int) -> tuple[float, int]:
    image_a = gray_thumbnail(path_a, size)
    image_b = gray_thumbnail(path_b, size)
    hist_a = cv2.calcHist([image_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([image_b], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    hist_corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

    orb = cv2.ORB_create(nfeatures=1000)
    kpa, desa = orb.detectAndCompute(image_a, None)
    kpb, desb = orb.detectAndCompute(image_b, None)
    good = 0
    if desa is not None and desb is not None and len(kpa) >= 8 and len(kpb) >= 8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(desa, desb, k=2)
        for first, second in matches:
            if first.distance < 0.75 * second.distance:
                good += 1
    return hist_corr, good


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["message"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--odom", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-frames", type=int, default=450)
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--min-path-m", type=float, default=7.0)
    parser.add_argument("--max-return-m", type=float, default=1.2)
    parser.add_argument("--max-yaw-diff-rad", type=float, default=1.0)
    parser.add_argument("--start-step", type=int, default=10)
    parser.add_argument("--end-step", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--thumb-size", type=int, default=320)
    args = parser.parse_args()

    rgb_items = list_rgb(args.rgb_dir)
    xy, yaw = associate_odom(rgb_items, read_odom(args.odom))
    path = cumulative_lengths(xy)

    odom_candidates: list[tuple[float, int, int, float, float, float]] = []
    for start in range(0, len(rgb_items), args.start_step):
        min_end = start + args.min_frames
        max_end = min(len(rgb_items) - 1, start + args.max_frames)
        if min_end > max_end:
            continue
        for end in range(min_end, max_end + 1, args.end_step):
            path_m = float(path[end] - path[start])
            if path_m < args.min_path_m:
                continue
            return_m = float(np.linalg.norm(xy[end] - xy[start]))
            if return_m > args.max_return_m:
                continue
            yaw_diff = angle_diff(float(yaw[end]), float(yaw[start]))
            if yaw_diff > args.max_yaw_diff_rad:
                continue
            odom_score = return_m + 0.35 * yaw_diff
            odom_candidates.append((odom_score, start, end, path_m, return_m, yaw_diff))

    odom_candidates.sort(key=lambda item: item[0])
    rows: list[dict[str, object]] = []
    for rank, (_, start, end, path_m, return_m, yaw_diff) in enumerate(odom_candidates[: args.max_candidates], start=1):
        hist_corr, orb_good = image_scores(rgb_items[start][2], rgb_items[end][2], args.thumb_size)
        visual_score = orb_good + 30.0 * max(hist_corr, 0.0) - 10.0 * return_m - 5.0 * yaw_diff
        rows.append(
            {
                "rank_by_odom": rank,
                "start_index": start,
                "end_index_inclusive": end,
                "end_index_exclusive": end + 1,
                "frames": end - start + 1,
                "path_length_m": f"{path_m:.3f}",
                "return_distance_m": f"{return_m:.3f}",
                "yaw_diff_rad": f"{yaw_diff:.3f}",
                "hist_corr": f"{hist_corr:.4f}",
                "orb_good_matches": orb_good,
                "visual_score": f"{visual_score:.3f}",
                "start_rgb": rgb_items[start][2].name,
                "end_rgb": rgb_items[end][2].name,
            }
        )
    rows.sort(key=lambda row: float(row["visual_score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_by_visual"] = rank
    write_csv(args.output, rows)
    print(f"rgb_frames: {len(rgb_items)}")
    print(f"odom_candidates: {len(odom_candidates)}")
    print(f"wrote: {args.output}")
    if rows:
        best = rows[0]
        print(
            "best: "
            f"{best['start_index']}..{best['end_index_inclusive']} "
            f"frames={best['frames']} return={best['return_distance_m']} "
            f"yaw={best['yaw_diff_rad']} orb={best['orb_good_matches']}"
        )


if __name__ == "__main__":
    main()
