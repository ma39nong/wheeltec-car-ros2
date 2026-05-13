#!/usr/bin/env python3
"""Run SuperPoint + LightGlue matching for the KR2 phase-2 pair schedule."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def add_local_lightglue_if_present() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    local_lightglue = repo_root / "LightGlue"
    if local_lightglue.exists() and str(local_lightglue) not in sys.path:
        sys.path.insert(0, str(local_lightglue))


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def require_lightglue() -> None:
    add_local_lightglue_if_present()
    if importlib.util.find_spec("lightglue") is None:
        raise RuntimeError(
            "The Python package 'lightglue' is not installed. "
            "Install LightGlue or clone it to /home/kiyi/COLMAP/LightGlue before running learned matching."
        )


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def read_schedule(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def load_rgb_for_visualization(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_match_visualization(
    image0_path: Path,
    image1_path: Path,
    keypoints0: np.ndarray,
    keypoints1: np.ndarray,
    matches: np.ndarray,
    inlier_mask: np.ndarray | None,
    output_path: Path,
) -> None:
    image0 = load_rgb_for_visualization(image0_path)
    image1 = load_rgb_for_visualization(image1_path)
    cv_matches = []
    cv_keypoints0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints0]
    cv_keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints1]
    for match_idx, (idx0, idx1) in enumerate(matches):
        cv_matches.append(cv2.DMatch(int(idx0), int(idx1), float(match_idx)))

    match_mask = None
    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    if inlier_mask is not None and len(inlier_mask) == len(cv_matches):
        match_mask = inlier_mask.astype(np.uint8).reshape(-1).tolist()

    visualization = cv2.drawMatches(
        image0,
        cv_keypoints0,
        image1,
        cv_keypoints1,
        cv_matches,
        None,
        matchesMask=match_mask,
        flags=flags,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)


def write_runtime_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    require_lightglue()
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd

    config = read_env(args.config)
    output_dir = Path(config["OUTPUT_DIR"])
    schedule_path = output_dir / "reports" / "pair_schedule.csv"
    metrics_path = output_dir / "reports" / "pair_metrics.csv"
    matches_dir = output_dir / "matches"
    features_dir = output_dir / "features"
    visualizations_dir = output_dir / "match_visualizations"
    reports_dir = output_dir / "reports"

    run_start = time.perf_counter()
    device = choose_device(config.get("DEVICE", "auto"))
    max_keypoints = int(config.get("MAX_KEYPOINTS", "2048"))
    min_matches_for_geometry = int(config.get("MIN_MATCHES_FOR_GEOMETRY", "8"))
    ransac_reproj_threshold = float(config.get("RANSAC_REPROJ_THRESHOLD", "1.5"))
    ransac_confidence = float(config.get("RANSAC_CONFIDENCE", "0.999"))
    max_visualizations = int(config.get("MAX_VISUALIZATIONS", "80"))

    rows = read_schedule(schedule_path)
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    feature_cache: dict[str, dict[str, torch.Tensor]] = {}
    feature_extract_time = 0.0
    feature_save_time = 0.0
    match_time = 0.0
    geometry_time = 0.0
    match_save_time = 0.0
    visualization_time = 0.0

    def get_features(image_path: Path) -> dict[str, torch.Tensor]:
        nonlocal feature_extract_time, feature_save_time
        image_key = image_path.name
        if image_key in feature_cache:
            return feature_cache[image_key]
        image_tensor = load_image(image_path).to(device)
        extract_start = time.perf_counter()
        with torch.inference_mode():
            features = extractor.extract(image_tensor)
        feature_extract_time += time.perf_counter() - extract_start
        feature_cache[image_key] = features

        save_start = time.perf_counter()
        feature_path = features_dir / f"{image_path.stem}.npz"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        compact = rbd(features)
        np.savez_compressed(
            feature_path,
            keypoints=tensor_to_numpy(compact["keypoints"]),
            descriptors=tensor_to_numpy(compact["descriptors"]),
            keypoint_scores=tensor_to_numpy(compact["keypoint_scores"]),
        )
        feature_save_time += time.perf_counter() - save_start
        return features

    metric_rows: list[dict[str, str | int | float]] = []
    for row_index, row in enumerate(rows):
        image0_path = Path(row["image0_path"])
        image1_path = Path(row["image1_path"])
        feats0 = get_features(image0_path)
        feats1 = get_features(image1_path)

        match_start = time.perf_counter()
        with torch.inference_mode():
            pred = matcher({"image0": feats0, "image1": feats1})
        match_time += time.perf_counter() - match_start
        pred = rbd(pred)

        matches = tensor_to_numpy(pred["matches"]).astype(np.int32)
        compact0 = rbd(feats0)
        compact1 = rbd(feats1)
        keypoints0 = tensor_to_numpy(compact0["keypoints"])
        keypoints1 = tensor_to_numpy(compact1["keypoints"])
        matched0 = keypoints0[matches[:, 0]] if len(matches) else np.empty((0, 2), dtype=np.float32)
        matched1 = keypoints1[matches[:, 1]] if len(matches) else np.empty((0, 2), dtype=np.float32)

        inlier_mask = None
        inlier_count = 0
        fundamental_ok = "no"
        if len(matches) >= min_matches_for_geometry:
            geometry_start = time.perf_counter()
            _, mask = cv2.findFundamentalMat(
                matched0,
                matched1,
                cv2.USAC_MAGSAC,
                ransac_reproj_threshold,
                ransac_confidence,
            )
            geometry_time += time.perf_counter() - geometry_start
            if mask is not None:
                inlier_mask = mask.reshape(-1).astype(bool)
                inlier_count = int(inlier_mask.sum())
                fundamental_ok = "yes" if inlier_count >= min_matches_for_geometry else "weak"

        pair_stem = f"pair_{int(row['pair_id']):05d}_s{row['stride']}_{image0_path.stem}_to_{image1_path.stem}"
        match_path = matches_dir / f"{pair_stem}.npz"
        save_start = time.perf_counter()
        match_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            match_path,
            matches=matches,
            matched_keypoints0=matched0,
            matched_keypoints1=matched1,
            inlier_mask=np.array([]) if inlier_mask is None else inlier_mask,
        )
        match_save_time += time.perf_counter() - save_start

        if max_visualizations == 0 or row_index < max_visualizations:
            visualization_start = time.perf_counter()
            save_match_visualization(
                image0_path,
                image1_path,
                keypoints0,
                keypoints1,
                matches,
                inlier_mask,
                visualizations_dir / f"{pair_stem}.jpg",
            )
            visualization_time += time.perf_counter() - visualization_start

        metric_rows.append(
            {
                "pair_id": int(row["pair_id"]),
                "stride": int(row["stride"]),
                "image0_name": row["image0_name"],
                "image1_name": row["image1_name"],
                "keypoints0": len(keypoints0),
                "keypoints1": len(keypoints1),
                "matches": len(matches),
                "inliers": inlier_count,
                "inlier_ratio": round(inlier_count / len(matches), 4) if len(matches) else 0.0,
                "fundamental_ok": fundamental_ok,
                "match_file": str(match_path),
            }
        )

        if (row_index + 1) % 25 == 0:
            print(f"processed {row_index + 1}/{len(rows)} pairs")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    total_time = time.perf_counter() - run_start
    runtime_report_path = reports_dir / "matching_runtime.txt"
    runtime_lines = [
        "SuperPoint + LightGlue runtime report",
        "",
        f"device: {device}",
        f"images: {len(feature_cache)}",
        f"pairs: {len(metric_rows)}",
        f"max_keypoints: {max_keypoints}",
        f"feature_extract_sec: {feature_extract_time:.3f}",
        f"feature_save_sec: {feature_save_time:.3f}",
        f"lightglue_match_sec: {match_time:.3f}",
        f"geometry_check_sec: {geometry_time:.3f}",
        f"match_save_sec: {match_save_time:.3f}",
        f"visualization_sec: {visualization_time:.3f}",
        f"total_sec: {total_time:.3f}",
        f"total_min: {total_time / 60:.3f}",
    ]
    write_runtime_report(runtime_report_path, runtime_lines)

    print(f"device: {device}")
    print(f"pairs: {len(metric_rows)}")
    print(f"wrote: {metrics_path}")
    print(f"runtime: {runtime_report_path}")
    print(f"total_sec: {total_time:.3f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
