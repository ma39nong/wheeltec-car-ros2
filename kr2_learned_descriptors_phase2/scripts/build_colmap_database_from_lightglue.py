#!/usr/bin/env python3
"""Create a COLMAP database from SuperPoint keypoints and LightGlue matches."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import subprocess
from pathlib import Path

import numpy as np


CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def array_to_blob(array: np.ndarray) -> bytes:
    return np.ascontiguousarray(array).tobytes()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def create_empty_colmap_database(output_db: Path) -> None:
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if output_db.exists():
        output_db.unlink()
    subprocess.run(
        ["colmap", "database_creator", "--database_path", str(output_db)],
        check=True,
    )


def copy_camera_and_image_metadata(source_db: Path, output_db: Path) -> None:
    if not source_db.exists():
        raise FileNotFoundError(f"Source database not found: {source_db}")

    src = sqlite3.connect(source_db)
    dst = sqlite3.connect(output_db)
    try:
        src_cur = src.cursor()
        dst_cur = dst.cursor()

        for row in src_cur.execute(
            "SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras ORDER BY camera_id"
        ):
            dst_cur.execute(
                """
                INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                row,
            )

        for image_id, name, camera_id in src_cur.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ):
            dst_cur.execute(
                """
                INSERT INTO images(
                    image_id, name, camera_id,
                    prior_qw, prior_qx, prior_qy, prior_qz,
                    prior_tx, prior_ty, prior_tz
                )
                VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
                """,
                (image_id, name, camera_id),
            )

        dst.commit()
    finally:
        src.close()
        dst.close()


def load_cameras_txt(path: Path) -> dict[int, tuple[int, int, int, np.ndarray]]:
    cameras: dict[int, tuple[int, int, int, np.ndarray]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Baseline cameras.txt not found: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        camera_id = int(parts[0])
        model_name = parts[1]
        if model_name not in CAMERA_MODEL_IDS:
            raise ValueError(f"Unsupported COLMAP camera model in {path}: {model_name}")
        width = int(parts[2])
        height = int(parts[3])
        params = np.array([float(value) for value in parts[4:]], dtype=np.float64)
        cameras[camera_id] = (CAMERA_MODEL_IDS[model_name], width, height, params)
    return cameras


def apply_refined_camera_intrinsics(connection: sqlite3.Connection, cameras_txt: Path) -> int:
    cameras = load_cameras_txt(cameras_txt)
    cursor = connection.cursor()
    updated = 0
    for camera_id, (model, width, height, params) in cameras.items():
        cursor.execute(
            """
            UPDATE cameras
            SET model = ?, width = ?, height = ?, params = ?, prior_focal_length = 1
            WHERE camera_id = ?
            """,
            (model, width, height, array_to_blob(params), camera_id),
        )
        updated += cursor.rowcount
    connection.commit()
    return updated


def read_image_id_by_name(connection: sqlite3.Connection) -> dict[str, int]:
    cursor = connection.cursor()
    return {name: image_id for image_id, name in cursor.execute("SELECT image_id, name FROM images")}


def insert_keypoints(connection: sqlite3.Connection, image_id_by_name: dict[str, int], features_dir: Path) -> int:
    cursor = connection.cursor()
    inserted = 0
    for image_name, image_id in sorted(image_id_by_name.items(), key=lambda item: item[1]):
        feature_path = features_dir / f"{Path(image_name).stem}.npz"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file missing for {image_name}: {feature_path}")
        with np.load(feature_path) as data:
            keypoints = data["keypoints"].astype(np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2:
            raise ValueError(f"Expected Nx2 keypoints in {feature_path}, got {keypoints.shape}")
        cursor.execute(
            "INSERT INTO keypoints(image_id, rows, cols, data) VALUES (?, ?, ?, ?)",
            (image_id, int(keypoints.shape[0]), 2, array_to_blob(keypoints)),
        )
        inserted += 1
    connection.commit()
    return inserted


def write_raw_match_list(metric_rows: list[dict[str, str]], output_path: Path) -> dict[str, int]:
    stats = {
        "raw_match_pairs": 0,
        "raw_matches": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in metric_rows:
            with np.load(row["match_file"]) as data:
                matches = data["matches"].astype(np.uint32)

            handle.write(f"{row['image0_name']} {row['image1_name']}\n")
            for idx0, idx1 in matches:
                handle.write(f"{int(idx0)} {int(idx1)}\n")
            handle.write("\n")

            stats["raw_match_pairs"] += 1
            stats["raw_matches"] += int(matches.shape[0])

    return stats


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = read_env(args.config)
    source_db = Path(config["SOURCE_DB"])
    output_db = Path(config["DB_PATH"])
    refined_cameras_txt = Path(config["BASELINE_CAMERAS_TXT"]) if config.get("BASELINE_CAMERAS_TXT") else None
    match_list_path = Path(config.get("MATCH_LIST_PATH", output_db.with_name("lightglue_raw_matches.txt")))
    lightglue_output_dir = Path(config["LIGHTGLUE_OUTPUT_DIR"])
    features_dir = lightglue_output_dir / "features"
    metrics_path = lightglue_output_dir / "reports" / "pair_metrics.csv"

    create_empty_colmap_database(output_db)
    copy_camera_and_image_metadata(source_db, output_db)
    rows = load_csv(metrics_path)

    connection = sqlite3.connect(output_db)
    try:
        refined_camera_count = 0
        if refined_cameras_txt is not None:
            refined_camera_count = apply_refined_camera_intrinsics(connection, refined_cameras_txt)
        image_id_by_name = read_image_id_by_name(connection)
        keypoint_count = insert_keypoints(connection, image_id_by_name, features_dir)
    finally:
        connection.close()
    stats = write_raw_match_list(rows, match_list_path)

    report_path = lightglue_output_dir / "reports" / "colmap_database_build.txt"
    lines = [
        "COLMAP learned-feature database build",
        "",
        f"source_db: {source_db}",
        f"baseline_cameras_txt: {refined_cameras_txt or ''}",
        f"output_db: {output_db}",
        f"raw_match_list: {match_list_path}",
        f"refined_camera_rows: {refined_camera_count}",
        f"keypoint_images: {keypoint_count}",
        f"raw_match_pairs: {stats['raw_match_pairs']}",
        f"raw_matches: {stats['raw_matches']}",
        "geometry_pairs: generated later by colmap matches_importer",
    ]
    write_report(report_path, lines)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
