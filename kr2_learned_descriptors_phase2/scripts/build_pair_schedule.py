#!/usr/bin/env python3
"""Build a deterministic image-pair schedule for KR2 learned matching."""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def parse_strides(value: str) -> list[int]:
    strides = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not strides:
        raise ValueError("PAIR_STRIDES must contain at least one integer stride")
    if any(stride <= 0 for stride in strides):
        raise ValueError("PAIR_STRIDES must only contain positive integers")
    return strides


def list_images(image_dir: Path) -> list[Path]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def build_pairs(images: list[Path], strides: list[int], max_pairs_per_stride: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    pair_id = 0
    for stride in strides:
        count_for_stride = 0
        for idx0 in range(0, len(images) - stride):
            if max_pairs_per_stride > 0 and count_for_stride >= max_pairs_per_stride:
                break
            idx1 = idx0 + stride
            rows.append(
                {
                    "pair_id": pair_id,
                    "stride": stride,
                    "image0_index": idx0,
                    "image1_index": idx1,
                    "image0_name": images[idx0].name,
                    "image1_name": images[idx1].name,
                    "image0_path": str(images[idx0]),
                    "image1_path": str(images[idx1]),
                }
            )
            pair_id += 1
            count_for_stride += 1
    return rows


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No pairs were generated")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def safe_tag(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def archive_previous_output(output_dir: Path, config: dict[str, str]) -> Path | None:
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return None

    archive_root = output_dir.parent / f"{output_dir.name}_archives"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_parts = [
        f"strides{config.get('PAIR_STRIDES', 'unknown').replace(',', '-')}",
        f"kp{config.get('MAX_KEYPOINTS', 'unknown')}",
        f"ransac{config.get('RANSAC_REPROJ_THRESHOLD', 'unknown')}",
    ]
    archive_dir = archive_root / safe_tag(f"{timestamp}_{'_'.join(tag_parts)}")
    archive_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, archive_dir)
    return archive_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = read_env(args.config)
    image_dir = Path(config["IMAGE_DIR"])
    output_dir = Path(config["OUTPUT_DIR"])
    strides = parse_strides(config.get("PAIR_STRIDES", "1,4,8"))
    max_pairs_per_stride = int(config.get("MAX_PAIRS_PER_STRIDE", "0"))

    if enabled(config.get("ARCHIVE_PREVIOUS_OUTPUTS"), default=True):
        archived = archive_previous_output(output_dir, config)
        if archived is not None:
            print(f"archived_previous_output: {archived}")

    images = list_images(image_dir)
    rows = build_pairs(images, strides, max_pairs_per_stride)
    schedule_path = output_dir / "reports" / "pair_schedule.csv"
    write_csv(schedule_path, rows)

    print(f"images: {len(images)}")
    print(f"strides: {','.join(str(stride) for stride in strides)}")
    print(f"pairs: {len(rows)}")
    print(f"wrote: {schedule_path}")


if __name__ == "__main__":
    main()
