#!/usr/bin/env python3
import argparse
import csv
from datetime import datetime
from pathlib import Path


def count_registered_images(images_txt: Path) -> int:
    count = 0
    with images_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 10:
                try:
                    int(parts[0])
                except ValueError:
                    continue
                count += 1
    return count


def count_sparse_points(points3d_txt: Path) -> int:
    count = 0
    with points3d_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Summarize COLMAP sparse model into one reproducible metrics row")
    parser.add_argument("--images-txt", required=True, help="Path to sparse_txt/images.txt")
    parser.add_argument("--points3d-txt", required=True, help="Path to sparse_txt/points3D.txt")
    parser.add_argument("--total-images", type=int, required=True, help="Total input image count")
    parser.add_argument("--duration-sec", type=float, required=True, help="End-to-end runtime in seconds")
    parser.add_argument("--scene-name", default="scene_01", help="Scene name")
    parser.add_argument("--changed-variable", default="none", help="Name of the single variable changed for this run")
    parser.add_argument("--variable-value", default="baseline", help="Value of the changed variable")
    parser.add_argument("--notes", default="", help="Free-form note for this run")
    parser.add_argument("--output", required=True, help="CSV output path")
    args = parser.parse_args()

    images_txt = Path(args.images_txt)
    points3d_txt = Path(args.points3d_txt)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    registered_images = count_registered_images(images_txt)
    sparse_points = count_sparse_points(points3d_txt)
    success = "yes" if registered_images > 0 and sparse_points > 0 else "no"

    row = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "scene_name": args.scene_name,
        "total_images": args.total_images,
        "registered_images": registered_images,
        "sparse_points": sparse_points,
        "duration_sec": f"{args.duration_sec:.2f}",
        "success": success,
        "changed_variable": args.changed_variable,
        "variable_value": args.variable_value,
        "notes": args.notes,
    }

    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists or output_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)

    print(f"registered_images={registered_images}")
    print(f"sparse_points={sparse_points}")
    print(f"success={success}")
    print(f"appended_metrics={output_path}")


if __name__ == "__main__":
    main()
