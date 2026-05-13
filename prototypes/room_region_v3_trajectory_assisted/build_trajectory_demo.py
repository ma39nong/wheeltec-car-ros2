#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(image_dir: Path):
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalize_quat(qw, qx, qy, qz):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / norm, qx / norm, qy / norm, qz / norm


def quat_to_rot(qw, qx, qy, qz):
    qw, qx, qy, qz = normalize_quat(qw, qx, qy, qz)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def parse_colmap_positions(images_txt: Path):
    positions = {}
    with images_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 10:
                continue
            try:
                int(parts[0])
                qw, qx, qy, qz = [float(value) for value in parts[1:5]]
                tx, ty, tz = [float(value) for value in parts[5:8]]
            except ValueError:
                continue
            name = parts[9]
            r_wc = quat_to_rot(qw, qx, qy, qz)
            r_cw = r_wc.T
            t = np.array([tx, ty, tz], dtype=np.float64)
            center = -r_cw @ t
            positions[name] = {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}
    return positions


def ordered_pose_rows(images, positions):
    rows = []
    for index, image in enumerate(images):
        position = positions.get(image.name)
        if position is None:
            continue
        rows.append({"index": index, "image": image, **position})
    return rows


def cumulative_distances(rows):
    distances = [0.0]
    total = 0.0
    for previous, current in zip(rows, rows[1:]):
        dx = current["x"] - previous["x"]
        dy = current["y"] - previous["y"]
        dz = current["z"] - previous["z"]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
        distances.append(total)
    return distances


def choose_path_boundaries(rows, num_regions):
    if len(rows) < num_regions:
        return []
    cumulative = cumulative_distances(rows)
    total = cumulative[-1]
    if total <= 0:
        step = math.ceil(len(rows) / num_regions)
        return [rows[index]["index"] for index in range(step, len(rows), step)][: num_regions - 1]
    boundaries = []
    for region_index in range(1, num_regions):
        target = total * region_index / num_regions
        local_index = min(range(len(cumulative)), key=lambda index: abs(cumulative[index] - target))
        boundaries.append(rows[local_index]["index"])
    return sorted(set(boundaries))


def regions_from_boundaries(images, boundaries):
    starts = [0] + boundaries
    ends = boundaries + [len(images)]
    regions = []
    for index, (start, end) in enumerate(zip(starts, ends), start=1):
        regions.append(
            {
                "id": f"trajectory_region_{index:02d}",
                "label": f"Trajectory Region {index}",
                "start_index": start,
                "end_index": end - 1,
                "images": images[start:end],
            }
        )
    return regions


def pick_representatives(images, samples_per_region):
    if len(images) <= samples_per_region:
        return images
    if samples_per_region == 1:
        return [images[len(images) // 2]]
    indexes = sorted({round(i * (len(images) - 1) / (samples_per_region - 1)) for i in range(samples_per_region)})
    return [images[index] for index in indexes]


def make_thumbnail(src: Path, dst: Path, max_size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {src}")
    height, width = image.shape[:2]
    scale = min(max_size / width, max_size / height, 1.0)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), resized, [cv2.IMWRITE_JPEG_QUALITY, 86])


def write_trajectory_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "image", "x", "y", "z"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"index": row["index"], "image": row["image"].name, "x": row["x"], "y": row["y"], "z": row["z"]})


def write_segments_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "region_id",
                "label",
                "start_index",
                "end_index",
                "frame_count",
                "start_image",
                "end_image",
                "path_length",
                "representatives",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def region_path_length(region, positions):
    total = 0.0
    region_images = region["images"]
    for previous, current in zip(region_images, region_images[1:]):
        a = positions.get(previous.name)
        b = positions.get(current.name)
        if not a or not b:
            continue
        total += math.sqrt((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2 + (b["z"] - a["z"]) ** 2)
    return total


def write_svg(path: Path, pose_rows, boundaries):
    if not pose_rows:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return
    xs = [row["x"] for row in pose_rows]
    zs = [row["z"] for row in pose_rows]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    width, height, pad = 880, 520, 36
    sx = (width - 2 * pad) / max(max_x - min_x, 1e-6)
    sz = (height - 2 * pad) / max(max_z - min_z, 1e-6)

    def project(row):
        x = pad + (row["x"] - min_x) * sx
        y = height - pad - (row["z"] - min_z) * sz
        return x, y

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (project(row) for row in pose_rows))
    markers = []
    boundary_set = set(boundaries)
    for row in pose_rows:
        x, y = project(row)
        color = "#c0392b" if row["index"] in boundary_set else "#2c7fb8"
        radius = 5 if row["index"] in boundary_set else 2.5
        markers.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{radius}' fill='{color}'><title>{html.escape(row['image'].name)}</title></circle>")

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f8fafc"/>
  <polyline points="{points}" fill="none" stroke="#334e68" stroke-width="2"/>
  {''.join(markers)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_html(path: Path, rows, svg_name, title):
    cards = []
    for row in rows:
        thumbs = "\n".join(
            f'<figure><img src="{html.escape(thumb)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
            for name, thumb in row["thumbnail_pairs"]
        )
        cards.append(
            f"""
      <section>
        <h2>{html.escape(row['label'])} <span>{html.escape(row['region_id'])}</span></h2>
        <p>Frames {row['start_index']} - {row['end_index']} · {row['frame_count']} images · path length {row['path_length']:.3f}</p>
        <div class="thumbs">{thumbs}</div>
      </section>
"""
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{ margin: 0; padding: 28px; font-family: Arial, sans-serif; background: #f5f7fa; color: #16202a; }}
    header, main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .note {{ color: #5d6878; line-height: 1.5; }}
    section {{ background: white; border: 1px solid #dfe5ec; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    h2 {{ margin: 0 0 6px; font-size: 19px; }}
    h2 span {{ float: right; color: #687586; font-size: 13px; font-weight: 400; }}
    p {{ color: #576474; margin: 0 0 14px; }}
    object {{ width: 100%; height: 420px; border: 1px solid #e0e5eb; border-radius: 6px; background: #f8fafc; }}
    .thumbs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; border: 1px solid #e0e5eb; border-radius: 6px; overflow: hidden; background: #fbfcfd; }}
    img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; }}
    figcaption {{ padding: 7px 8px; font-size: 12px; color: #4f5b69; overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="note">Boundaries are selected from COLMAP camera-center trajectory positions by splitting cumulative path length into balanced regions.</p>
  </header>
  <main>
    <section>
      <h2>Trajectory Overview</h2>
      <object data="{html.escape(svg_name)}" type="image/svg+xml"></object>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def count_tum_rows(path: Path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip() and not line.startswith("#"))


def main():
    parser = argparse.ArgumentParser(description="Build trajectory-assisted room region demo.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--images-txt", required=True, help="COLMAP sparse_txt/images.txt with image names and poses")
    parser.add_argument("--trajectory-tum", required=True, help="COLMAP-derived TUM trajectory for provenance/checking")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-regions", type=int, default=5)
    parser.add_argument("--samples-per-region", type=int, default=6)
    parser.add_argument("--thumbnail-size", type=int, default=360)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    images_txt = Path(args.images_txt)
    trajectory_tum = Path(args.trajectory_tum)
    output_dir = Path(args.output)
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    positions = parse_colmap_positions(images_txt)
    pose_rows = ordered_pose_rows(images, positions)
    if not pose_rows:
        raise SystemExit("No image poses could be matched to input images.")

    boundaries = choose_path_boundaries(pose_rows, args.num_regions)
    regions = regions_from_boundaries(images, boundaries)
    rows = []
    json_regions = []

    for region in regions:
        reps = pick_representatives(region["images"], args.samples_per_region)
        thumbnail_pairs = []
        for src in reps:
            rel_thumb = Path("thumbnails") / region["id"] / f"{src.stem}.jpg"
            make_thumbnail(src, output_dir / rel_thumb, args.thumbnail_size)
            thumbnail_pairs.append((src.name, rel_thumb.as_posix()))
        path_length = region_path_length(region, positions)
        row = {
            "region_id": region["id"],
            "label": region["label"],
            "start_index": region["start_index"],
            "end_index": region["end_index"],
            "frame_count": len(region["images"]),
            "start_image": region["images"][0].name,
            "end_image": region["images"][-1].name,
            "path_length": path_length,
            "representatives": ";".join(path.name for path in reps),
            "thumbnail_pairs": thumbnail_pairs,
        }
        rows.append(row)
        json_regions.append({key: row[key] for key in row if key not in {"thumbnail_pairs"}})

    write_segments_csv(output_dir / "segments.csv", [{key: row[key] for key in row if key != "thumbnail_pairs"} for row in rows])
    write_trajectory_csv(output_dir / "trajectory_positions.csv", pose_rows)
    write_svg(output_dir / "trajectory_overview.svg", pose_rows, boundaries)
    (output_dir / "segments.json").write_text(
        json.dumps(
            {
                "image_dir": str(image_dir),
                "images_txt": str(images_txt),
                "trajectory_tum": str(trajectory_tum),
                "trajectory_tum_rows": count_tum_rows(trajectory_tum),
                "matched_pose_count": len(pose_rows),
                "image_count": len(images),
                "strategy": "colmap_trajectory_path_length",
                "boundaries": boundaries,
                "regions": json_regions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_html(output_dir / "index.html", rows, "trajectory_overview.svg", "V3 Trajectory-Assisted Region Demo")
    print(f"images={len(images)}")
    print(f"matched_poses={len(pose_rows)}")
    print(f"tum_rows={count_tum_rows(trajectory_tum)}")
    print(f"boundaries={boundaries}")
    print(f"regions={len(regions)}")
    print(f"html={output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
