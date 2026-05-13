#!/usr/bin/env python3
import argparse
import csv
import html
import json
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


def image_feature(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_hist = cv2.resize(edges, (20, 15), interpolation=cv2.INTER_AREA).flatten().astype(np.float32) / 255.0
    return np.concatenate([hist.astype(np.float32), edge_hist])


def visual_distances(images):
    features = [image_feature(path) for path in images]
    distances = []
    for index in range(1, len(features)):
        color_texture_distance = float(np.linalg.norm(features[index] - features[index - 1]))
        distances.append({"after_index": index - 1, "before_index": index, "score": color_texture_distance})
    return distances


def choose_boundaries(distances, image_count, num_regions, min_segment_len):
    candidates = sorted(distances, key=lambda row: row["score"], reverse=True)
    boundaries = []
    for candidate in candidates:
        boundary = candidate["before_index"]
        proposed = sorted(boundaries + [boundary])
        starts = [0] + proposed
        ends = proposed + [image_count]
        if all((end - start) >= min_segment_len for start, end in zip(starts, ends)):
            boundaries.append(boundary)
        if len(boundaries) >= num_regions - 1:
            break
    return sorted(boundaries)


def regions_from_boundaries(images, boundaries):
    starts = [0] + boundaries
    ends = boundaries + [len(images)]
    regions = []
    for index, (start, end) in enumerate(zip(starts, ends), start=1):
        regions.append(
            {
                "id": f"visual_region_{index:02d}",
                "label": f"Visual Region {index}",
                "start_index": start,
                "end_index": end - 1,
                "images": images[start:end],
            }
        )
    return regions


def parse_labels(label_text, region_count):
    if not label_text:
        return []
    labels = [item.strip() for item in label_text.split("|") if item.strip()]
    if len(labels) != region_count:
        raise SystemExit(f"--labels expects {region_count} labels separated by '|', got {len(labels)}")
    return labels


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


def make_boundary_pair(before: Path, after: Path, dst: Path, max_size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    before_image = cv2.imread(str(before), cv2.IMREAD_COLOR)
    after_image = cv2.imread(str(after), cv2.IMREAD_COLOR)
    if before_image is None:
        raise RuntimeError(f"Failed to read image: {before}")
    if after_image is None:
        raise RuntimeError(f"Failed to read image: {after}")
    target_width = max_size
    target_height = int(max_size * 0.75)
    before_resized = cv2.resize(before_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    after_resized = cv2.resize(after_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    separator = np.full((target_height, 8, 3), 245, dtype=np.uint8)
    pair = np.hstack([before_resized, separator, after_resized])
    cv2.imwrite(str(dst), pair, [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_score_curve_svg(path: Path, distances, boundaries):
    width, height = 980, 320
    pad_left, pad_right, pad_top, pad_bottom = 46, 18, 26, 38
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    max_score = max((item["score"] for item in distances), default=1.0)
    max_index = max((item["before_index"] for item in distances), default=1)
    boundary_set = set(boundaries)

    def x_for(index):
        return pad_left + (index / max(max_index, 1)) * plot_width

    def y_for(score):
        return pad_top + plot_height - (score / max(max_score, 1e-6)) * plot_height

    points = " ".join(f"{x_for(item['before_index']):.2f},{y_for(item['score']):.2f}" for item in distances)
    boundary_lines = []
    for boundary in boundaries:
        x = x_for(boundary)
        boundary_lines.append(
            f"<line x1='{x:.2f}' y1='{pad_top}' x2='{x:.2f}' y2='{pad_top + plot_height}' stroke='#c0392b' stroke-width='2'/>"
            f"<text x='{x + 4:.2f}' y='{pad_top + 14}' fill='#8f2f24' font-size='12'>b={boundary}</text>"
        )
    score_points = []
    for item in distances:
        if item["before_index"] in boundary_set:
            score_points.append(
                f"<circle cx='{x_for(item['before_index']):.2f}' cy='{y_for(item['score']):.2f}' r='4' fill='#c0392b'/>"
            )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfcfd"/>
  <line x1="{pad_left}" y1="{pad_top + plot_height}" x2="{pad_left + plot_width}" y2="{pad_top + plot_height}" stroke="#8a97a8"/>
  <line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top + plot_height}" stroke="#8a97a8"/>
  <text x="{pad_left}" y="{height - 8}" fill="#536273" font-size="13">frame index</text>
  <text x="8" y="{pad_top + 12}" fill="#536273" font-size="13">score</text>
  {''.join(boundary_lines)}
  <polyline points="{points}" fill="none" stroke="#2c7fb8" stroke-width="2"/>
  {''.join(score_points)}
  <text x="{pad_left}" y="18" fill="#536273" font-size="13">Visual change score curve; red lines are selected boundaries</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_boundary_reviews(images, distances, boundaries, output_dir: Path, pair_size: int):
    scores = {item["before_index"]: item["score"] for item in distances}
    reviews = []
    for boundary in boundaries:
        before_index = boundary - 1
        after_index = boundary
        if before_index < 0 or after_index >= len(images):
            continue
        rel_pair = Path("boundary_pairs") / f"boundary_{before_index:03d}_to_{after_index:03d}.jpg"
        make_boundary_pair(images[before_index], images[after_index], output_dir / rel_pair, pair_size)
        reviews.append(
            {
                "boundary": boundary,
                "before_index": before_index,
                "after_index": after_index,
                "before_image": images[before_index].name,
                "after_image": images[after_index].name,
                "score": scores.get(boundary, 0.0),
                "pair_image": rel_pair.as_posix(),
            }
        )
    return reviews


def write_csv(path: Path, rows):
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
                "boundary_reason",
                "representatives",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows, distances, boundary_reviews, score_curve_svg, title):
    score_rows = "\n".join(
        f"<tr><td>{item['after_index']} -> {item['before_index']}</td><td>{item['score']:.4f}</td></tr>"
        for item in sorted(distances, key=lambda row: row["score"], reverse=True)[:12]
    )
    boundary_cards = "\n".join(
        f"""
      <article class="boundary-card">
        <h3>Boundary {review['before_index']} -> {review['after_index']}</h3>
        <p>score = {review['score']:.4f}</p>
        <img src="{html.escape(review['pair_image'])}" alt="Boundary {review['before_index']} to {review['after_index']}">
        <div class="boundary-names">
          <span>{html.escape(review['before_image'])}</span>
          <span>{html.escape(review['after_image'])}</span>
        </div>
      </article>
"""
        for review in boundary_reviews
    )
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
        <p>Frames {row['start_index']} - {row['end_index']} · {row['frame_count']} images · {html.escape(row['boundary_reason'])}</p>
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
    header {{ margin-bottom: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .note {{ margin: 0; color: #5d6878; line-height: 1.5; }}
    section {{ background: white; border: 1px solid #dfe5ec; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
    h2 {{ margin: 0 0 6px; font-size: 19px; }}
    h3 {{ margin: 0 0 6px; font-size: 15px; }}
    h2 span {{ float: right; color: #687586; font-size: 13px; font-weight: 400; }}
    p {{ color: #576474; margin: 0 0 14px; }}
    .thumbs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }}
    .boundary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
    .boundary-card {{ border: 1px solid #e0e5eb; border-radius: 6px; padding: 12px; background: #fbfcfd; }}
    .boundary-card img {{ aspect-ratio: 8 / 3; object-fit: cover; border: 1px solid #e5e9ef; border-radius: 4px; }}
    .boundary-names {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 8px; color: #4f5b69; font-size: 12px; overflow-wrap: anywhere; }}
    object {{ width: 100%; height: 320px; border: 1px solid #e0e5eb; border-radius: 6px; background: #fbfcfd; }}
    figure {{ margin: 0; border: 1px solid #e0e5eb; border-radius: 6px; overflow: hidden; background: #fbfcfd; }}
    img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: cover; }}
    figcaption {{ padding: 7px 8px; font-size: 12px; color: #4f5b69; overflow-wrap: anywhere; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #e5e9ef; padding: 8px; text-align: left; font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="note">Boundaries are selected from the largest adjacent-frame color and texture changes, with a minimum segment length constraint.</p>
  </header>
  <main>
    <section>
      <h2>Visual Change Score Curve</h2>
      <object data="{html.escape(score_curve_svg)}" type="image/svg+xml"></object>
    </section>
    <section>
      <h2>Selected Boundary Reviews</h2>
      <div class="boundary-grid">{boundary_cards}</div>
    </section>
    <section>
      <h2>Top Visual Change Scores</h2>
      <table><thead><tr><th>Boundary</th><th>Score</th></tr></thead><tbody>{score_rows}</tbody></table>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build visual-change room region demo.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-regions", type=int, default=5)
    parser.add_argument("--min-segment-len", type=int, default=24)
    parser.add_argument("--samples-per-region", type=int, default=6)
    parser.add_argument("--thumbnail-size", type=int, default=360)
    parser.add_argument("--boundary-pair-size", type=int, default=360)
    parser.add_argument("--labels", default="", help="Optional region labels separated by '|'. Count must match generated regions.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output)
    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images found: {image_dir}")

    distances = visual_distances(images)
    boundaries = choose_boundaries(distances, len(images), args.num_regions, args.min_segment_len)
    boundary_scores = {item["before_index"]: item["score"] for item in distances}
    boundary_reviews = build_boundary_reviews(images, distances, boundaries, output_dir, args.boundary_pair_size)
    write_score_curve_svg(output_dir / "visual_change_curve.svg", distances, boundaries)
    regions = regions_from_boundaries(images, boundaries)
    labels = parse_labels(args.labels, len(regions))
    for index, label in enumerate(labels):
        regions[index]["label"] = label
    rows = []
    json_regions = []

    for region in regions:
        reps = pick_representatives(region["images"], args.samples_per_region)
        thumbnail_pairs = []
        for src in reps:
            rel_thumb = Path("thumbnails") / region["id"] / f"{src.stem}.jpg"
            make_thumbnail(src, output_dir / rel_thumb, args.thumbnail_size)
            thumbnail_pairs.append((src.name, rel_thumb.as_posix()))
        boundary_reason = "sequence start"
        if region["start_index"] > 0:
            boundary_reason = f"visual change score={boundary_scores.get(region['start_index'], 0.0):.4f}"
        row = {
            "region_id": region["id"],
            "label": region["label"],
            "start_index": region["start_index"],
            "end_index": region["end_index"],
            "frame_count": len(region["images"]),
            "start_image": region["images"][0].name,
            "end_image": region["images"][-1].name,
            "boundary_reason": boundary_reason,
            "representatives": ";".join(path.name for path in reps),
            "thumbnail_pairs": thumbnail_pairs,
        }
        rows.append(row)
        json_regions.append({key: row[key] for key in row if key not in {"thumbnail_pairs"}})

    write_csv(output_dir / "segments.csv", [{key: row[key] for key in row if key != "thumbnail_pairs"} for row in rows])
    (output_dir / "segments.json").write_text(
        json.dumps(
            {
                "image_dir": str(image_dir),
                "image_count": len(images),
                "strategy": "visual_color_texture_change",
                "boundaries": boundaries,
                "boundary_reviews": boundary_reviews,
                "regions": json_regions,
                "top_change_scores": sorted(distances, key=lambda row: row["score"], reverse=True)[:20],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_html(output_dir / "index.html", rows, distances, boundary_reviews, "visual_change_curve.svg", "V2 Visual-Change Region Demo")
    print(f"images={len(images)}")
    print(f"boundaries={boundaries}")
    print(f"regions={len(regions)}")
    print(f"html={output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
