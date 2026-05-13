#!/usr/bin/env python3
import argparse
import csv
import html
import json
import math
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(image_dir: Path):
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_regions(images, num_regions: int):
    if num_regions <= 0:
        raise ValueError("--num-regions must be positive")
    region_size = math.ceil(len(images) / num_regions)
    regions = []
    for region_index in range(num_regions):
        start = region_index * region_size
        end = min(start + region_size, len(images))
        if start >= len(images):
            break
        regions.append(
            {
                "id": f"region_{region_index + 1:02d}",
                "label": f"Region {region_index + 1}",
                "start_index": start,
                "end_index": end - 1,
                "images": images[start:end],
            }
        )
    return regions


def pick_representatives(region, samples_per_region: int):
    images = region["images"]
    if len(images) <= samples_per_region:
        return images
    if samples_per_region == 1:
        return [images[len(images) // 2]]
    step = (len(images) - 1) / (samples_per_region - 1)
    indexes = sorted({round(i * step) for i in range(samples_per_region)})
    return [images[index] for index in indexes]


def make_thumbnail(src: Path, dst: Path, max_size: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.thumbnail((max_size, max_size))
        image.convert("RGB").save(dst, quality=86)


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
                "representatives",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_index_html(path: Path, title: str, rows):
    cards = []
    for row in rows:
        thumbs = "\n".join(
            f'<figure><img src="{html.escape(thumb)}" alt="{html.escape(name)}">'
            f'<figcaption>{html.escape(name)}</figcaption></figure>'
            for name, thumb in row["thumbnail_pairs"]
        )
        cards.append(
            f"""
      <section class="region-card">
        <div class="region-header">
          <h2>{html.escape(row["label"])}</h2>
          <span>{html.escape(row["region_id"])}</span>
        </div>
        <dl>
          <div><dt>Frame range</dt><dd>{row["start_index"]} - {row["end_index"]}</dd></div>
          <div><dt>Frame count</dt><dd>{row["frame_count"]}</dd></div>
          <div><dt>Start</dt><dd>{html.escape(row["start_image"])}</dd></div>
          <div><dt>End</dt><dd>{html.escape(row["end_image"])}</dd></div>
        </dl>
        <div class="thumb-grid">
          {thumbs}
        </div>
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
    :root {{
      color-scheme: light;
      font-family: Inter, Arial, sans-serif;
      color: #16202a;
      background: #f6f7f9;
    }}
    body {{
      margin: 0;
      padding: 28px;
    }}
    header {{
      max-width: 1180px;
      margin: 0 auto 22px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      font-weight: 700;
    }}
    .summary {{
      color: #596575;
      margin: 0;
      line-height: 1.5;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      display: grid;
      gap: 16px;
    }}
    .region-card {{
      background: #ffffff;
      border: 1px solid #dde2e8;
      border-radius: 8px;
      padding: 18px;
    }}
    .region-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid #e8ecf1;
      padding-bottom: 12px;
      margin-bottom: 14px;
    }}
    h2 {{
      margin: 0;
      font-size: 19px;
    }}
    .region-header span {{
      color: #667384;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
    }}
    dl {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 0 0 16px;
    }}
    dl div {{
      min-width: 0;
    }}
    dt {{
      color: #6a7483;
      font-size: 12px;
      margin-bottom: 4px;
    }}
    dd {{
      margin: 0;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .thumb-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 12px;
    }}
    figure {{
      margin: 0;
      border: 1px solid #e1e6ec;
      border-radius: 6px;
      overflow: hidden;
      background: #fbfcfd;
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
    }}
    figcaption {{
      padding: 7px 8px;
      font-size: 12px;
      color: #4e5b69;
      overflow-wrap: anywhere;
    }}
    @media (max-width: 720px) {{
      body {{
        padding: 16px;
      }}
      dl {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="summary">Fixed-frame image strategy prototype for room-level region partitioning. Region names are editable labels for manual review.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def build_demo(args):
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output)
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory not found: {image_dir}")

    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images found in: {image_dir}")

    if output_dir.exists() and args.clean:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = output_dir / "thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    json_regions = []
    regions = split_regions(images, args.num_regions)
    for region in regions:
        representatives = pick_representatives(region, args.samples_per_region)
        thumbnail_pairs = []
        for src in representatives:
            rel_thumb = Path("thumbnails") / region["id"] / f"{src.stem}.jpg"
            make_thumbnail(src, output_dir / rel_thumb, args.thumbnail_size)
            thumbnail_pairs.append((src.name, rel_thumb.as_posix()))

        row = {
            "region_id": region["id"],
            "label": region["label"],
            "start_index": region["start_index"],
            "end_index": region["end_index"],
            "frame_count": len(region["images"]),
            "start_image": region["images"][0].name,
            "end_image": region["images"][-1].name,
            "representatives": ";".join(path.name for path in representatives),
            "thumbnail_pairs": thumbnail_pairs,
        }
        rows.append(row)
        json_regions.append(
            {
                "region_id": row["region_id"],
                "label": row["label"],
                "start_index": row["start_index"],
                "end_index": row["end_index"],
                "frame_count": row["frame_count"],
                "start_image": row["start_image"],
                "end_image": row["end_image"],
                "representatives": [path.name for path in representatives],
            }
        )

    write_segments_csv(
        output_dir / "segments.csv",
        [{key: value for key, value in row.items() if key != "thumbnail_pairs"} for row in rows],
    )
    (output_dir / "segments.json").write_text(
        json.dumps(
            {
                "image_dir": str(image_dir),
                "image_count": len(images),
                "num_regions": len(regions),
                "strategy": "fixed_frame_count",
                "regions": json_regions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_index_html(output_dir / "index.html", args.title, rows)

    print(f"images={len(images)}")
    print(f"regions={len(regions)}")
    print(f"output={output_dir}")
    print(f"html={output_dir / 'index.html'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a room-level region partition demo from an image sequence.")
    parser.add_argument("--image-dir", required=True, help="Input RGB image directory")
    parser.add_argument("--output", required=True, help="Output demo directory")
    parser.add_argument("--num-regions", type=int, default=5, help="Number of fixed-frame regions")
    parser.add_argument("--samples-per-region", type=int, default=6, help="Representative thumbnails per region")
    parser.add_argument("--thumbnail-size", type=int, default=360, help="Maximum thumbnail side length")
    parser.add_argument("--title", default="Scene 09 Room-Level Region Demo", help="HTML page title")
    parser.add_argument("--clean", action="store_true", help="Remove output directory before generating")
    return parser.parse_args()


def main():
    build_demo(parse_args())


if __name__ == "__main__":
    main()
