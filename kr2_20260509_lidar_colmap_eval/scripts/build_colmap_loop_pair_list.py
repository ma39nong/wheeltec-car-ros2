#!/usr/bin/env python3
"""Build a focused COLMAP pair list for first/last keyframe loop closure."""

from __future__ import annotations

import argparse
from pathlib import Path


def list_images(image_dir: Path) -> list[str]:
    names = sorted(path.name for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not names:
        raise RuntimeError(f"No images found: {image_dir}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=int, default=40, help="Number of images from each end to connect.")
    parser.add_argument("--band", type=int, default=8, help="Match each first-window image to nearby reversed last-window images.")
    parser.add_argument("--dense-endpoints", action="store_true", help="Also add all first-window x last-window pairs.")
    args = parser.parse_args()

    if args.window < 1:
        raise ValueError("--window must be >= 1")
    if args.band < 0:
        raise ValueError("--band must be >= 0")

    names = list_images(args.image_dir)
    window = min(args.window, len(names) // 2)
    first = names[:window]
    last = names[-window:]

    pairs: set[tuple[str, str]] = set()
    for i, name_a in enumerate(first):
        # A loop closes with opposite temporal direction: first[0] is closest to last[-1].
        center = window - 1 - i
        lo = max(0, center - args.band)
        hi = min(window - 1, center + args.band)
        for j in range(lo, hi + 1):
            pairs.add((name_a, last[j]))

    if args.dense_endpoints:
        for name_a in first:
            for name_b in last:
                pairs.add((name_a, name_b))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for name_a, name_b in sorted(pairs):
            handle.write(f"{name_a} {name_b}\n")

    print(f"images: {len(names)}")
    print(f"window: {window}")
    print(f"band: {args.band}")
    print(f"pairs: {len(pairs)}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
