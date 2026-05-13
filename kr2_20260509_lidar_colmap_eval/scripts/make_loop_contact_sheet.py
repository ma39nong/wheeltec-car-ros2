#!/usr/bin/env python3
"""Create a contact sheet for visually checking a proposed loop endpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def find_image(rgb_dir: Path, index: int) -> Path:
    matches = sorted(rgb_dir.glob(f"{index:06d}_*.png"))
    if not matches:
        raise FileNotFoundError(f"No RGB image found for index {index:06d} in {rgb_dir}")
    return matches[0]


def make_thumb(path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    scale = width / image.shape[1]
    resized = cv2.resize(image, (width, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y = max(0, (height - resized.shape[0]) // 2)
    canvas[y : y + min(height, resized.shape[0]), :] = resized[:height]
    cv2.putText(canvas, path.name[:6], (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=20)
    args = parser.parse_args()

    start_indices = [args.start, args.start + 10, args.start + 20]
    end_indices = [max(args.start, args.end - args.radius), args.end, args.end + args.radius]
    paths = [find_image(args.rgb_dir, idx) for idx in start_indices + end_indices]
    thumbs = [make_thumb(path, 300, 240) for path in paths]
    sheet = np.vstack([np.hstack(thumbs[:3]), np.hstack(thumbs[3:])])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
