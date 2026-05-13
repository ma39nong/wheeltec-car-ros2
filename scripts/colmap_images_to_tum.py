#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


def normalize_quat(qw, qx, qy, qz):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("Zero-norm quaternion")
    return qw / norm, qx / norm, qy / norm, qz / norm


def quat_to_rot(qw, qx, qy, qz):
    qw, qx, qy, qz = normalize_quat(qw, qx, qy, qz)
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def rot_to_quat(r):
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        qw = (r[2][1] - r[1][2]) / s
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        qw = (r[0][2] - r[2][0]) / s
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        qw = (r[1][0] - r[0][1]) / s
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s
    return normalize_quat(qw, qx, qy, qz)


def parse_colmap_images(images_txt):
    entries = []
    with images_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 10:
                continue
            try:
                image_id = int(parts[0])
                qw = float(parts[1])
                qx = float(parts[2])
                qy = float(parts[3])
                qz = float(parts[4])
                tx = float(parts[5])
                ty = float(parts[6])
                tz = float(parts[7])
            except ValueError:
                continue
            entries.append((image_id, qw, qx, qy, qz, tx, ty, tz, parts[9]))
    entries.sort(key=lambda row: (filename_timestamp(row[8]) is None, filename_timestamp(row[8]) or row[0]))
    return entries


def filename_timestamp(name):
    stem = Path(name).stem
    candidates = [stem]
    if "_" in stem:
        candidates.insert(0, stem.split("_")[-1])

    for candidate in candidates:
        try:
            timestamp = float(candidate)
        except ValueError:
            continue
        if timestamp > 1e12:
            timestamp /= 1e9
        return timestamp
    return None


def convert_to_tum(entries, fps_fallback):
    tum_rows = []
    current_ts = 0.0
    dt = 1.0 / fps_fallback

    for _, qw, qx, qy, qz, tx, ty, tz, name in entries:
        r_wc = quat_to_rot(qw, qx, qy, qz)
        r_cw = [
            [r_wc[0][0], r_wc[1][0], r_wc[2][0]],
            [r_wc[0][1], r_wc[1][1], r_wc[2][1]],
            [r_wc[0][2], r_wc[1][2], r_wc[2][2]],
        ]
        cx = -(r_cw[0][0] * tx + r_cw[0][1] * ty + r_cw[0][2] * tz)
        cy = -(r_cw[1][0] * tx + r_cw[1][1] * ty + r_cw[1][2] * tz)
        cz = -(r_cw[2][0] * tx + r_cw[2][1] * ty + r_cw[2][2] * tz)
        qw_cw, qx_cw, qy_cw, qz_cw = rot_to_quat(r_cw)

        ts = filename_timestamp(name)
        if ts is None:
            ts = current_ts
            current_ts += dt

        tum_rows.append((ts, cx, cy, cz, qx_cw, qy_cw, qz_cw, qw_cw))

    return tum_rows


def main():
    parser = argparse.ArgumentParser(description="Convert COLMAP images.txt to TUM trajectory format")
    parser.add_argument("--input", required=True, help="Path to COLMAP images.txt")
    parser.add_argument("--output", required=True, help="Output TUM file path")
    parser.add_argument("--fps-fallback", type=float, default=30.0, help="Fallback FPS if filenames are not timestamps")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries = parse_colmap_images(input_path)
    if not entries:
        raise RuntimeError(f"No image poses found in {input_path}")

    tum_rows = convert_to_tum(entries, args.fps_fallback)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in tum_rows:
            handle.write("{:.6f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n".format(*row))

    print(f"Wrote {len(tum_rows)} poses to {output_path}")


if __name__ == "__main__":
    main()
