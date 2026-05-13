#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def max_float(rows, key):
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return max(values) if values else 0.0


def avg_float(rows, key):
    values = [float(row[key]) for row in rows if row.get(key, "") != ""]
    return sum(values) / len(values) if values else 0.0


def gpu_summary(rows):
    if not rows:
        return {"gpu_detected": False}
    util_values = [float(row["utilization_gpu"]) for row in rows if row.get("utilization_gpu", "") != ""]
    mem_values = [float(row["memory_used_mb"]) for row in rows if row.get("memory_used_mb", "") != ""]
    return {
        "gpu_detected": True,
        "gpu_samples": len(rows),
        "gpu_names": sorted({row["name"] for row in rows if row.get("name")}),
        "max_gpu_utilization_percent": max(util_values) if util_values else 0.0,
        "avg_gpu_utilization_percent": (sum(util_values) / len(util_values)) if util_values else 0.0,
        "max_gpu_memory_used_mb": max(mem_values) if mem_values else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize COLMAP resource monitor CSV logs")
    parser.add_argument("--resource-dir", required=True, help="Directory containing system_usage.csv/process_usage.csv/gpu_usage.csv")
    args = parser.parse_args()

    resource_dir = Path(args.resource_dir)
    system_rows = read_rows(resource_dir / "system_usage.csv")
    process_rows = read_rows(resource_dir / "process_usage.csv")
    gpu_rows = read_rows(resource_dir / "gpu_usage.csv")

    summary = {
        "resource_dir": str(resource_dir),
        "system_samples": len(system_rows),
        "process_samples": len(process_rows),
        "max_system_cpu_percent": max_float(system_rows, "cpu_percent"),
        "avg_system_cpu_percent": avg_float(system_rows, "cpu_percent"),
        "max_system_mem_used_gb": max_float(system_rows, "mem_used_kb") / 1024 / 1024,
        "avg_system_mem_used_gb": avg_float(system_rows, "mem_used_kb") / 1024 / 1024,
        "max_colmap_proc_count": max_float(process_rows, "colmap_proc_count"),
        "max_colmap_cpu_percent": max_float(process_rows, "colmap_cpu_percent"),
        "avg_colmap_cpu_percent": avg_float(process_rows, "colmap_cpu_percent"),
        "max_colmap_rss_gb": max_float(process_rows, "colmap_rss_kb") / 1024 / 1024,
        "avg_colmap_rss_gb": avg_float(process_rows, "colmap_rss_kb") / 1024 / 1024,
        "max_colmap_threads": max_float(process_rows, "colmap_threads"),
    }
    summary.update(gpu_summary(gpu_rows))

    out_path = resource_dir / "resource_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
