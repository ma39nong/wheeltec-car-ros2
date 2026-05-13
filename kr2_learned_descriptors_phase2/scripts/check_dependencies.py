#!/usr/bin/env python3
"""Write a small dependency report for the KR2 learned-descriptor workspace."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def add_local_lightglue_if_present() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    local_lightglue = repo_root / "LightGlue"
    if local_lightglue.exists() and str(local_lightglue) not in sys.path:
        sys.path.insert(0, str(local_lightglue))


def module_status(name: str) -> str:
    return "available" if importlib.util.find_spec(name) is not None else "missing"


def main() -> None:
    add_local_lightglue_if_present()
    output = Path("kr2_learned_descriptors_phase2/outputs/scene_09_superpoint_lightglue/reports/dependency_check.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "KR2 Phase 2 dependency check",
        "",
        f"torch: {module_status('torch')}",
        f"cv2: {module_status('cv2')}",
        f"numpy: {module_status('numpy')}",
        f"lightglue: {module_status('lightglue')}",
        "",
        "The pair schedule can be generated without lightglue.",
        "SuperPoint + LightGlue matching requires the lightglue package.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
