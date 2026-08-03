#!/usr/bin/env python3
"""Print face counts for an organic project. Usage: python scripts/diagnose_organic.py shoe_cli_full"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    slug = sys.argv[1] if len(sys.argv) > 1 else "shoe_cli_full"
    project = ROOT / "data" / "raw" / "organic" / slug
    if not project.is_dir():
        raise SystemExit(f"Missing project: {project}")

    names = [
        "source.glb",
        "source.stl",
        "source_120mm.stl",
        "source_repaired.stl",
        "parts/part_01.stl",
        "parts/part_02.stl",
    ]
    for name in names:
        fp = project / name
        if not fp.exists():
            print(f"{name}: MISSING")
            continue
        mesh = trimesh.load(fp, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        print(f"{name}: faces={len(mesh.faces)} extents={mesh.extents.round(2)}")

    print("---")
    print("git:", subprocess.check_output(["git", "log", "-1", "--oneline"], cwd=ROOT, text=True).strip())
    print("branch:", subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip())


if __name__ == "__main__":
    main()
