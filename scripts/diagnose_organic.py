#!/usr/bin/env python3
"""Diagnose face counts + quality gates for an organic project.

Usage (repo root, macOS/Linux):
  cd /path/to/3D-Model-Segmentation-AI
  python scripts/diagnose_organic.py shoe_cli_full
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import trimesh
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_mesh(fp: Path) -> trimesh.Trimesh | None:
    if not fp.exists():
        return None
    mesh = trimesh.load(fp, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    return mesh


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
    meshes: dict[str, trimesh.Trimesh] = {}
    for name in names:
        fp = project / name
        mesh = _load_mesh(fp)
        if mesh is None:
            print(f"{name}: MISSING")
            continue
        meshes[name] = mesh
        print(f"{name}: faces={len(mesh.faces)} extents={mesh.extents.round(2)}")

    print("--- quality ratios ---")
    src = meshes.get("source_120mm.stl") or meshes.get("source.stl")
    repaired = meshes.get("source_repaired.stl")
    p1 = meshes.get("parts/part_01.stl")
    p2 = meshes.get("parts/part_02.stl")
    ok = True
    if src and repaired:
        ratio = len(repaired.faces) / max(len(src.faces), 1)
        gate = ratio >= 0.70
        ok = ok and gate
        print(f"repaired/source faces: {ratio:.2f}  [{'OK' if gate else 'FAIL'}] (min 0.70)")
    if src and p1 and p2:
        part_sum = len(p1.faces) + len(p2.faces)
        ratio = part_sum / max(len(src.faces), 1)
        # Pins may add faces; collapse is the failure mode
        gate = ratio >= 0.70
        ok = ok and gate
        print(
            f"parts_sum/source faces: {ratio:.2f}  [{'OK' if gate else 'FAIL'}] "
            f"(p1={len(p1.faces)} p2={len(p2.faces)})"
        )
        for label, part in (("part_01", p1), ("part_02", p2)):
            pr = len(part.faces) / max(len(src.faces), 1)
            gate_p = pr >= 0.15
            ok = ok and gate_p
            print(f"  {label}/source: {pr:.2f}  [{'OK' if gate_p else 'FAIL'}]")

    meta_path = project / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text()) or {}
        q = meta.get("quality") or {}
        print("--- meta.quality ---")
        print(f"ok: {q.get('ok')}")
        for stage in q.get("stages") or []:
            print(f"  [{stage.get('stage')}] ok={stage.get('ok')}")
            for c in stage.get("checks") or []:
                mark = "OK" if c.get("ok") else "FAIL"
                print(f"    [{mark}] {c.get('name')}: {c.get('message')}")
        org = meta.get("organic_cut") or {}
        if org:
            print(
                f"pins: applied={org.get('pins_applied')} method={org.get('pin_method')} "
                f"remesh={org.get('pin_used_remesh')}"
            )

    print("---")
    print("git:", subprocess.check_output(["git", "log", "-1", "--oneline"], cwd=ROOT, text=True).strip())
    print("branch:", subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip())
    print(f"diagnose: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
