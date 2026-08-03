"""Organic / aesthetic cut track: repair, mid-seam splits, P1S validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml

from .cut import plane_cut, validate_parts
from .printer import (
    aabb_extents,
    default_profile_path,
    load_printer_profile,
    part_fits_build_plate,
)
from .repair import RepairMode, repair_mesh


@dataclass
class OrganicCutResult:
    project_dir: Path
    parts_dir: Path
    part_paths: list[Path]
    all_fit: bool
    watertight: bool
    fit_reports: list[str]
    repaired_path: Path | None = None


def _load_print_mesh(project: Path) -> tuple[trimesh.Trimesh, Path]:
    meta_path = project / "meta.yaml"
    candidates: list[Path] = []
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text()) or {}
        print_stl = meta.get("print_stl")
        if print_stl:
            candidates.append(project / print_stl)
    candidates.extend(sorted(project.glob("source_*mm.stl")))
    candidates.append(project / "source.stl")

    for path in candidates:
        if path.is_file():
            mesh = trimesh.load(path, force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.to_geometry()
            return mesh, path
    raise FileNotFoundError(
        f"No print STL found in {project}. Run generate-from-image first."
    )


def longest_axis_midplane(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Aesthetic starter seam: mid-plane normal to the longest AABB axis."""
    verts = np.asarray(mesh.vertices, dtype=float)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    extents = maxs - mins
    axis = int(np.argmax(extents))
    origin = (mins + maxs) / 2.0
    normal = np.zeros(3, dtype=float)
    normal[axis] = 1.0
    return origin, normal


def split_until_fits(
    mesh: trimesh.Trimesh,
    profile,
    *,
    max_splits: int,
    force_split: bool,
) -> list[trimesh.Trimesh]:
    """
    Recursively mid-plane split along longest axis until parts fit (or budget).

    Organic heuristic only — not mechanical registers/pins.
    """
    ext = aabb_extents(np.asarray(mesh.vertices))
    fits, _ = part_fits_build_plate(ext, profile, allow_rotation=True)

    queue: list[tuple[trimesh.Trimesh, int]] = [(mesh, 0)]
    done: list[trimesh.Trimesh] = []

    # Optional first split even when the whole object already fits
    if force_split and fits and max_splits >= 1:
        origin, normal = longest_axis_midplane(mesh)
        parts = plane_cut(mesh, origin=origin, normal=normal)
        if len(parts) >= 2:
            queue = [(p, 1) for p in parts]
        else:
            return [mesh]

    while queue:
        current, depth = queue.pop(0)
        ext = aabb_extents(np.asarray(current.vertices))
        ok, _ = part_fits_build_plate(ext, profile, allow_rotation=True)
        if ok or depth >= max_splits:
            done.append(current)
            continue

        origin, normal = longest_axis_midplane(current)
        parts = plane_cut(current, origin=origin, normal=normal)
        if len(parts) < 2:
            done.append(current)
            continue
        for p in parts:
            queue.append((p, depth + 1))

    return done if done else [mesh]


def cut_organic_project(
    project_dir: str | Path,
    *,
    max_splits: int = 3,
    force_split: bool = False,
    repair_mode: RepairMode = "auto",
    voxel_resolution: int = 64,
    pitch_mm: float | None = None,
) -> OrganicCutResult:
    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(project)

    profile = load_printer_profile(default_profile_path())
    mesh, source_path = _load_print_mesh(project)
    print(f"Loaded {source_path.name}  faces={len(mesh.faces)}")

    mesh, repair = repair_mesh(
        mesh,
        mode=repair_mode,
        pitch_mm=pitch_mm,
        voxel_resolution=voxel_resolution,
    )
    print(
        f"Repair ({repair.mode}): watertight {repair.watertight_before} → {repair.watertight_after} "
        f"faces {repair.faces_before} → {repair.faces_after}"
    )
    if repair.pitch_mm is not None:
        print(f"  voxel pitch_mm={repair.pitch_mm:.4f}")
    for note in repair.notes:
        print(f"  {note}")

    repaired_path = project / "source_repaired.stl"
    mesh.export(repaired_path)
    print(f"Wrote {repaired_path}")

    parts = split_until_fits(
        mesh,
        profile,
        max_splits=max_splits,
        force_split=force_split,
    )
    result = validate_parts(parts, profile)
    for line in result.fit_reports:
        print(line)
    print("All fit:", result.all_fit)

    parts_dir = project / "parts"
    if parts_dir.exists():
        for old in parts_dir.glob("part_*.stl"):
            old.unlink()
    parts_dir.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    part_meta = []
    for i, part in enumerate(result.parts, start=1):
        path = parts_dir / f"part_{i:02d}.stl"
        part.export(path)
        part_paths.append(path)
        ext = aabb_extents(np.asarray(part.vertices))
        ok, msg = part_fits_build_plate(ext, profile, allow_rotation=True)
        part_meta.append(
            {
                "file": path.name,
                "extents_mm": [float(x) for x in ext.tolist()],
                "fits_build_plate": bool(ok),
                "message": msg,
            }
        )
        print(f"Wrote {path}")

    meta_path = project / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    meta = meta or {}
    meta["domain"] = "organic"
    meta["printer"] = profile.name
    meta["cut_track"] = "organic"
    meta["cut_source_stl"] = source_path.name
    meta["repair"] = {
        "mode": repair.mode,
        "watertight_before": repair.watertight_before,
        "watertight_after": repair.watertight_after,
        "faces_before": repair.faces_before,
        "faces_after": repair.faces_after,
        "pitch_mm": repair.pitch_mm,
        "repaired_stl": repaired_path.name,
        "notes": repair.notes,
    }
    meta["organic_cut"] = {
        "strategy": "longest_axis_midplane",
        "max_splits": max_splits,
        "force_split": force_split,
    }
    meta["part_count"] = len(part_meta)
    meta["parts"] = part_meta
    meta["all_parts_fit_build_plate"] = bool(result.all_fit)
    meta["watertight"] = bool(repair.watertight_after)
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"Updated {meta_path}")

    return OrganicCutResult(
        project_dir=project,
        parts_dir=parts_dir,
        part_paths=part_paths,
        all_fit=bool(result.all_fit),
        watertight=bool(repair.watertight_after),
        fit_reports=result.fit_reports,
        repaired_path=repaired_path,
    )
