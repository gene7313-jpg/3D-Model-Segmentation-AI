"""Mesh repair helpers for organic / TRELLIS meshes before cutting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import trimesh

RepairMode = Literal["basic", "voxel", "auto"]


@dataclass
class RepairReport:
    watertight_before: bool
    watertight_after: bool
    faces_before: int
    faces_after: int
    mode: str
    notes: list[str] = field(default_factory=list)
    pitch_mm: float | None = None


def _basic_cleanup(mesh: trimesh.Trimesh, notes: list[str]) -> trimesh.Trimesh:
    m = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=True,
    )
    try:
        trimesh.repair.fix_normals(m)
        notes.append("fix_normals")
    except Exception as exc:
        notes.append(f"fix_normals skipped: {exc}")

    try:
        trimesh.repair.fix_winding(m)
        notes.append("fix_winding")
    except Exception as exc:
        notes.append(f"fix_winding skipped: {exc}")

    try:
        trimesh.repair.fix_inversion(m)
        notes.append("fix_inversion")
    except Exception as exc:
        notes.append(f"fix_inversion skipped: {exc}")

    try:
        filled = trimesh.repair.fill_holes(m)
        notes.append(f"fill_holes={'ok' if filled else 'partial'}")
    except Exception as exc:
        notes.append(f"fill_holes skipped: {exc}")

    try:
        m.update_faces(m.nondegenerate_faces())
        m.remove_unreferenced_vertices()
        notes.append("nondegenerate_faces")
    except Exception as exc:
        notes.append(f"nondegenerate skipped: {exc}")

    try:
        m.merge_vertices()
        notes.append("merge_vertices")
    except Exception as exc:
        notes.append(f"merge_vertices skipped: {exc}")

    # Keep largest connected component (drops loose scraps)
    try:
        parts = m.split(only_watertight=False)
        if len(parts) > 1:
            parts = sorted(parts, key=lambda p: len(p.faces), reverse=True)
            m = parts[0]
            notes.append(f"keep_largest_of_{len(parts)}_components")
    except Exception as exc:
        notes.append(f"split skipped: {exc}")

    return m


def _voxel_watertight(
    mesh: trimesh.Trimesh,
    *,
    pitch_mm: float | None,
    resolution: int,
    notes: list[str],
) -> tuple[trimesh.Trimesh, float]:
    """
    Force a watertight shell via voxelization + marching cubes.

    Requires scikit-image (trimesh marching_cubes backend).
    """
    extents = np.asarray(mesh.extents, dtype=float)
    longest = float(np.max(extents))
    if longest <= 0:
        raise ValueError("mesh has zero extent")

    if pitch_mm is None or pitch_mm <= 0:
        pitch_mm = longest / float(max(resolution, 8))

    # Soft cap so huge meshes don't explode memory
    min_pitch = longest / 128.0
    pitch_mm = max(float(pitch_mm), min_pitch)

    centroid = np.asarray(mesh.centroid, dtype=float)
    vg = mesh.voxelized(pitch=pitch_mm)
    try:
        vg = vg.fill()
        notes.append("voxel_fill")
    except Exception as exc:
        notes.append(f"voxel_fill skipped: {exc}")

    solid = vg.marching_cubes.copy()
    solid.apply_transform(vg.transform)

    # Re-center to original centroid (voxelization can drift slightly)
    delta = centroid - np.asarray(solid.centroid, dtype=float)
    solid.apply_translation(delta)

    notes.append(f"voxel_remesh pitch_mm={pitch_mm:.4f} faces={len(solid.faces)}")
    if not solid.is_watertight:
        # One more cleanup pass on the remesh
        solid = _basic_cleanup(solid, notes)
    return solid, pitch_mm


def repair_mesh(
    mesh: trimesh.Trimesh,
    *,
    mode: RepairMode = "auto",
    pitch_mm: float | None = None,
    voxel_resolution: int = 64,
) -> tuple[trimesh.Trimesh, RepairReport]:
    """
    Repair for print/cut readiness.

    Modes:
      - basic: normals / hole fill / cleanup (may stay open on TRELLIS meshes)
      - voxel: always voxel-remesh to a watertight shell (needs scikit-image)
      - auto:  basic first; if still open, voxel-remesh
    """
    notes: list[str] = []
    before_w = bool(mesh.is_watertight)
    faces_before = len(mesh.faces)
    used_pitch: float | None = None

    m = _basic_cleanup(mesh, notes)
    notes.append(f"mode_requested={mode}")

    need_voxel = mode == "voxel" or (mode == "auto" and not bool(m.is_watertight))
    if need_voxel:
        try:
            m, used_pitch = _voxel_watertight(
                m,
                pitch_mm=pitch_mm,
                resolution=voxel_resolution,
                notes=notes,
            )
            notes.append("voxel_stage=applied")
        except ModuleNotFoundError as exc:
            notes.append(
                f"voxel_stage unavailable ({exc}); install scikit-image for watertight remesh"
            )
        except Exception as exc:
            notes.append(f"voxel_stage failed: {exc}")

    after_w = bool(m.is_watertight)
    return m, RepairReport(
        watertight_before=before_w,
        watertight_after=after_w,
        faces_before=faces_before,
        faces_after=len(m.faces),
        mode=mode,
        notes=notes,
        pitch_mm=used_pitch,
    )
