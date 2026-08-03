"""Mesh repair helpers for organic / TRELLIS meshes before cutting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class RepairReport:
    watertight_before: bool
    watertight_after: bool
    faces_before: int
    faces_after: int
    notes: list[str]


def repair_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, RepairReport]:
    """
    Best-effort repair for print/cut readiness.

    TRELLIS.mac often leaves small holes (cumesh skipped). We fill what we can;
    plane_cut (slice+cap) still works when not fully watertight.
    """
    notes: list[str] = []
    m = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=True,
    )
    before_w = bool(m.is_watertight)
    faces_before = len(m.faces)

    try:
        trimesh.repair.fix_normals(m)
        notes.append("fix_normals")
    except Exception as exc:
        notes.append(f"fix_normals skipped: {exc}")

    try:
        filled = trimesh.repair.fill_holes(m)
        notes.append(f"fill_holes={'ok' if filled else 'partial'}")
    except Exception as exc:
        notes.append(f"fill_holes skipped: {exc}")

    # Drop degenerate / duplicate faces when possible
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

    after_w = bool(m.is_watertight)
    return m, RepairReport(
        watertight_before=before_w,
        watertight_after=after_w,
        faces_before=faces_before,
        faces_after=len(m.faces),
        notes=notes,
    )
