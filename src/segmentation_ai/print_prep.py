"""Print-prep helpers for open organic meshes (Mac-friendly; no Bambu Fix)."""

from __future__ import annotations

import numpy as np
import trimesh
from trimesh import Trimesh


def consolidate_for_print(
    mesh: Trimesh,
    *,
    keep_min_faces: int = 0,
) -> tuple[Trimesh, list[str]]:
    """
    Drop disconnected scraps that trigger slicer "floating region" warnings.

    Keeps the largest connected component (by face count). Optionally also keeps
    other components with >= keep_min_faces (default: largest only).
    """
    notes: list[str] = []
    m = Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )
    try:
        m.merge_vertices()
        m.remove_unreferenced_vertices()
    except Exception as exc:
        notes.append(f"merge_vertices skipped: {exc}")

    try:
        comps = list(m.split(only_watertight=False))
    except Exception as exc:
        notes.append(f"split skipped: {exc}")
        return m, notes

    if len(comps) <= 1:
        notes.append("print_prep: single component")
        return m, notes

    comps = sorted(comps, key=lambda p: len(p.faces), reverse=True)
    kept = [comps[0]]
    for c in comps[1:]:
        if keep_min_faces > 0 and len(c.faces) >= keep_min_faces:
            kept.append(c)

    dropped = len(comps) - len(kept)
    if len(kept) == 1:
        out = kept[0]
    else:
        out = trimesh.util.concatenate(kept)
        out.merge_vertices()

    notes.append(
        f"print_prep: kept {len(kept)}/{len(comps)} component(s) "
        f"faces={len(out.faces)} (dropped {dropped} floating)"
    )
    return out, notes
