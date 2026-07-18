"""Synthetic mechanical primitives for Phase 0 (build complexity later)."""

from __future__ import annotations

import trimesh


def box(size=(80.0, 40.0, 30.0)) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=size)


def cylinder(radius=20.0, height=60.0, sections=64) -> trimesh.Trimesh:
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


def l_bracket(
    arm_length=60.0,
    arm_width=20.0,
    thickness=4.0,
) -> trimesh.Trimesh:
    """Simple L-bracket as union of two boxes (mechanical starter shape)."""
    horizontal = trimesh.creation.box(extents=(arm_length, arm_width, thickness))
    vertical = trimesh.creation.box(extents=(thickness, arm_width, arm_length))
    vertical.apply_translation(
        (
            (arm_length / 2.0) - (thickness / 2.0),
            0.0,
            (arm_length / 2.0) + (thickness / 2.0),
        )
    )
    return trimesh.util.concatenate([horizontal, vertical])


def enclosure_shell(
    outer=(100.0, 70.0, 40.0),
    wall=2.0,
) -> trimesh.Trimesh:
    """Hollow box (outer minus inner) — mechanical enclosure starter."""
    outer_mesh = trimesh.creation.box(extents=outer)
    inner = (
        max(outer[0] - 2 * wall, wall),
        max(outer[1] - 2 * wall, wall),
        max(outer[2] - wall, wall),  # open-ish top feel via thinner Z cavity
    )
    inner_mesh = trimesh.creation.box(extents=inner)
    try:
        return outer_mesh.difference(inner_mesh)
    except Exception:
        # Fallback if boolean backend missing: return solid outer
        return outer_mesh
