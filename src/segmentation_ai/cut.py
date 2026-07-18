"""Contoured / planar mesh cutting with build-plate validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .printer import PrinterProfile, aabb_extents, part_fits_build_plate


@dataclass
class CutResult:
    parts: list[trimesh.Trimesh]
    fit_reports: list[str]
    all_fit: bool


def plane_cut(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    normal: np.ndarray,
) -> list[trimesh.Trimesh]:
    """
    Split mesh with a plane. Returns one or two solid-ish meshes.

    Contoured cuts will replace/extend this in later Phase 0 work;
    plane cut is the correctness baseline.
    """
    origin = np.asarray(origin, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # slice_plane keeps one side; run twice and cap
    pos = mesh.slice_plane(plane_normal=normal, plane_origin=origin, cap=True)
    neg = mesh.slice_plane(plane_normal=-normal, plane_origin=origin, cap=True)

    parts: list[trimesh.Trimesh] = []
    for piece in (pos, neg):
        if piece is None:
            continue
        if isinstance(piece, trimesh.Scene):
            geoms = [g for g in piece.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                continue
            piece = trimesh.util.concatenate(geoms)
        if len(piece.vertices) == 0 or len(piece.faces) == 0:
            continue
        parts.append(piece)
    return parts


def contour_cut_polyline(
    mesh: trimesh.Trimesh,
    polyline_xyz: np.ndarray,
    *,
    extrusion_depth: float = 500.0,
) -> list[trimesh.Trimesh]:
    """
    Contoured cut defined by a 3D polyline.

    Phase 0 approach: extrude the polyline into a thin cutting ribbon / slab
    via a prism approximation and boolean-difference both halves.

    polyline_xyz: (N, 3) points lying near the intended seam (N >= 2).
    """
    pts = np.asarray(polyline_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
        raise ValueError("polyline_xyz must be (N, 3) with N >= 2")

    # Build a cutting blade: sweep a thin rectangle along the polyline
    # using successive boxes — robust starter; refine later with proper surface.
    blade_bits: list[trimesh.Trimesh] = []
    thickness = max(mesh.scale * 0.002, 0.4)

    for a, b in zip(pts[:-1], pts[1:]):
        direction = b - a
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        mid = (a + b) / 2.0
        # Local frame: X along segment, Z world-up preference
        x_axis = direction / length
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(x_axis, up)) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(up, x_axis)
        y_axis /= np.linalg.norm(y_axis) + 1e-12
        z_axis = np.cross(x_axis, y_axis)

        box = trimesh.creation.box(extents=(length + thickness, thickness, extrusion_depth))
        transform = np.eye(4)
        transform[:3, 0] = x_axis
        transform[:3, 1] = y_axis
        transform[:3, 2] = z_axis
        transform[:3, 3] = mid
        box.apply_transform(transform)
        blade_bits.append(box)

    if not blade_bits:
        raise ValueError("polyline produced no cutting geometry")

    blade = trimesh.util.concatenate(blade_bits)

    try:
        left = mesh.difference(blade, engine="manifold")
    except Exception:
        left = mesh.difference(blade)

    # Approximate "other side" by intersecting with a large bbox around the mesh
    # then subtracting left — better dual-boolean comes in next iteration.
    # For Phase 0 demo correctness on simple shapes, prefer plane_cut when
    # the polyline is nearly planar; otherwise return left + remnant.
    try:
        remnant = mesh.difference(left, engine="manifold")
    except Exception:
        try:
            remnant = mesh.difference(left)
        except Exception:
            remnant = None

    parts = [p for p in (left, remnant) if p is not None and len(getattr(p, "faces", [])) > 0]
    if len(parts) < 2:
        # Fall back: treat polyline as plane through first three points / endpoints
        p0, p1 = pts[0], pts[-1]
        mid = pts[len(pts) // 2]
        normal = np.cross(p1 - p0, mid - p0)
        if np.linalg.norm(normal) < 1e-8:
            normal = np.array([1.0, 0.0, 0.0])
        return plane_cut(mesh, origin=(p0 + p1) / 2.0, normal=normal)
    return parts


def validate_parts(
    parts: list[trimesh.Trimesh],
    profile: PrinterProfile,
) -> CutResult:
    reports: list[str] = []
    all_fit = True
    for i, part in enumerate(parts):
        ext = aabb_extents(np.asarray(part.vertices))
        ok, msg = part_fits_build_plate(ext, profile, allow_rotation=True)
        reports.append(f"part[{i}]: {msg}")
        all_fit = all_fit and ok
    return CutResult(parts=parts, fit_reports=reports, all_fit=all_fit)
