"""Mating pin / socket geometry for split organic parts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class PinSpec:
    diameter_mm: float = 4.0
    clearance_mm: float = 0.25
    length_mm: float = 8.0
    count: int = 2


@dataclass
class PinReport:
    applied: bool
    pin_count: int
    notes: list[str]


def _boolean(a: trimesh.Trimesh, b: trimesh.Trimesh, op: str) -> trimesh.Trimesh:
    try:
        if op == "union":
            out = a.union(b, engine="manifold")
        elif op == "difference":
            out = a.difference(b, engine="manifold")
        else:
            raise ValueError(op)
    except Exception:
        out = a.union(b) if op == "union" else a.difference(b)
    if isinstance(out, trimesh.Scene):
        geoms = [g for g in out.geometry.values() if isinstance(g, trimesh.Trimesh)]
        out = trimesh.util.concatenate(geoms)
    return out


def _is_volume(mesh: trimesh.Trimesh) -> bool:
    if bool(getattr(mesh, "is_volume", False)):
        return True
    return bool(mesh.is_watertight and mesh.is_winding_consistent)


def _fine_volume(mesh: trimesh.Trimesh, pitch_mm: float, notes: list[str], label: str) -> trimesh.Trimesh:
    """Fine voxel remesh for CSG only — pitch should stay ≤ ~0.4 mm for organic parts."""
    m = mesh.copy()
    try:
        trimesh.repair.fix_normals(m)
        trimesh.repair.fill_holes(m)
    except Exception:
        pass
    if _is_volume(m):
        notes.append(f"{label}: volume_ok faces={len(m.faces)}")
        return m

    longest = float(np.max(m.extents))
    pitch = float(np.clip(pitch_mm, 0.2, max(0.2, longest / 180.0)))
    centroid = np.asarray(m.centroid, dtype=float)
    vg = m.voxelized(pitch=pitch).fill()
    solid = vg.marching_cubes.copy()
    solid.apply_transform(vg.transform)
    solid.apply_translation(centroid - np.asarray(solid.centroid, dtype=float))
    notes.append(
        f"{label}: fine_voxel pitch_mm={pitch:.3f} faces {len(m.faces)}→{len(solid.faces)}"
    )
    return solid


def _basis_from_normal(normal: np.ndarray) -> np.ndarray:
    n = normal / (np.linalg.norm(normal) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    major = np.cross(up, n)
    major /= np.linalg.norm(major) + 1e-12
    return major


def _oriented_cylinder(
    *,
    base: np.ndarray,
    tip: np.ndarray,
    radius: float,
) -> trimesh.Trimesh:
    """Cylinder from base point to tip point (centerline)."""
    axis_vec = np.asarray(tip, dtype=float) - np.asarray(base, dtype=float)
    height = float(np.linalg.norm(axis_vec))
    if height < 1e-6:
        raise ValueError("zero-length cylinder")
    direction = axis_vec / height
    center = (np.asarray(base, dtype=float) + np.asarray(tip, dtype=float)) * 0.5

    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, direction)) < 0.999:
        rot_axis = np.cross(z, direction)
        rot_axis /= np.linalg.norm(rot_axis) + 1e-12
        angle = float(np.arccos(np.clip(np.dot(z, direction), -1.0, 1.0)))
        cyl.apply_transform(trimesh.transformations.rotation_matrix(angle, rot_axis))
    cyl.apply_translation(center)
    return cyl


def add_mating_pins(
    male: trimesh.Trimesh,
    female: trimesh.Trimesh,
    *,
    cut_normal: np.ndarray,
    cut_origin: np.ndarray,
    spec: PinSpec | None = None,
    allow_remesh: bool = False,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, PinReport]:
    """
    Add pin(s) on ``male`` and sockets on ``female``.

    Quality-first: by default does **not** voxel-remesh. Male pins are
    concatenated onto the high-res mesh. Female sockets require a volume
    boolean — if the part is not a volume and ``allow_remesh`` is False,
    sockets are skipped (pins-only) or pins are skipped entirely when both fail.
    """
    spec = spec or PinSpec()
    notes: list[str] = []
    if spec.count < 1:
        return male, female, PinReport(False, 0, ["pin_count<1"])

    n = np.asarray(cut_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    origin = np.asarray(cut_origin, dtype=float)

    # Male on +n side of plane, female on −n side
    if np.dot(np.asarray(male.centroid) - origin, n) < np.dot(
        np.asarray(female.centroid) - origin, n
    ):
        male, female = female, male
        notes.append("swapped_male_female_by_plane_side")

    major = _basis_from_normal(n)
    span = float(np.ptp(np.asarray(male.vertices) @ major))
    spread = min(span * 0.35, span * 0.5)
    if spec.count == 1 or spread < spec.diameter_mm * 2.5:
        offsets = [0.0]
        notes.append("single_pin_center")
    else:
        offsets = list(np.linspace(-0.5 * spread, 0.5 * spread, spec.count))
        notes.append(f"pin_spread_mm={spread:.2f}")

    embed = min(2.5, 0.30 * spec.length_mm)
    protrude = max(spec.length_mm - embed, 0.70 * spec.length_mm)
    pin_r = spec.diameter_mm / 2.0
    hole_r = (spec.diameter_mm + spec.clearance_mm) / 2.0
    hole_depth = protrude + 1.5

    female_out = female.copy()
    if _is_volume(female_out):
        notes.append("female: volume_ok")
    elif allow_remesh:
        female_out = _fine_volume(
            female_out,
            pitch_mm=min(0.35, spec.diameter_mm / 10.0),
            notes=notes,
            label="female",
        )
    else:
        notes.append(
            "female: not a volume; sockets skipped (pass --pin-remesh to allow fine remesh)"
        )

    male_out = male.copy()
    applied = 0
    pins_for_male: list[trimesh.Trimesh] = []

    for off in offsets:
        base = origin + major * float(off) + n * embed
        tip = origin + major * float(off) - n * protrude
        pin = _oriented_cylinder(base=base, tip=tip, radius=pin_r)
        pins_for_male.append(pin)

        if _is_volume(female_out) or (
            female_out.is_watertight and female_out.is_winding_consistent
        ):
            hole_base = origin + major * float(off)
            hole_tip = origin + major * float(off) - n * hole_depth
            hole = _oriented_cylinder(base=hole_base, tip=hole_tip, radius=hole_r)
            try:
                female_out = _boolean(female_out, hole, "difference")
                applied += 1
                notes.append(f"socket_ok @ {off:.2f}")
            except Exception as exc:
                notes.append(f"socket difference failed @ {off:.2f}: {exc}")

    if pins_for_male:
        # Always prefer concatenate on male to preserve TRELLIS detail
        male_out = trimesh.util.concatenate([male_out, *pins_for_male])
        notes.append("male_pins_concatenate_preserve_detail")
        if applied == 0:
            # Still count male pins as applied for visibility even if sockets skipped
            applied = len(pins_for_male)
            notes.append("male_pins_only_no_sockets")

    notes.append(f"pins_applied={applied}")
    notes.append(f"pin_embed_mm={embed:.2f} pin_protrude_mm={protrude:.2f}")
    return male_out, female_out, PinReport(applied > 0, applied, notes)
