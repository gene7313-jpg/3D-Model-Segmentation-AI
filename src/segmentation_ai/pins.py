"""Mating pin / socket geometry for split organic parts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass
class PinSpec:
    diameter_mm: float = 3.0
    clearance_mm: float = 0.2
    length_mm: float = 6.0
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


def _ensure_volume(
    mesh: trimesh.Trimesh,
    *,
    pitch_mm: float,
    notes: list[str],
    label: str,
) -> trimesh.Trimesh:
    """
    Manifold boolean requires watertight volumes. Prefer light repair; if still
    open, do a *fine* voxel remesh of this part only (not the whole pre-cut mesh).
    """
    m = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=True,
    )
    try:
        trimesh.repair.fix_normals(m)
        trimesh.repair.fill_holes(m)
    except Exception:
        pass

    if bool(getattr(m, "is_volume", False)) or (
        m.is_watertight and m.is_winding_consistent
    ):
        notes.append(f"{label}: already_volume faces={len(m.faces)}")
        return m

    longest = float(np.max(m.extents))
    pitch = max(float(pitch_mm), longest / 200.0)
    centroid = np.asarray(m.centroid, dtype=float)
    try:
        vg = m.voxelized(pitch=pitch).fill()
        solid = vg.marching_cubes.copy()
        solid.apply_transform(vg.transform)
        solid.apply_translation(centroid - np.asarray(solid.centroid, dtype=float))
        notes.append(
            f"{label}: fine_voxel_for_pins pitch_mm={pitch:.3f} "
            f"faces {len(m.faces)}→{len(solid.faces)} watertight={solid.is_watertight}"
        )
        return solid
    except Exception as exc:
        notes.append(f"{label}: volume_prepare_failed ({exc})")
        return m


def _basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / (np.linalg.norm(normal) + 1e-12)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    major = np.cross(up, n)
    major /= np.linalg.norm(major) + 1e-12
    minor = np.cross(n, major)
    minor /= np.linalg.norm(minor) + 1e-12
    return major, minor


def _cut_face_centroid(mesh: trimesh.Trimesh, inward_normal: np.ndarray) -> np.ndarray:
    """Centroid of faces whose outward normal opposes inward_normal (cut cap)."""
    n = inward_normal / (np.linalg.norm(inward_normal) + 1e-12)
    outward = -n
    align = mesh.face_normals @ outward
    mask = align > 0.85
    if np.any(mask):
        tris = mesh.vertices[mesh.faces[mask]]
        return tris.reshape(-1, 3).mean(axis=0)
    return np.asarray(mesh.centroid, dtype=float)


def _oriented_cylinder(
    *,
    position: np.ndarray,
    axis: np.ndarray,
    radius: float,
    height: float,
) -> trimesh.Trimesh:
    """Cylinder centered at ``position``, +Z aligned to ``axis``."""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    z = np.array([0.0, 0.0, 1.0])
    a = axis / (np.linalg.norm(axis) + 1e-12)
    if abs(np.dot(z, a)) < 0.999:
        rot_axis = np.cross(z, a)
        rot_axis /= np.linalg.norm(rot_axis) + 1e-12
        angle = float(np.arccos(np.clip(np.dot(z, a), -1.0, 1.0)))
        cyl.apply_transform(trimesh.transformations.rotation_matrix(angle, rot_axis))
    cyl.apply_translation(position)
    return cyl


def add_mating_pins(
    male: trimesh.Trimesh,
    female: trimesh.Trimesh,
    *,
    cut_normal: np.ndarray,
    cut_origin: np.ndarray,
    spec: PinSpec | None = None,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, PinReport]:
    """
    Add pin(s) on ``male`` and sockets on ``female``.

    ``cut_normal`` is the plane normal used for the split (male assumed on the
    +normal side / ``plane_cut`` positive half; female on the −normal side).
    Pins protrude from male toward female (−normal).
    """
    spec = spec or PinSpec()
    notes: list[str] = []
    if spec.count < 1:
        return male, female, PinReport(False, 0, ["pin_count<1"])

    n = np.asarray(cut_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    origin = np.asarray(cut_origin, dtype=float)

    male_side = np.dot(np.asarray(male.centroid) - origin, n)
    female_side = np.dot(np.asarray(female.centroid) - origin, n)
    if male_side < female_side:
        male, female = female, male
        notes.append("swapped_male_female_by_plane_side")

    # Fine pitch for pin booleans only (~pin radius / 2, clamped)
    pin_pitch = float(np.clip(spec.diameter_mm / 6.0, 0.25, 0.6))
    male = _ensure_volume(male, pitch_mm=pin_pitch, notes=notes, label="male")
    female = _ensure_volume(female, pitch_mm=pin_pitch, notes=notes, label="female")

    major, _ = _basis_from_normal(n)
    cap_c = _cut_face_centroid(male, inward_normal=-n)
    origin = 0.5 * (origin + cap_c)

    verts_m = np.asarray(male.vertices, dtype=float)
    span = float(np.ptp(verts_m @ major))
    spread = min(span * 0.35, span * 0.5)
    if spec.count == 1 or spread < spec.diameter_mm * 2.5:
        offsets = [0.0]
        notes.append("single_pin_center")
    else:
        offsets = list(np.linspace(-0.5 * spread, 0.5 * spread, spec.count))
        notes.append(f"pin_spread_mm={spread:.2f}")

    pin_r = spec.diameter_mm / 2.0
    hole_r = (spec.diameter_mm + spec.clearance_mm) / 2.0
    pin_h = spec.length_mm
    hole_h = spec.length_mm + 2.0
    protrude = 0.65 * pin_h
    embed = pin_h - protrude

    male_out = male
    female_out = female
    applied = 0

    for off in offsets:
        pos = origin + major * float(off)
        pin_center = pos - n * (embed - pin_h * 0.5)
        pin = _oriented_cylinder(
            position=pin_center, axis=-n, radius=pin_r, height=pin_h
        )
        try:
            male_out = _boolean(male_out, pin, "union")
        except Exception as exc:
            # Fallback: glue pin geometry (overlap) so pins are still printable
            try:
                male_out = trimesh.util.concatenate([male_out, pin])
                notes.append(f"pin union fallback concatenate @ {off:.2f}: {exc}")
            except Exception as exc2:
                notes.append(f"pin union failed @ {off:.2f}: {exc2}")
                continue

        hole_center = pos - n * (hole_h * 0.5 - 0.5)
        hole = _oriented_cylinder(
            position=hole_center, axis=-n, radius=hole_r, height=hole_h
        )
        try:
            female_out = _boolean(female_out, hole, "difference")
            applied += 1
        except Exception as exc:
            notes.append(f"socket difference failed @ {off:.2f}: {exc}")

    notes.append(f"pins_applied={applied}")
    return male_out, female_out, PinReport(applied > 0, applied, notes)
