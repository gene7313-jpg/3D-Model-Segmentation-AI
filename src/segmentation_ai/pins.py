"""Mating pin/socket features for cut-plane alignment (quality-preserving).

Male: high-res cylinders concatenated onto the cut face (never remeshed).
Female: circular holes cut into the existing planar cut-cap triangles by face
removal — no wafer plate, no voxel remesh, outline stays the shoe seam.

``--pin-remesh`` enables a lossy whole-part boolean fallback only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from trimesh import Trimesh
from trimesh.creation import cylinder


@dataclass
class PinSpec:
    diameter_mm: float = 4.0
    length_mm: float = 8.0
    clearance_mm: float = 0.25
    count: int = 2
    edge_margin_mm: float = 4.0
    sections: int = 48  # high-res pin geometry

    @property
    def radius_mm(self) -> float:
        return 0.5 * self.diameter_mm


@dataclass
class PinReport:
    pin_count: int
    used_remesh: bool
    method: str  # "cap_punch" | "boolean" | "male_only" | "none"
    notes: list[str] = field(default_factory=list)


@dataclass
class PinApplyResult:
    parts: list[Trimesh]
    pins_applied: int
    used_remesh: bool
    method: str
    notes: list[str]


def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    v /= np.linalg.norm(v) + 1e-12
    return u, v


def _cut_face_mask(
    mesh: Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    *,
    cos_thresh: float = 0.85,
    plane_tol_mm: float = 0.6,
) -> np.ndarray:
    """Planar cut-cap faces: near the seam plane and aligned with ±normal."""
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    origin = np.asarray(plane_origin, dtype=float)
    aligned = np.abs(mesh.face_normals @ n) >= cos_thresh
    centers = mesh.triangles_center
    dist = np.abs((centers - origin) @ n)
    return aligned & (dist <= plane_tol_mm)


def _cut_uv_bounds(
    mesh: Trimesh,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    face_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float] | None:
    if not np.any(face_mask):
        return None
    u, v = _orthonormal_basis(plane_normal)
    verts = mesh.vertices[np.unique(mesh.faces[face_mask].ravel())]
    rel = verts - plane_origin
    uu = rel @ u
    vv = rel @ v
    return u, v, float(uu.min()), float(uu.max()), float(vv.min()), float(vv.max())


def _pin_centers_in_bounds(
    umin: float,
    umax: float,
    vmin: float,
    vmax: float,
    count: int,
    edge_margin: float,
) -> list[tuple[float, float]]:
    usable_u = (umax - umin) - 2 * edge_margin
    usable_v = (vmax - vmin) - 2 * edge_margin
    if usable_u <= 0 or usable_v <= 0:
        return []
    centers: list[tuple[float, float]] = []
    if count == 1:
        centers.append((0.5 * (umin + umax), 0.5 * (vmin + vmax)))
    elif usable_u >= usable_v:
        for i in range(count):
            t = (i + 0.5) / count
            centers.append((umin + edge_margin + t * usable_u, 0.5 * (vmin + vmax)))
    else:
        for i in range(count):
            t = (i + 0.5) / count
            centers.append((0.5 * (umin + umax), vmin + edge_margin + t * usable_v))
    return centers


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty and g.area > 0]
        return max(polys, key=lambda g: g.area) if polys else None
    try:
        return _largest_polygon(unary_union(geom))
    except Exception:
        return None


def _cut_face_polygon_2d(
    mesh: Trimesh,
    face_mask: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> Polygon | None:
    if not np.any(face_mask):
        return None
    polys: list[Polygon] = []
    for face in mesh.faces[face_mask]:
        pts = mesh.vertices[face]
        coords = [
            (float(np.dot(p - origin, u)), float(np.dot(p - origin, v))) for p in pts
        ]
        try:
            poly = Polygon(coords)
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        poly = _largest_polygon(poly)
        if poly is not None and poly.area > 1e-6:
            polys.append(poly)
    if not polys:
        return None
    return _largest_polygon(unary_union(polys))


def _pin_centers_in_polygon(
    polygon: Polygon,
    count: int,
    edge_margin: float,
    pin_radius: float,
) -> list[tuple[float, float]]:
    inset_dist = max(edge_margin, pin_radius + 0.75)
    inset = _largest_polygon(polygon.buffer(-inset_dist))
    if inset is None:
        inset = _largest_polygon(polygon.buffer(-0.5 * inset_dist))
    if inset is None:
        return []
    minx, miny, maxx, maxy = inset.bounds
    candidates = _pin_centers_in_bounds(minx, maxx, miny, maxy, count, edge_margin=0.0)
    kept: list[tuple[float, float]] = []
    for cu, cv in candidates:
        if inset.contains(Point(cu, cv)):
            kept.append((cu, cv))
        else:
            rp = inset.representative_point()
            mid = (0.5 * (cu + rp.x), 0.5 * (cv + rp.y))
            if inset.contains(Point(*mid)):
                kept.append(mid)
    if len(kept) >= count:
        return kept[:count]
    # Spread along minimum rotated rectangle major axis
    try:
        mrr = inset.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [
            (np.array(coords[i]), np.array(coords[i + 1]))
            for i in range(len(coords) - 1)
        ]
        lengths = [float(np.linalg.norm(b - a)) for a, b in edges]
        i = int(np.argmax(lengths))
        a, b = edges[i]
        kept = []
        for k in range(count):
            t = (k + 0.5) / count
            p = a + t * (b - a)
            pt = Point(float(p[0]), float(p[1]))
            if not inset.contains(pt):
                pt = inset.representative_point()
            kept.append((float(pt.x), float(pt.y)))
    except Exception:
        pass
    return kept


def _make_cylinder(
    base: np.ndarray,
    tip: np.ndarray,
    radius: float,
    sections: int,
) -> Trimesh:
    axis = tip - base
    height = float(np.linalg.norm(axis))
    if height < 1e-6:
        raise ValueError("degenerate cylinder")
    direction = axis / height
    cyl = cylinder(radius=radius, height=height, sections=sections)
    mid = 0.5 * (base + tip)
    z = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z, direction)) > 0.999:
        rot = np.eye(3) if np.dot(z, direction) > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        v = np.cross(z, direction)
        s = np.linalg.norm(v)
        c = float(np.dot(z, direction))
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-12))
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = mid
    cyl.apply_transform(T)
    return cyl


def _boolean_difference(a: Trimesh, b: Trimesh) -> Trimesh | None:
    try:
        out = a.difference(b, engine="manifold")
        if isinstance(out, list):
            out = trimesh.util.concatenate(out) if out else None
        if out is None or len(out.faces) == 0:
            return None
        return out
    except Exception:
        try:
            out = a.difference(b)
            if isinstance(out, list):
                out = trimesh.util.concatenate(out) if out else None
            if out is None or len(out.faces) == 0:
                return None
            return out
        except Exception:
            return None


def _ensure_volume_local(mesh: Trimesh, pitch: float) -> Trimesh:
    """Lossy remesh — only for explicit --pin-remesh fallback."""
    try:
        vox = mesh.voxelized(pitch=pitch)
        filled = vox.fill()
        return filled.marching_cubes
    except Exception:
        return mesh


def _side_sign(part: Trimesh, origin: np.ndarray, normal: np.ndarray) -> float:
    c = np.asarray(part.centroid, dtype=float)
    return float(np.dot(c - origin, normal))


def _punch_sockets_in_cap(
    mesh: Trimesh,
    face_mask: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    centers_uv: list[tuple[float, float]],
    hole_radius: float,
) -> tuple[Trimesh, int]:
    """
    Remove cut-cap triangles whose centroids fall inside any socket circle.

    Preserves the organic cut outline and all non-cap triangulation. No remesh.
    """
    if not np.any(face_mask) or not centers_uv:
        return mesh.copy(), 0

    centers = mesh.triangles_center
    remove = np.zeros(len(mesh.faces), dtype=bool)
    punched = 0
    for i in np.where(face_mask)[0]:
        rel = centers[i] - origin
        cu = float(np.dot(rel, u))
        cv = float(np.dot(rel, v))
        for pu, pv in centers_uv:
            if (cu - pu) ** 2 + (cv - pv) ** 2 <= hole_radius**2:
                remove[i] = True
                punched += 1
                break

    if not np.any(remove):
        return mesh.copy(), 0

    keep = ~remove
    try:
        out = mesh.submesh([np.where(keep)[0]], append=True)
    except Exception:
        out = mesh.copy()
        out.update_faces(keep)
        out.remove_unreferenced_vertices()
    return out, int(punched)


def apply_mating_pins(
    parts: list[Trimesh],
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    male_index: int | None = None,
    female_index: int | None = None,
    spec: PinSpec | None = None,
    allow_remesh: bool = False,
    remesh_pitch_mm: float = 0.8,
) -> PinApplyResult:
    """
    Male (+n): concatenate pin cylinders.
    Female (-n): punch socket holes in the existing cut-cap faces (no wafer).
    """
    notes: list[str] = []
    if spec is None:
        spec = PinSpec()
    if len(parts) < 2:
        return PinApplyResult(parts, 0, False, "none", ["need >=2 parts"])

    n = np.asarray(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    origin = np.asarray(plane_origin, dtype=float)

    if male_index is None or female_index is None:
        s0 = _side_sign(parts[0], origin, n)
        s1 = _side_sign(parts[1], origin, n)
        if s0 >= s1:
            male_index, female_index = 0, 1
        else:
            male_index, female_index = 1, 0
        notes.append(f"male=part[{male_index}] (+n) female=part[{female_index}] (-n)")

    male = parts[male_index].copy()
    female = parts[female_index].copy()
    female_ext_before = np.asarray(female.extents, dtype=float)

    f_mask = _cut_face_mask(female, origin, n, cos_thresh=0.80, plane_tol_mm=0.75)
    m_mask = _cut_face_mask(male, origin, n, cos_thresh=0.80, plane_tol_mm=0.75)
    notes.append(
        f"cut-cap faces: female={int(np.count_nonzero(f_mask))} "
        f"male={int(np.count_nonzero(m_mask))}"
    )

    layout_mesh, layout_mask = (female, f_mask) if np.any(f_mask) else (male, m_mask)
    if not np.any(layout_mask):
        return PinApplyResult(parts, 0, False, "none", ["no cut-face caps found"])

    bounds = _cut_uv_bounds(layout_mesh, origin, n, layout_mask)
    if bounds is None:
        return PinApplyResult(parts, 0, False, "none", ["no cut-face bounds"])

    u, v, umin, umax, vmin, vmax = bounds
    cut_poly = _cut_face_polygon_2d(layout_mesh, layout_mask, origin, u, v)
    if cut_poly is not None:
        centers = _pin_centers_in_polygon(
            cut_poly, spec.count, spec.edge_margin_mm, spec.radius_mm
        )
        notes.append(f"cut outline area={cut_poly.area:.1f} mm²")
    else:
        centers = _pin_centers_in_bounds(
            umin, umax, vmin, vmax, spec.count, spec.edge_margin_mm
        )
        notes.append("polygon layout failed; using AABB pin layout")

    if not centers:
        return PinApplyResult(parts, 0, False, "none", ["cut face too small for pins"])

    pin_meshes: list[Trimesh] = []
    hole_meshes: list[Trimesh] = []
    for cu, cv in centers:
        base = origin + u * cu + v * cv
        tip = base - n * spec.length_mm
        pin_meshes.append(
            _make_cylinder(base + n * 0.2, tip, spec.radius_mm, spec.sections)
        )
        hole_r = spec.radius_mm + spec.clearance_mm
        hole_meshes.append(
            _make_cylinder(base + n * 0.5, tip - n * 0.5, hole_r, spec.sections)
        )

    male_out = trimesh.util.concatenate([male] + pin_meshes)
    male_out.merge_vertices()
    notes.append(
        f"male: concatenated {len(pin_meshes)} pins "
        f"(Ø{spec.diameter_mm}mm × {spec.length_mm}mm, {spec.sections}-gon)"
    )

    used_remesh = False
    method = "cap_punch"
    hole_r = spec.radius_mm + spec.clearance_mm

    if np.any(f_mask):
        female_out, n_removed = _punch_sockets_in_cap(
            female, f_mask, origin, u, v, centers, hole_r
        )
        if n_removed == 0:
            method = "male_only"
            female_out = female
            notes.append("cap punch removed 0 faces — sockets skipped, male pins only")
        else:
            notes.append(
                f"female: cap_punch removed {n_removed} cut-cap triangles "
                f"for {len(centers)} sockets (no remesh, no wafer)"
            )
    elif allow_remesh:
        used_remesh = True
        method = "boolean"
        pitch = remesh_pitch_mm
        f_vol = _ensure_volume_local(female, pitch=pitch)
        notes.append(f"female whole-part remesh pitch={pitch}mm (allow_remesh)")
        for hole in hole_meshes:
            punched = _boolean_difference(f_vol, hole)
            if punched is None:
                notes.append("remesh boolean still failed")
                method = "male_only"
                female_out = female
                break
            f_vol = punched
        else:
            female_out = f_vol
    else:
        female_out = female
        method = "male_only"
        notes.append(
            "female cut cap not found — sockets skipped "
            "(male pins only; use --pin-remesh for lossy fallback)"
        )

    # Hard reject: female must not grow into a giant slab
    female_ext_after = np.asarray(female_out.extents, dtype=float)
    if np.any(female_ext_after > female_ext_before * 1.15 + 5.0):
        notes.append(
            f"female extent blow-up {female_ext_before.round(1)} → "
            f"{female_ext_after.round(1)}; reverting female (male pins kept)"
        )
        female_out = female
        method = "male_only"

    out = list(parts)
    out[male_index] = male_out
    out[female_index] = female_out
    return PinApplyResult(
        parts=out,
        pins_applied=len(pin_meshes),
        used_remesh=used_remesh,
        method=method,
        notes=notes,
    )


def add_mating_pins(
    male: Trimesh,
    female: Trimesh,
    *,
    cut_normal: np.ndarray,
    cut_origin: np.ndarray,
    spec: PinSpec | None = None,
    allow_remesh: bool = False,
    remesh_pitch_mm: float = 0.8,
) -> tuple[Trimesh, Trimesh, PinReport]:
    result = apply_mating_pins(
        [male, female],
        plane_origin=cut_origin,
        plane_normal=cut_normal,
        male_index=None,
        female_index=None,
        spec=spec,
        allow_remesh=allow_remesh,
        remesh_pitch_mm=remesh_pitch_mm,
    )
    report = PinReport(
        pin_count=result.pins_applied,
        used_remesh=result.used_remesh,
        method=result.method,
        notes=result.notes,
    )
    return result.parts[0], result.parts[1], report
