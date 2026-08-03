"""Mating pin/socket features for cut-plane alignment (quality-preserving).

Male pins: high-res cylinders concatenated onto the male cut face (no remesh).
Female sockets: holes punched into a thin watertight *cut-cap wafer* shaped to the
cut-face outline (not a bounding rectangle) — body triangulation is never remeshed.

Use --pin-remesh only as a last-resort whole-part remesh for boolean fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from trimesh import Trimesh
from trimesh.creation import cylinder, extrude_polygon


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
    method: str  # "wafer" | "boolean" | "male_only" | "none"
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


def _cut_face_mask(mesh: Trimesh, plane_normal: np.ndarray, cos_thresh: float = 0.85) -> np.ndarray:
    n = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    fn = mesh.face_normals
    dots = fn @ n
    return dots >= cos_thresh


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
    else:
        if usable_u >= usable_v:
            for i in range(count):
                t = (i + 0.5) / count
                centers.append((umin + edge_margin + t * usable_u, 0.5 * (vmin + vmax)))
        else:
            for i in range(count):
                t = (i + 0.5) / count
                centers.append((0.5 * (umin + umax), vmin + edge_margin + t * usable_v))
    return centers


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


def _largest_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if not g.is_empty and g.area > 0]
        if not polys:
            return None
        return max(polys, key=lambda g: g.area)
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
    """Union cut-face triangles in plane UV → outline polygon (non-rectangular)."""
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


def _make_outline_wafer(
    polygon: Polygon,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    thickness: float,
    margin: float,
) -> Trimesh:
    """
    Extrude the cut-face outline into a thin solid seated in the female (-normal).
    Local +Z maps to world +normal; top face sits just proud of the cut plane.
    """
    poly = polygon.buffer(margin) if margin > 0 else polygon
    poly = _largest_polygon(poly)
    if poly is None:
        raise ValueError("empty cut-face polygon after buffer")
    solid = extrude_polygon(poly, height=thickness)
    R = np.column_stack([u, v, normal])
    T = np.eye(4)
    T[:3, :3] = R
    # z=thickness → near plane (+0.05 toward male); z=0 deep in female
    T[:3, 3] = origin - normal * (thickness - 0.05)
    solid.apply_transform(T)
    return solid


def _make_box_wafer(
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    umin: float,
    umax: float,
    vmin: float,
    vmax: float,
    thickness: float,
    margin: float,
) -> Trimesh:
    """Fallback AABB wafer if outline reconstruction fails."""
    cu = 0.5 * (umin + umax)
    cv = 0.5 * (vmin + vmax)
    wu = (umax - umin) + 2 * margin
    wv = (vmax - vmin) + 2 * margin
    box = trimesh.creation.box(extents=[wu, wv, thickness])
    R = np.column_stack([u, v, normal])
    T = np.eye(4)
    T[:3, :3] = R
    center = origin + u * cu + v * cv - normal * (0.5 * thickness - 0.05)
    T[:3, 3] = center
    box.apply_transform(T)
    return box


def _pin_centers_in_polygon(
    polygon: Polygon,
    count: int,
    edge_margin: float,
    pin_radius: float,
) -> list[tuple[float, float]]:
    """Place pins inside an inset of the cut outline along its longer axis."""
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
            # Snap toward polygon representative point
            rp = inset.representative_point()
            mid = (0.5 * (cu + rp.x), 0.5 * (cv + rp.y))
            if inset.contains(Point(*mid)):
                kept.append(mid)
    if len(kept) < count and count > 0:
        # Fall back: spread along minimum rotated rectangle major axis
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
    """Last-resort remesh of a *small* mesh (wafer / tool), not the whole part."""
    try:
        vox = mesh.voxelized(pitch=pitch)
        filled = vox.fill()
        return filled.marching_cubes
    except Exception:
        return mesh


def _side_sign(part: Trimesh, origin: np.ndarray, normal: np.ndarray) -> float:
    c = np.asarray(part.centroid, dtype=float)
    return float(np.dot(c - origin, normal))


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
    Apply mating pins without remeshing organic body triangles.

    Male (+n side): concatenate high-res pin cylinders pointing into female.
    Female (-n side): punch holes in a cut-cap wafer; keep body triangulation.
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
        # Male on +n, female on -n
        if s0 >= s1:
            male_index, female_index = 0, 1
        else:
            male_index, female_index = 1, 0
        notes.append(f"male=part[{male_index}] (+n) female=part[{female_index}] (-n)")

    male = parts[male_index].copy()
    female = parts[female_index].copy()

    # Female cut faces point roughly +n (toward male / out of female)
    f_mask = _cut_face_mask(female, n, cos_thresh=0.80)
    m_mask = _cut_face_mask(male, -n, cos_thresh=0.80)
    outline_mesh = female
    outline_mask = f_mask
    bounds = _cut_uv_bounds(female, origin, n, f_mask)
    if bounds is None or not np.any(f_mask):
        outline_mesh = male
        outline_mask = m_mask
        bounds = _cut_uv_bounds(male, origin, n, m_mask)
        notes.append("used male cut-face outline (female cut faces not found)")
    if bounds is None:
        return PinApplyResult(parts, 0, False, "none", ["no cut-face bounds"])

    u, v, umin, umax, vmin, vmax = bounds
    cut_poly = _cut_face_polygon_2d(outline_mesh, outline_mask, origin, u, v)
    if cut_poly is not None:
        centers = _pin_centers_in_polygon(
            cut_poly, spec.count, spec.edge_margin_mm, spec.radius_mm
        )
        notes.append(
            f"cut outline polygon area={cut_poly.area:.1f} mm² "
            f"(not AABB rectangle)"
        )
    else:
        centers = _pin_centers_in_bounds(
            umin, umax, vmin, vmax, spec.count, spec.edge_margin_mm
        )
        notes.append("cut outline polygon failed; using AABB pin layout")

    if not centers:
        return PinApplyResult(parts, 0, False, "none", ["cut face too small for pins"])

    pin_meshes: list[Trimesh] = []
    hole_meshes: list[Trimesh] = []
    for cu, cv in centers:
        base = origin + u * cu + v * cv
        # Pins grow from plane into female (-n)
        tip = base - n * spec.length_mm
        pin = _make_cylinder(base + n * 0.2, tip, spec.radius_mm, spec.sections)
        pin_meshes.append(pin)
        hole_r = spec.radius_mm + spec.clearance_mm
        hole = _make_cylinder(
            base + n * 0.5,
            tip - n * 0.5,
            hole_r,
            spec.sections,
        )
        hole_meshes.append(hole)

    male_out = trimesh.util.concatenate([male] + pin_meshes)
    male_out.merge_vertices()
    notes.append(
        f"male: concatenated {len(pin_meshes)} pins "
        f"(Ø{spec.diameter_mm}mm × {spec.length_mm}mm, {spec.sections}-gon)"
    )

    used_remesh = False
    method = "wafer"
    thickness = max(1.2, spec.length_mm * 0.35)
    wafer_margin = 0.35
    wafer_kind = "outline"
    try:
        if cut_poly is None:
            raise ValueError("no polygon")
        wafer = _make_outline_wafer(
            cut_poly, origin, n, u, v, thickness=thickness, margin=wafer_margin
        )
    except Exception as exc:
        wafer_kind = "box"
        notes.append(f"outline wafer failed ({exc}); falling back to AABB box")
        wafer = _make_box_wafer(
            origin, n, u, v, umin, umax, vmin, vmax, thickness=thickness, margin=0.5
        )
    if not getattr(wafer, "is_volume", False):
        wafer = _ensure_volume_local(wafer, pitch=min(0.4, remesh_pitch_mm))
        notes.append("wafer local remesh for volume")

    holed = wafer
    for hole in hole_meshes:
        tool = hole
        if not getattr(tool, "is_volume", False):
            tool = _ensure_volume_local(tool, pitch=min(0.35, remesh_pitch_mm))
        punched = _boolean_difference(holed, tool)
        if punched is None:
            tool = _ensure_volume_local(hole, pitch=0.3)
            punched = _boolean_difference(holed, tool)
        if punched is None:
            method = "male_only"
            notes.append("socket punch failed — male pins only")
            break
        holed = punched

    female_out = female
    if method != "male_only":
        if np.any(f_mask):
            keep = ~f_mask
            try:
                female_body = female.submesh([np.where(keep)[0]], append=True)
            except Exception:
                female_body = female.copy()
                female_body.update_faces(keep)
                female_body.remove_unreferenced_vertices()
        else:
            female_body = female.copy()

        female_out = trimesh.util.concatenate([female_body, holed])
        female_out.merge_vertices()
        notes.append(
            f"female: {wafer_kind} cut-cap wafer with {len(pin_meshes)} sockets "
            "(body unremeshed)"
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
        notes.append("sockets skipped (wafer failed; pass allow_remesh/--pin-remesh for lossy fallback)")

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
    """
    Convenience wrapper: caller-supplied male/female ordering.

    Note: side assignment still prefers +n = male / -n = female; if the
    caller swapped them, centroids are used to re-order for correct pin direction.
    """
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
