"""Mating pin/socket features for cut-plane alignment (quality-preserving).

FROZEN (shoe_cli_full verified 2026-08-03 — do not change casually):

- Male: ``add_male_pins_frozen`` — concatenate high-res cylinders on the cut face.
- Female: ``apply_female_cap_and_sleeves`` — rebuild cut-cap with circular holes +
  recessed tube sleeves entirely on the female side of the plane.

CLI: ``--with-pins --with-pin-holes`` → method ``male+cap_sleeves``.

Rejected approaches (do not revive without a new design review):
whole-part voxel remesh, AABB/outline wafers, face-delete punch, embossed ROI plugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from trimesh import Trimesh
from trimesh.creation import cylinder, triangulate_polygon


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
    method: str  # "male_only" | "roi_plugs" | "cap_rebuild_step" | "boolean" | "none"
    notes: list[str] = field(default_factory=list)
    centers_uv: list[tuple[float, float]] = field(default_factory=list)
    layout_origin: np.ndarray | None = None
    layout_normal: np.ndarray | None = None
    layout_u: np.ndarray | None = None
    layout_v: np.ndarray | None = None


@dataclass
class PinApplyResult:
    parts: list[Trimesh]
    pins_applied: int
    used_remesh: bool
    method: str
    notes: list[str]
    centers_uv: list[tuple[float, float]] = field(default_factory=list)
    layout_origin: np.ndarray | None = None
    layout_normal: np.ndarray | None = None
    layout_u: np.ndarray | None = None
    layout_v: np.ndarray | None = None


@dataclass
class _PinLayout:
    male_index: int
    female_index: int
    origin: np.ndarray
    normal: np.ndarray
    u: np.ndarray
    v: np.ndarray
    centers_uv: list[tuple[float, float]]
    f_mask: np.ndarray
    m_mask: np.ndarray
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
    """Lossy remesh — ROI plugs or explicit --pin-remesh fallback."""
    try:
        vox = mesh.voxelized(pitch=pitch)
        filled = vox.fill()
        out = filled.marching_cubes
        # marching_cubes is in voxel index space; map back to world
        if hasattr(filled, "transform") and filled.transform is not None:
            out = out.copy()
            out.apply_transform(filled.transform)
        return out
    except Exception:
        return mesh


def _roi_face_mask(
    mesh: Trimesh,
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    center_uv: tuple[float, float],
    roi_radius: float,
    depth_mm: float,
) -> np.ndarray:
    """Faces whose centroids lie in a cylinder into the female (-normal)."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    pu, pv = center_uv
    rel = mesh.triangles_center - origin
    axial = -(rel @ n)  # >=0 into female
    cu = rel @ u
    cv = rel @ v
    radial2 = (cu - pu) ** 2 + (cv - pv) ** 2
    return (
        (axial >= -0.8)
        & (axial <= depth_mm + 0.8)
        & (radial2 <= (roi_radius + 0.15) ** 2)
    )


def _build_roi_seed(
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    center_uv: tuple[float, float],
    roi_radius: float,
    depth_mm: float,
    sections: int,
) -> Trimesh:
    """Solid cylinder seed entirely on the female side of the cut (-normal)."""
    pu, pv = center_uv
    # Flush/slightly recessed at the cut plane — never proud toward the male.
    top = origin + u * pu + v * pv - normal * 0.05
    bot = origin + u * pu + v * pv - normal * (depth_mm + 0.05)
    return _make_cylinder(top, bot, roi_radius, max(24, sections // 2))


def _make_recessed_sleeve(
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    center_uv: tuple[float, float],
    hole_radius: float,
    depth_mm: float,
    wall_mm: float = 1.2,
    sections: int = 48,
) -> Trimesh | None:
    """
    Tube sleeve for pin depth: entirely recessed into female (-n).

    Uses analytic cylinders + boolean (no voxel) so rims stay clean.
    """
    outer_r = hole_radius + wall_mm
    pu, pv = center_uv
    top = origin + u * pu + v * pv - normal * 0.05
    bot = origin + u * pu + v * pv - normal * (depth_mm + 0.05)
    outer = _make_cylinder(top, bot, outer_r, sections)
    # Oversize the bore tool slightly past the sleeve ends for a clean through-hole
    bore = _make_cylinder(
        top + normal * 0.4,
        bot - normal * 0.4,
        hole_radius,
        sections,
    )
    punched = _boolean_difference(outer, bore)
    return punched


def _make_roi_plug(
    female: Trimesh,
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    center_uv: tuple[float, float],
    hole_radius: float,
    depth_mm: float,
    pitch_mm: float,
    sections: int,
) -> tuple[Trimesh | None, np.ndarray, list[str]]:
    """
    Build a locally remeshed plug with a socket hole.

    Returns (plug_or_None, roi_face_mask, notes).
    """
    notes: list[str] = []
    # Wall around the hole so the plug is a ring/solid with depth
    roi_radius = max(hole_radius + 1.8, hole_radius * 1.75)
    face_mask = _roi_face_mask(
        female,
        origin=origin,
        normal=normal,
        u=u,
        v=v,
        center_uv=center_uv,
        roi_radius=roi_radius,
        depth_mm=depth_mm,
    )
    n_faces = int(np.count_nonzero(face_mask))
    notes.append(f"roi faces={n_faces} r={roi_radius:.2f}mm depth={depth_mm:.2f}mm")

    seed = _build_roi_seed(
        origin=origin,
        normal=normal,
        u=u,
        v=v,
        center_uv=center_uv,
        roi_radius=roi_radius,
        depth_mm=depth_mm,
        sections=sections,
    )

    pitch = float(np.clip(pitch_mm, 0.25, 0.7))
    # Solid cylindrical seed → local volume (surface scraps not voxelized)
    plug = _ensure_volume_local(seed, pitch=pitch)
    if plug is None or len(plug.faces) == 0:
        notes.append("roi voxel plug empty")
        return None, face_mask, notes
    notes.append(f"roi plug remesh pitch={pitch:.2f}mm faces={len(plug.faces)}")

    pu, pv = center_uv
    base = origin + u * pu + v * pv + normal * 0.6
    tip = origin + u * pu + v * pv - normal * (depth_mm + 0.6)
    hole = _make_cylinder(base, tip, hole_radius, sections)
    if not getattr(hole, "is_volume", False):
        hole = _ensure_volume_local(hole, pitch=min(0.35, pitch))

    punched = _boolean_difference(plug, hole)
    if punched is None:
        notes.append("roi boolean difference failed")
        return None, face_mask, notes

    # Keep only the component nearest the pin axis (drop stray voxel scraps)
    try:
        parts = punched.split(only_watertight=False)
        if len(parts) > 1:
            axis_pt = origin + u * pu + v * pv - normal * (0.5 * depth_mm)

            def _dist(p: Trimesh) -> float:
                return float(np.linalg.norm(np.asarray(p.centroid) - axis_pt))

            punched = min(parts, key=_dist)
            notes.append(f"roi kept nearest of {len(parts)} components")
    except Exception:
        pass

    # Hard clip: reject plugs that escaped the ROI bounding sphere
    axis_pt = origin + u * pu + v * pv - normal * (0.5 * depth_mm)
    if float(np.linalg.norm(np.asarray(punched.centroid) - axis_pt)) > roi_radius * 2.5:
        notes.append("roi plug centroid outside ROI; rejected")
        return None, face_mask, notes

    return punched, face_mask, notes


def apply_female_roi_plugs(
    female: Trimesh,
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    centers_uv: list[tuple[float, float]],
    hole_radius: float,
    depth_mm: float,
    pitch_mm: float = 0.4,
    sections: int = 48,
) -> tuple[Trimesh, list[str], int]:
    """
    Stepwise local ROI plugs: remesh+boolean only around each pin, stitch back.

    Body triangles outside each ROI are preserved. Local remesh is confined to
    the plug (not whole-part).
    """
    notes: list[str] = []
    out = female.copy()
    ext0 = np.asarray(female.extents, dtype=float)
    plugs_ok = 0
    # Union of ROI masks to strip once at the end per step (strip per hole)
    for step, center in enumerate(centers_uv, start=1):
        before = out
        faces_before = len(before.faces)
        plug, roi_mask, plug_notes = _make_roi_plug(
            before,
            origin=origin,
            normal=normal,
            u=u,
            v=v,
            center_uv=center,
            hole_radius=hole_radius,
            depth_mm=depth_mm,
            pitch_mm=pitch_mm,
            sections=sections,
        )
        for n in plug_notes:
            notes.append(f"hole[{step}] {n}")
        if plug is None:
            notes.append(f"hole[{step}/{len(centers_uv)}] WARN roi plug failed")
            continue

        # Recompute mask on current mesh (indices valid for `before`)
        roi_mask = _roi_face_mask(
            before,
            origin=origin,
            normal=normal,
            u=u,
            v=v,
            center_uv=center,
            roi_radius=max(hole_radius + 1.8, hole_radius * 1.75),
            depth_mm=depth_mm,
        )
        body = _strip_faces(before, roi_mask)
        try:
            merged = trimesh.util.concatenate([body, plug])
            merged.merge_vertices()
        except Exception as exc:
            notes.append(f"hole[{step}] stitch failed ({exc})")
            continue

        ext1 = np.asarray(merged.extents, dtype=float)
        if np.any(ext1 > ext0 * 1.2 + 8.0):
            notes.append(
                f"hole[{step}] aborted extent {ext0.round(1)}→{ext1.round(1)}"
            )
            continue

        out = merged
        plugs_ok += 1
        notes.append(
            f"hole[{step}/{len(centers_uv)}] center=({center[0]:.1f},{center[1]:.1f}) "
            f"roi plug stitched ({faces_before}→{len(out.faces)}) [ok]"
        )

    if plugs_ok == 0:
        notes.append("roi plugs: 0 sockets applied")
        return female.copy(), notes, 0

    notes.append(
        f"female: {plugs_ok} ROI socket plug(s) "
        f"(local remesh+boolean only; body outside ROI preserved)"
    )
    return out, notes, plugs_ok


def _side_sign(part: Trimesh, origin: np.ndarray, normal: np.ndarray) -> float:
    c = np.asarray(part.centroid, dtype=float)
    return float(np.dot(c - origin, normal))


def _resolve_layout(
    parts: list[Trimesh],
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    spec: PinSpec,
) -> _PinLayout | None:
    notes: list[str] = []
    n = np.asarray(plane_normal, dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    origin = np.asarray(plane_origin, dtype=float)

    s0 = _side_sign(parts[0], origin, n)
    s1 = _side_sign(parts[1], origin, n)
    if s0 >= s1:
        male_index, female_index = 0, 1
    else:
        male_index, female_index = 1, 0
    notes.append(f"male=part[{male_index}] (+n) female=part[{female_index}] (-n)")

    male = parts[male_index]
    female = parts[female_index]
    f_mask = _cut_face_mask(female, origin, n, cos_thresh=0.80, plane_tol_mm=0.75)
    m_mask = _cut_face_mask(male, origin, n, cos_thresh=0.80, plane_tol_mm=0.75)
    notes.append(
        f"cut-cap faces: female={int(np.count_nonzero(f_mask))} "
        f"male={int(np.count_nonzero(m_mask))}"
    )

    layout_mesh, layout_mask = (female, f_mask) if np.any(f_mask) else (male, m_mask)
    if not np.any(layout_mask):
        notes.append("no cut-face caps found")
        return None

    bounds = _cut_uv_bounds(layout_mesh, origin, n, layout_mask)
    if bounds is None:
        notes.append("no cut-face bounds")
        return None

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
        notes.append("cut face too small for pins")
        return None

    # Polygon inset can land in gaps (shoe opening / missing cap tris).
    # Snap every center onto a real female (or layout) cut-cap triangle.
    snap_mesh = female if np.any(f_mask) else layout_mesh
    snap_mask = f_mask if np.any(f_mask) else layout_mask
    centers, snap_notes = _snap_centers_onto_cap(
        snap_mesh, snap_mask, origin, u, v, centers
    )
    notes.extend(snap_notes)

    return _PinLayout(
        male_index=male_index,
        female_index=female_index,
        origin=origin,
        normal=n,
        u=u,
        v=v,
        centers_uv=centers,
        f_mask=f_mask,
        m_mask=m_mask,
        notes=notes,
    )


def _snap_centers_onto_cap(
    mesh: Trimesh,
    face_mask: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    centers: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[str]]:
    """Ensure each pin UV lies on an actual cut-cap triangle (not empty outline)."""
    notes: list[str] = []
    if not np.any(face_mask):
        return centers, ["snap skipped: no cut-cap faces"]

    cap_ids = np.where(face_mask)[0]
    cap_centroids_uv: list[tuple[float, float]] = []
    for i in cap_ids:
        cu, cv = _uv(mesh.triangles_center[i], origin, u, v)
        cap_centroids_uv.append((cu, cv))

    snapped: list[tuple[float, float]] = []
    for pu, pv in centers:
        on_cap = False
        for i in cap_ids:
            tri = mesh.vertices[mesh.faces[i]]
            tri_uv = np.array([_uv(p, origin, u, v) for p in tri], dtype=float)
            if _point_in_triangle_2d(pu, pv, tri_uv):
                snapped.append((pu, pv))
                on_cap = True
                break
        if on_cap:
            continue
        # Nearest cut-face centroid
        dists = [(cu - pu) ** 2 + (cv - pv) ** 2 for cu, cv in cap_centroids_uv]
        j = int(np.argmin(dists))
        spu, spv = cap_centroids_uv[j]
        notes.append(
            f"snapped pin center ({pu:.1f},{pv:.1f})→({spu:.1f},{spv:.1f}) onto cut-cap"
        )
        snapped.append((spu, spv))
    return snapped, notes


# ---------------------------------------------------------------------------
# FROZEN: male pin concatenation — visually verified; do not "improve" casually.
# ---------------------------------------------------------------------------
def add_male_pins_frozen(
    male: Trimesh,
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    centers_uv: list[tuple[float, float]],
    spec: PinSpec,
) -> tuple[Trimesh, list[str]]:
    """
    FROZEN male-pin path (shoe_cli_full verified).

    Concatenate high-res cylinders from the cut plane into the female (-normal).
    Never remesh the male body.
    """
    notes: list[str] = []
    pin_meshes: list[Trimesh] = []
    for cu, cv in centers_uv:
        base = origin + u * cu + v * cv
        tip = base - normal * spec.length_mm
        pin_meshes.append(
            _make_cylinder(base + normal * 0.2, tip, spec.radius_mm, spec.sections)
        )
    out = trimesh.util.concatenate([male] + pin_meshes)
    out.merge_vertices()
    notes.append(
        f"male: concatenated {len(pin_meshes)} pins "
        f"(Ø{spec.diameter_mm}mm × {spec.length_mm}mm, {spec.sections}-gon) [FROZEN]"
    )
    return out, notes


def _uv(p: np.ndarray, origin: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    rel = p - origin
    return float(np.dot(rel, u)), float(np.dot(rel, v))


def _face_intersects_circle(
    mesh: Trimesh,
    face_index: int,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pu: float,
    pv: float,
    radius: float,
) -> bool:
    """True if any vertex is inside or the centroid is inside the circle."""
    pts = mesh.vertices[mesh.faces[face_index]]
    uvs = [_uv(p, origin, u, v) for p in pts]
    if any((a - pu) ** 2 + (b - pv) ** 2 <= radius**2 for a, b in uvs):
        return True
    cu = sum(a for a, _ in uvs) / 3.0
    cv = sum(b for _, b in uvs) / 3.0
    return (cu - pu) ** 2 + (cv - pv) ** 2 <= radius**2


def _face_area(mesh: Trimesh, face_index: int) -> float:
    pts = mesh.vertices[mesh.faces[face_index]]
    return float(0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0])))


def _refine_cut_near_hole(
    mesh: Trimesh,
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    center_uv: tuple[float, float],
    hole_radius: float,
    max_face_area: float = 0.6,
    max_rounds: int = 6,
) -> Trimesh:
    """
    Locally subdivide cut-cap faces near one hole so punch removes small
    triangles only — prevents deleting one giant triangle (the wedge void).
    """
    pu, pv = center_uv
    out = mesh.copy()
    nrm = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    for _ in range(max_rounds):
        f_mask = _cut_face_mask(
            out, plane_origin, plane_normal, cos_thresh=0.80, plane_tol_mm=0.75
        )
        # Also consider near-plane faces with odd winding (needed for stubborn hole[1])
        band = np.abs((out.triangles_center - plane_origin) @ nrm) <= 1.5
        f_mask = f_mask | band
        split_ids = [
            int(i)
            for i in np.where(f_mask)[0]
            if _face_intersects_circle(out, int(i), plane_origin, u, v, pu, pv, hole_radius)
            and _face_area(out, int(i)) > max_face_area
        ]
        if not split_ids:
            break

        verts = [np.asarray(out.vertices, dtype=float)]
        # Use list of faces; mark split faces for replacement
        faces = np.asarray(out.faces, dtype=np.int64)
        keep = np.ones(len(faces), dtype=bool)
        new_faces: list[list[int]] = []
        vcount = len(out.vertices)
        # Midpoint cache for edges
        mid_cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            nonlocal vcount
            key = (a, b) if a < b else (b, a)
            if key in mid_cache:
                return mid_cache[key]
            m = 0.5 * (out.vertices[a] + out.vertices[b])
            verts.append(m.reshape(1, 3))
            mid_cache[key] = vcount
            vcount += 1
            return mid_cache[key]

        for fi in split_ids:
            keep[fi] = False
            a, b, c = (int(x) for x in faces[fi])
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend(
                [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            )

        base_faces = faces[keep].tolist()
        all_faces = np.asarray(base_faces + new_faces, dtype=np.int64)
        all_verts = np.vstack(verts)
        out = Trimesh(vertices=all_verts, faces=all_faces, process=False)

    return out


def _point_in_triangle_2d(
    pu: float, pv: float, tri_uv: np.ndarray, eps: float = 1e-9
) -> bool:
    """Barycentric point-in-triangle test in UV."""
    (ax, ay), (bx, by), (cx, cy) = tri_uv
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = pu - ax, pv - ay
    den = v0x * v1y - v1x * v0y
    if abs(den) < eps:
        return False
    u = (v2x * v1y - v1x * v2y) / den
    v = (v0x * v2y - v2x * v0y) / den
    return u >= -eps and v >= -eps and (u + v) <= 1.0 + eps


def _remove_cap_faces_in_circle(
    mesh: Trimesh,
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    center_uv: tuple[float, float],
    hole_radius: float,
    max_delete_area: float,
) -> tuple[Trimesh, int]:
    f_mask = _cut_face_mask(
        mesh, plane_origin, plane_normal, cos_thresh=0.80, plane_tol_mm=0.75
    )
    if not np.any(f_mask):
        return mesh, 0

    pu, pv = center_uv
    r2 = hole_radius**2
    remove = np.zeros(len(mesh.faces), dtype=bool)
    centers = mesh.triangles_center
    for i in np.where(f_mask)[0]:
        if _face_area(mesh, int(i)) > max_delete_area:
            continue
        cu, cv = _uv(centers[i], plane_origin, u, v)
        if (cu - pu) ** 2 + (cv - pv) ** 2 <= r2:
            remove[i] = True

    n_removed = int(np.count_nonzero(remove))
    if n_removed == 0:
        return mesh, 0

    keep = ~remove
    try:
        out = mesh.submesh([np.where(keep)[0]], append=True)
    except Exception:
        out = mesh.copy()
        out.update_faces(keep)
        out.remove_unreferenced_vertices()
    return out, n_removed


def _punch_one_socket(
    mesh: Trimesh,
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    center_uv: tuple[float, float],
    hole_radius: float,
) -> tuple[Trimesh, int]:
    """
    Refine near one hole, then remove cut-cap faces whose centroid is inside.

    Retries with tighter refinement if the first pass removes nothing (common
    when a pin sits on a large/odd cut triangle).
    """
    pu, pv = center_uv
    attempts = [
        # (search_radius_scale, max_face_area, rounds, max_delete_area)
        (1.05, 0.6, 6, 1.25),
        (1.25, 0.25, 10, 2.0),
        (1.40, 0.12, 12, 3.0),
    ]
    best = mesh
    best_removed = 0

    for scale, max_area, rounds, max_del in attempts:
        refined = _refine_cut_near_hole(
            mesh,
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            u=u,
            v=v,
            center_uv=center_uv,
            hole_radius=hole_radius * scale,
            max_face_area=max_area,
            max_rounds=rounds,
        )

        # If no face intersects the search circle, force-split the cut triangle
        # that contains the pin center (if any).
        f_mask = _cut_face_mask(
            refined, plane_origin, plane_normal, cos_thresh=0.80, plane_tol_mm=0.75
        )
        hits = [
            int(i)
            for i in np.where(f_mask)[0]
            if _face_intersects_circle(
                refined,
                int(i),
                plane_origin,
                u,
                v,
                pu,
                pv,
                hole_radius * scale,
            )
        ]
        if not hits:
            for i in np.where(f_mask)[0]:
                tri = refined.vertices[refined.faces[i]]
                tri_uv = np.array([_uv(p, plane_origin, u, v) for p in tri], dtype=float)
                if _point_in_triangle_2d(pu, pv, tri_uv):
                    # One forced refine round around a slightly larger radius
                    refined = _refine_cut_near_hole(
                        refined,
                        plane_origin=plane_origin,
                        plane_normal=plane_normal,
                        u=u,
                        v=v,
                        center_uv=center_uv,
                        hole_radius=max(hole_radius * 1.5, 3.0),
                        max_face_area=0.1,
                        max_rounds=8,
                    )
                    break

        punched, n_removed = _remove_cap_faces_in_circle(
            refined,
            plane_origin=plane_origin,
            plane_normal=plane_normal,
            u=u,
            v=v,
            center_uv=center_uv,
            hole_radius=hole_radius * 0.99,
            max_delete_area=max_del,
        )
        if n_removed > best_removed:
            best, best_removed = punched, n_removed
        if n_removed > 0:
            return punched, n_removed

    # Last resort: snap to nearest near-plane face and carve a UV disk there.
    forced = _force_carve_uv_disk(
        mesh,
        plane_origin=plane_origin,
        plane_normal=plane_normal,
        u=u,
        v=v,
        center_uv=center_uv,
        hole_radius=hole_radius,
    )
    if forced[1] > 0:
        return forced

    return best, best_removed


def _force_carve_uv_disk(
    mesh: Trimesh,
    *,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    center_uv: tuple[float, float],
    hole_radius: float,
) -> tuple[Trimesh, int]:
    """
    Guaranteed carve attempt: move to nearest near-plane face centroid if needed,
    refine, delete every small face whose centroid lies in the UV disk.
    """
    nrm = plane_normal / (np.linalg.norm(plane_normal) + 1e-12)
    dist = np.abs((mesh.triangles_center - plane_origin) @ nrm)
    band_ids = np.where(dist <= 1.5)[0]
    if len(band_ids) == 0:
        return mesh, 0

    pu, pv = center_uv
    # Prefer a band face that contains the point; else nearest band centroid.
    target = (pu, pv)
    containing = None
    for i in band_ids:
        tri_uv = np.array(
            [_uv(p, plane_origin, u, v) for p in mesh.vertices[mesh.faces[i]]],
            dtype=float,
        )
        if _point_in_triangle_2d(pu, pv, tri_uv):
            containing = int(i)
            break
    if containing is None:
        best_i = int(
            min(
                band_ids,
                key=lambda i: (
                    (_uv(mesh.triangles_center[i], plane_origin, u, v)[0] - pu) ** 2
                    + (_uv(mesh.triangles_center[i], plane_origin, u, v)[1] - pv) ** 2
                ),
            )
        )
        target = _uv(mesh.triangles_center[best_i], plane_origin, u, v)

    refined = _refine_cut_near_hole(
        mesh,
        plane_origin=plane_origin,
        plane_normal=plane_normal,
        u=u,
        v=v,
        center_uv=target,
        hole_radius=hole_radius * 1.6,
        max_face_area=0.1,
        max_rounds=14,
    )
    dist2 = np.abs((refined.triangles_center - plane_origin) @ nrm)
    r_hit = hole_radius * 1.05
    tpu, tpv = target
    remove = np.zeros(len(refined.faces), dtype=bool)
    for i in range(len(refined.faces)):
        if dist2[i] > 1.5:
            continue
        if _face_intersects_circle(
            refined, i, plane_origin, u, v, tpu, tpv, r_hit
        ):
            remove[i] = True
            continue
        tri_uv = np.array(
            [_uv(p, plane_origin, u, v) for p in refined.vertices[refined.faces[i]]],
            dtype=float,
        )
        if _point_in_triangle_2d(tpu, tpv, tri_uv):
            remove[i] = True
    n_removed = int(np.count_nonzero(remove))
    if n_removed == 0:
        return mesh, 0
    keep = ~remove
    try:
        out = refined.submesh([np.where(keep)[0]], append=True)
    except Exception:
        out = refined.copy()
        out.update_faces(keep)
        out.remove_unreferenced_vertices()
    return out, n_removed


def _strip_faces(mesh: Trimesh, face_mask: np.ndarray) -> Trimesh:
    if not np.any(face_mask):
        return mesh.copy()
    keep = ~face_mask
    try:
        return mesh.submesh([np.where(keep)[0]], append=True)
    except Exception:
        out = mesh.copy()
        out.update_faces(keep)
        out.remove_unreferenced_vertices()
        return out


def _triangulate_polygon_2d(polygon: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a (possibly holed) polygon; earcut/triangle if present else Delaunay."""
    for engine in ("earcut", "triangle", None):
        try:
            if engine is None:
                break
            return triangulate_polygon(polygon, engine=engine)
        except Exception:
            continue

    # Fallback: boundary Delaunay kept only where centroid is inside polygon
    from scipy.spatial import Delaunay

    ring_pts: list[list[float]] = []
    coords = list(polygon.exterior.coords)[:-1]
    # Densify exterior a bit for stabler triangles
    for a, b in zip(coords, coords[1:] + coords[:1]):
        ring_pts.append([float(a[0]), float(a[1])])
        mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        ring_pts.append([float(mid[0]), float(mid[1])])
    for interior in polygon.interiors:
        ic = list(interior.coords)[:-1]
        for a, b in zip(ic, ic[1:] + ic[:1]):
            ring_pts.append([float(a[0]), float(a[1])])
            mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            ring_pts.append([float(mid[0]), float(mid[1])])
    pts = np.asarray(ring_pts, dtype=float)
    # Unique rows
    pts = np.unique(np.round(pts, 6), axis=0)
    if len(pts) < 3:
        raise ValueError("not enough polygon samples to triangulate")
    delaunay = Delaunay(pts)
    faces: list[list[int]] = []
    for simplex in delaunay.simplices:
        c = pts[simplex].mean(axis=0)
        if polygon.contains(Point(float(c[0]), float(c[1]))):
            faces.append([int(simplex[0]), int(simplex[1]), int(simplex[2])])
    if not faces:
        raise ValueError("Delaunay produced no interior triangles")
    return pts, np.asarray(faces, dtype=np.int64)


def _cap_mesh_from_polygon(
    polygon: Polygon,
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> Trimesh:
    """Planar retriangulation of a (possibly holed) cut-cap polygon."""
    verts2, faces = _triangulate_polygon_2d(polygon)
    verts3 = (
        origin
        + np.outer(verts2[:, 0], u)
        + np.outer(verts2[:, 1], v)
        - normal * 0.02
    )
    cap = Trimesh(vertices=verts3, faces=faces, process=True)
    if len(cap.faces) and float(np.dot(cap.face_normals.mean(axis=0), normal)) < 0:
        cap.invert()
    return cap


def punch_female_sockets_stepwise(
    female: Trimesh,
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    centers_uv: list[tuple[float, float]],
    hole_radius: float,
) -> tuple[Trimesh, list[str], int]:
    """
    Rebuild the cut-cap with clean circular holes, one hole at a time.

    Uses shapely polygon difference + earcut retriangulation so the cut face
    stays a clean seam outline — no subdivided face-delete streaks.
    """
    notes: list[str] = []
    f_mask = _cut_face_mask(female, origin, normal, cos_thresh=0.80, plane_tol_mm=0.75)
    if not np.any(f_mask):
        notes.append("female cut cap not found — sockets skipped")
        return female.copy(), notes, 0

    poly = _cut_face_polygon_2d(female, f_mask, origin, u, v)
    if poly is None:
        notes.append("cut-cap polygon failed — sockets skipped")
        return female.copy(), notes, 0

    working = poly
    holes_ok = 0
    ext0 = np.asarray(female.extents, dtype=float)

    for step, (pu, pv) in enumerate(centers_uv, start=1):
        disk = Point(float(pu), float(pv)).buffer(float(hole_radius), resolution=24)
        if working.intersection(disk).is_empty:
            # Snap disk onto polygon if center missed solid cap
            rp = working.representative_point()
            # nearest point on polygon to desired center
            try:
                from shapely.ops import nearest_points

                nearest = nearest_points(working, Point(float(pu), float(pv)))[0]
                pu, pv = float(nearest.x), float(nearest.y)
                disk = Point(pu, pv).buffer(float(hole_radius), resolution=24)
                notes.append(
                    f"hole[{step}] center snapped onto cap ({pu:.1f},{pv:.1f})"
                )
            except Exception:
                notes.append(f"hole[{step}/{len(centers_uv)}] WARN empty (no cap under center)")
                continue

        nxt = working.difference(disk)
        nxt = _largest_polygon(nxt)
        if nxt is None or nxt.is_empty:
            notes.append(f"hole[{step}/{len(centers_uv)}] WARN difference empty; skipped")
            continue
        if nxt.area >= working.area - 1e-6:
            notes.append(f"hole[{step}/{len(centers_uv)}] WARN no area change; skipped")
            continue

        # Prefer a single polygon that still has an interior hole when possible
        n_interiors = len(getattr(nxt, "interiors", []) or [])
        working = nxt
        holes_ok += 1
        notes.append(
            f"hole[{step}/{len(centers_uv)}] center=({pu:.1f},{pv:.1f}) "
            f"polygon hole ok (area {poly.area:.1f}→{working.area:.1f}, "
            f"interiors={n_interiors}) [ok]"
        )

    if holes_ok == 0:
        notes.append("cap rebuild: 0 holes applied")
        return female.copy(), notes, 0

    try:
        cap = _cap_mesh_from_polygon(
            working, origin=origin, normal=normal, u=u, v=v
        )
    except Exception as exc:
        notes.append(f"cap retriangulation failed ({exc}); female unchanged")
        return female.copy(), notes, 0

    body = _strip_faces(female, f_mask)
    out = trimesh.util.concatenate([body, cap])
    out.merge_vertices()

    ext1 = np.asarray(out.extents, dtype=float)
    if np.any(ext1 > ext0 * 1.15 + 5.0):
        notes.append(
            f"cap rebuild aborted (extent {ext0.round(1)}→{ext1.round(1)}); "
            "female unchanged"
        )
        return female.copy(), notes, 0

    notes.append(
        f"female: rebuilt cut-cap with {holes_ok} clean circular hole(s) "
        f"(earcut, no face-delete shred, no remesh)"
    )
    return out, notes, holes_ok


def apply_female_cap_and_sleeves(
    female: Trimesh,
    *,
    origin: np.ndarray,
    normal: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    centers_uv: list[tuple[float, float]],
    hole_radius: float,
    depth_mm: float,
    sections: int = 48,
) -> tuple[Trimesh, list[str], int]:
    """
    FROZEN female-socket path (shoe_cli_full verified).

    1) Rebuild cut-cap with clean circular holes (planar earcut).
    2) Add recessed tube sleeves entirely on the female side for pin depth.
    Never strips organic side-wall faces (avoids wedge gaps).
    """
    notes: list[str] = []
    ext0 = np.asarray(female.extents, dtype=float)

    capped, cap_notes, n_holes = punch_female_sockets_stepwise(
        female,
        origin=origin,
        normal=normal,
        u=u,
        v=v,
        centers_uv=centers_uv,
        hole_radius=hole_radius,
    )
    notes.extend(cap_notes)
    if n_holes == 0:
        notes.append("cap+sleeves aborted: no cut-cap holes")
        return female.copy(), notes, 0

    sleeves: list[Trimesh] = [capped]
    sleeves_ok = 0
    for step, center in enumerate(centers_uv, start=1):
        sleeve = _make_recessed_sleeve(
            origin=origin,
            normal=normal,
            u=u,
            v=v,
            center_uv=center,
            hole_radius=hole_radius,
            depth_mm=depth_mm,
            sections=sections,
        )
        if sleeve is None or len(sleeve.faces) == 0:
            notes.append(f"sleeve[{step}] boolean failed; hole kept without tube")
            continue
        # Guard: sleeve must not stick out past the cut toward male
        n = normal / (np.linalg.norm(normal) + 1e-12)
        proud = float(np.max((sleeve.vertices - origin) @ n))
        if proud > 0.15:
            sleeve.vertices = sleeve.vertices - n * (proud + 0.05)
            notes.append(f"sleeve[{step}] pushed flush (was proud {proud:.2f}mm)")
        sleeves.append(sleeve)
        sleeves_ok += 1
        notes.append(
            f"sleeve[{step}/{len(centers_uv)}] center=({center[0]:.1f},{center[1]:.1f}) "
            f"recessed tube ok [ok]"
        )

    out = trimesh.util.concatenate(sleeves)
    out.merge_vertices()
    ext1 = np.asarray(out.extents, dtype=float)
    if np.any(ext1 > ext0 * 1.15 + 8.0):
        notes.append(
            f"cap+sleeves aborted (extent {ext0.round(1)}→{ext1.round(1)}); "
            "keeping cap-only"
        )
        return capped, notes, n_holes

    notes.append(
        f"female: {n_holes} clean cap hole(s) + {sleeves_ok} recessed sleeve(s) "
        f"(no body ROI strip, no embossed plugs)"
    )
    return out, notes, n_holes


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
    apply_male: bool = True,
    apply_holes: bool = False,
    centers_uv: list[tuple[float, float]] | None = None,
) -> PinApplyResult:
    """
    Apply mating features in separate phases.

    - apply_male=True  → FROZEN male pin concatenate
    - apply_holes=True → female clean cap holes + recessed sleeves
    - centers_uv       → reuse male-phase pin centers so holes align
    """
    notes: list[str] = []
    if spec is None:
        spec = PinSpec()
    if len(parts) < 2:
        return PinApplyResult(parts, 0, False, "none", ["need >=2 parts"])
    if not apply_male and not apply_holes:
        return PinApplyResult(parts, 0, False, "none", ["nothing to apply"])

    layout = _resolve_layout(
        parts, plane_origin=plane_origin, plane_normal=plane_normal, spec=spec
    )
    if layout is None:
        return PinApplyResult(parts, 0, False, "none", ["layout failed"])

    # Allow caller override of indices only when both provided
    if male_index is not None and female_index is not None:
        layout.male_index = male_index
        layout.female_index = female_index

    if centers_uv is not None and len(centers_uv) > 0:
        layout.centers_uv = list(centers_uv)
        notes.append(f"reusing {len(centers_uv)} pin center(s) from male phase")

    notes.extend(layout.notes)
    out = list(parts)
    male = out[layout.male_index].copy()
    female = out[layout.female_index].copy()

    # Re-snap onto the live female cut cap (outline gaps / shoe openings).
    if apply_holes and np.any(layout.f_mask):
        layout.centers_uv, snap_notes = _snap_centers_onto_cap(
            female, layout.f_mask, layout.origin, layout.u, layout.v, layout.centers_uv
        )
        notes.extend(snap_notes)

    pins_applied = 0
    used_remesh = False
    method = "none"

    if apply_male:
        male, male_notes = add_male_pins_frozen(
            male,
            origin=layout.origin,
            normal=layout.normal,
            u=layout.u,
            v=layout.v,
            centers_uv=layout.centers_uv,
            spec=spec,
        )
        notes.extend(male_notes)
        pins_applied = len(layout.centers_uv)
        method = "male_only"
        out[layout.male_index] = male

    if apply_holes:
        hole_r = spec.radius_mm + spec.clearance_mm
        # Primary: clean cap holes + recessed sleeves (no embossed ROI plugs)
        female, hole_notes, n_ok = apply_female_cap_and_sleeves(
            female,
            origin=layout.origin,
            normal=layout.normal,
            u=layout.u,
            v=layout.v,
            centers_uv=layout.centers_uv,
            hole_radius=hole_r,
            depth_mm=spec.length_mm,
            sections=spec.sections,
        )
        notes.extend(hole_notes)
        if n_ok > 0:
            method = "cap_sleeves" if not apply_male else "male+cap_sleeves"
        elif allow_remesh:
            used_remesh = True
            method = "boolean"
            pitch = remesh_pitch_mm
            f_vol = _ensure_volume_local(out[layout.female_index].copy(), pitch=pitch)
            notes.append(f"female whole-part remesh pitch={pitch}mm (allow_remesh)")
            ok = True
            for i, (cu, cv) in enumerate(layout.centers_uv, start=1):
                base = layout.origin + layout.u * cu + layout.v * cv
                tip = base - layout.normal * spec.length_mm
                hole = _make_cylinder(
                    base + layout.normal * 0.5,
                    tip - layout.normal * 0.5,
                    hole_r,
                    spec.sections,
                )
                punched = _boolean_difference(f_vol, hole)
                if punched is None:
                    notes.append(f"hole[{i}] remesh boolean failed")
                    ok = False
                    break
                f_vol = punched
                notes.append(f"hole[{i}] remesh boolean ok")
            female = f_vol if ok else out[layout.female_index].copy()
            if not ok:
                notes.append("whole-part remesh boolean aborted")
        else:
            female = out[layout.female_index].copy()
            notes.append("sockets skipped (cap+sleeves failed; male pins unchanged)")
        out[layout.female_index] = female

    return PinApplyResult(
        parts=out,
        pins_applied=pins_applied if apply_male else len(layout.centers_uv),
        used_remesh=used_remesh,
        method=method,
        notes=notes,
        centers_uv=list(layout.centers_uv),
        layout_origin=layout.origin,
        layout_normal=layout.normal,
        layout_u=layout.u,
        layout_v=layout.v,
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
    apply_male: bool = True,
    apply_holes: bool = False,
    centers_uv: list[tuple[float, float]] | None = None,
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
        apply_male=apply_male,
        apply_holes=apply_holes,
        centers_uv=centers_uv,
    )
    report = PinReport(
        pin_count=result.pins_applied,
        used_remesh=result.used_remesh,
        method=result.method,
        notes=result.notes,
        centers_uv=result.centers_uv,
        layout_origin=result.layout_origin,
        layout_normal=result.layout_normal,
        layout_u=result.layout_u,
        layout_v=result.layout_v,
    )
    return result.parts[0], result.parts[1], report
