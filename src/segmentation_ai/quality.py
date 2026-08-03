"""Mesh quality gates for organic repair / cut / pin pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh


@dataclass
class QualityCheck:
    name: str
    ok: bool
    message: str


@dataclass
class QualityReport:
    ok: bool
    checks: list[QualityCheck] = field(default_factory=list)

    def print(self) -> None:
        status = "PASS" if self.ok else "FAIL"
        print(f"Quality: {status}")
        for c in self.checks:
            mark = "OK" if c.ok else "FAIL"
            print(f"  [{mark}] {c.name}: {c.message}")


def _check(
    name: str, ok: bool, message: str, checks: list[QualityCheck]
) -> None:
    checks.append(QualityCheck(name=name, ok=ok, message=message))


def check_repair_quality(
    *,
    faces_before: int,
    faces_after: int,
    repair_mode: str,
    used_voxel: bool,
    min_face_keep_ratio: float = 0.70,
) -> QualityReport:
    """Gate after repair — block silent detail destruction."""
    checks: list[QualityCheck] = []
    ratio = faces_after / max(faces_before, 1)
    _check(
        "face_keep_ratio",
        ratio >= min_face_keep_ratio or repair_mode == "voxel",
        f"faces {faces_before}→{faces_after} (ratio={ratio:.2f}, min={min_face_keep_ratio})",
        checks,
    )
    if repair_mode != "voxel":
        _check(
            "no_unexpected_voxel",
            not used_voxel,
            "voxel remesh used" if used_voxel else "no whole-model voxel remesh",
            checks,
        )
    else:
        _check("voxel_requested", True, "repair_mode=voxel (lossy OK if explicit)", checks)

    ok = all(c.ok for c in checks)
    return QualityReport(ok=ok, checks=checks)


def check_cut_quality(
    *,
    input_mesh: trimesh.Trimesh,
    parts: list[trimesh.Trimesh],
    min_part_face_ratio: float = 0.15,
    max_extent_drift: float = 0.08,
) -> QualityReport:
    """
    Gate after split — each part should retain substantial face count vs input,
    and extents along each axis should not explode.
    """
    checks: list[QualityCheck] = []
    in_faces = max(len(input_mesh.faces), 1)
    in_ext = np.asarray(input_mesh.extents, dtype=float)

    total_part_faces = sum(len(p.faces) for p in parts)
    _check(
        "parts_face_sum",
        total_part_faces >= 0.70 * in_faces,
        f"sum(part faces)={total_part_faces} vs input={in_faces}",
        checks,
    )

    for i, part in enumerate(parts):
        ratio = len(part.faces) / in_faces
        _check(
            f"part[{i}]_face_ratio",
            ratio >= min_part_face_ratio,
            f"faces={len(part.faces)} ratio={ratio:.2f} (min={min_part_face_ratio})",
            checks,
        )
        ext = np.asarray(part.extents, dtype=float)
        # No axis should be hugely larger than the input (pins may add a few mm)
        drift_ok = bool(np.all(ext <= in_ext * (1.0 + max_extent_drift) + 10.0))
        _check(
            f"part[{i}]_extent_bounds",
            drift_ok,
            f"extents={np.round(ext, 2)} input={np.round(in_ext, 2)}",
            checks,
        )

    ok = all(c.ok for c in checks)
    return QualityReport(ok=ok, checks=checks)


def check_pin_quality(
    *,
    parts_before: list[trimesh.Trimesh],
    parts_after: list[trimesh.Trimesh],
    pins_applied: int,
    used_remesh: bool,
    allow_remesh: bool,
    min_face_keep_ratio: float = 0.85,
    max_extent_growth: float = 1.15,
) -> QualityReport:
    """Gate after pins — detail must not collapse; no extent blow-up / remesh."""
    checks: list[QualityCheck] = []
    _check(
        "pins_applied",
        pins_applied > 0,
        f"pins_applied={pins_applied}",
        checks,
    )
    _check(
        "remesh_policy",
        (not used_remesh) or allow_remesh,
        "remesh used without allow" if used_remesh and not allow_remesh else (
            "remesh allowed" if used_remesh else "no remesh"
        ),
        checks,
    )
    for i, (before, after) in enumerate(zip(parts_before, parts_after)):
        ratio = len(after.faces) / max(len(before.faces), 1)
        ok = ratio >= min_face_keep_ratio or len(after.faces) > len(before.faces)
        _check(
            f"part[{i}]_pin_face_keep",
            ok,
            f"faces {len(before.faces)}→{len(after.faces)} ratio={ratio:.2f}",
            checks,
        )
        be = np.asarray(before.extents, dtype=float)
        ae = np.asarray(after.extents, dtype=float)
        # Pins may add length; reject wafer/remesh slabs that balloon XY
        extent_ok = bool(np.all(ae <= be * max_extent_growth + 12.0))
        _check(
            f"part[{i}]_pin_extent",
            extent_ok,
            f"extents {np.round(be, 2)}→{np.round(ae, 2)}",
            checks,
        )
    ok = all(c.ok for c in checks)
    return QualityReport(ok=ok, checks=checks)
