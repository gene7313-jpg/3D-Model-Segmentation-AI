"""Organic / aesthetic cut track: repair, mid-seam splits, pins, P1S validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
import yaml

from .cut import plane_cut, validate_parts
from .pins import PinSpec, add_mating_pins
from .printer import (
    aabb_extents,
    default_profile_path,
    load_printer_profile,
    part_fits_build_plate,
)
from .quality import (
    QualityReport,
    check_cut_quality,
    check_pin_quality,
    check_repair_quality,
)
from .repair import RepairMode, repair_mesh


@dataclass
class OrganicCutResult:
    project_dir: Path
    parts_dir: Path
    part_paths: list[Path]
    all_fit: bool
    watertight: bool
    fit_reports: list[str]
    repaired_path: Path | None = None
    pins_applied: int = 0
    notes: list[str] = field(default_factory=list)
    quality_ok: bool = True
    quality_reports: list[QualityReport] = field(default_factory=list)


def _load_print_mesh(project: Path) -> tuple[trimesh.Trimesh, Path]:
    """
    Prefer the scaled high-res print mesh. Never use source_repaired.stl as input
    (that file may be a lossy voxel remesh from an earlier run).
    """
    meta_path = project / "meta.yaml"
    skip_names = {"source_repaired.stl"}
    candidates: list[Path] = []

    # 1) Scaled print meshes first (detail)
    candidates.extend(sorted(project.glob("source_*mm.stl")))
    # 2) meta print_stl if it isn't a repaired/voxel artifact
    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text()) or {}
        print_stl = meta.get("print_stl")
        if print_stl and print_stl not in skip_names:
            candidates.append(project / str(print_stl))
    # 3) raw source
    candidates.append(project / "source.stl")

    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve() if path.exists() else path
        if path in seen or not path.is_file():
            continue
        if path.name in skip_names:
            continue
        seen.add(path)
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        print(f"Using input mesh: {path.name}  faces={len(mesh.faces)}")
        return mesh, path
    raise FileNotFoundError(
        f"No print STL found in {project}. Run generate-from-image first."
    )


def longest_axis_midplane(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Aesthetic starter seam: mid-plane normal to the longest AABB axis."""
    verts = np.asarray(mesh.vertices, dtype=float)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    extents = maxs - mins
    axis = int(np.argmax(extents))
    origin = (mins + maxs) / 2.0
    normal = np.zeros(3, dtype=float)
    normal[axis] = 1.0
    return origin, normal


def split_until_fits(
    mesh: trimesh.Trimesh,
    profile,
    *,
    max_splits: int,
    force_split: bool,
) -> tuple[list[trimesh.Trimesh], np.ndarray | None, np.ndarray | None]:
    """
    Recursively mid-plane split along longest axis until parts fit (or budget).

    Returns parts plus the first cut plane (origin, normal) when a split occurred.
    """
    ext = aabb_extents(np.asarray(mesh.vertices))
    fits, _ = part_fits_build_plate(ext, profile, allow_rotation=True)

    queue: list[tuple[trimesh.Trimesh, int]] = [(mesh, 0)]
    done: list[trimesh.Trimesh] = []
    first_origin: np.ndarray | None = None
    first_normal: np.ndarray | None = None

    if force_split and fits and max_splits >= 1:
        origin, normal = longest_axis_midplane(mesh)
        parts = plane_cut(mesh, origin=origin, normal=normal)
        if len(parts) >= 2:
            first_origin, first_normal = origin, normal
            queue = [(p, 1) for p in parts]
        else:
            return [mesh], None, None

    while queue:
        current, depth = queue.pop(0)
        ext = aabb_extents(np.asarray(current.vertices))
        ok, _ = part_fits_build_plate(ext, profile, allow_rotation=True)
        if ok or depth >= max_splits:
            done.append(current)
            continue

        origin, normal = longest_axis_midplane(current)
        parts = plane_cut(current, origin=origin, normal=normal)
        if len(parts) < 2:
            done.append(current)
            continue
        if first_origin is None:
            first_origin, first_normal = origin, normal
        for p in parts:
            queue.append((p, depth + 1))

    return (done if done else [mesh]), first_origin, first_normal


def _pin_spec_from_config(cfg: dict) -> PinSpec:
    mech = cfg.get("mechanical") or {}
    pins = (cfg.get("organic_cut") or {}).get("pins") or {}
    return PinSpec(
        diameter_mm=float(pins.get("diameter_mm", mech.get("pin_diameter_mm", 3.0))),
        clearance_mm=float(pins.get("clearance_mm", mech.get("pin_clearance_mm", 0.2))),
        length_mm=float(pins.get("length_mm", 6.0)),
        count=int(pins.get("count", 2)),
    )


def cut_organic_project(
    project_dir: str | Path,
    *,
    max_splits: int = 3,
    force_split: bool = False,
    repair_mode: RepairMode = "basic",
    voxel_resolution: int = 160,
    pitch_mm: float | None = None,
    with_pins: bool = False,
    with_pin_holes: bool = False,
    pin_remesh: bool = False,
    pin_spec: PinSpec | None = None,
    enforce_quality: bool = True,
) -> OrganicCutResult:
    from .generate_trellis import load_defaults

    project = Path(project_dir).expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(project)

    defaults = load_defaults()
    profile = load_printer_profile(default_profile_path())
    mesh, source_path = _load_print_mesh(project)
    print(f"Loaded {source_path.name}  faces={len(mesh.faces)}")

    # Quality-first: never voxel-remesh the whole model unless user asked for voxel.
    if repair_mode == "voxel":
        print(
            "WARNING: --repair-mode voxel remeshes the whole model and will look blocky. "
            "Prefer --repair-mode basic for organic detail."
        )
    if with_pins or with_pin_holes:
        print(
            "Pins: male path is FROZEN (concatenate only). "
            "Female holes use local ROI plugs (--with-pin-holes)."
        )

    quality_reports: list[QualityReport] = []

    mesh, repair = repair_mesh(
        mesh,
        mode=repair_mode,
        pitch_mm=pitch_mm,
        voxel_resolution=voxel_resolution,
    )
    print(
        f"Repair ({repair.mode}): watertight {repair.watertight_before} → {repair.watertight_after} "
        f"faces {repair.faces_before} → {repair.faces_after}"
    )
    if repair.pitch_mm is not None:
        print(f"  voxel pitch_mm={repair.pitch_mm:.4f}")
        print("  WARNING: voxel remesh is lossy/blocky — prefer --repair-mode basic for organic detail")
    for note in repair.notes:
        print(f"  {note}")

    used_voxel = repair.pitch_mm is not None or repair.mode == "voxel"
    q_repair = check_repair_quality(
        faces_before=repair.faces_before,
        faces_after=repair.faces_after,
        repair_mode=str(repair.mode),
        used_voxel=used_voxel,
    )
    q_repair.print()
    quality_reports.append(q_repair)
    if not q_repair.ok:
        print(
            "QUALITY FAIL (repair): detail likely destroyed. "
            "Prefer --repair-mode basic (avoid voxel)."
        )

    repaired_path = project / "source_repaired.stl"
    mesh.export(repaired_path)
    print(f"Wrote {repaired_path} (working copy for this cut; input was {source_path.name})")

    parts, cut_origin, cut_normal = split_until_fits(
        mesh,
        profile,
        max_splits=max_splits,
        force_split=force_split,
    )

    q_cut = check_cut_quality(input_mesh=mesh, parts=parts)
    q_cut.print()
    quality_reports.append(q_cut)
    if not q_cut.ok:
        print("QUALITY FAIL (cut): part face counts collapsed vs repaired input.")

    pins_applied = 0
    pin_notes: list[str] = []
    pin_method = "none"
    pin_used_remesh = False
    if with_pins or with_pin_holes:
        if len(parts) == 2 and cut_origin is not None and cut_normal is not None:
            spec = pin_spec or _pin_spec_from_config(defaults)
            print(
                f"Pins: diameter={spec.diameter_mm}mm clearance={spec.clearance_mm}mm "
                f"length={spec.length_mm}mm count={spec.count} "
                f"male={with_pins} holes={with_pin_holes} pin_remesh={pin_remesh}"
            )
            parts_before = [p.copy() for p in parts]

            # Phase A — frozen male pins (optional)
            shared_centers: list[tuple[float, float]] | None = None
            if with_pins:
                male, female, preport = add_mating_pins(
                    parts[0],
                    parts[1],
                    cut_normal=cut_normal,
                    cut_origin=cut_origin,
                    spec=spec,
                    allow_remesh=False,
                    apply_male=True,
                    apply_holes=False,
                )
                parts = [male, female]
                pins_applied = preport.pin_count
                pin_notes.extend(preport.notes)
                pin_method = preport.method
                shared_centers = preport.centers_uv or None
                for note in preport.notes:
                    print(f"  pin: {note}")
                print(f"Pins (male) applied: {pins_applied} method={pin_method}")

            # Phase B — stepwise female holes (optional, separate from male)
            if with_pin_holes:
                male, female, hreport = add_mating_pins(
                    parts[0],
                    parts[1],
                    cut_normal=cut_normal,
                    cut_origin=cut_origin,
                    spec=spec,
                    allow_remesh=pin_remesh,
                    apply_male=False,
                    apply_holes=True,
                    centers_uv=shared_centers,
                )
                parts = [male, female]
                pin_notes.extend(hreport.notes)
                pin_method = hreport.method if hreport.method != "none" else pin_method
                pin_used_remesh = hreport.used_remesh
                for note in hreport.notes:
                    print(f"  hole: {note}")
                print(f"Holes method={hreport.method}")

            q_pins = check_pin_quality(
                parts_before=parts_before,
                parts_after=parts,
                pins_applied=pins_applied if with_pins else max(pins_applied, 1),
                used_remesh=pin_used_remesh,
                allow_remesh=pin_remesh,
            )
            # If male-only, require pins; if holes-only, don't fail on pins_applied==0 from male
            if with_pins:
                q_pins.print()
                quality_reports.append(q_pins)
                if not q_pins.ok:
                    print(
                        "QUALITY FAIL (pins): remesh collapsed detail or pins not applied."
                    )
            else:
                # Holes-only: still run extent/face checks
                q_pins.print()
                quality_reports.append(q_pins)
        else:
            msg = (
                f"pins skipped (need exactly 2 parts from one mid-plane cut; got {len(parts)})"
            )
            pin_notes.append(msg)
            print(msg)

    result = validate_parts(parts, profile)
    for line in result.fit_reports:
        print(line)
    print("All fit:", result.all_fit)

    parts_dir = project / "parts"
    if parts_dir.exists():
        for old in parts_dir.glob("part_*.stl"):
            old.unlink()
    parts_dir.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    part_meta = []
    for i, part in enumerate(result.parts, start=1):
        path = parts_dir / f"part_{i:02d}.stl"
        part.export(path)
        part_paths.append(path)
        ext = aabb_extents(np.asarray(part.vertices))
        ok, msg = part_fits_build_plate(ext, profile, allow_rotation=True)
        part_meta.append(
            {
                "file": path.name,
                "faces": int(len(part.faces)),
                "extents_mm": [float(x) for x in ext.tolist()],
                "fits_build_plate": bool(ok),
                "message": msg,
            }
        )
        print(f"Wrote {path}  faces={len(part.faces)}")

    quality_ok = all(r.ok for r in quality_reports)
    stage_names: list[str] = []
    for r in quality_reports:
        first = r.checks[0].name if r.checks else ""
        if first.startswith("face_keep") or first.startswith("no_unexpected"):
            stage_names.append("repair")
        elif first.startswith("parts_face") or first.startswith("part["):
            stage_names.append("cut")
        elif first.startswith("pins_") or "pin_face" in first:
            stage_names.append("pins")
        else:
            stage_names.append("unknown")
    quality_meta = [
        {
            "stage": stage_names[i] if i < len(stage_names) else f"stage_{i}",
            "ok": r.ok,
            "checks": [
                {"name": c.name, "ok": c.ok, "message": c.message} for c in r.checks
            ],
        }
        for i, r in enumerate(quality_reports)
    ]

    meta_path = project / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
    meta = meta or {}
    meta["domain"] = "organic"
    meta["printer"] = profile.name
    meta["cut_track"] = "organic"
    meta["cut_source_stl"] = source_path.name
    meta["repair"] = {
        "mode": repair.mode,
        "watertight_before": repair.watertight_before,
        "watertight_after": repair.watertight_after,
        "faces_before": repair.faces_before,
        "faces_after": repair.faces_after,
        "pitch_mm": repair.pitch_mm,
        "repaired_stl": repaired_path.name,
        "notes": repair.notes,
    }
    meta["organic_cut"] = {
        "strategy": "longest_axis_midplane",
        "max_splits": max_splits,
        "force_split": force_split,
        "with_pins": with_pins,
        "with_pin_holes": with_pin_holes,
        "pins_applied": pins_applied,
        "pin_method": pin_method,
        "pin_remesh": pin_remesh,
        "pin_used_remesh": pin_used_remesh,
        "pin_notes": pin_notes,
    }
    meta["quality"] = {"ok": quality_ok, "stages": quality_meta}
    meta["part_count"] = len(part_meta)
    meta["parts"] = part_meta
    meta["all_parts_fit_build_plate"] = bool(result.all_fit)
    meta["watertight"] = bool(repair.watertight_after)
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"Updated {meta_path}")
    print(f"Quality overall: {'PASS' if quality_ok else 'FAIL'}")
    if not quality_ok and enforce_quality:
        print("Quality gates failed (use --allow-quality-fail to ignore exit code).")

    return OrganicCutResult(
        project_dir=project,
        parts_dir=parts_dir,
        part_paths=part_paths,
        all_fit=bool(result.all_fit),
        watertight=bool(repair.watertight_after),
        fit_reports=result.fit_reports,
        repaired_path=repaired_path,
        pins_applied=pins_applied,
        notes=pin_notes,
        quality_ok=quality_ok,
        quality_reports=quality_reports,
    )


def iter_organic_projects(root: str | Path) -> list[Path]:
    """List slug dirs under data/raw/organic that look processable."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        has_stl = (
            (child / "source.stl").is_file()
            or any(child.glob("source_*mm.stl"))
            or (
                (child / "meta.yaml").is_file()
                and "print_stl"
                in (yaml.safe_load((child / "meta.yaml").read_text()) or {})
            )
        )
        if has_stl:
            out.append(child)
    return out


def process_organic_batch(
    root: str | Path,
    *,
    fail_fast: bool = False,
    **cut_kwargs,
) -> list[OrganicCutResult]:
    results: list[OrganicCutResult] = []
    projects = iter_organic_projects(root)
    print(f"Found {len(projects)} organic project(s) under {root}")
    for project in projects:
        print(f"\n=== {project.name} ===")
        try:
            results.append(cut_organic_project(project, **cut_kwargs))
        except Exception as exc:
            print(f"ERROR {project.name}: {exc}")
            if fail_fast:
                raise
    return results
