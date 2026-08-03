"""CLI entry points for Phase 0 demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
import yaml

from . import shapes
from .cut import plane_cut, validate_parts
from .printer import (
    aabb_extents,
    default_profile_path,
    load_printer_profile,
    part_fits_build_plate,
)


def _out_dir() -> Path:
    d = Path(__file__).resolve().parents[2] / "data" / "synthetic"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_demo_basic_shapes(_: argparse.Namespace) -> None:
    profile = load_printer_profile(default_profile_path())
    out = _out_dir()

    mesh = shapes.box(size=(200.0, 80.0, 40.0))
    # Mid cut along X — two 100mm halves, both within P1S
    parts = plane_cut(
        mesh,
        origin=np.array([0.0, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]),
    )
    result = validate_parts(parts, profile)

    print(f"Printer: {profile.display_name}")
    print(
        f"Usable volume: {profile.max_part.x:.0f}×{profile.max_part.y:.0f}×{profile.max_part.z:.0f} mm"
    )
    print(f"Parts: {len(result.parts)}")
    for line in result.fit_reports:
        print(" ", line)
    print("All fit:", result.all_fit)

    for i, part in enumerate(result.parts):
        path = out / f"box_cut_part_{i}.stl"
        part.export(path)
        print(f"Wrote {path}")


def cmd_check_oversized(_: argparse.Namespace) -> None:
    """Intentionally oversized single part — must fail build-plate check."""
    profile = load_printer_profile(default_profile_path())
    mesh = shapes.box(size=(400.0, 100.0, 50.0))
    result = validate_parts([mesh], profile)
    for line in result.fit_reports:
        print(line)
    print("All fit:", result.all_fit)
    if result.all_fit:
        raise SystemExit("expected failure for oversized part")


def cmd_ingest_3mf(args: argparse.Namespace) -> None:
    """Ingest a Bambu Studio / generic 3MF into data/raw/<domain>/<slug>/."""
    import shutil

    from .ingest_3mf import load_3mf

    src = Path(args.file).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"File not found: {src}")

    slug = args.slug or src.stem.lower().replace(" ", "-").replace("_", "-")
    domain = args.domain
    root = Path(__file__).resolve().parents[2]
    project = root / "data" / "raw" / domain / slug
    parts_dir = project / "parts"
    source_dir = project / "source"
    parts_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    result = load_3mf(src)
    if not result.parts:
        raise SystemExit(f"No build objects found in {src.name}")

    shutil.copy2(src, source_dir / src.name)
    print(f"Ingesting {src.name} → {project.relative_to(root)}")
    print(f"Units: {result.unit}, objects: {len(result.parts)}")
    if result.cut_info:
        print(f"Cut records: {result.cut_info}")

    for i, part in enumerate(result.parts, start=1):
        out = parts_dir / f"part_{i:02d}.stl"
        part.mesh.export(out)
        print(f"  part_{i:02d}.stl  ← {part.name} ({len(part.mesh.faces)} faces)")

    meta_path = project / "meta.yaml"
    meta = {
        "domain": domain,
        "printer": "bambu_p1s",
        "success": True,
        "source_of_cut": "bambu_studio",
        "source_file": src.name,
        "object_names": {f"part_{i:02d}": p.name for i, p in enumerate(result.parts, start=1)},
        "cut_info": result.cut_info,
        "notes": "",
    }
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"Wrote {meta_path.relative_to(root)}")

    # Chain into build-plate validation
    args.project_dir = str(project)
    cmd_ingest_validate(args)


def _organic_cut_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        max_splits=args.max_splits,
        force_split=args.force_split,
        repair_mode=args.repair_mode,
        voxel_resolution=args.voxel_resolution,
        pitch_mm=args.pitch_mm,
        with_pins=args.with_pins,
    )


def _apply_organic_defaults(args: argparse.Namespace) -> None:
    from .generate_trellis import load_defaults

    org = load_defaults().get("organic_cut") or {}
    if getattr(args, "max_splits", None) is None:
        args.max_splits = int(org.get("max_splits", 3))
    if getattr(args, "repair_mode", None) is None:
        args.repair_mode = str(org.get("repair_mode", "auto"))
    if getattr(args, "voxel_resolution", None) is None:
        args.voxel_resolution = int(org.get("voxel_resolution", 64))
    if not hasattr(args, "with_pins") or args.with_pins is False:
        # only auto-enable from config if user didn't pass the flag; flag is store_true
        if not args.with_pins:
            args.with_pins = bool(org.get("with_pins", False))


def cmd_cut_organic(args: argparse.Namespace) -> None:
    """Repair + aesthetic mid-plane splits for an organic project folder."""
    from .organic_cut import cut_organic_project

    result = cut_organic_project(args.project_dir, **_organic_cut_kwargs(args))
    if not result.all_fit:
        raise SystemExit(1)


def cmd_process_organic(args: argparse.Namespace) -> None:
    """Alias for cut-organic — apply repair/split/pins nodes to an existing slug."""
    cmd_cut_organic(args)


def cmd_process_organic_batch(args: argparse.Namespace) -> None:
    """Apply repair/split/pins to every processable slug under an organic root."""
    from .organic_cut import process_organic_batch

    results = process_organic_batch(
        args.root_dir,
        fail_fast=args.fail_fast,
        **_organic_cut_kwargs(args),
    )
    failed = [r for r in results if not r.all_fit]
    print(f"\nBatch done: {len(results)} processed, {len(failed)} with fit failures")
    if failed:
        raise SystemExit(1)


def cmd_generate_from_image(args: argparse.Namespace) -> None:
    """Shell out to trellis-mac and stage an organic project under data/raw/organic/."""
    from .generate_trellis import generate_from_image

    generate_from_image(
        image=Path(args.image),
        slug=args.slug,
        trellis_root=args.trellis_root,
        pipeline_type=args.pipeline_type,
        seed=args.seed,
        texture_size=args.texture_size,
        target_longest_mm=args.target_mm,
        skip_generate=args.skip_generate,
        existing_glb=Path(args.glb) if args.glb else None,
    )


def cmd_ingest_validate(args: argparse.Namespace) -> None:
    """Validate part STLs in a mechanical project folder against the build plate."""
    project = Path(args.project_dir).expanduser().resolve()
    parts_dir = project / "parts"
    meta_path = project / "meta.yaml"

    if not parts_dir.is_dir():
        raise SystemExit(f"Missing parts/ directory: {parts_dir}")

    stls = sorted(parts_dir.glob("*.stl")) + sorted(parts_dir.glob("*.STL"))
    if not stls:
        raise SystemExit(f"No STL files in {parts_dir}")

    profile = load_printer_profile(default_profile_path())
    print(f"Project: {project.name}")
    print(f"Printer: {profile.display_name}")

    part_entries = []
    all_fit = True
    for path in stls:
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            mesh = trimesh.util.concatenate(geoms)
        ext = aabb_extents(np.asarray(mesh.vertices))
        ok, msg = part_fits_build_plate(ext, profile, allow_rotation=True)
        all_fit = all_fit and ok
        print(f"  {path.name}: {msg}")
        part_entries.append(
            {
                "file": path.name,
                "extents_mm": [float(x) for x in ext.tolist()],
                "fits_build_plate": ok,
            }
        )

    if meta_path.exists():
        meta = yaml.safe_load(meta_path.read_text()) or {}
    else:
        meta = {"domain": "mechanical", "printer": profile.name, "success": True}

    meta["part_count"] = len(part_entries)
    meta["parts"] = part_entries
    meta["all_parts_fit_build_plate"] = all_fit
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"Updated {meta_path}")
    print("All fit:", all_fit)
    if not all_fit:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="segmentation_ai")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("demo-basic-shapes", help="Cut a box and enforce P1S limits")
    p1.set_defaults(func=cmd_demo_basic_shapes)

    p2 = sub.add_parser("check-oversized", help="Show build-plate rejection")
    p2.set_defaults(func=cmd_check_oversized)

    p3 = sub.add_parser(
        "ingest-validate",
        help="Validate part STLs in data/raw/mechanical/<project>/ against P1S",
    )
    p3.add_argument(
        "project_dir",
        help="Path to project folder containing parts/ and meta.yaml",
    )
    p3.set_defaults(func=cmd_ingest_validate)

    p4 = sub.add_parser(
        "ingest-3mf",
        help="Ingest a Bambu Studio 3MF: export objects as part STLs + validate",
    )
    p4.add_argument("file", help="Path to the .3mf file")
    p4.add_argument("--slug", default=None, help="Project folder name (default: from filename)")
    p4.add_argument(
        "--domain",
        default="mechanical",
        choices=["mechanical", "organic"],
        help="Dataset track (default: mechanical)",
    )
    p4.set_defaults(func=cmd_ingest_3mf)

    p5 = sub.add_parser(
        "generate-from-image",
        help=(
            "Run sibling trellis-mac (subprocess) and stage data/raw/organic/<slug>/ "
            "with scaled print STL + meta.yaml"
        ),
    )
    p5.add_argument("image", help="Path to input image (PNG/JPG)")
    p5.add_argument("--slug", default=None, help="Project folder name (default: from image name)")
    p5.add_argument(
        "--trellis-root",
        default=None,
        help="Path to trellis-mac checkout (default: config/defaults.yaml or ../trellis-mac)",
    )
    p5.add_argument("--pipeline-type", default=None, choices=["512", "1024", "1024_cascade"])
    p5.add_argument("--seed", type=int, default=None)
    p5.add_argument("--texture-size", type=int, default=None, choices=[512, 1024, 2048])
    p5.add_argument(
        "--target-mm",
        type=float,
        default=None,
        help="Scale longest axis to this many mm (default: 120)",
    )
    p5.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip trellis-mac; stage/scale an existing GLB instead",
    )
    p5.add_argument(
        "--glb",
        default=None,
        help="With --skip-generate, path to an existing .glb to stage",
    )
    p5.set_defaults(func=cmd_generate_from_image)

    p6 = sub.add_parser(
        "cut-organic",
        help=(
            "Phase D: repair organic print STL, mid-plane seam splits, "
            "write parts/ + update meta.yaml"
        ),
    )
    p6.add_argument(
        "project_dir",
        help="Path to data/raw/organic/<slug>/ (needs print STL from generate-from-image)",
    )
    p6.add_argument(
        "--max-splits",
        type=int,
        default=None,
        help="Max recursive mid-plane splits for oversized parts (default: config)",
    )
    p6.add_argument(
        "--force-split",
        action="store_true",
        help="Split once along longest axis even if the whole mesh already fits P1S",
    )
    p6.add_argument(
        "--repair-mode",
        default=None,
        choices=["basic", "voxel", "auto"],
        help="Mesh repair: basic | voxel (watertight remesh) | auto (default)",
    )
    p6.add_argument(
        "--voxel-resolution",
        type=int,
        default=None,
        help="Voxel grid resolution along longest axis when remeshing (default: 64)",
    )
    p6.add_argument(
        "--pitch-mm",
        type=float,
        default=None,
        help="Optional explicit voxel pitch in mm (overrides --voxel-resolution)",
    )
    p6.add_argument(
        "--with-pins",
        action="store_true",
        help="Add mating pin/socket on the cut face (2-part splits only)",
    )
    p6.set_defaults(func=cmd_cut_organic)

    def _add_organic_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-splits", type=int, default=None)
        p.add_argument("--force-split", action="store_true")
        p.add_argument(
            "--repair-mode", default=None, choices=["basic", "voxel", "auto"]
        )
        p.add_argument("--voxel-resolution", type=int, default=None)
        p.add_argument("--pitch-mm", type=float, default=None)
        p.add_argument("--with-pins", action="store_true")

    p7 = sub.add_parser(
        "process-organic",
        help="Apply repair/split/pins pipeline nodes to an existing organic project",
    )
    p7.add_argument("project_dir", help="Path to data/raw/organic/<slug>/")
    _add_organic_flags(p7)
    p7.set_defaults(func=cmd_process_organic)

    p8 = sub.add_parser(
        "process-organic-batch",
        help="Apply repair/split/pins to all organic slugs under a directory",
    )
    p8.add_argument(
        "root_dir",
        nargs="?",
        default=None,
        help="Root folder (default: data/raw/organic)",
    )
    p8.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first project error",
    )
    _add_organic_flags(p8)
    p8.set_defaults(func=cmd_process_organic_batch)

    args = parser.parse_args()

    if getattr(args, "func", None) in {
        cmd_cut_organic,
        cmd_process_organic,
        cmd_process_organic_batch,
    }:
        _apply_organic_defaults(args)
        if getattr(args, "root_dir", None) is None and args.func is cmd_process_organic_batch:
            args.root_dir = str(
                Path(__file__).resolve().parents[2] / "data" / "raw" / "organic"
            )

    args.func(args)


if __name__ == "__main__":
    main()
