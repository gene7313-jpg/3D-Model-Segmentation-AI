"""Shell out to sibling trellis-mac; stage organic assets for this repo.

Does not import PyTorch / TRELLIS into this package — generation runs in
trellis-mac's own virtualenv via subprocess.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml

from .printer import (
    aabb_extents,
    default_profile_path,
    load_printer_profile,
    part_fits_build_plate,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_defaults() -> dict:
    path = project_root() / "config" / "defaults.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def slugify(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[-\s]+", "-", s)
    return s.strip("-") or "organic-asset"


@dataclass
class GenerateResult:
    project_dir: Path
    glb_path: Path
    stl_path: Path
    print_stl_path: Path
    meta_path: Path
    extents_mm: list[float]
    fits_p1s: bool
    watertight: bool
    scale_factor: float


def resolve_trellis_root(explicit: str | None) -> Path:
    defaults = load_defaults()
    trellis_cfg = defaults.get("trellis") or {}
    raw = explicit or trellis_cfg.get("root") or "../trellis-mac"
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (project_root() / root).resolve()
    else:
        root = root.resolve()
    return root


def _trellis_python(trellis_root: Path) -> Path:
    py = trellis_root / ".venv" / "bin" / "python"
    if not py.is_file():
        raise FileNotFoundError(
            f"trellis-mac venv not found at {py}. "
            "Run bash setup.sh inside the trellis-mac checkout first."
        )
    return py


def run_trellis_generate(
    *,
    image: Path,
    trellis_root: Path,
    output_stem: Path,
    pipeline_type: str,
    seed: int,
    texture_size: int,
) -> Path:
    """Run trellis-mac generate.py; return path to produced .glb."""
    generate_py = trellis_root / "generate.py"
    if not generate_py.is_file():
        raise FileNotFoundError(f"Missing {generate_py}")

    py = _trellis_python(trellis_root)
    cmd = [
        str(py),
        str(generate_py),
        str(image),
        "--output",
        str(output_stem),
        "--pipeline-type",
        pipeline_type,
        "--seed",
        str(seed),
        "--texture-size",
        str(texture_size),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(trellis_root), check=True)

    glb = output_stem.with_suffix(".glb")
    if not glb.is_file():
        # Some runs may write relative to cwd
        alt = trellis_root / f"{output_stem.name}.glb"
        if alt.is_file():
            return alt
        raise FileNotFoundError(f"Expected GLB not found at {glb}")
    return glb


def glb_to_stl(glb_path: Path, stl_path: Path) -> tuple[trimesh.Trimesh, bool]:
    loaded = trimesh.load(glb_path, force="scene")
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception as exc:  # networkx optional at runtime; continue without fill
        print(f"Warning: fill_holes skipped ({exc})")
    trimesh.repair.fix_normals(mesh)
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(stl_path)
    return mesh, bool(mesh.is_watertight)


def scale_to_longest_axis(
    mesh: trimesh.Trimesh, target_mm: float
) -> tuple[trimesh.Trimesh, float, np.ndarray]:
    ext = aabb_extents(np.asarray(mesh.vertices))
    longest = float(np.max(ext))
    if longest <= 0:
        raise ValueError("Mesh has zero extent; cannot scale")
    scale = target_mm / longest
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    return mesh, scale, aabb_extents(np.asarray(mesh.vertices))


def generate_from_image(
    *,
    image: Path,
    slug: str | None = None,
    trellis_root: str | None = None,
    pipeline_type: str | None = None,
    seed: int | None = None,
    texture_size: int | None = None,
    target_longest_mm: float | None = None,
    skip_generate: bool = False,
    existing_glb: Path | None = None,
) -> GenerateResult:
    defaults = load_defaults()
    trellis_cfg = defaults.get("trellis") or {}

    image = image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")

    slug = slugify(slug or image.stem)
    root = project_root()
    project = root / "data" / "raw" / "organic" / slug
    project.mkdir(parents=True, exist_ok=True)

    pipeline_type = str(pipeline_type or trellis_cfg.get("pipeline_type") or "512")
    seed = int(seed if seed is not None else trellis_cfg.get("seed", 42))
    texture_size = int(
        texture_size if texture_size is not None else trellis_cfg.get("texture_size", 1024)
    )
    target_longest_mm = float(
        target_longest_mm
        if target_longest_mm is not None
        else trellis_cfg.get("target_longest_mm", 120.0)
    )

    staged_image = project / "source_image.png"
    if image.suffix.lower() == ".png":
        shutil.copy2(image, staged_image)
    else:
        # Keep original extension if not PNG; still record as source_image.*
        staged_image = project / f"source_image{image.suffix.lower()}"
        shutil.copy2(image, staged_image)

    glb_path = project / "source.glb"
    if skip_generate:
        src_glb = existing_glb.expanduser().resolve() if existing_glb else glb_path
        if not src_glb.is_file():
            raise FileNotFoundError(
                f"--skip-generate requires an existing GLB at {src_glb}"
            )
        if src_glb != glb_path:
            shutil.copy2(src_glb, glb_path)
    else:
        t_root = resolve_trellis_root(trellis_root)
        # Write GLB into a temp stem under the project, then ensure source.glb
        out_stem = project / "_trellis_out"
        produced = run_trellis_generate(
            image=image,
            trellis_root=t_root,
            output_stem=out_stem,
            pipeline_type=pipeline_type,
            seed=seed,
            texture_size=texture_size,
        )
        if produced.resolve() != glb_path.resolve():
            shutil.copy2(produced, glb_path)
            # Remove temp output next to the project if we copied away from it
            if produced.name.startswith("_trellis_out"):
                try:
                    produced.unlink()
                except OSError:
                    pass
        for p in project.glob("_trellis_out*"):
            if p.resolve() != glb_path.resolve():
                try:
                    p.unlink()
                except OSError:
                    pass

    stl_path = project / "source.stl"
    mesh, watertight = glb_to_stl(glb_path, stl_path)

    scaled, scale_factor, extents = scale_to_longest_axis(mesh, target_longest_mm)
    print_stl = project / f"source_{int(target_longest_mm)}mm.stl"
    scaled.export(print_stl)

    profile = load_printer_profile(default_profile_path())
    ok, fit_msg = part_fits_build_plate(extents, profile, allow_rotation=True)
    print(fit_msg)

    meta = {
        "domain": "organic",
        "source": "trellis-mac",
        "generator": "microsoft/TRELLIS.2-4B",
        "pipeline_type": pipeline_type,
        "texture_size": texture_size,
        "seed": seed,
        "printer": profile.name,
        "source_image": staged_image.name,
        "source_glb": glb_path.name,
        "source_stl": stl_path.name,
        "print_stl": print_stl.name,
        "scale_to_mm": {
            "target_longest_axis_mm": target_longest_mm,
            "scale_factor": float(scale_factor),
            "extents_mm": [float(x) for x in extents.tolist()],
            "fits_p1s": bool(ok),
        },
        "watertight": bool(watertight),
        "notes": (
            "Generated via segmentation-ai generate-from-image. "
            "Not cut yet (Phase 4). Repair before boolean cuts if needed."
        ),
    }
    meta_path = project / "meta.yaml"
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    print(f"Staged organic project: {project}")
    print(f"Print STL: {print_stl}  extents_mm={extents.round(2)}  Fits P1S: {ok}")
    print(f"Watertight: {watertight}")

    return GenerateResult(
        project_dir=project,
        glb_path=glb_path,
        stl_path=stl_path,
        print_stl_path=print_stl,
        meta_path=meta_path,
        extents_mm=[float(x) for x in extents.tolist()],
        fits_p1s=bool(ok),
        watertight=bool(watertight),
        scale_factor=float(scale_factor),
    )
