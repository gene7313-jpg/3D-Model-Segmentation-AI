"""Printer build-volume profiles and part size enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class BuildVolume:
    x: float
    y: float
    z: float

    def usable(self, margin_mm: float) -> "BuildVolume":
        return BuildVolume(
            x=self.x - 2.0 * margin_mm,
            y=self.y - 2.0 * margin_mm,
            z=self.z - 2.0 * margin_mm,
        )


@dataclass(frozen=True)
class PrinterProfile:
    name: str
    display_name: str
    build_volume: BuildVolume
    margin_mm: float = 2.0

    @property
    def max_part(self) -> BuildVolume:
        return self.build_volume.usable(self.margin_mm)


def load_printer_profile(path: Path | str) -> PrinterProfile:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    bv = data["build_volume_mm"]
    return PrinterProfile(
        name=data["name"],
        display_name=data.get("display_name", data["name"]),
        build_volume=BuildVolume(float(bv["x"]), float(bv["y"]), float(bv["z"])),
        margin_mm=float(data.get("margin_mm", 2.0)),
    )


def default_profile_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "printers" / "bambu_p1s.yaml"


def aabb_extents(vertices: np.ndarray) -> np.ndarray:
    """Return XYZ extents of an axis-aligned bounding box."""
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    return maxs - mins


def part_fits_build_plate(
    extents_mm: Iterable[float],
    profile: PrinterProfile,
    *,
    allow_rotation: bool = True,
) -> tuple[bool, str]:
    """
    Check whether a part AABB fits the usable build volume.

    If allow_rotation is True, any axis permutation of the AABB may be used
    (common when re-orienting parts on the bed).
    """
    ext = np.asarray(list(extents_mm), dtype=float)
    if ext.shape != (3,):
        raise ValueError("extents_mm must be length 3 (X, Y, Z)")

    limit = np.array(
        [profile.max_part.x, profile.max_part.y, profile.max_part.z], dtype=float
    )

    candidates = [ext]
    if allow_rotation:
        # Unique permutations of XYZ
        from itertools import permutations

        candidates = [np.array(p, dtype=float) for p in set(permutations(ext.tolist()))]

    for cand in candidates:
        if np.all(cand <= limit + 1e-6):
            return True, f"fits as {cand[0]:.1f}×{cand[1]:.1f}×{cand[2]:.1f} mm"

    return (
        False,
        f"part {ext[0]:.1f}×{ext[1]:.1f}×{ext[2]:.1f} mm exceeds usable "
        f"{limit[0]:.1f}×{limit[1]:.1f}×{limit[2]:.1f} mm "
        f"({profile.display_name}, margin {profile.margin_mm} mm)",
    )
