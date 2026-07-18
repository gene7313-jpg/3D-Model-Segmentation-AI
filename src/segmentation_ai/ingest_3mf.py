"""Ingest Bambu Studio / generic 3MF project files.

Handles both plain 3MF (mesh inside 3D/3dmodel.model) and the Production
extension used by Bambu Studio (objects referenced via p:path into
3D/Objects/*.model). Also reads Bambu's Metadata/model_settings.config for
object names and Metadata/cut_information.xml for cut/connector history.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_PROD_NS = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _transform_to_matrix(text: str | None) -> np.ndarray:
    """3MF transform: 12 floats (row-vector convention) -> 4x4 matrix."""
    if not text:
        return np.eye(4)
    vals = np.array([float(v) for v in text.split()], dtype=float)
    if vals.size != 12:
        return np.eye(4)
    rows = vals.reshape(4, 3)
    matrix = np.eye(4)
    matrix[:3, :3] = rows[:3, :].T
    matrix[:3, 3] = rows[3, :]
    return matrix


@dataclass
class _ObjectDef:
    """Raw object definition from a .model file."""

    mesh: trimesh.Trimesh | None = None
    # (path_of_referenced_model, objectid, transform)
    components: list[tuple[str | None, str, np.ndarray]] = field(default_factory=list)


@dataclass
class IngestedPart:
    name: str
    mesh: trimesh.Trimesh


@dataclass
class Ingested3MF:
    parts: list[IngestedPart]
    object_names: dict[str, str]
    cut_info: list[dict]
    unit: str


def _parse_model_xml(data: bytes) -> tuple[dict[str, _ObjectDef], list[tuple[str, np.ndarray]], str]:
    """Parse one .model XML. Returns (objects_by_id, build_items, unit)."""
    root = ET.fromstring(data)
    unit = root.get("unit", "millimeter")
    objects: dict[str, _ObjectDef] = {}
    build_items: list[tuple[str, np.ndarray]] = []

    for obj in root.iter(f"{{{_CORE_NS}}}object"):
        obj_id = obj.get("id")
        if obj_id is None:
            continue
        odef = _ObjectDef()
        for child in obj:
            tag = _local(child.tag)
            if tag == "mesh":
                verts = []
                faces = []
                for sub in child:
                    sub_tag = _local(sub.tag)
                    if sub_tag == "vertices":
                        for v in sub:
                            verts.append(
                                (float(v.get("x")), float(v.get("y")), float(v.get("z")))
                            )
                    elif sub_tag == "triangles":
                        for t in sub:
                            faces.append(
                                (int(t.get("v1")), int(t.get("v2")), int(t.get("v3")))
                            )
                if verts and faces:
                    odef.mesh = trimesh.Trimesh(
                        vertices=np.array(verts), faces=np.array(faces), process=False
                    )
            elif tag == "components":
                for comp in child:
                    ref_path = comp.get(f"{{{_PROD_NS}}}path")
                    ref_id = comp.get("objectid")
                    matrix = _transform_to_matrix(comp.get("transform"))
                    if ref_id is not None:
                        odef.components.append((ref_path, ref_id, matrix))
        objects[obj_id] = odef

    build = root.find(f"{{{_CORE_NS}}}build")
    if build is not None:
        for item in build:
            obj_id = item.get("objectid")
            if obj_id is None:
                continue
            build_items.append((obj_id, _transform_to_matrix(item.get("transform"))))

    return objects, build_items, unit


def _read_object_names(zf: zipfile.ZipFile) -> dict[str, str]:
    """Bambu Metadata/model_settings.config: object id -> display name."""
    names: dict[str, str] = {}
    try:
        data = zf.read("Metadata/model_settings.config")
    except KeyError:
        return names
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return names
    for obj in root.iter("object"):
        obj_id = obj.get("id")
        if obj_id is None:
            continue
        for meta in obj.iter("metadata"):
            if meta.get("key") == "name" and meta.get("value"):
                names[obj_id] = meta.get("value")
                break
    return names


def _read_cut_info(zf: zipfile.ZipFile) -> list[dict]:
    """Bambu Metadata/cut_information.xml -> list of cut records."""
    records: list[dict] = []
    try:
        data = zf.read("Metadata/cut_information.xml")
    except KeyError:
        return records
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return records
    for obj in root.iter("object"):
        entry: dict = {"object_id": obj.get("id")}
        for cut in obj.iter("cut_id"):
            entry["cut_id"] = cut.get("id")
            entry["connectors_cnt"] = cut.get("connectors_cnt")
        records.append(entry)
    return records


def load_3mf(path: Path | str) -> Ingested3MF:
    """Load every build item of a 3MF as a separate part mesh."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        model_entries = [n for n in zf.namelist() if n.lower().endswith(".model")]
        if "3D/3dmodel.model" not in model_entries:
            raise ValueError(f"{path.name}: missing 3D/3dmodel.model")

        parsed: dict[str, tuple[dict[str, _ObjectDef], list[tuple[str, np.ndarray]], str]] = {}
        for entry in model_entries:
            parsed[entry] = _parse_model_xml(zf.read(entry))

        object_names = _read_object_names(zf)
        cut_info = _read_cut_info(zf)

    root_objects, build_items, unit = parsed["3D/3dmodel.model"]

    def resolve(entry: str, obj_id: str, matrix: np.ndarray) -> list[trimesh.Trimesh]:
        objects = parsed.get(entry, (None,))[0]
        if not objects or obj_id not in objects:
            return []
        odef = objects[obj_id]
        meshes: list[trimesh.Trimesh] = []
        if odef.mesh is not None:
            mesh = odef.mesh.copy()
            mesh.apply_transform(matrix)
            meshes.append(mesh)
        for ref_path, ref_id, comp_matrix in odef.components:
            # p:path is absolute inside the archive ("/3D/Objects/x.model")
            target_entry = ref_path.lstrip("/") if ref_path else entry
            meshes.extend(resolve(target_entry, ref_id, matrix @ comp_matrix))
        return meshes

    parts: list[IngestedPart] = []
    for i, (obj_id, matrix) in enumerate(build_items, start=1):
        meshes = resolve("3D/3dmodel.model", obj_id, matrix)
        if not meshes:
            continue
        combined = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        name = object_names.get(obj_id, f"object_{obj_id}")
        parts.append(IngestedPart(name=name, mesh=combined))

    return Ingested3MF(parts=parts, object_names=object_names, cut_info=cut_info, unit=unit)
