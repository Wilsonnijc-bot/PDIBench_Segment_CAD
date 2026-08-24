"""Strict COLLADA loading for the official Franka CAD meshes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ElementTree

import numpy as np


FRANKA_EXPECTED_GEOMETRY = {
    "link2": (1, np.array([0.110033, 0.249024, 0.184393])),
    "link3": (4, np.array([0.192511, 0.166063, 0.176002])),
    "link4": (4, np.array([0.192507, 0.179000, 0.166053])),
    "link5": (3, np.array([0.109996, 0.184930, 0.311199])),
    "link6": (17, np.array([0.179926, 0.132863, 0.100243])),
    "link7": (8, np.array([0.125333, 0.125297, 0.054800])),
}


@dataclass(frozen=True)
class ColladaAssetMetadata:
    unit_name: str
    unit_meter: float
    up_axis: str


@dataclass(frozen=True)
class CadMesh:
    name: str
    path: Path
    sha256: str
    asset: ColladaAssetMetadata
    instance_count: int
    mesh: Any
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray
    extents: np.ndarray
    diameter: float


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_collada_asset_metadata(path: str | Path) -> ColladaAssetMetadata:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"CAD mesh is missing: {source}")
    try:
        root = ElementTree.parse(source).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"CAD mesh is not valid XML: {source}") from exc
    unit = root.find("./{*}asset/{*}unit")
    up_axis = root.find("./{*}asset/{*}up_axis")
    if unit is None or up_axis is None or up_axis.text is None:
        raise ValueError(f"CAD mesh lacks COLLADA unit/up_axis metadata: {source}")
    try:
        unit_meter = float(unit.attrib["meter"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"CAD mesh has invalid COLLADA meter metadata: {source}") from exc
    return ColladaAssetMetadata(
        unit_name=str(unit.attrib.get("name", "")),
        unit_meter=unit_meter,
        up_axis=up_axis.text.strip(),
    )


def load_cad_mesh(
    path: str | Path,
    *,
    name: str | None = None,
    expected_sha256: str | None = None,
    expected_instance_count: int | None = None,
    expected_extents: np.ndarray | None = None,
    extent_tolerance: float = 1e-5,
) -> CadMesh:
    """Load a DAE scene and bake every scene-graph instance transform."""
    source = Path(path).resolve()
    mesh_name = name or source.stem
    asset = read_collada_asset_metadata(source)
    if asset.unit_name != "meter" or not np.isclose(asset.unit_meter, 1.0):
        raise ValueError(f"{source.name} must declare meter=1, got {asset}")
    if asset.up_axis != "Z_UP":
        raise ValueError(f"{source.name} must declare Z_UP, got {asset.up_axis!r}")
    actual_sha256 = sha256_file(source)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"CAD checksum mismatch for {source.name}: {actual_sha256}"
        )
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("CAD loading requires trimesh and pycollada") from exc

    loaded = trimesh.load(source, force="scene", process=False, maintain_order=True)
    instances = []
    if isinstance(loaded, trimesh.Trimesh):
        instances.append(loaded.copy())
    else:
        for node_name in loaded.graph.nodes_geometry:
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry[geometry_name].copy()
            geometry.apply_transform(np.asarray(transform, dtype=np.float64))
            instances.append(geometry)
    if not instances:
        raise ValueError(f"CAD mesh contains no triangle geometry: {source}")
    combined = trimesh.util.concatenate(instances)
    vertices = np.asarray(combined.vertices, dtype=np.float64)
    faces = np.asarray(combined.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError(f"CAD mesh has invalid vertices: {source}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
        raise ValueError(f"CAD mesh is not triangulated: {source}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"CAD mesh has non-finite vertices: {source}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"CAD mesh has out-of-range triangle indices: {source}")
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    extents = bounds[1] - bounds[0]
    diameter = float(np.linalg.norm(extents))
    if not np.isfinite(diameter) or diameter <= 0.0:
        raise ValueError(f"CAD mesh has zero or invalid extent: {source}")

    frank_expectation = FRANKA_EXPECTED_GEOMETRY.get(mesh_name)
    if frank_expectation is not None:
        expected_instance_count = (
            frank_expectation[0]
            if expected_instance_count is None
            else expected_instance_count
        )
        expected_extents = (
            frank_expectation[1] if expected_extents is None else expected_extents
        )
    if expected_instance_count is not None and len(instances) != expected_instance_count:
        raise ValueError(
            f"{mesh_name} has {len(instances)} geometry instances; "
            f"expected {expected_instance_count}"
        )
    if expected_extents is not None and not np.allclose(
        extents,
        np.asarray(expected_extents, dtype=np.float64),
        atol=extent_tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            f"{mesh_name} baked extents {extents.tolist()} do not match "
            f"{np.asarray(expected_extents).tolist()}"
        )

    _ = combined.vertex_normals
    return CadMesh(
        name=mesh_name,
        path=source,
        sha256=actual_sha256,
        asset=asset,
        instance_count=len(instances),
        mesh=combined,
        vertices=vertices,
        faces=faces,
        bounds=bounds,
        extents=extents,
        diameter=diameter,
    )


def load_cad_manifest(
    manifest_path: str | Path,
    *,
    link_names: Iterable[str],
) -> dict[str, CadMesh]:
    import yaml

    source = Path(manifest_path).resolve()
    manifest = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = source.parent.parent
    cad = manifest["cad"]
    specifications = {item["name"]: item for item in cad["meshes"]}
    requested = tuple(link_names)
    missing = sorted(set(requested).difference(specifications))
    if missing:
        raise ValueError(f"CAD manifest is missing links: {missing}")
    mesh_root = root / cad["mesh_root"]
    return {
        name: load_cad_mesh(
            mesh_root / specifications[name]["file"],
            name=name,
            expected_sha256=specifications[name]["sha256"],
        )
        for name in requested
    }
