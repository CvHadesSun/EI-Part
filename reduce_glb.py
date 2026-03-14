#!/usr/bin/env python3
"""
Recursively simplify GLB files with pymeshlab while:
1. Applying node transforms from the source scene.
2. Preserving separate meshes instead of concatenating them.
3. Treating --target-faces as a per-GLB total face budget.
4. Skipping simplification for meshes already below their allocated budget.
5. Normalizing the full output scene into [-1, 1].
6. Optionally dropping textures/materials to reduce file size.
7. Skipping simplification when the original GLB is already under a size threshold.

Examples:
    python reduce_glb.py --input /path/to/file.glb --output /path/to/out --target-faces 5000
    python reduce_glb.py --input /path/to/glb_dir --output /path/to/out_dir --target-faces 10000
    python reduce_glb.py --input /path/to/glb_dir --target-faces 10000
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplify GLB files recursively with pymeshlab.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input GLB file or a directory to scan recursively.",
    )
    parser.add_argument(
        "--output",
        help="Output directory. If omitted, files are overwritten in-place.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        required=True,
        help="Target total face count per GLB after simplification.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--keep-visuals",
        action="store_true",
        help="Keep original visual/material data. By default visuals are stripped to reduce file size.",
    )
    parser.add_argument(
        "--skip-simplify-below-mb",
        type=float,
        default=50.0,
        help="If the original GLB size is at or below this threshold, only normalize it without simplification.",
    )
    parser.add_argument(
        "--backend",
        default="pymeshlab",
        choices=["pymeshlab"],
        help="Mesh simplification backend.",
    )
    return parser.parse_args()


def collect_glb_files(input_path: Path) -> Tuple[Path, List[Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".glb":
            raise ValueError(f"Input file must be a .glb: {input_path}")
        return input_path.parent, [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    files = sorted(p for p in input_path.rglob("*.glb") if p.is_file())
    return input_path, files


def as_scene(obj: trimesh.Scene | trimesh.Trimesh) -> trimesh.Scene:
    if isinstance(obj, trimesh.Scene):
        return obj
    scene = trimesh.Scene()
    scene.add_geometry(obj)
    return scene


def read_glb_json(input_file: Path) -> dict:
    payload = input_file.read_bytes()
    if payload[:4] != b"glTF":
        raise ValueError(f"Not a valid GLB file: {input_file}")

    offset = 12
    chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
    if chunk_type != 0x4E4F534A:
        raise ValueError(f"GLB JSON chunk missing: {input_file}")
    offset += 8
    return json.loads(payload[offset : offset + chunk_length].decode("utf-8"))


def read_glb_binary_chunk(input_file: Path) -> memoryview:
    payload = input_file.read_bytes()
    if payload[:4] != b"glTF":
        raise ValueError(f"Not a valid GLB file: {input_file}")

    json_length, _json_type = struct.unpack_from("<II", payload, 12)
    offset = 20 + json_length
    if offset % 4:
        offset += 4 - (offset % 4)
    bin_length, bin_type = struct.unpack_from("<II", payload, offset)
    if bin_type != 0x004E4942:
        raise ValueError(f"GLB BIN chunk missing: {input_file}")
    offset += 8
    return memoryview(payload)[offset : offset + bin_length]


def quaternion_to_matrix(quat: Sequence[float]) -> np.ndarray:
    x, y, z, w = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4)

    matrix = np.eye(4, dtype=np.float64)
    if "translation" in node:
        matrix[:3, 3] = np.asarray(node["translation"], dtype=np.float64)
    if "rotation" in node:
        matrix = matrix @ quaternion_to_matrix(node["rotation"])
    if "scale" in node:
        scale = np.asarray(node["scale"], dtype=np.float64)
        scale_matrix = np.eye(4, dtype=np.float64)
        scale_matrix[0, 0] = scale[0]
        scale_matrix[1, 1] = scale[1]
        scale_matrix[2, 2] = scale[2]
        matrix = matrix @ scale_matrix
    return matrix


def compute_gltf_world_transforms(gltf_json: dict) -> dict[str, np.ndarray]:
    nodes = gltf_json.get("nodes", [])
    scenes = gltf_json.get("scenes", [])
    scene_index = gltf_json.get("scene", 0)
    if scene_index >= len(scenes):
        return {}

    transform_map: dict[str, np.ndarray] = {}

    def walk(node_index: int, parent_matrix: np.ndarray) -> None:
        node = nodes[node_index]
        local = node_local_matrix(node)
        world = parent_matrix @ local
        node_name = node.get("name", f"node_{node_index}")
        transform_map[node_name] = world
        for child_index in node.get("children", []):
            walk(child_index, world)

    for root_index in scenes[scene_index].get("nodes", []):
        walk(root_index, np.eye(4, dtype=np.float64))

    return transform_map


def read_gltf_accessor(gltf_json: dict, binary_chunk: memoryview, accessor_index: int) -> np.ndarray:
    accessor = gltf_json["accessors"][accessor_index]
    buffer_view = gltf_json["bufferViews"][accessor["bufferView"]]
    component_map = {
        5120: np.int8,
        5121: np.uint8,
        5122: np.int16,
        5123: np.uint16,
        5125: np.uint32,
        5126: np.float32,
    }
    type_size = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }
    dtype = component_map[accessor["componentType"]]
    width = type_size[accessor["type"]]
    offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    itemsize = np.dtype(dtype).itemsize
    raw = binary_chunk[offset : offset + count * width * itemsize]
    return np.frombuffer(raw, dtype=dtype).reshape(count, width)


def to_rgba_uint8(colors: np.ndarray) -> np.ndarray:
    array = np.asarray(colors)
    if array.size == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    source_max = float(np.max(array))
    if array.shape[1] == 3:
        alpha_value = 1.0 if source_max <= 1.0 else 255.0
        alpha = np.full((array.shape[0], 1), alpha_value, dtype=np.float64)
        array = np.concatenate([array.astype(np.float64), alpha], axis=1)
    else:
        array = array[:, :4].astype(np.float64)
    if source_max <= 1.0:
        array[:, :3] *= 255.0
        if np.max(array[:, 3]) <= 1.0:
            array[:, 3] *= 255.0
    return np.asarray(np.round(array), dtype=np.uint8)


def compute_node_color_map(gltf_json: dict, binary_chunk: memoryview) -> dict[str, np.ndarray]:
    node_color_map: dict[str, np.ndarray] = {}
    meshes = gltf_json.get("meshes", [])
    for node_index, node in enumerate(gltf_json.get("nodes", [])):
        mesh_index = node.get("mesh")
        if mesh_index is None or mesh_index >= len(meshes):
            continue
        primitives = meshes[mesh_index].get("primitives", [])
        if not primitives:
            continue
        attributes = primitives[0].get("attributes", {})
        color_accessor = attributes.get("COLOR_0")
        if color_accessor is None:
            continue
        colors = to_rgba_uint8(read_gltf_accessor(gltf_json, binary_chunk, color_accessor))
        if len(colors) == 0:
            continue
        unique, counts = np.unique(colors.reshape(-1, 4), axis=0, return_counts=True)
        rgba = unique[int(np.argmax(counts))]
        node_name = node.get("name", f"node_{node_index}")
        node_color_map[node_name] = rgba
    return node_color_map


def scene_to_world_meshes(
    scene: trimesh.Scene, transform_map: dict[str, np.ndarray], node_color_map: dict[str, np.ndarray]
) -> List[trimesh.Trimesh]:
    meshes: List[trimesh.Trimesh] = []

    for node_name in scene.graph.nodes_geometry:
        _transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh):
            continue

        mesh = geom.copy()
        if mesh.vertices is None or len(mesh.vertices) == 0:
            continue
        if mesh.faces is None or len(mesh.faces) == 0:
            continue

        transform = transform_map.get(str(node_name))
        if transform is None:
            transform = np.eye(4, dtype=np.float64)
        mesh.apply_transform(np.asarray(transform, dtype=np.float64))

        metadata = dict(mesh.metadata) if mesh.metadata else {}
        metadata["source_node_name"] = str(node_name)
        metadata["source_geom_name"] = str(geom_name)
        if str(node_name) in node_color_map:
            metadata["source_rgba"] = node_color_map[str(node_name)].tolist()
        mesh.metadata = metadata
        meshes.append(mesh)

    return meshes

def representative_rgba(mesh: trimesh.Trimesh) -> np.ndarray:
    default = np.array([200, 200, 200, 255], dtype=np.uint8)
    metadata_color = (mesh.metadata or {}).get("source_rgba")
    if metadata_color is not None:
        return np.asarray(metadata_color, dtype=np.uint8)
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return default

    def dominant_color(colors: np.ndarray) -> np.ndarray:
        array = to_rgba_uint8(colors)
        if len(array) == 0:
            return default
        unique, counts = np.unique(array.reshape(-1, 4), axis=0, return_counts=True)
        return unique[int(np.argmax(counts))]

    face_colors = getattr(visual, "face_colors", None)
    if face_colors is not None and len(face_colors) > 0:
        return dominant_color(face_colors)

    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) > 0:
        return dominant_color(vertex_colors)

    material = getattr(visual, "material", None)
    for attr in ("baseColorFactor", "main_color", "diffuse"):
        value = getattr(material, attr, None)
        if value is None:
            continue
        arr = to_rgba_uint8(np.asarray(value))
        if len(arr) > 0:
            return arr[0]

    return default


def apply_flat_color(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
    face_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.faces), 1))
    vertex_colors = np.tile(np.asarray(rgba, dtype=np.uint8), (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=face_colors,
        vertex_colors=vertex_colors,
    )


def simplify_mesh_with_pymeshlab(
    mesh: trimesh.Trimesh, target_faces: int, keep_visuals: bool
) -> trimesh.Trimesh:
    rgba = representative_rgba(mesh)
    face_count = len(mesh.faces)
    if face_count <= target_faces:
        out = mesh.copy()
        if keep_visuals and mesh.visual is not None:
            try:
                out.visual = mesh.visual.copy()
            except BaseException:
                apply_flat_color(out, rgba)
        else:
            apply_flat_color(out, rgba)
        return out

    try:
        import pymeshlab  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            "pymeshlab is required. Install it with `pip install pymeshlab` before running this script."
        ) from exc

    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ),
        "mesh",
    )
    meshset.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=False,
    )

    reduced = meshset.current_mesh()
    vertices = reduced.vertex_matrix()
    faces = reduced.face_matrix()

    simplified = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )

    # Topology changes make UV/material transfer unreliable. Preserve a stable sub-mesh color instead.
    apply_flat_color(simplified, rgba)

    if mesh.metadata:
        simplified.metadata = dict(mesh.metadata)

    return simplified


def simplify_mesh(
    mesh: trimesh.Trimesh, target_faces: int, keep_visuals: bool, backend: str
) -> trimesh.Trimesh:
    if backend == "pymeshlab":
        return simplify_mesh_with_pymeshlab(mesh, target_faces, keep_visuals)
    raise ValueError(f"Unsupported backend: {backend}")


def minimum_faces_for_mesh(mesh: trimesh.Trimesh) -> int:
    # Keep a small floor so aggressive decimation does not collapse tiny pieces.
    return min(len(mesh.faces), 4)


def clamp_target(value: int, original_faces: int, minimum_faces: int) -> int:
    return max(minimum_faces, min(original_faces, value))


def allocate_face_budget(face_counts: Sequence[int], minimum_faces: Sequence[int], total_budget: int) -> List[int]:
    total_input_faces = sum(face_counts)
    if total_input_faces <= total_budget:
        return list(face_counts)

    min_total = sum(minimum_faces)
    if total_budget <= min_total:
        targets = list(minimum_faces)
        overflow = sum(targets) - total_budget
        for index in sorted(range(len(targets)), key=lambda i: targets[i] - 1, reverse=True):
            if overflow <= 0:
                break
            reducible = max(0, targets[index] - 1)
            step = min(reducible, overflow)
            targets[index] -= step
            overflow -= step
        return targets

    remaining = set(range(len(face_counts)))
    targets = [0] * len(face_counts)
    remaining_budget = total_budget

    while remaining:
        denom = sum(face_counts[i] for i in remaining)
        changed = False

        for index in list(remaining):
            proposed = int(round(face_counts[index] * remaining_budget / denom))
            clamped = clamp_target(proposed, face_counts[index], minimum_faces[index])
            if clamped == face_counts[index] or clamped == minimum_faces[index]:
                targets[index] = clamped
                remaining_budget -= clamped
                remaining.remove(index)
                changed = True

        if not remaining:
            break

        if not changed:
            for index in remaining:
                proposed = int(round(face_counts[index] * remaining_budget / denom))
                targets[index] = clamp_target(proposed, face_counts[index], minimum_faces[index])
            break

    current_total = sum(targets)
    if current_total > total_budget:
        overflow = current_total - total_budget
        for index in sorted(range(len(targets)), key=lambda i: targets[i] - minimum_faces[i], reverse=True):
            if overflow <= 0:
                break
            reducible = max(0, targets[index] - minimum_faces[index])
            step = min(reducible, overflow)
            targets[index] -= step
            overflow -= step
    elif current_total < total_budget:
        deficit = total_budget - current_total
        for index in sorted(range(len(targets)), key=lambda i: face_counts[i] - targets[i], reverse=True):
            if deficit <= 0:
                break
            increasable = max(0, face_counts[index] - targets[index])
            step = min(increasable, deficit)
            targets[index] += step
            deficit -= step

    return targets


def simplify_meshes_to_total_budget(
    meshes: Sequence[trimesh.Trimesh], total_budget: int, keep_visuals: bool, backend: str
) -> List[trimesh.Trimesh]:
    face_counts = [len(mesh.faces) for mesh in meshes]
    minimum_faces = [minimum_faces_for_mesh(mesh) for mesh in meshes]
    targets = allocate_face_budget(face_counts, minimum_faces, total_budget)

    simplified = [simplify_mesh(mesh, target, keep_visuals, backend) for mesh, target in zip(meshes, targets)]

    for _ in range(3):
        actual_counts = [len(mesh.faces) for mesh in simplified]
        actual_total = sum(actual_counts)
        if actual_total <= total_budget:
            break

        next_targets = allocate_face_budget(actual_counts, minimum_faces, total_budget)
        if next_targets == actual_counts:
            break
        simplified = [simplify_mesh(mesh, target, keep_visuals, backend) for mesh, target in zip(meshes, next_targets)]

    return simplified


def compute_normalization(meshes: Sequence[trimesh.Trimesh]) -> Tuple[np.ndarray, float]:
    mins = []
    maxs = []
    for mesh in meshes:
        bounds = mesh.bounds
        mins.append(bounds[0])
        maxs.append(bounds[1])

    global_min = np.min(np.stack(mins, axis=0), axis=0)
    global_max = np.max(np.stack(maxs, axis=0), axis=0)
    center = (global_min + global_max) * 0.5
    extent = global_max - global_min
    max_extent = float(np.max(extent))
    scale = 1.0 if max_extent <= 0.0 else 2.0 / max_extent
    return center, scale


def normalize_meshes(meshes: Sequence[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
    if not meshes:
        return []

    center, scale = compute_normalization(meshes)
    normalized: List[trimesh.Trimesh] = []
    for mesh in meshes:
        new_mesh = mesh.copy()
        new_mesh.vertices = (np.asarray(new_mesh.vertices, dtype=np.float64) - center) * scale
        normalized.append(new_mesh)
    return normalized


def build_output_scene(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for index, mesh in enumerate(meshes):
        scene.add_geometry(mesh, node_name=f"mesh_{index:04d}", geom_name=f"mesh_{index:04d}")
    return scene


def process_glb_file(
    input_file: Path,
    output_file: Path,
    target_faces: int,
    overwrite: bool,
    keep_visuals: bool,
    skip_simplify_below_mb: float,
    backend: str,
) -> None:
    if output_file.exists() and not overwrite:
        print(f"[skip] {output_file} already exists")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    loaded = trimesh.load(input_file, force="scene", process=False)
    scene = as_scene(loaded)
    gltf_json = read_glb_json(input_file)
    binary_chunk = read_glb_binary_chunk(input_file)
    transform_map = compute_gltf_world_transforms(gltf_json)
    node_color_map = compute_node_color_map(gltf_json, binary_chunk)
    meshes = scene_to_world_meshes(scene, transform_map, node_color_map)
    if not meshes:
        raise ValueError(f"No mesh geometry found in {input_file}")

    input_size_mb = input_file.stat().st_size / 1024 / 1024
    should_skip_simplify = input_size_mb <= skip_simplify_below_mb

    if should_skip_simplify:
        simplified = []
        for mesh in meshes:
            out = mesh.copy()
            if keep_visuals and mesh.visual is not None:
                try:
                    out.visual = mesh.visual.copy()
                except BaseException:
                    apply_flat_color(out, representative_rgba(mesh))
            else:
                apply_flat_color(out, representative_rgba(mesh))
            simplified.append(out)
    else:
        simplified = simplify_meshes_to_total_budget(meshes, target_faces, keep_visuals, backend)

    normalized = normalize_meshes(simplified)
    out_scene = build_output_scene(normalized)
    out_scene.export(output_file)

    input_faces = sum(len(mesh.faces) for mesh in meshes)
    output_faces = sum(len(mesh.faces) for mesh in normalized)
    output_size_mb = output_file.stat().st_size / 1024 / 1024
    mode = "normalize-only" if should_skip_simplify else "simplified"
    print(
        f"[ok] {input_file} -> {output_file} "
        f"mode {mode}, "
        f"faces {input_faces} -> {output_faces}, "
        f"size {input_size_mb:.2f}MB -> {output_size_mb:.2f}MB"
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve() if args.output else None

    if args.target_faces <= 0:
        raise ValueError("--target-faces must be > 0")

    scan_root, glb_files = collect_glb_files(input_path)
    if not glb_files:
        print("No .glb files found.", file=sys.stderr)
        return 1

    for input_file in glb_files:
        relative = input_file.relative_to(scan_root)
        output_file = input_file if output_root is None else output_root / relative
        process_glb_file(
            input_file,
            output_file,
            args.target_faces,
            args.overwrite,
            args.keep_visuals,
            args.skip_simplify_below_mb,
            args.backend,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
