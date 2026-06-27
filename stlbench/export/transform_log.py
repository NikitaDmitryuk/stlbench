from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

SCHEMA_VERSION = 1


def matrix_to_list(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float64)
    return [[float(v) for v in row] for row in arr.tolist()]


def translation_matrix(values: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    out[:3, 3] = v[:3]
    return out


def uniform_scale_matrix(scale: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[0, 0] = float(scale)
    out[1, 1] = float(scale)
    out[2, 2] = float(scale)
    return out


def z_rotation_matrix(angle_deg: float) -> np.ndarray:
    return np.asarray(
        trimesh.transformations.rotation_matrix(np.radians(float(angle_deg)), [0.0, 0.0, 1.0]),
        dtype=np.float64,
    )


def bounds_to_list(bounds: np.ndarray | None) -> list[list[float]] | None:
    if bounds is None:
        return None
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.shape != (2, 3):
        return None
    return [[float(v) for v in row] for row in arr.tolist()]


def mesh_bounds(mesh: trimesh.Trimesh) -> list[list[float]]:
    return bounds_to_list(np.asarray(mesh.bounds, dtype=np.float64)) or []


def transform_bounds(bounds: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(bounds, dtype=np.float64)
    mins = arr[0]
    maxs = arr[1]
    corners = np.array(
        [
            [mins[0], mins[1], mins[2], 1.0],
            [mins[0], mins[1], maxs[2], 1.0],
            [mins[0], maxs[1], mins[2], 1.0],
            [mins[0], maxs[1], maxs[2], 1.0],
            [maxs[0], mins[1], mins[2], 1.0],
            [maxs[0], mins[1], maxs[2], 1.0],
            [maxs[0], maxs[1], mins[2], 1.0],
            [maxs[0], maxs[1], maxs[2], 1.0],
        ],
        dtype=np.float64,
    )
    transformed = (np.asarray(matrix, dtype=np.float64) @ corners.T).T[:, :3]
    return np.array([transformed.min(axis=0), transformed.max(axis=0)], dtype=np.float64)


def safe_inverse(matrix: np.ndarray | None) -> np.ndarray | None:
    if matrix is None:
        return None
    try:
        return np.linalg.inv(np.asarray(matrix, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None


def transform_step(
    name: str,
    *,
    matrix: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    available: bool = True,
) -> dict[str, Any]:
    out: dict[str, Any] = {"name": name, "available": bool(available)}
    if params:
        out["params"] = params
    matrix_payload = matrix_to_list(matrix)
    if matrix_payload is not None:
        out["matrix"] = matrix_payload
    return out


def placement_transform_for_mesh(
    mesh: trimesh.Trimesh, rect: Any
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return current-mesh-to-plate transform matching export_plate_3mf_lazy."""
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    normalize = translation_matrix(-bounds[0])
    rotation_deg = float(getattr(rect, "rotation_deg", 0.0))
    rotate = z_rotation_matrix(rotation_deg)
    after_rotate = transform_bounds(transform_bounds(bounds, normalize), rotate)
    post_rotate_normalize = translation_matrix([-after_rotate[0, 0], -after_rotate[0, 1], 0.0])
    plate_translation = translation_matrix(
        [float(getattr(rect, "x", 0.0)), float(getattr(rect, "y", 0.0)), 0.0]
    )
    matrix = plate_translation @ post_rotate_normalize @ rotate @ normalize
    steps = [
        transform_step("normalize_to_origin", matrix=normalize),
        transform_step("packer_z_rotation", matrix=rotate, params={"rotation_deg": rotation_deg}),
        transform_step("post_rotation_normalize", matrix=post_rotate_normalize),
        transform_step(
            "plate_translation",
            matrix=plate_translation,
            params={
                "x_mm": float(getattr(rect, "x", 0.0)),
                "y_mm": float(getattr(rect, "y", 0.0)),
            },
        ),
    ]
    return matrix, steps


def shared_geometry_placement_transform_for_mesh(
    mesh: trimesh.Trimesh,
    rect: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return normalized-shared-geometry-to-plate transform for fill-style instancing."""
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    rotation_deg = float(getattr(rect, "rotation_deg", 0.0))
    rotate = z_rotation_matrix(rotation_deg)
    after_rotate = transform_bounds(bounds, rotate)
    post_rotate_normalize = translation_matrix([-after_rotate[0, 0], -after_rotate[0, 1], 0.0])
    plate_translation = translation_matrix(
        [float(getattr(rect, "x", 0.0)), float(getattr(rect, "y", 0.0)), 0.0]
    )
    matrix = plate_translation @ post_rotate_normalize @ rotate
    steps = [
        transform_step("packer_z_rotation", matrix=rotate, params={"rotation_deg": rotation_deg}),
        transform_step("post_rotation_normalize", matrix=post_rotate_normalize),
        transform_step(
            "plate_translation",
            matrix=plate_translation,
            params={
                "x_mm": float(getattr(rect, "x", 0.0)),
                "y_mm": float(getattr(rect, "y", 0.0)),
            },
        ),
    ]
    return matrix, steps


def transform_entry(
    *,
    source_path: Path | str,
    source_name: str,
    output_name: str,
    output_file: Path | str,
    source_bounds_mm: list[list[float]] | None,
    final_bounds_mm: list[list[float]] | None,
    source_to_export_matrix: np.ndarray | None,
    steps: list[dict[str, Any]],
    index: int | None = None,
    plate_index: int | None = None,
    plate_file: Path | str | None = None,
    scale_factor: float | None = None,
    plate_x_mm: float | None = None,
    plate_y_mm: float | None = None,
    rotation_deg: float | None = None,
    source_transform_available: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_path": str(source_path),
        "source_name": source_name,
        "output_name": output_name,
        "output_file": str(output_file),
        "source_bounds_mm": source_bounds_mm,
        "final_bounds_mm": final_bounds_mm,
        "steps": steps,
        "source_transform_available": bool(source_transform_available),
        "source_to_export_matrix": matrix_to_list(source_to_export_matrix),
        "export_to_source_matrix": matrix_to_list(safe_inverse(source_to_export_matrix)),
    }
    optional: dict[str, Any] = {
        "index": index,
        "plate_index": plate_index,
        "plate_file": None if plate_file is None else str(plate_file),
        "scale_factor": scale_factor,
        "plate_x_mm": plate_x_mm,
        "plate_y_mm": plate_y_mm,
        "rotation_deg": rotation_deg,
    }
    payload.update({k: v for k, v in optional.items() if v is not None})
    return payload


def write_transform_log(
    path: Path,
    *,
    command: str,
    output_files: Sequence[Path | str],
    parts: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "created_at": datetime.now(UTC).isoformat(),
        "units": "mm",
        "output_files": [str(p) for p in output_files],
        "parts": parts,
    }
    if metadata:
        payload["metadata"] = metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
