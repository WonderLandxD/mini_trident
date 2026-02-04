from __future__ import annotations

import json
import os
import socket
from typing import List, Optional

import h5py

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon

ENV_TRIDENT_HOME = "TRIDENT_HOME"
ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
DEFAULT_CACHE_DIR = "~/.cache"
_cache_dir: Optional[str] = None


def get_dir() -> str:
    """
    Return the cache directory used to store downloaded segmentation checkpoints.
    Defaults to $TRIDENT_HOME or ~/.cache/trident if the environment variables are
    not set.
    """
    if _cache_dir is not None:
        return _cache_dir
    trident_home = os.path.expanduser(
        os.getenv(
            ENV_TRIDENT_HOME,
            os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "trident"),
        )
    )
    return trident_home


def set_dir(path: str) -> None:
    """Override the default cache directory."""
    global _cache_dir
    _cache_dir = os.path.expanduser(path)


def has_internet_connection(timeout: float = 3.0) -> bool:
    """Lightweight internet connectivity check used before attempting downloads."""
    endpoint = os.environ.get("HF_ENDPOINT", "huggingface.co")
    if endpoint.startswith(("http://", "https://")):
        from urllib.parse import urlparse

        endpoint = urlparse(endpoint).netloc

    try:
        socket.create_connection((endpoint, 443), timeout=timeout)
        return True
    except OSError:
        return False


def get_weights_path(model_name: str) -> str:
    """
    Look up a local checkpoint path for the given model.
    The registry lives next to this file in local_ckpts.json.
    """
    registry_path = os.path.join(os.path.dirname(__file__), "local_ckpts.json")
    if not os.path.exists(registry_path):
        return ""

    with open(registry_path, "r") as f:
        registry = json.load(f)

    path = registry.get(model_name, "") or ""
    if path:
        path = path if os.path.isabs(path) else os.path.abspath(
            os.path.join(os.path.dirname(registry_path), path)
        )
        if not os.path.exists(path):
            path = ""

    return path


def read_coords(coords_path: str):
    """
    Read coordinates saved in HDF5 format and return (attrs, coords).
    Expects a dataset named 'coords' with attributes such as patch_size,
    level0_magnification, target_magnification, and overlap.
    """
    with h5py.File(coords_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' dataset not found in {coords_path}")
        coords = np.array(f["coords"])
        attrs = dict(f["coords"].attrs)

    # Decode bytes attributes to str for readability
    clean_attrs = {}
    for k, v in attrs.items():
        if isinstance(v, bytes):
            clean_attrs[k] = v.decode()
        else:
            clean_attrs[k] = v
    return clean_attrs, coords


def coords_to_h5(
    coords: List[List[int]],
    save_path: str,
    patch_size: int,
    src_mag: int,
    target_mag: int,
    save_coords_dir: str,
    width: int,
    height: int,
    name: str,
    overlap: int,
):
    """
    Save patch coordinates to an HDF5 file with Trident-compatible attributes.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, "w") as f:
        dset = f.create_dataset("coords", data=np.array(coords, dtype=np.int32))
        dset.attrs["patch_size"] = patch_size
        dset.attrs["patch_size_level0"] = patch_size * src_mag // target_mag
        dset.attrs["level0_magnification"] = src_mag
        dset.attrs["target_magnification"] = target_mag
        dset.attrs["overlap"] = overlap
        dset.attrs["name"] = name
        dset.attrs["savetodir"] = save_coords_dir
        dset.attrs["level0_width"] = width
        dset.attrs["level0_height"] = height


def filter_contours(contours, hierarchy, filter_params, pixel_size):
    """Filter raw OpenCV contours, keeping only sufficiently large regions and holes."""
    if hierarchy is None or not hierarchy.size:
        return [], []

    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    foreground_indices = np.flatnonzero(hierarchy[:, 1] == -1)
    filtered_foregrounds, filtered_holes = [], []

    for cont_idx in foreground_indices:
        contour = contours[cont_idx]
        hole_indices = np.flatnonzero(hierarchy[:, 1] == cont_idx)

        contour_area = cv2.contourArea(contour)
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in hole_indices]
        net_area = (contour_area - sum(hole_areas)) * (pixel_size ** 2)
        if net_area <= 0 or net_area <= filter_params["a_t"]:
            continue

        if filter_params.get("filter_color_mode") not in [None, "none"]:
            raise Exception("Unsupported filter_color_mode")

        filtered_foregrounds.append(contour)

        valid_holes = [
            contours[hole_idx]
            for hole_idx in hole_indices
            if cv2.contourArea(contours[hole_idx]) * (pixel_size ** 2)
            > filter_params["min_hole_area"]
        ]
        valid_holes = sorted(valid_holes, key=cv2.contourArea, reverse=True)[
            : filter_params["max_n_holes"]
        ]
        filtered_holes.append(valid_holes)

    return filtered_foregrounds, filtered_holes


def make_valid(polygon: Polygon) -> Polygon:
    """Try to fix invalid polygons by buffering them slightly."""
    for i in [0, 0.1, -0.1, 0.2]:
        new_polygon = polygon.buffer(i)
        if isinstance(new_polygon, Polygon) and new_polygon.is_valid:
            return new_polygon
    raise Exception("Failed to make a valid polygon")


def scale_contours(contours, scale, is_nested: bool = False):
    """Scale OpenCV contours by a constant factor."""
    if is_nested:
        return [
            [np.array(hole * scale, dtype="int32") for hole in holes] for holes in contours
        ]
    return [np.array(cont * scale, dtype="int32") for cont in contours]


def mask_to_gdf(
    mask: np.ndarray,
    keep_ids: List[int] | None = None,
    exclude_ids: List[int] | None = None,
    max_nb_holes: int = 0,
    min_contour_area: float = 1000,
    pixel_size: float = 1.0,
    contour_scale: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Convert a binary mask into a GeoDataFrame of polygons.
    Non-zero pixels are treated as tissue; holes can be preserved or discarded.
    """
    keep_ids = keep_ids or []
    exclude_ids = exclude_ids or []

    target_edge_size = 2000
    scale = target_edge_size / mask.shape[0]
    downscaled_mask = cv2.resize(
        mask, (round(mask.shape[1] * scale), round(mask.shape[0] * scale))
    )

    mode = cv2.RETR_TREE if max_nb_holes == 0 else cv2.RETR_CCOMP
    contours, hierarchy = cv2.findContours(downscaled_mask, mode, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        hierarchy = np.array([])

    filter_params = {
        "filter_color_mode": "none",
        "max_n_holes": max_nb_holes,
        "a_t": min_contour_area * pixel_size**2,
        "min_hole_area": 4000 * pixel_size**2,
    }

    foreground_contours, hole_contours = filter_contours(
        contours, hierarchy, filter_params, pixel_size
    )
    if len(foreground_contours) == 0:
        return gpd.GeoDataFrame(columns=["tissue_id", "geometry"])

    contours_tissue = scale_contours(foreground_contours, contour_scale / scale, is_nested=False)
    contours_holes = scale_contours(hole_contours, contour_scale / scale, is_nested=True)

    contour_ids = set(keep_ids) - set(exclude_ids) if keep_ids else set(
        np.arange(len(contours_tissue))
    ) - set(exclude_ids)

    tissue_ids = [i for i in contour_ids]
    polygons = []
    for i in contour_ids:
        holes = [contours_holes[i][j].squeeze(1) for j in range(len(contours_holes[i]))] if len(
            contours_holes[i]
        ) > 0 else None
        polygon = Polygon(contours_tissue[i].squeeze(1), holes=holes)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        polygons.append(polygon)

    gdf_contours = gpd.GeoDataFrame(pd.DataFrame(tissue_ids, columns=["tissue_id"]), geometry=polygons)
    return gdf_contours


def overlay_gdf_on_thumbnail(
    gdf_contours: gpd.GeoDataFrame,
    thumbnail: np.ndarray,
    contours_saveto: str,
    scale: float,
    tissue_color=(0, 255, 0),
    hole_color=(255, 0, 0),
):
    """Draw polygons onto a thumbnail and save the annotated image."""
    if len(gdf_contours) == 0:
        cv2.imwrite(contours_saveto, cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR))
        return

    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
    for _, row in gdf_contours.iterrows():
        polygon = row.geometry
        if polygon.is_empty:
            continue

        exterior_coords = np.array(polygon.exterior.coords) / scale
        cv2.polylines(
            thumbnail, [exterior_coords.astype(np.int32)], isClosed=True, color=tissue_color, thickness=2
        )

        for interior in polygon.interiors:
            interior_coords = np.array(interior.coords) / scale
            cv2.polylines(
                thumbnail, [interior_coords.astype(np.int32)], isClosed=True, color=hole_color, thickness=2
            )

    nz = np.nonzero(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2GRAY))
    xmin, xmax, ymin, ymax = np.min(nz[1]), np.max(nz[1]), np.min(nz[0]), np.max(nz[0])
    cropped_annotated = thumbnail[ymin:ymax, xmin:xmax]

    os.makedirs(os.path.dirname(contours_saveto), exist_ok=True)
    cropped_annotated = cv2.cvtColor(cropped_annotated, cv2.COLOR_BGR2RGB)
    cv2.imwrite(contours_saveto, cropped_annotated)


def get_num_workers(
    batch_size: int, factor: float = 0.75, fallback: int = 16, max_workers: int | None = None
) -> int:
    """Heuristic for torch DataLoader workers with a soft cap."""
    if os.name == "nt":
        return 0

    num_cores = os.cpu_count() or fallback
    num_workers = int(factor * num_cores)
    max_workers = max_workers or (2 * batch_size)
    num_workers = np.clip(num_workers, 1, max_workers)
    return int(num_workers)
