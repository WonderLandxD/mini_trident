from __future__ import annotations

import os
import io
import json
from typing import Any, List, Literal, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .io_utils import (
    get_num_workers,
    mask_to_gdf,
    overlay_gdf_on_thumbnail,
    read_coords,
    coords_to_h5,
)
from .models import SegmentationModel
from .patcher import WSIPatcher
from .patcher_dataset import WSIPatcherDataset

ReadMode = Literal["pil", "numpy"]


class OpenSlideWSI:
    """
    Minimal WSI wrapper for tissue segmentation using the OpenSlide backend.
    """

    def __init__(
        self,
        slide_path: str,
        tissue_seg_path: Optional[str] = None,
        custom_mpp_keys: Optional[List[str]] = None,
        mpp: Optional[float] = None,
        max_workers: Optional[int] = None,
    ):
        self.slide_path = slide_path
        self.name, self.ext = os.path.splitext(os.path.basename(slide_path))
        self.custom_mpp_keys = custom_mpp_keys
        self.max_workers = max_workers

        self.img = openslide.OpenSlide(self.slide_path)
        self.dimensions = self.img.dimensions
        self.width, self.height = self.dimensions
        self.level_count = self.img.level_count
        self.level_downsamples = self.img.level_downsamples
        self.level_dimensions = self.img.level_dimensions
        self.properties = self.img.properties
        self.mpp = mpp if mpp is not None else self._fetch_mpp(custom_mpp_keys)
        self.mag = self._fetch_magnification(custom_mpp_keys)

        self.gdf_contours = None
        self.tissue_seg_path = None
        if tissue_seg_path is not None:
            self.gdf_contours = gpd.read_file(tissue_seg_path)
            self.tissue_seg_path = tissue_seg_path

    def _fetch_mpp(self, custom_mpp_keys: Optional[List[str]] = None) -> float:
        mpp_keys = [
            openslide.PROPERTY_NAME_MPP_X,
            "openslide.mirax.MPP",
            "aperio.MPP",
            "hamamatsu.XResolution",
            "openslide.comment",
        ]
        if custom_mpp_keys:
            mpp_keys.extend(custom_mpp_keys)

        for key in mpp_keys:
            if key in self.img.properties:
                try:
                    return round(float(self.img.properties[key]), 4)
                except ValueError:
                    continue

        x_resolution = self.img.properties.get("tiff.XResolution")
        unit = self.img.properties.get("tiff.ResolutionUnit")
        if x_resolution and unit:
            try:
                if unit.lower() == "centimeter":
                    return round(10000 / float(x_resolution), 4)
                if unit.upper() == "INCH":
                    return round(25400 / float(x_resolution), 4)
            except ValueError:
                pass

        raise ValueError(
            f"Unable to extract MPP from slide metadata: '{self.slide_path}'. "
            "Provide `mpp` directly or supply `custom_mpp_keys`."
        )

    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:
        mpp_x = self.mpp if self.mpp is not None else self._fetch_mpp(custom_mpp_keys)
        if mpp_x < 0.16:
            return 80
        if mpp_x < 0.2:
            return 60
        if mpp_x < 0.3:
            return 40
        if mpp_x < 0.6:
            return 20
        if mpp_x < 1.2:
            return 10
        if mpp_x < 2.4:
            return 5
        raise ValueError(f"Identified mpp is very low: mpp={mpp_x}. Most WSIs are at 20x or 40x magnification.")

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int], read_as: ReadMode = "pil"
    ):
        region = self.img.read_region(location, level, size).convert("RGB")
        if read_as == "pil":
            return region
        if read_as == "numpy":
            return np.array(region)
        raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'.")

    def get_dimensions(self) -> Tuple[int, int]:
        return self.img.dimensions

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        return self.img.get_thumbnail(size).convert("RGB")

    def get_best_level_and_custom_downsample(self, downsample: float, tolerance: float = 0.01) -> Tuple[int, float]:
        level_downsamples = self.level_downsamples

        for level, level_downsample in enumerate(level_downsamples):
            if abs(level_downsample - downsample) <= tolerance:
                return level, 1.0

        if downsample >= level_downsamples[0]:
            closest_level, closest_downsample = None, None
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample <= downsample:
                    closest_level = level
                    closest_downsample = level_downsample
                else:
                    break
            if closest_level is not None:
                custom_downsample = downsample / closest_downsample
                return closest_level, custom_downsample
        else:
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample >= downsample:
                    custom_downsample = level_downsample / downsample
                    return level, custom_downsample

        raise ValueError(f"No suitable level found for downsample {downsample}.")

    @torch.inference_mode()
    def _segment_semantic(
        self,
        segmentation_model: SegmentationModel,
        target_mag: int,
        verbose: bool,
        device: str,
        batch_size: int,
    ):
        destination_mpp = 10 / target_mag
        patcher = WSIPatcher(
            self,
            patch_size=segmentation_model.input_size,
            src_pixel_size=self.mpp,
            dst_pixel_size=destination_mpp,
            mask=self.gdf_contours,
        )
        precision = segmentation_model.precision
        eval_transforms = segmentation_model.eval_transforms
        dataset = WSIPatcherDataset(patcher, eval_transforms)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,  # get_num_workers(batch_size, max_workers=self.max_workers)
            pin_memory=True,
        )

        mpp_reduction_factor = self.mpp / destination_mpp
        width, height = self.get_dimensions()
        width, height = int(round(width * mpp_reduction_factor)), int(round(height * mpp_reduction_factor))
        predicted_mask = np.zeros((height, width), dtype=np.uint8)

        total_patches = len(dataset)
        processed_patches = 0

        for batch in dataloader:
            imgs, (xcoords, ycoords) = batch
            device_type = device.split(":")[0]
            autocast_enabled = (precision != torch.float32) and device_type == "cuda"
            dtype_for_tensor = precision if autocast_enabled else torch.float32
            with torch.autocast(device_type=device_type, dtype=precision, enabled=autocast_enabled):
                imgs = imgs.to(device, dtype=dtype_for_tensor)
                preds = segmentation_model(imgs).cpu().numpy()
            if verbose:
                processed_patches += imgs.shape[0]
                print(f"\r[Seg] {processed_patches}/{total_patches} patches", end="", flush=True)

            x_starts = np.clip(np.round(xcoords.numpy() * mpp_reduction_factor).astype(int), 0, width - 1)
            y_starts = np.clip(np.round(ycoords.numpy() * mpp_reduction_factor).astype(int), 0, height - 1)
            x_ends = np.clip(x_starts + segmentation_model.input_size, 0, width)
            y_ends = np.clip(y_starts + segmentation_model.input_size, 0, height)

            for i in range(len(preds)):
                x_start, x_end = x_starts[i], x_ends[i]
                y_start, y_end = y_starts[i], y_ends[i]
                if x_start >= x_end or y_start >= y_end:
                    continue
                patch_pred = preds[i][: y_end - y_start, : x_end - x_start]
                predicted_mask[y_start:y_end, x_start:x_end] += patch_pred
        if verbose:
            print()
        return predicted_mask, mpp_reduction_factor

    @torch.inference_mode()
    def segment_tissue(
        self,
        segmentation_model: SegmentationModel,
        target_mag: int = 10,
        holes_are_tissue: bool = True,
        job_dir: Optional[str] = None,
        batch_size: int = 16,
        device: str = "cuda:0",
        verbose: bool = False,
    ) -> Union[str, gpd.GeoDataFrame]:
        """
        Run tissue segmentation, optionally saving thumbnails/contours to disk.
        """
        segmentation_model.to(device)
        max_dimension = 1000
        if self.width > self.height:
            thumbnail_width = max_dimension
            thumbnail_height = int(thumbnail_width * self.height / self.width)
        else:
            thumbnail_height = max_dimension
            thumbnail_width = int(thumbnail_height * self.width / self.height)
        thumbnail = self.get_thumbnail((thumbnail_width, thumbnail_height))

        predicted_mask, mpp_reduction_factor = self._segment_semantic(
            segmentation_model,
            target_mag,
            verbose,
            device,
            batch_size,
        )

        predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255
        gdf_contours = mask_to_gdf(
            mask=predicted_mask,
            max_nb_holes=0 if holes_are_tissue else 20,
            min_contour_area=1000,
            pixel_size=self.mpp,
            contour_scale=1 / mpp_reduction_factor,
        )

        if job_dir is None:
            return gdf_contours

        thumbnail_saveto = os.path.join(job_dir, "thumbnails", f"{self.name}.jpg")
        os.makedirs(os.path.dirname(thumbnail_saveto), exist_ok=True)
        thumbnail.save(thumbnail_saveto)

        gdf_saveto = os.path.join(job_dir, "contours_geojson", f"{self.name}.geojson")
        os.makedirs(os.path.dirname(gdf_saveto), exist_ok=True)
        gdf_contours.set_crs("EPSG:3857", inplace=True)
        gdf_contours.to_file(gdf_saveto, driver="GeoJSON")
        self.gdf_contours = gdf_contours
        self.tissue_seg_path = gdf_saveto

        contours_saveto = os.path.join(job_dir, "contours", f"{self.name}.jpg")
        annotated = np.array(thumbnail)
        overlay_gdf_on_thumbnail(gdf_contours, annotated, contours_saveto, thumbnail_width / self.width)

        return gdf_saveto

    def visualize_coords(self, coords_path: str, save_patch_viz: str) -> str:
        """
        Overlay stored patch coordinates onto a thumbnail and save the visualization.
        """
        return _visualize_coords_impl(self, coords_path, save_patch_viz)

    def save_patches(
        self,
        coords_to_keep: List[Tuple[int, int]],
        target_mag: int,
        patch_size: int,
        save_coords: str,
        overlap: int,
        min_tissue_proportion: float,
        save_patches_type: str,
        save_patches_verbose: bool,
    ) -> None:
        save_patches_type = save_patches_type.lower()
        if save_patches_type not in {"tar", "jpg", "none"}:
            raise ValueError(f"Invalid save_patches_type: {save_patches_type}")

        if save_patches_type == "none" or len(coords_to_keep) == 0:
            return

        patcher_imgs = WSIPatcher(
            self,
            patch_size=patch_size,
            src_mag=self.mag,
            dst_mag=target_mag,
            mask=self.gdf_contours if hasattr(self, "gdf_contours") else None,
            coords_only=False,
            custom_coords=np.array(coords_to_keep),
            overlap=overlap,
            threshold=min_tissue_proportion,
            pil=True,
        )
        total_patches = len(coords_to_keep)
        if save_patches_type == "tar":
            patches_root = os.path.join(save_coords, "patches_webdataset", f"{self.name}")
            os.makedirs(patches_root, exist_ok=True)
            pattern_base = os.path.join(patches_root, self.name)
            wds_pattern = f"{pattern_base}-%06d.tar"
            try:
                import webdataset as wds  # type: ignore
            except ImportError as exc:
                raise ImportError("webdataset is required for save_patches_type=tar") from exc

            if save_patches_verbose:
                print(f"[{self.name}] Saving patches (wds) -> {wds_pattern}, total={total_patches}")
            with wds.ShardWriter(wds_pattern, maxcount=total_patches + 1, compress=False) as sink:
                for idx, (tile, x, y) in enumerate(patcher_imgs):
                    key = f"{self.name}-{idx:06d}-x{x}-y{y}"
                    img_buf = io.BytesIO()
                    tile.save(img_buf, format="JPEG")
                    meta = {
                        "slide": self.name,
                        "x": int(x),
                        "y": int(y),
                        "patch_size": patch_size,
                        "target_mag": target_mag,
                        "overlap": overlap,
                    }
                    sink.write(
                        {
                            "__key__": key,
                            "jpg": img_buf.getvalue(),
                            "json": json.dumps(meta).encode("utf-8"),
                        }
                    )
                    if save_patches_verbose and (idx % 500 == 0 or idx == total_patches - 1):
                        print(f"\r[{self.name}] saved {idx + 1}/{total_patches}", end="", flush=True)
        else:
            patches_root = os.path.join(save_coords, "patches_jpg", f"{self.name}")
            os.makedirs(patches_root, exist_ok=True)
            if save_patches_verbose:
                print(f"[{self.name}] Saving patches (jpg) -> {patches_root}, total={total_patches}")
            for idx, (tile, x, y) in enumerate(patcher_imgs):
                key = f"{self.name}-{idx:06d}-x{x}-y{y}"
                tile_path = os.path.join(patches_root, f"{key}.jpg")
                tile.save(tile_path, format="JPEG")
                if save_patches_verbose and (idx % 500 == 0 or idx == total_patches - 1):
                    print(f"\r[{self.name}] saved {idx + 1}/{total_patches}", end="", flush=True)

    def extract_tissue_coords(
        self,
        target_mag: int,
        patch_size: int,
        save_coords: str,
        overlap: int = 0,
        min_tissue_proportion: float = 0.0,
    ) -> str:
        """
        Extract patch coordinates from tissue regions and save them to HDF5.
        """
        patcher = WSIPatcher(
            self,
            patch_size=patch_size,
            src_mag=self.mag,
            dst_mag=target_mag,
            mask=self.gdf_contours if hasattr(self, "gdf_contours") else None,
            coords_only=True,
            overlap=overlap,
            threshold=min_tissue_proportion,
        )

        coords_to_keep = [(x, y) for x, y in patcher]

        os.makedirs(os.path.join(save_coords, "patches"), exist_ok=True)
        out_fname = os.path.join(save_coords, "patches", f"{self.name}_patches.h5")
        coords_to_h5(
            coords=coords_to_keep,
            save_path=out_fname,
            patch_size=patch_size,
            src_mag=self.mag,
            target_mag=target_mag,
            save_coords_dir=save_coords,
            width=self.width,
            height=self.height,
            name=self.name,
            overlap=overlap,
        )

        return out_fname


def _visualize_coords_impl(wsi_obj: Any, coords_path: str, save_patch_viz: str) -> str:
    coords_attrs, coords = read_coords(coords_path)
    patch_size = coords_attrs.get("patch_size", None)
    level0_magnification = coords_attrs.get("level0_magnification", None)
    target_magnification = coords_attrs.get("target_magnification", None)
    if None in (patch_size, level0_magnification, target_magnification):
        raise KeyError("Missing essential attributes (patch_size, level0_magnification, target_magnification) in coords file.")

    patcher = WSIPatcher(
        wsi=wsi_obj,
        patch_size=patch_size,
        src_mag=level0_magnification,
        dst_mag=target_magnification,
        custom_coords=coords,
        coords_only=True,
    )
    img = patcher.visualize()
    os.makedirs(save_patch_viz, exist_ok=True)
    viz_coords_path = os.path.join(save_patch_viz, f"{wsi_obj.name}.jpg")
    img.save(viz_coords_path)
    return viz_coords_path


def load_wsi(slide_path: str, **kwargs) -> Union[OpenSlideWSI, "SDPCWSI"]:
    """
    Convenience wrapper to create a WSI reader.
    Supports OpenSlide-backed formats plus `.sdpc` (via opensdpc).
    """
    ext = os.path.splitext(slide_path)[1].lower()
    if ext == ".sdpc":
        return SDPCWSI(slide_path=slide_path, **kwargs)
    return OpenSlideWSI(slide_path=slide_path, **kwargs)


class SDPCWSI(OpenSlideWSI):
    """
    SDPC reader using opensdpc with the same interface as OpenSlideWSI for segmentation.
    """

    def __init__(
        self,
        slide_path: str,
        tissue_seg_path: Optional[str] = None,
        custom_mpp_keys: Optional[List[str]] = None,
        mpp: Optional[float] = None,
        max_workers: Optional[int] = None,
    ):
        try:
            import opensdpc
        except ImportError as e:
            raise ImportError("opensdpc is required to read .sdpc slides. See https://github.com/WonderLandxD/opensdpc for installation instructions.") from e

        self.slide_path = slide_path
        self.name, self.ext = os.path.splitext(os.path.basename(slide_path))
        self.custom_mpp_keys = custom_mpp_keys
        self.max_workers = max_workers

        self.img = opensdpc.OpenSdpc(self.slide_path)
        self.dimensions = self.get_dimensions()
        self.width, self.height = self.dimensions
        self.level_count = self.img.level_count
        self.level_downsamples = self.img.level_downsamples
        self.level_dimensions = self.img.level_dimensions
        self.properties = None
        self.mpp = mpp if mpp is not None else self.img.readSdpc(self.slide_path).contents.picHead.contents.ruler
        self.mag = self.img.scan_magnification

        self.gdf_contours = None
        self.tissue_seg_path = None
        if tissue_seg_path is not None:
            self.gdf_contours = gpd.read_file(tissue_seg_path)
            self.tissue_seg_path = tissue_seg_path

    def _fetch_magnification(self, custom_mpp_keys: Optional[List[str]] = None) -> int:
        mpp_x = self.mpp
        if mpp_x < 0.16:
            return 80
        if mpp_x < 0.2:
            return 60
        if mpp_x < 0.3:
            return 40
        if mpp_x < 0.6:
            return 20
        if mpp_x < 1.2:
            return 10
        if mpp_x < 2.4:
            return 5
        raise ValueError(f"Identified mpp is very low: mpp={mpp_x}. Most WSIs are at 20x or 40x magnification.")

    def _get_closest_thumbnail_level(self, size: Tuple[int, int]) -> int:
        for level in range(self.level_count):
            level_width, level_height = self.level_dimensions[level]
            if level_width <= size[0] and level_height <= size[1]:
                return max(0, level - 1)
        return self.level_count - 1

    def read_region(
        self, location: Tuple[int, int], level: int, size: Tuple[int, int], read_as: ReadMode = "pil"
    ):
        region = self.img.read_region(location, level, size).convert("RGB")
        if read_as == "pil":
            return region
        if read_as == "numpy":
            return np.array(region)
        raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'.")

    def get_dimensions(self) -> Tuple[int, int]:
        return self.img.level_dimensions[0]

    def get_thumbnail(self, size: tuple[int, int]) -> Image.Image:
        closest_level = self._get_closest_thumbnail_level(size)
        level_width, level_height = self.level_dimensions[closest_level]
        thumbnail = self.read_region((0, 0), closest_level, (level_width, level_height), read_as="pil")
        thumbnail = thumbnail.resize(size, Image.LANCZOS)
        return thumbnail

    def get_best_level_and_custom_downsample(self, downsample: float, tolerance: float = 0.01) -> Tuple[int, float]:
        level_downsamples = self.level_downsamples

        for level, level_downsample in enumerate(level_downsamples):
            if abs(level_downsample - downsample) <= tolerance:
                return level, 1.0

        if downsample >= level_downsamples[0]:
            closest_level, closest_downsample = None, None
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample <= downsample:
                    closest_level = level
                    closest_downsample = level_downsample
                else:
                    break
            if closest_level is not None:
                custom_downsample = downsample / closest_downsample
                return closest_level, custom_downsample
        else:
            for level, level_downsample in enumerate(level_downsamples):
                if level_downsample >= downsample:
                    custom_downsample = level_downsample / downsample
                    return level, custom_downsample

        raise ValueError(f"No suitable level found for downsample {downsample}.")
