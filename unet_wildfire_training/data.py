"""Dataset, raster/label matching, and DataLoader construction."""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box, mapping
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from unet_wildfire_training.config import TrainingConfig


_DATE_RE = re.compile(r"(\d{8})")
_MAX_TILE_RETRIES = 10


def extract_date_string(path: str | Path) -> Optional[str]:
    """Return the first 8-digit date (YYYYMMDD) embedded in the filename, if any."""
    match = _DATE_RE.search(os.path.basename(str(path)))
    return match.group(1) if match else None


def percentile_normalize(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Clip each band to its ``[low, high]`` percentile range, then scale to ``[0, 1]``.

    Operates in place on a copy and returns the result. A flat band (max equal to
    min within the clipped range) is zeroed out to keep the network input well-defined.
    """
    out = image.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        band = out[c]
        lo, hi = np.percentile(band, [low, high])
        if hi - lo > 1e-6:
            out[c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
        else:
            out[c] = 0.0
    return out


# ---------------------------------------------------------------------------
# Sentinel-2 spectral indices
# ---------------------------------------------------------------------------

def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise division returning 0 where ``|denominator| < 1e-6``."""
    out = np.zeros(numerator.shape, dtype=np.float32)
    mask = np.abs(denominator) > 1e-6
    np.divide(numerator, denominator, out=out, where=mask)
    return out


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Sentinel-2 bands B08 (NIR), B04 (Red)."""
    return _safe_divide(nir - red, nir + red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR). Sentinel-2 bands B03 (Green), B08 (NIR)."""
    return _safe_divide(green - nir, green + nir)


def compute_savi(red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray:
    """SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L). Sentinel-2 bands B08, B04."""
    return _safe_divide(nir - red, nir + red + L) * (1.0 + L)


def compute_bais2(
    red: np.ndarray,
    red_edge_2: np.ndarray,
    red_edge_3: np.ndarray,
    narrow_nir: np.ndarray,
    swir_2: np.ndarray,
) -> np.ndarray:
    """BAIS2 (Filipponi 2018) burned-area index for Sentinel-2.

    ``(1 - sqrt(B06 * B07 * B8A / B04)) * ((B12 - B8A) / sqrt(B12 + B8A) + 1)``
    """
    inner = _safe_divide(red_edge_2 * red_edge_3 * narrow_nir, red)
    inner = np.clip(inner, 0.0, None)
    term1 = 1.0 - np.sqrt(inner)

    sum_nir_swir = np.clip(swir_2 + narrow_nir, 1e-6, None)
    term2 = (swir_2 - narrow_nir) / np.sqrt(sum_nir_swir) + 1.0
    return (term1 * term2).astype(np.float32)


SENTINEL2_INDEX_NAMES: Tuple[str, ...] = ("NDVI", "NDWI", "SAVI", "BAIS2")
_REQUIRED_INDEX_BANDS: Tuple[str, ...] = ("B03", "B04", "B06", "B07", "B08", "B8A", "B12")


def compute_sentinel2_indices(
    image: np.ndarray,
    band_layout: Dict[str, int],
    reflectance_scale: float = 10000.0,
) -> np.ndarray:
    """Stack NDVI, NDWI, SAVI, and BAIS2 as a ``(4, H, W)`` float32 array.

    Args:
        image: Raw band stack shaped ``(C, H, W)``.
        band_layout: 0-indexed channel positions for Sentinel-2 bands within
            ``image``. Must include keys ``B03, B04, B06, B07, B08, B8A, B12``.
        reflectance_scale: Divisor applied before computing indices. Sentinel-2
            L2A ships as uint16 with a quantification value of 10000. Pass
            ``1.0`` if the rasters already hold [0, 1] reflectance.

    Returns:
        ``(4, H, W)`` float32 array in the order of :data:`SENTINEL2_INDEX_NAMES`.
    """
    missing = [b for b in _REQUIRED_INDEX_BANDS if b not in band_layout]
    if missing:
        raise KeyError(f"band_layout is missing required Sentinel-2 bands: {missing}")

    image_f32 = image.astype(np.float32, copy=False)
    if reflectance_scale and reflectance_scale != 1.0:
        image_f32 = image_f32 / float(reflectance_scale)

    green = image_f32[band_layout["B03"]]
    red = image_f32[band_layout["B04"]]
    red_edge_2 = image_f32[band_layout["B06"]]
    red_edge_3 = image_f32[band_layout["B07"]]
    nir = image_f32[band_layout["B08"]]
    narrow_nir = image_f32[band_layout["B8A"]]
    swir_2 = image_f32[band_layout["B12"]]

    return np.stack(
        [
            compute_ndvi(red, nir),
            compute_ndwi(green, nir),
            compute_savi(red, nir),
            compute_bais2(red, red_edge_2, red_edge_3, narrow_nir, swir_2),
        ],
        axis=0,
    ).astype(np.float32)


class SegmentationDataset(Dataset):
    """PyTorch dataset producing ``(image_tile, binary_mask)`` tensor pairs.

    Args:
        image_paths: List of raster file paths.
        label_paths: List of vector label file paths (parallel to ``image_paths``).
        windows: List of ``(image_index, rasterio.windows.Window)`` tuples
            identifying each tile to read.
        target_size: ``(H, W)`` to which tiles are resized before being returned.
        bands: ``bands[i]`` is the list of 1-indexed band numbers to read from
            raster ``i``. All entries must have equal length so tensors stack
            into a batch.
        normalization_percentiles: ``(low, high)`` percentile pair used by
            ``percentile_normalize``.
    """

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        label_paths: Sequence[str | Path],
        windows: Sequence[Tuple[int, Window]],
        target_size: Tuple[int, int] = (256, 256),
        bands: Optional[Sequence[Sequence[int]]] = None,
        normalization_percentiles: Tuple[float, float] = (2.0, 98.0),
        band_layout: Optional[Dict[str, int]] = None,
        reflectance_scale: float = 10000.0,
        gdfs: Optional[Sequence[gpd.GeoDataFrame]] = None,
        packed_masks: Optional[Sequence[np.ndarray]] = None,
        image_widths: Optional[Sequence[int]] = None,
    ):
        if bands is None:
            raise ValueError("`bands` is required so the channel count is known")
        if len({len(b) for b in bands}) != 1:
            raise ValueError("All entries in `bands` must have the same length")

        self.image_paths = [str(p) for p in image_paths]
        self.label_paths = [str(p) for p in label_paths]
        self.windows = list(windows)
        self.target_size = target_size
        self.bands = [list(b) for b in bands]
        self.num_bands = len(self.bands[0])
        self.norm_low, self.norm_high = normalization_percentiles
        self.band_layout = dict(band_layout) if band_layout else None
        self.reflectance_scale = reflectance_scale
        self.output_channels = self.num_bands + (
            len(SENTINEL2_INDEX_NAMES) if self.band_layout is not None else 0
        )

        self.crs_list = []
        self.gdf_list = []
        self.packed_masks = None
        self.image_widths = None

        if packed_masks is not None and image_widths is not None:
            self.packed_masks = list(packed_masks)
            self.image_widths = list(image_widths)
        elif gdfs is not None:
            self.gdf_list = list(gdfs)
        else:
            for image_path, label_path in zip(self.image_paths, self.label_paths):
                with rasterio.open(image_path) as src:
                    crs = src.crs
                self.crs_list.append(crs)

                gdf = gpd.read_file(label_path)
                if gdf.crs != crs:
                    gdf = gdf.to_crs(crs)
                self.gdf_list.append(gdf)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng()
        last_error: Optional[BaseException] = None

        for attempt in range(_MAX_TILE_RETRIES + 1):
            try:
                return self._read_tile(idx)
            except Exception as exc:
                last_error = exc
                image_path = self.image_paths[self.windows[idx][0]]
                print(
                    f"Warning: Error reading tile at index {idx} from {image_path}: {exc}",
                    flush=True,
                )
                idx = int(rng.integers(0, len(self.windows)))

        raise RuntimeError(
            f"Exceeded maximum retries ({_MAX_TILE_RETRIES}) reading a valid tile. "
            f"Last error: {last_error}"
        )

    def _read_tile(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx, window = self.windows[idx]
        image_path = self.image_paths[img_idx]

        with rasterio.open(image_path) as src:
            image = src.read(self.bands[img_idx], window=window)
            tile_transform = src.window_transform(window)
            bounds = window_bounds(window, src.transform)

        height = window.height
        width = window.width

        if self.packed_masks is not None and self.image_widths is not None:
            packed_mask = self.packed_masks[img_idx]
            r_start = int(window.row_off)
            r_end = r_start + height
            c_start = int(window.col_off)
            c_end = c_start + width
            
            packed_rows = packed_mask[r_start:r_end, :]
            unpacked_rows = np.unpackbits(packed_rows, axis=-1)[:, :self.image_widths[img_idx]]
            mask = unpacked_rows[:, c_start:c_end]
        else:
            gdf = self.gdf_list[img_idx]
            clipped = gdf.clip(bounds)
            clipped_geom = clipped.geometry.buffer(0)
            shapes = [(mapping(geom), 1) for geom in clipped_geom if not geom.is_empty]
            mask = rasterize(
                shapes,
                out_shape=(height, width),
                transform=tile_transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            )

        if self.band_layout is not None:
            indices = compute_sentinel2_indices(
                image, self.band_layout, self.reflectance_scale
            )
            image = np.concatenate(
                [image.astype(np.float32, copy=False), indices], axis=0
            )

        image = resize(
            image.transpose(1, 2, 0),
            self.target_size + (image.shape[0],),
            mode="reflect",
            anti_aliasing=True,
        ).transpose(2, 0, 1).astype(np.float32)
        image = percentile_normalize(image, self.norm_low, self.norm_high)

        mask = resize(mask, self.target_size, order=0, anti_aliasing=False).astype(np.uint8)
        mask = mask[None, :, :]

        return torch.from_numpy(image), torch.from_numpy(mask)


def match_raster_shapefile(
    image_base_dir: str | Path,
    label_base_dir: str | Path,
) -> List[Tuple[str, str]]:
    """Match each GeoTIFF in ``image_base_dir`` to a spatially-overlapping shapefile.

    A shapefile is considered a match if (a) its filename contains the same
    YYYYMMDD date as the raster (when both have one) and (b) any of its
    geometries intersect the raster's bounding box. Shapefiles are read once
    and cached so the cost is linear in the number of label files.

    Returns a list of ``(image_path, label_path)`` pairs.
    """
    image_files = sorted(glob.glob(os.path.join(str(image_base_dir), "**", "*.tif"), recursive=True))
    label_files = sorted(glob.glob(os.path.join(str(label_base_dir), "**", "*.shp"), recursive=True))

    label_dates: Dict[str, Optional[str]] = {p: extract_date_string(p) for p in label_files}
    gdf_cache: Dict[str, Optional[gpd.GeoDataFrame]] = {}

    def load_gdf(path: str) -> Optional[gpd.GeoDataFrame]:
        if path not in gdf_cache:
            try:
                gdf = gpd.read_file(path)
                if gdf.empty:
                    gdf_cache[path] = None
                else:
                    gdf_cache[path] = gdf
            except Exception as exc:
                print(f"Warning: Could not load shapefile {path}: {exc}", flush=True)
                gdf_cache[path] = None
        return gdf_cache[path]

    pairs: List[Tuple[str, str]] = []
    for idx, img_path in enumerate(image_files, start=1):
        print(
            f"[{idx}/{len(image_files)}] Scanning {os.path.basename(img_path)}...",
            flush=True,
        )
        img_date = extract_date_string(img_path)
        try:
            with rasterio.open(img_path) as src:
                img_polygon = box(*src.bounds)
                img_crs = src.crs
        except Exception as exc:
            print(f"Warning: Could not process raster {img_path}: {exc}", flush=True)
            continue

        for label_path in label_files:
            label_date = label_dates[label_path]
            if img_date and label_date and img_date != label_date:
                continue

            gdf = load_gdf(label_path)
            if gdf is None:
                continue

            if gdf.crs != img_crs:
                gdf = gdf.to_crs(img_crs)

            if gdf.geometry.intersects(img_polygon).any():
                pairs.append((img_path, label_path))
                print(
                    f"Matched: {os.path.basename(img_path)} <-> {os.path.basename(label_path)}",
                    flush=True,
                )
                break
        else:
            print(f"No overlapping shapefile found for: {os.path.basename(img_path)}", flush=True)

    print(f"\nTotal matched pairs: {len(pairs)}", flush=True)
    return pairs


def collect_tile_windows(
    pairs: Sequence[Tuple[str, str]],
    tile_size: int,
) -> Tuple[
    List[str],
    List[str],
    List[List[int]],
    List[Tuple[int, Window]],
    List[int],
    int,
    List[gpd.GeoDataFrame],
]:
    """Enumerate non-overlapping tile windows for every matched raster.

    For each raster the function:
      * reads its CRS/transform/band count,
      * loads the matching shapefile and reprojects it to the raster CRS,
      * generates ``tile_size`` × ``tile_size`` windows that fit entirely
        inside the raster,
      * labels each window as positive (intersects any burn polygon) or
        negative using the shapefile's spatial index.

    Returns:
        image_paths, label_paths, bands_per_image, windows, tile_labels, max_channels, gdfs_per_image

        - ``windows[i]`` is ``(image_index, Window)``.
        - ``tile_labels[i]`` is 1 for burned tiles and 0 for unburned.
        - ``bands_per_image[i]`` is padded later so all rasters expose the same
          channel count.
        - ``gdfs_per_image[i]`` is the loaded and reprojected GeoDataFrame for image i.
    """
    image_paths: List[str] = []
    label_paths: List[str] = []
    gdfs_per_image: List[gpd.GeoDataFrame] = []
    bands_per_image: List[List[int]] = []
    burn_windows: List[Tuple[int, Window]] = []
    unburn_windows: List[Tuple[int, Window]] = []
    max_channels = 0

    # Cache for loaded and reprojected GeoDataFrames
    # Key: (label_path, img_crs)
    gdf_cache: Dict[Tuple[str, str], gpd.GeoDataFrame] = {}

    for image_path, label_path in pairs:
        with rasterio.open(image_path) as src:
            height, width = src.shape
            band_indices = list(range(1, src.count + 1))
            img_transform = src.transform
            img_crs = src.crs

        cache_key = (str(label_path), str(img_crs))
        if cache_key in gdf_cache:
            gdf = gdf_cache[cache_key]
        else:
            gdf = gpd.read_file(label_path)
            if gdf.crs != img_crs:
                gdf = gdf.to_crs(img_crs)
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
            gdf_cache[cache_key] = gdf

        if gdf.empty:
            continue

        windows = [
            Window(j, i, tile_size, tile_size)
            for i in range(0, height, tile_size)
            for j in range(0, width, tile_size)
            if i + tile_size <= height and j + tile_size <= width
        ]

        sindex = gdf.sindex
        image_index = len(image_paths)
        for window in windows:
            left, bottom, right, top = window_bounds(window, img_transform)
            tile_poly = box(left, bottom, right, top)
            if list(sindex.query(tile_poly, predicate="intersects")):
                burn_windows.append((image_index, window))
            else:
                unburn_windows.append((image_index, window))

        image_paths.append(image_path)
        label_paths.append(label_path)
        gdfs_per_image.append(gdf)
        bands_per_image.append(band_indices)
        max_channels = max(max_channels, len(band_indices))

    all_windows = burn_windows + unburn_windows
    tile_labels = [1] * len(burn_windows) + [0] * len(unburn_windows)
    return image_paths, label_paths, bands_per_image, all_windows, tile_labels, max_channels, gdfs_per_image


def _pad_bands(bands_per_image: Sequence[Sequence[int]], max_channels: int) -> List[List[int]]:
    """Pad shorter band lists by repeating the last band so every entry has ``max_channels`` indices."""
    padded: List[List[int]] = []
    for b in bands_per_image:
        b = list(b)
        if len(b) < max_channels:
            b = b + [b[-1]] * (max_channels - len(b))
        padded.append(b[:max_channels])
    return padded


def build_dataloaders(
    config: TrainingConfig,
    pairs: Sequence[Tuple[str, str]],
    band_layout: Optional[Dict[str, int]] = None,
    reflectance_scale: float = 10000.0,
) -> Tuple[DataLoader, DataLoader, int]:
    """Build stratified train/validation DataLoaders from matched raster/label pairs.

    When ``band_layout`` is supplied, each tile is augmented with the four
    Sentinel-2 indices (NDVI, NDWI, SAVI, BAIS2) computed by
    :func:`compute_sentinel2_indices`, and the returned channel count reflects
    the added bands so the model can be sized accordingly.

    Returns ``(train_loader, val_loader, model_input_channels)``. Raises
    ``RuntimeError`` when no burned tiles are found (training would be degenerate).
    """
    image_paths, label_paths, bands_per_image, windows, tile_labels, max_channels, gdfs_per_image = collect_tile_windows(
        pairs, config.tile_size
    )

    if not windows:
        raise RuntimeError("No tiles collected from the matched pairs.")
    if 1 not in tile_labels:
        raise RuntimeError(
            "No burned tiles found. Check your shapefiles and spatial overlap logic."
        )

    print(f" -> Total dataset size (no tiles discarded): {len(windows)} tiles", flush=True)

    train_windows, val_windows, _, _ = train_test_split(
        windows,
        tile_labels,
        test_size=config.val_split,
        random_state=config.random_seed,
        stratify=tile_labels,
    )

    padded_bands = _pad_bands(bands_per_image, max_channels)

    print(" -> Pre-rasterizing shapefiles to image grids...", flush=True)
    packed_masks = []
    image_widths = []
    for idx, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        gdf = gdfs_per_image[idx]
        with rasterio.open(image_path) as src:
            out_shape = src.shape
            transform = src.transform
            width = src.width
        image_widths.append(width)

        valid_geoms = gdf.geometry.buffer(0)
        shapes = [(mapping(geom), 1) for geom in valid_geoms if not geom.is_empty]
        if shapes:
            mask = rasterize(
                shapes,
                out_shape=out_shape,
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8,
            )
        else:
            mask = np.zeros(out_shape, dtype=np.uint8)
        packed_mask = np.packbits(mask, axis=-1)
        packed_masks.append(packed_mask)
    print(" -> Pre-rasterization completed successfully.", flush=True)

    train_dataset = SegmentationDataset(
        image_paths,
        label_paths,
        train_windows,
        target_size=config.target_size,
        bands=padded_bands,
        normalization_percentiles=config.normalization_percentiles,
        band_layout=band_layout,
        reflectance_scale=reflectance_scale,
        packed_masks=packed_masks,
        image_widths=image_widths,
    )
    val_dataset = SegmentationDataset(
        image_paths,
        label_paths,
        val_windows,
        target_size=config.target_size,
        bands=padded_bands,
        normalization_percentiles=config.normalization_percentiles,
        band_layout=band_layout,
        reflectance_scale=reflectance_scale,
        packed_masks=packed_masks,
        image_widths=image_widths,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    model_input_channels = max_channels + (
        len(SENTINEL2_INDEX_NAMES) if band_layout else 0
    )
    return train_loader, val_loader, model_input_channels
