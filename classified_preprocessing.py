"""Sentinel-2 raster preprocessing: resample + stack + index + cloud mask.

This module is the merge of the previous ``classified_image_processing.py``
(resampling, spectral band stacking, vegetation/water/burn indices, raster
pyramids) and ``classified_cloud_mask.py`` (SCL-based cloud masking).

Top-level entry points:

* :func:`prepare_sentinel2_raw` — unpacks ``.zip`` / ``.SAFE.zip`` archives
  and flattens ``*.SAFE`` Sentinel-2 L2A products into per-granule folders
  containing only the JP2 bands the rest of the pipeline expects.
* :func:`run_image_processing` — walks the prepared folder recursively,
  turning each Sentinel-2 product folder of ``.jp2`` bands into a compressed
  multi-band GeoTIFF in ``Raster_Classified`` (plus a matching ``_SCL.tif``
  in ``SCL_Classified``).
* :func:`run_cloud_masking` — pairs the stacked rasters with their SCL files
  and writes cloud/shadow-masked rasters to ``Raster_Classified_Cloud_Mask``.

The latter two are also exposed under their legacy aliases ``main``,
``resampling`` / ``cloud_mask`` so existing callers keep working.
"""

from __future__ import annotations

import gc
import logging
import os
import shutil
import sys
import time
import typing
import zipfile
from pathlib import Path

# Ensure pyproj uses the Conda-shipped PROJ database before importing GDAL/rasterio.
_proj_path = os.path.join(sys.prefix, "Library", "share", "proj")
if os.path.exists(_proj_path):
    os.environ.setdefault("PROJ_LIB", _proj_path)

import numpy as np
import rasterio
from osgeo import gdal
from rasterio.windows import Window


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    """Configure root logging to file (``sentinel_processing.log``) and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("sentinel_processing.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0. Raw archive / SAFE preparation
# ---------------------------------------------------------------------------

# Bands the rest of the pipeline reads, mapped to the SAFE IMG_DATA subdir
# where the highest-resolution copy lives in Sentinel-2 L2A products.
_SENTINEL2_REQUIRED_BANDS: dict[str, str] = {
    "B02": "R10m", "B03": "R10m", "B04": "R10m", "B08": "R10m",
    "B05": "R20m", "B06": "R20m", "B07": "R20m", "B8A": "R20m",
    "B11": "R20m", "B12": "R20m", "SCL": "R20m",
}


def _is_safe_dir(path: Path) -> bool:
    """True if ``path`` is a Sentinel-2 ``*.SAFE`` directory."""
    return path.is_dir() and path.suffix.upper() == ".SAFE"


def _find_safe_dirs(root: Path) -> list[Path]:
    """Recursively locate ``*.SAFE`` directories under ``root``."""
    if _is_safe_dir(root):
        return [root]
    return [p for p in root.rglob("*.SAFE") if _is_safe_dir(p)]


def _collect_safe_band_files(safe_dir: Path) -> dict[str, Path]:
    """Return a {band_code: jp2_path} mapping for the required Sentinel-2 bands."""
    found: dict[str, Path] = {}
    for img_data in safe_dir.glob("GRANULE/*/IMG_DATA"):
        for band, res_dir in _SENTINEL2_REQUIRED_BANDS.items():
            if band in found:
                continue
            sub = img_data / res_dir
            if not sub.is_dir():
                continue
            matches = list(sub.glob(f"*_{band}_*.jp2"))
            if matches:
                found[band] = matches[0]
    return found


def _extract_archive(archive: Path, dest: Path, logger: logging.Logger) -> None:
    """Extract a ``.zip`` / ``.SAFE.zip`` archive into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive.name} -> {dest}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)


def _stage_safe_dir(
    safe_dir: Path,
    staging_dir: Path,
    logger: logging.Logger,
) -> typing.Optional[Path]:
    """Copy required Sentinel-2 bands from a SAFE folder into a flat staging subdir."""
    bands = _collect_safe_band_files(safe_dir)
    if not bands:
        logger.warning(f"No Sentinel-2 bands found inside {safe_dir}")
        return None

    sample = next(iter(bands.values()))
    parts = sample.stem.split("_")
    if len(parts) < 2:
        logger.warning(f"Unexpected JP2 filename, cannot derive granule key: {sample.name}")
        return None

    granule_key = f"{parts[0]}_{parts[1]}"
    target_dir = staging_dir / granule_key
    target_dir.mkdir(parents=True, exist_ok=True)

    if "SCL" not in bands:
        logger.warning(
            f"{granule_key}: SCL band missing — cloud masking will skip this "
            "granule. Pass --skip_cloud_mask if that's intentional."
        )

    copied = 0
    for jp2 in bands.values():
        dest = target_dir / jp2.name
        if dest.exists():
            continue
        shutil.copy2(jp2, dest)
        copied += 1

    logger.info(
        f"Staged {len(bands)} bands ({copied} new) from {safe_dir.name} -> {target_dir}"
    )
    return target_dir


def prepare_sentinel2_raw(
    raw_input: str | Path,
    staging_dir: str | Path | None = None,
    extracted_dir: str | Path | None = None,
    logger: typing.Optional[logging.Logger] = None,
) -> Path:
    """Normalize raw Sentinel-2 inputs into a folder of per-granule flat JP2 dirs.

    ``raw_input`` may be:

    * A directory containing any mix of ``.zip`` / ``.SAFE.zip`` archives,
      ``*.SAFE`` folders, and pre-flattened JP2 folders.
    * A single ``.zip`` (or ``.SAFE.zip``) archive.
    * A single ``*.SAFE`` folder.

    Archives are extracted under ``extracted_dir`` (default
    ``<raw_input>_Extracted``). For each ``*.SAFE`` product the required bands
    (B02-B12, B8A, SCL) are copied from ``GRANULE/*/IMG_DATA/{R10m,R20m}``
    into a flat folder named after the granule under ``staging_dir`` (default
    ``<raw_input>_Prepared``).

    When ``raw_input`` contains no archives or SAFE folders the call is a
    no-op and returns ``raw_input`` unchanged, so existing pre-flattened
    layouts keep working without modification.

    Returns the directory that should be passed to :func:`run_image_processing`.
    """
    log = logger or logging.getLogger(__name__)
    raw_input = Path(raw_input)
    if not raw_input.exists():
        raise FileNotFoundError(f"Raw input does not exist: {raw_input}")

    archives: list[Path] = []
    safe_dirs: list[Path] = []
    flat_dirs: list[Path] = []

    if raw_input.is_file():
        if raw_input.suffix.lower() != ".zip":
            raise ValueError(
                f"Unsupported input file: {raw_input}. Expected .zip / .SAFE.zip."
            )
        archives = [raw_input]
        anchor = raw_input.parent
        anchor_name = raw_input.stem
    elif _is_safe_dir(raw_input):
        safe_dirs = [raw_input]
        anchor = raw_input.parent
        anchor_name = raw_input.stem
    else:
        anchor = raw_input.parent
        anchor_name = raw_input.name
        for child in raw_input.iterdir():
            if child.is_file() and child.suffix.lower() == ".zip":
                archives.append(child)
            elif _is_safe_dir(child):
                safe_dirs.append(child)
            elif child.is_dir() and any(child.glob("*.jp2")):
                flat_dirs.append(child)

    if not archives and not safe_dirs:
        log.info(
            f"No .zip or .SAFE inputs in {raw_input}; using it as-is "
            f"({len(flat_dirs)} pre-flattened JP2 folders found)."
        )
        return raw_input

    staging_dir = Path(staging_dir) if staging_dir else anchor / f"{anchor_name}_Prepared"
    extracted_dir = Path(extracted_dir) if extracted_dir else anchor / f"{anchor_name}_Extracted"
    staging_dir.mkdir(parents=True, exist_ok=True)

    if flat_dirs:
        log.warning(
            f"{raw_input} mixes archives/SAFE products with {len(flat_dirs)} "
            "pre-flattened JP2 folder(s). Only archives/SAFE will be staged "
            f"automatically; move flat folders into {staging_dir} manually if "
            "you want them included."
        )

    prepared = 0

    for archive in archives:
        target = extracted_dir / archive.stem
        if target.exists() and any(target.iterdir()):
            log.info(f"Reusing already-extracted archive: {target}")
        else:
            _extract_archive(archive, target, log)
        for safe in _find_safe_dirs(target):
            if _stage_safe_dir(safe, staging_dir, log) is not None:
                prepared += 1

    for safe in safe_dirs:
        if _stage_safe_dir(safe, staging_dir, log) is not None:
            prepared += 1

    if prepared == 0:
        log.error(
            f"No Sentinel-2 SAFE products were successfully prepared from "
            f"{raw_input}. Check the SAFE folder structure."
        )
    else:
        log.info(f"Prepared {prepared} Sentinel-2 product(s) under {staging_dir}")

    return staging_dir


# ---------------------------------------------------------------------------
# 1. Resampling, stacking, and index computation
# ---------------------------------------------------------------------------

def resample_image(input_path: str, output_path: str, target_resolution: int = 10) -> bool:
    """Resample a single raster to ``target_resolution`` metres using GDAL.

    Output is written as a tiled, LZW-compressed BigTIFF.
    """
    try:
        print(f"Resampling image: {input_path} to {output_path} at {target_resolution}m resolution.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        src_ds = gdal.Open(input_path)
        if not src_ds:
            raise ValueError(f"Could not open {input_path}")

        gt = src_ds.GetGeoTransform()
        input_res = gt[1]

        src_xsize = src_ds.RasterXSize
        src_ysize = src_ds.RasterYSize
        dst_xsize = int(src_xsize * (input_res / target_resolution))
        dst_ysize = int(src_ysize * (input_res / target_resolution))

        translate_options = gdal.TranslateOptions(
            format="GTiff",
            width=dst_xsize,
            height=dst_ysize,
            resampleAlg=gdal.GRA_NearestNeighbour,
            creationOptions=[
                "COMPRESS=LZW",
                "PREDICTOR=2",
                "TILED=YES",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "BIGTIFF=YES",
            ],
        )

        gdal.Translate(destName=output_path, srcDS=src_ds, options=translate_options)
        src_ds = None

        print(f"Resampling completed with compression: {output_path}")
        return True

    except Exception as exc:
        print(f"Error resampling image {input_path}: {exc}")
        return False


def safe_remove(file_path, max_attempts: int = 5, delay: float = 1.0) -> bool:
    """Remove a file with retries to dodge transient Windows file locks."""
    file_path = Path(file_path)
    if not file_path.exists():
        return True

    for attempt in range(max_attempts):
        try:
            file_path.unlink()
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            continue
        except Exception as exc:
            print(f"Error removing file {file_path}: {exc}")
            return False

    return False


def build_pyramids_nearest(
    raster_path: str,
    overview_levels: typing.Sequence[int] = (2, 4, 8, 16, 32),
    resample_alg: str = "NEAREST",
) -> bool:
    """Build embedded raster overviews (no external ``.ovr``) with nearest-neighbour."""
    try:
        print(f"Building pyramids for: {raster_path}")
        dataset = gdal.Open(raster_path, gdal.GA_Update)
        if not dataset:
            raise ValueError(f"Could not open raster for pyramid building: {raster_path}")

        dataset.BuildOverviews(resample_alg, list(overview_levels))
        dataset = None

        print(f"Successfully built pyramids: {list(overview_levels)} using {resample_alg}")
        return True

    except Exception as exc:
        print(f"Error building pyramids for {raster_path}: {exc}")
        return False


def process_bands(input_folder, output_folder, scl_output_folder=None) -> None:
    """Resample + stack Sentinel-2 bands and compute NDVI/NDWI/SAVI/BAIS2.

    Produces one multi-band GeoTIFF in ``output_folder`` and (optionally) a
    matching ``_SCL.tif`` in ``scl_output_folder``. Pyramids are built on the
    combined raster for fast visualization in QGIS.
    """
    temp_folder = None

    try:
        print(f"Processing bands in folder: {input_folder}")

        input_folder = Path(input_folder)
        if not input_folder.exists():
            raise ValueError(f"Input folder does not exist: {input_folder}")

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        if scl_output_folder:
            scl_output_folder = Path(scl_output_folder)
            scl_output_folder.mkdir(parents=True, exist_ok=True)

        jp2_files = list(input_folder.glob("*.jp2"))
        if not jp2_files:
            print("No JP2 files found in the input folder.")
            return

        temp_folder = output_folder / "temp"
        temp_folder.mkdir(parents=True, exist_ok=True)

        resampled_files: dict = {}
        scl_file = None

        band_map = {
            "B02": "B02", "B03": "B03", "B04": "B04", "B05": "B05",
            "B06": "B06", "B07": "B07", "B08": "B08", "B8A": "B8A",
            "B11": "B11", "B12": "B12",
        }

        for jp2_file in jp2_files:
            output_path = temp_folder / f"{jp2_file.stem}_resampled.tif"
            if not resample_image(str(jp2_file), str(output_path)):
                continue
            if "SCL" in jp2_file.name:
                scl_file = str(output_path)
                continue
            for band_key in band_map:
                if band_key in jp2_file.name:
                    resampled_files[band_key] = str(output_path)
                    break

        ordered_bands = ["B02", "B03", "B04", "B05", "B06",
                         "B07", "B08", "B8A", "B11", "B12"]
        final_resampled_files = [resampled_files[b] for b in ordered_bands if b in resampled_files]

        if not final_resampled_files:
            raise ValueError("No valid band files were processed")

        sample_filename = jp2_files[0].name
        parts = sample_filename.split("_")
        tile_date_timestamp = f"{parts[0]}_{parts[1]}"
        output_filename = f"{tile_date_timestamp}.tif"
        output_path = output_folder / output_filename

        print(f"Creating compressed output file: {output_path}")

        band_data: dict = {}
        for band_name, path in resampled_files.items():
            ds = gdal.Open(path)
            band_data[band_name] = ds.ReadAsArray().astype(np.float32)
            ds = None

        eps = 1e-6

        # BAIS2 — Burned Area Index for Sentinel-2
        term1 = 1 - np.sqrt((band_data["B06"] * band_data["B07"] * band_data["B8A"]) / (band_data["B04"] + eps))
        term2 = ((band_data["B12"] - band_data["B8A"]) /
                 np.sqrt(band_data["B12"] + band_data["B8A"] + eps)) + 1
        bais2 = term1 * term2

        # NDVI
        ndvi = (band_data["B08"] - band_data["B04"]) / (band_data["B08"] + band_data["B04"] + eps)

        # NDWI
        ndwi = (band_data["B03"] - band_data["B08"]) / (band_data["B03"] + band_data["B08"] + eps)

        # SAVI
        L = 1.0
        savi = ((band_data["B08"] - band_data["B04"]) /
                (band_data["B08"] + band_data["B04"] + L)) * (1 + L)

        indices = {
            "BAIS2": bais2,
            "NDVI": ndvi,
            "NDWI": ndwi,
            "SAVI": savi,
        }

        ref_ds = gdal.Open(final_resampled_files[0])
        driver = gdal.GetDriverByName("GTiff")
        total_bands = len(final_resampled_files) + len(indices)

        out_ds = driver.Create(
            str(output_path),
            ref_ds.RasterXSize,
            ref_ds.RasterYSize,
            total_bands,
            gdal.GDT_Float32,
            options=[
                "COMPRESS=LZW",
                "PREDICTOR=2",
                "TILED=YES",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "BIGTIFF=YES",
            ],
        )

        out_ds.SetProjection(ref_ds.GetProjection())
        out_ds.SetGeoTransform(ref_ds.GetGeoTransform())

        for idx, band_path in enumerate(final_resampled_files, start=1):
            ds_band = gdal.Open(band_path)
            arr = ds_band.ReadAsArray().astype(np.float32)
            out_band = out_ds.GetRasterBand(idx)
            out_band.WriteArray(arr)
            out_band.SetDescription(ordered_bands[idx - 1])
            ds_band = None

        for i, (name, arr) in enumerate(indices.items(), start=len(final_resampled_files) + 1):
            out_band = out_ds.GetRasterBand(i)
            out_band.WriteArray(arr.astype(np.float32))
            out_band.SetDescription(name)

        out_ds.FlushCache()
        out_ds = None
        ref_ds = None

        build_pyramids_nearest(str(output_path))

        if scl_file and scl_output_folder:
            scl_output_filename = f"{tile_date_timestamp}_SCL.tif"
            scl_output_path = scl_output_folder / scl_output_filename

            translate_options_scl = gdal.TranslateOptions(
                format="GTiff",
                creationOptions=[
                    "COMPRESS=LZW",
                    "PREDICTOR=2",
                    "TILED=YES",
                    "BLOCKXSIZE=256",
                    "BLOCKYSIZE=256",
                ],
            )
            gdal.Translate(
                destName=str(scl_output_path),
                srcDS=scl_file,
                options=translate_options_scl,
            )
            print(f"Exported SCL file: {scl_output_path}")

        print("Cleaning up temporary files.")
        for path in list(resampled_files.values()):
            safe_remove(path)
        if scl_file:
            safe_remove(scl_file)
        if temp_folder and temp_folder.exists():
            try:
                temp_folder.rmdir()
            except Exception as exc:
                print(f"Warning: Could not remove temp folder: {exc}")

    except Exception as exc:
        print(f"Error processing bands: {exc}")
        raise

    finally:
        if temp_folder and temp_folder.exists():
            try:
                for file in temp_folder.glob("*"):
                    safe_remove(file)
                temp_folder.rmdir()
            except Exception as exc:
                print(f"Warning: Failed final cleanup: {exc}")


# ---------------------------------------------------------------------------
# 2. Cloud masking with SCL
# ---------------------------------------------------------------------------

class SentinelCloudMasker:
    """Memory-efficient SCL-driven cloud/shadow/water masker for Sentinel-2.

    Reads chunks of each stacked raster + its matching SCL file, sets pixels
    flagged as cloud / cloud-shadow / water (see :attr:`CLOUD_CLASSES`) to
    ``NaN``, then writes a tiled, LZW-compressed output GeoTIFF.
    """

    CLOUD_CLASSES = {
        3: "Cloud shadows",
        6: "Water",
        8: "Cloud medium probability",
        9: "Cloud high probability",
    }

    def __init__(
        self,
        scl_dir: str,
        band_dir: str,
        output_dir: str,
        chunk_size: int = 1024,
        log_level: int = logging.INFO,
    ):
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.chunk_size = chunk_size

        self._validate_directories(scl_dir, band_dir, output_dir)

        self.scl_dir = scl_dir
        self.band_dir = band_dir
        self.output_dir = output_dir

        self.scl_files = self._get_file_mapping(scl_dir, "_SCL.tif")
        self.band_files = self._get_file_mapping(band_dir, ".tif")

    def _validate_directories(self, *dirs: str) -> None:
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                raise ValueError(f"Directory does not exist: {dir_path}")
            if not os.path.isdir(dir_path):
                raise ValueError(f"Not a directory: {dir_path}")
            if not os.access(dir_path, os.R_OK):
                raise ValueError(f"Directory not readable: {dir_path}")

    def _get_file_mapping(self, directory: str, suffix: str) -> dict:
        return {
            os.path.basename(f).split(suffix)[0]: os.path.join(root, f)
            for root, _, files in os.walk(directory)
            for f in files if f.endswith(suffix)
        }

    def create_cloud_mask(self, scl_data: np.ndarray) -> np.ndarray:
        """Return a boolean mask where ``True`` = valid pixel (not cloud/water/shadow)."""
        return ~np.isin(scl_data, list(self.CLOUD_CLASSES.keys()))

    def _get_chunk_windows(self, width: int, height: int) -> typing.Iterator[Window]:
        for y in range(0, height, self.chunk_size):
            for x in range(0, width, self.chunk_size):
                yield Window(
                    x, y,
                    min(self.chunk_size, width - x),
                    min(self.chunk_size, height - y),
                )

    def process_chunk(self, scl_data: np.ndarray, band_data: np.ndarray) -> np.ndarray:
        cloud_mask = self.create_cloud_mask(scl_data)
        masked_chunk = np.where(cloud_mask, band_data, np.nan)
        return masked_chunk.astype(np.float32)

    def process_files(self) -> None:
        """Iterate every SCL/band pair, apply masking in chunks, write compressed output."""
        os.makedirs(self.output_dir, exist_ok=True)
        processed_count = 0

        for identifier, scl_path in self.scl_files.items():
            if identifier not in self.band_files:
                self.logger.warning(f"No matching band file for {identifier}")
                continue

            try:
                self.logger.info(f"Processing: {identifier}")
                band_path = self.band_files[identifier]

                with rasterio.open(scl_path) as scl_ds, \
                     rasterio.open(band_path) as band_ds:

                    out_meta = band_ds.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "dtype": "float32",
                        "nodata": np.nan,
                    })

                    temp_output_path = os.path.join(
                        self.output_dir,
                        f"{identifier}_masked_uncompressed.tif",
                    )

                    with rasterio.open(temp_output_path, "w", **out_meta) as dest:
                        for window in self._get_chunk_windows(band_ds.width, band_ds.height):
                            scl_chunk = scl_ds.read(1, window=window)
                            band_chunks = band_ds.read(window=window)

                            for band_idx in range(band_chunks.shape[0]):
                                processed_chunk = self.process_chunk(
                                    scl_chunk,
                                    band_chunks[band_idx],
                                )
                                dest.write(processed_chunk, window=window, indexes=band_idx + 1)

                            del scl_chunk, band_chunks
                            gc.collect()

                output_path = os.path.join(self.output_dir, f"{identifier}_masked.tif")
                gdal.Translate(
                    destName=output_path,
                    srcDS=temp_output_path,
                    options=gdal.TranslateOptions(
                        format="GTiff",
                        creationOptions=[
                            "COMPRESS=LZW",
                            "PREDICTOR=3",
                            "TILED=YES",
                            "BLOCKXSIZE=256",
                            "BLOCKYSIZE=256",
                            "BIGTIFF=YES",
                        ],
                    ),
                )

                os.remove(temp_output_path)
                processed_count += 1
                self.logger.info(f"Completed processing: {identifier}")

            except Exception as exc:
                self.logger.error(f"Error processing {identifier}: {exc}")
            finally:
                gc.collect()

        self.logger.info(f"Processed {processed_count} files")


# ---------------------------------------------------------------------------
# Folder-level entry points
# ---------------------------------------------------------------------------

def find_and_process_folders(
    root_folder,
    output_folder,
    scl_output_folder=None,
    logger: typing.Optional[logging.Logger] = None,
) -> int:
    """Walk ``root_folder`` recursively and call :func:`process_bands` on every
    sub-folder containing ``.jp2`` files. Returns the number of folders processed.
    """
    log = logger or logging.getLogger(__name__)
    root_folder = Path(root_folder)
    output_folder = Path(output_folder)
    scl_output_folder = Path(scl_output_folder) if scl_output_folder else None

    log.info(f"Searching for folders in: {root_folder}")
    processed_folders = 0

    for dirpath in root_folder.rglob("*"):
        if dirpath.is_dir() and any(f.suffix == ".jp2" for f in dirpath.iterdir()):
            relative_path = dirpath.relative_to(root_folder)
            current_output_folder = output_folder / relative_path
            current_scl_output_folder = (
                scl_output_folder / relative_path if scl_output_folder else None
            )
            log.info(f"Found JP2 files in: {dirpath}. Processing...")
            process_bands(dirpath, current_output_folder, current_scl_output_folder)
            processed_folders += 1

    if processed_folders == 0:
        log.warning("No folders with JP2 files were processed.")
    else:
        log.info(f"Processed {processed_folders} folders.")
    return processed_folders


def run_image_processing(
    root_folder: str | Path = "Sentinel2_Raw",
    output_folder: str | Path = "Raster_Classified",
    scl_output_folder: str | Path = "SCL_Classified",
) -> None:
    """Top-level entry: resample, stack, index, and pyramid-build the Sentinel-2 archive."""
    logger = setup_logging()
    try:
        gdal.UseExceptions()
        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")

        root_folder = Path(root_folder)
        output_folder = Path(output_folder)
        scl_output_folder = Path(scl_output_folder)

        if not root_folder.exists():
            logger.error(f"Input folder does not exist: {root_folder}")
            logger.info(
                f"Please create a '{root_folder.name}' folder and place Sentinel-2 JP2 files inside."
            )
            sys.exit(1)

        output_folder.mkdir(parents=True, exist_ok=True)
        scl_output_folder.mkdir(parents=True, exist_ok=True)

        jp2_files = list(root_folder.rglob("*.jp2"))
        if not jp2_files:
            logger.error(f"No JP2 files found in {root_folder}")
            logger.info(
                f"Ensure Sentinel-2 JP2 files are present in the '{root_folder.name}' folder."
            )
            sys.exit(1)

        logger.info(f"Found {len(jp2_files)} JP2 files to process")

        find_and_process_folders(root_folder, output_folder, scl_output_folder, logger=logger)

        logger.info("Processing complete.")

    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        sys.exit(1)


def run_cloud_masking(
    scl_dir: str | Path = "SCL_Classified",
    band_dir: str | Path = "Raster_Classified",
    output_dir: str | Path = "Raster_Classified_Cloud_Mask",
    chunk_size: int = 512,
) -> None:
    """Top-level entry: SCL-driven cloud masking on every stacked raster."""
    try:
        cloud_masker = SentinelCloudMasker(
            scl_dir=str(scl_dir),
            band_dir=str(band_dir),
            output_dir=str(output_dir),
            chunk_size=chunk_size,
            log_level=logging.INFO,
        )
        cloud_masker.process_files()
    except Exception as exc:
        logging.error(f"Unexpected error in cloud masking: {exc}")


# Legacy aliases so older imports keep working.
resampling = run_image_processing
cloud_mask = run_cloud_masking


def main() -> None:
    """Run the full preprocessing pipeline: image processing then cloud masking."""
    run_image_processing()
    run_cloud_masking()


if __name__ == "__main__":
    main()
