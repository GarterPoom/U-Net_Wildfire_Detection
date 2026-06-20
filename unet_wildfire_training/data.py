"""Dataset, raster/label matching, and DataLoader construction.""" # Module docstring describing the purpose of the file

from __future__ import annotations # Enable postponed evaluation of type annotations

import glob # Import glob for filesystem pattern matching
import os # Import os for operating system operations
import re # Import re for regular expression operations
from pathlib import Path # Import Path for object-oriented filesystem paths
from typing import Dict, List, Optional, Sequence, Tuple # Import typing for type hinting

import geopandas as gpd # Import geopandas for geospatial data handling
import numpy as np # Import numpy for numerical computations
import rasterio # Import rasterio for reading and writing raster data
import torch # Import torch for deep learning operations
from rasterio.features import rasterize # Import rasterize to convert vector to raster
from rasterio.windows import Window, bounds as window_bounds # Import Window and bounds for raster windowing
from shapely.geometry import box, mapping # Import box and mapping for geometric operations
from skimage.transform import resize # Import resize for image resizing
from sklearn.model_selection import train_test_split # Import train_test_split for data splitting
from torch.utils.data import DataLoader, Dataset # Import DataLoader and Dataset for PyTorch data handling

from unet_wildfire_training.config import TrainingConfig # Import TrainingConfig from the config module


_DATE_RE = re.compile(r"(\d{8})") # Compile regex for 8-digit date extraction
_MAX_TILE_RETRIES = 10 # Define maximum number of retries for tile reading


def extract_date_string(path: str | Path) -> Optional[str]: # Function to extract date string from path
    """Return the first 8-digit date (YYYYMMDD) embedded in the filename, if any.""" # Docstring for extract_date_string
    match = _DATE_RE.search(os.path.basename(str(path))) # Search for date pattern in filename
    return match.group(1) if match else None # Return matched group or None if no match


def percentile_normalize(image: np.ndarray, low: float, high: float) -> np.ndarray: # Function for percentile normalization
    """Clip each band to its ``[low, high]`` percentile range, then scale to ``[0, 1]``.

    Operates in place on a copy and returns the result. A flat band (max equal to
    min within the clipped range) is zeroed out to keep the network input well-defined.
    """ # Docstring for percentile_normalize
    out = image.astype(np.float32, copy=True) # Create a float32 copy of the image
    for c in range(out.shape[0]): # Iterate through each channel
        band = out[c] # Select the current band
        # Ignore NaNs (cloud-masked pixels) when computing statistics # Comment about NaN handling
        lo, hi = np.nanpercentile(band, [low, high]) # Calculate percentiles ignoring NaNs
        if not np.isnan(lo) and not np.isnan(hi) and hi - lo > 1e-6: # Check if percentiles are valid and range is non-zero
            out[c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0) # Clip and scale the band to [0, 1]
        else: # If band is invalid or flat
            out[c] = 0.0 # Set the band to zero
    # Convert any remaining NaNs to zero for model input # Comment about NaN conversion
    out = np.nan_to_num(out, nan=0.0) # Replace all NaNs with 0.0
    return out # Return the normalized image


# --------------------------------------------------------------------------- # Separator for Sentinel-2 indices
# Sentinel-2 spectral indices # Header for spectral indices
# --------------------------------------------------------------------------- # Separator for Sentinel-2 indices

def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray: # Helper function for safe division
    """Element-wise division returning 0 where ``|denominator| < 1e-6``.""" # Docstring for _safe_divide
    out = np.zeros(numerator.shape, dtype=np.float32) # Initialize output array with zeros
    mask = np.abs(denominator) > 1e-6 # Create mask for non-zero denominators
    np.divide(numerator, denominator, out=out, where=mask) # Perform division where mask is true
    return out # Return the result of the division


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray: # Function to compute NDVI
    """NDVI = (NIR - Red) / (NIR + Red). Sentinel-2 bands B08 (NIR), B04 (Red).""" # Docstring for compute_ndvi
    return _safe_divide(nir - red, nir + red) # Calculate and return NDVI


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray: # Function to compute NDWI
    """NDWI = (Green - NIR) / (Green + NIR). Sentinel-2 bands B03 (Green), B08 (NIR).""" # Docstring for compute_ndwi
    return _safe_divide(green - nir, green + nir) # Calculate and return NDWI


def compute_savi(red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray: # Function to compute SAVI
    """SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L). Sentinel-2 bands B08, B04.""" # Docstring for compute_savi
    return _safe_divide(nir - red, nir + red + L) * (1.0 + L) # Calculate and return SAVI


def compute_bais2( # Function to compute BAIS2 index
    red: np.ndarray, # Input red band
    red_edge_2: np.ndarray, # Input red edge band 2
    red_edge_3: np.ndarray, # Input red edge band 3
    narrow_nir: np.ndarray, # Input narrow NIR band
    swir_2: np.ndarray, # Input SWIR band 2
) -> np.ndarray: # Return type is numpy array
    """BAIS2 (Filipponi 2018) burned-area index for Sentinel-2.

    ``(1 - sqrt(B06 * B07 * B8A / B04)) * ((B12 - B8A) / sqrt(B12 + B8A) + 1)``
    """ # Docstring for compute_bais2
    inner = _safe_divide(red_edge_2 * red_edge_3 * narrow_nir, red) # Compute inner term of the formula
    inner = np.clip(inner, 0.0, None) # Clip inner term to be non-negative
    term1 = 1.0 - np.sqrt(inner) # Compute the first term

    sum_nir_swir = np.clip(swir_2 + narrow_nir, 1e-6, None) # Compute sum of NIR and SWIR with clipping
    term2 = (swir_2 - narrow_nir) / np.sqrt(sum_nir_swir) + 1.0 # Compute the second term
    return (term1 * term2).astype(np.float32) # Return the product as float32


SENTINEL2_INDEX_NAMES: Tuple[str, ...] = ("NDVI", "NDWI", "SAVI", "BAIS2") # Constant for index names
_REQUIRED_INDEX_BANDS: Tuple[str, ...] = ("B03", "B04", "B06", "B07", "B08", "B8A", "B12") # Constant for required bands


def compute_sentinel2_indices( # Function to compute all Sentinel-2 indices
    image: np.ndarray, # Input image array
    band_layout: Dict[str, int], # Dictionary mapping band names to indices
    reflectance_scale: float = 10000.0, # Scaling factor for reflectance
) -> np.ndarray: # Return type is numpy array
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
    """ # Docstring for compute_sentinel2_indices
    missing = [b for b in _REQUIRED_INDEX_BANDS if b not in band_layout] # Find missing bands
    if missing: # If bands are missing
        raise KeyError(f"band_layout is missing required Sentinel-2 bands: {missing}") # Raise KeyError

    image_f32 = image.astype(np.float32, copy=False) # Convert image to float32
    if reflectance_scale and reflectance_scale != 1.0: # If scale is not 1.0
        image_f32 = image_f32 / float(reflectance_scale) # Apply scaling factor

    green = image_f32[band_layout["B03"]] # Extract green band
    red = image_f32[band_layout["B04"]] # Extract red band
    red_edge_2 = image_f32[band_layout["B06"]] # Extract red edge 2 band
    red_edge_3 = image_f32[band_layout["B07"]] # Extract red edge 3 band
    nir = image_f32[band_layout["B08"]] # Extract NIR band
    narrow_nir = image_f32[band_layout["B8A"]] # Extract narrow NIR band
    swir_2 = image_f32[band_layout["B12"]] # Extract SWIR 2 band

    return np.stack( # Stack indices into a single array
        [
            compute_ndvi(red, nir), # Compute NDVI
            compute_ndwi(green, nir), # Compute NDWI
            compute_savi(red, nir), # Compute SAVI
            compute_bais2(red, red_edge_2, red_edge_3, narrow_nir, swir_2), # Compute BAIS2
        ],
        axis=0, # Stack along the first axis
    ).astype(np.float32) # Ensure output is float32


class SegmentationDataset(Dataset): # PyTorch dataset class for segmentation
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
    """ # Docstring for SegmentationDataset

    def __init__( # Constructor for SegmentationDataset
        self, # Self parameter
        image_paths: Sequence[str | Path], # Input image paths
        label_paths: Sequence[str | Path], # Input label paths
        windows: Sequence[Tuple[int, Window]], # Input windows for tiling
        target_size: Tuple[int, int] = (256, 256), # Target tile size
        bands: Optional[Sequence[Sequence[int]]] = None, # Input bands to read
        normalization_percentiles: Tuple[float, float] = (2.0, 98.0), # Normalization percentiles
        band_layout: Optional[Dict[str, int]] = None, # Band layout for indices
        reflectance_scale: float = 10000.0, # Reflectance scaling factor
        gdfs: Optional[Sequence[gpd.GeoDataFrame]] = None, # Input GeoDataFrames
        packed_masks: Optional[Sequence[np.ndarray]] = None, # Input packed masks
        image_widths: Optional[Sequence[int]] = None, # Input image widths
    ): # End of __init__ arguments
        if bands is None: # If bands is not provided
            raise ValueError("`bands` is required so the channel count is known") # Raise ValueError
        if len({len(b) for b in bands}) != 1: # If band counts are inconsistent
            raise ValueError("All entries in `bands` must have the same length") # Raise ValueError

        self.image_paths = [str(p) for p in image_paths] # Store image paths as strings
        self.label_paths = [str(p) for p in label_paths] # Store label paths as strings
        self.windows = list(windows) # Store windows as a list
        self.target_size = target_size # Store target size
        self.bands = [list(b) for b in bands] # Store bands as a list of lists
        self.num_bands = len(self.bands[0]) # Store number of bands
        self.norm_low, self.norm_high = normalization_percentiles # Store normalization percentiles
        self.band_layout = dict(band_layout) if band_layout else None # Store band layout
        self.reflectance_scale = reflectance_scale # Store reflectance scale
        self.output_channels = self.num_bands + ( # Calculate total output channels
            len(SENTINEL2_INDEX_NAMES) if self.band_layout is not None else 0 # Add indices if layout provided
        ) # End of output_channels calculation

        self.crs_list = [] # Initialize CRS list
        self.gdf_list = [] # Initialize GDF list
        self.packed_masks = None # Initialize packed masks
        self.image_widths = None # Initialize image widths

        if packed_masks is not None and image_widths is not None: # If packed masks provided
            self.packed_masks = list(packed_masks) # Store packed masks
            self.image_widths = list(image_widths) # Store image widths
        elif gdfs is not None: # Else if GDFs provided
            self.gdf_list = list(gdfs) # Store GDFs
        else: # Otherwise load from files
            for image_path, label_path in zip(self.image_paths, self.label_paths): # Iterate through paths
                with rasterio.open(image_path) as src: # Open raster file
                    crs = src.crs # Get CRS from raster
                self.crs_list.append(crs) # Append CRS to list

                gdf = gpd.read_file(label_path) # Read shapefile
                if gdf.crs != crs: # If CRS doesn't match
                    gdf = gdf.to_crs(crs) # Reproject GDF to raster CRS
                self.gdf_list.append(gdf) # Append GDF to list

    def __len__(self) -> int: # Method to get dataset length
        return len(self.windows) # Return number of windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: # Method to get a single item
        rng = np.random.default_rng() # Initialize random number generator
        last_error: Optional[BaseException] = None # Initialize error tracker

        for attempt in range(_MAX_TILE_RETRIES + 1): # Retry loop for reading tiles
            try: # Try reading the tile
                return self._read_tile(idx) # Return tile if successful
            except Exception as exc: # If error occurs
                last_error = exc # Store last error
                image_path = self.image_paths[self.windows[idx][0]] # Get image path of failed tile
                print( # Print warning message
                    f"Warning: Error reading tile at index {idx} from {image_path}: {exc}", # Warning text
                    flush=True, # Flush output
                ) # End of print
                idx = int(rng.integers(0, len(self.windows))) # Pick a random index for retry

        raise RuntimeError( # If all retries fail
            f"Exceeded maximum retries ({_MAX_TILE_RETRIES}) reading a valid tile. " # Error message
            f"Last error: {last_error}" # Include last error
        ) # End of RuntimeError

    def _read_tile(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]: # Private method to read a tile
        img_idx, window = self.windows[idx] # Get image index and window
        image_path = self.image_paths[img_idx] # Get image path

        with rasterio.open(image_path) as src: # Open raster file
            image = src.read(self.bands[img_idx], window=window) # Read specified bands for window
            tile_transform = src.window_transform(window) # Get transform for the window
            bounds = window_bounds(window, src.transform) # Get spatial bounds of the window

        height = window.height # Window height
        width = window.width # Window width

        if self.packed_masks is not None and self.image_widths is not None: # If using packed masks
            packed_mask = self.packed_masks[img_idx] # Get packed mask
            r_start = int(window.row_off) # Starting row
            r_end = r_start + height # Ending row
            c_start = int(window.col_off) # Starting column
            c_end = c_start + width # Ending column
            
            packed_rows = packed_mask[r_start:r_end, :] # Slice packed rows
            unpacked_rows = np.unpackbits(packed_rows, axis=-1)[:, :self.image_widths[img_idx]] # Unpack bits to get mask
            mask = unpacked_rows[:, c_start:c_end] # Slice mask to current window
        else: # Otherwise use GeoDataFrame
            gdf = self.gdf_list[img_idx] # Get GDF
            clipped = gdf.clip(bounds) # Clip GDF to window bounds
            clipped_geom = clipped.geometry.buffer(0) # Fix geometry with buffer(0)
            shapes = [(mapping(geom), 1) for geom in clipped_geom if not geom.is_empty] # Prepare shapes for rasterization
            mask = rasterize( # Rasterize shapes to mask
                shapes, # Geometric shapes
                out_shape=(height, width), # Output shape
                transform=tile_transform, # Transform for window
                fill=0, # Fill value
                all_touched=True, # Include all touched pixels
                dtype=np.uint8, # Data type
            ) # End of rasterize

        if self.band_layout is not None: # If indices are requested
            indices = compute_sentinel2_indices( # Compute Sentinel-2 indices
                image, self.band_layout, self.reflectance_scale # Pass arguments
            ) # End of compute
            image = np.concatenate( # Concatenate bands and indices
                [image.astype(np.float32, copy=False), indices], axis=0 # Stack along channel axis
            ) # End of concatenate

        image = resize( # Resize image
            image.transpose(1, 2, 0), # Transpose to (H, W, C) for resize
            self.target_size + (image.shape[0],), # Target dimensions
            mode="reflect", # Use reflect mode
            anti_aliasing=True, # Enable anti-aliasing
        ).transpose(2, 0, 1).astype(np.float32) # Transpose back to (C, H, W) and convert to float32
        image = percentile_normalize(image, self.norm_low, self.norm_high) # Normalize image

        mask = resize(mask, self.target_size, order=0, anti_aliasing=False).astype(np.uint8) # Resize mask
        mask = mask[None, :, :] # Add channel dimension to mask

        return torch.from_numpy(image), torch.from_numpy(mask) # Return tensors


def match_raster_shapefile( # Function to match rasters with shapefiles
    image_base_dir: str | Path, # Directory for images
    label_base_dir: str | Path, # Directory for labels
) -> List[Tuple[str, str]]: # Returns list of matched pairs
    """Match each GeoTIFF in ``image_base_dir`` to a spatially-overlapping shapefile.

    A shapefile is considered a match if (a) its filename contains the same
    YYYYMMDD date as the raster (when both have one) and (b) any of its
    geometries intersect the raster's bounding box. Shapefiles are read once
    and cached so the cost is linear in the number of label files.

    Returns a list of ``(image_path, label_path)`` pairs.
    """ # Docstring for match_raster_shapefile
    image_files = sorted(glob.glob(os.path.join(str(image_base_dir), "**", "*.tif"), recursive=True)) # Find all TIF files
    label_files = sorted(glob.glob(os.path.join(str(label_base_dir), "**", "*.shp"), recursive=True)) # Find all SHP files

    label_dates: Dict[str, Optional[str]] = {p: extract_date_string(p) for p in label_files} # Map label files to dates
    gdf_cache: Dict[str, Optional[gpd.GeoDataFrame]] = {} # Cache for GDFs

    def load_gdf(path: str) -> Optional[gpd.GeoDataFrame]: # Helper to load GDF with caching
        if path not in gdf_cache: # If not in cache
            try: # Try loading
                gdf = gpd.read_file(path) # Read shapefile
                if gdf.empty: # If empty
                    gdf_cache[path] = None # Cache as None
                else: # If not empty
                    gdf_cache[path] = gdf # Cache GDF
            except Exception as exc: # On error
                print(f"Warning: Could not load shapefile {path}: {exc}", flush=True) # Print warning
                gdf_cache[path] = None # Cache as None
        return gdf_cache[path] # Return cached GDF

    pairs: List[Tuple[str, str]] = [] # Initialize matched pairs list
    for idx, img_path in enumerate(image_files, start=1): # Iterate through image files
        print( # Print progress
            f"[{idx}/{len(image_files)}] Scanning {os.path.basename(img_path)}...", # Progress message
            flush=True, # Flush output
        ) # End of print
        img_date = extract_date_string(img_path) # Get image date
        try: # Try processing raster
            with rasterio.open(img_path) as src: # Open raster
                img_polygon = box(*src.bounds) # Create bounding box polygon
                img_crs = src.crs # Get CRS
        except Exception as exc: # On error
            print(f"Warning: Could not process raster {img_path}: {exc}", flush=True) # Print warning
            continue # Skip to next image

        for label_path in label_files: # Iterate through labels
            label_date = label_dates[label_path] # Get label date
            if img_date and label_date and img_date != label_date: # If dates mismatch
                continue # Skip to next label

            gdf = load_gdf(label_path) # Load GDF
            if gdf is None: # If GDF is invalid
                continue # Skip to next label

            if gdf.crs != img_crs: # If CRS doesn't match
                gdf = gdf.to_crs(img_crs) # Reproject GDF

            if gdf.geometry.intersects(img_polygon).any(): # If spatial intersection exists
                pairs.append((img_path, label_path)) # Add pair to list
                print( # Print match message
                    f"Matched: {os.path.basename(img_path)} <-> {os.path.basename(label_path)}", # Match info
                    flush=True, # Flush output
                ) # End of print
                break # Stop searching labels for this image
        else: # If no labels matched the image
            print(f"No overlapping shapefile found for: {os.path.basename(img_path)}", flush=True) # Print warning

    print(f"\nTotal matched pairs: {len(pairs)}", flush=True) # Print final count
    return pairs # Return matched pairs


def collect_tile_windows( # Function to create tile windows
    pairs: Sequence[Tuple[str, str]], # Matched pairs
    tile_size: int, # Desired tile size
) -> Tuple[ # Return tuple of several lists
    List[str],
    List[str],
    List[List[int]],
    List[Tuple[int, Window]],
    List[int],
    int,
    List[gpd.GeoDataFrame],
]: # End of return type
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
    """ # Docstring for collect_tile_windows
    image_paths: List[str] = [] # List of image paths
    label_paths: List[str] = [] # List of label paths
    gdfs_per_image: List[gpd.GeoDataFrame] = [] # List of GDFs
    bands_per_image: List[List[int]] = [] # List of band indices per image
    burn_windows: List[Tuple[int, Window]] = [] # List of positive windows
    unburn_windows: List[Tuple[int, Window]] = [] # List of negative windows
    max_channels = 0 # Tracker for max channels found

    # Cache for loaded and reprojected GeoDataFrames
    # Key: (label_path, img_crs)
    gdf_cache: Dict[Tuple[str, str], gpd.GeoDataFrame] = {} # Cache for GDFs

    for image_path, label_path in pairs: # Iterate through pairs
        with rasterio.open(image_path) as src: # Open raster
            height, width = src.shape # Get dimensions
            band_indices = list(range(1, src.count + 1)) # Get 1-indexed band indices
            img_transform = src.transform # Get transform
            img_crs = src.crs # Get CRS

        cache_key = (str(label_path), str(img_crs)) # Create cache key
        if cache_key in gdf_cache: # If in cache
            gdf = gdf_cache[cache_key] # Retrieve GDF
        else: # Otherwise load and reproject
            gdf = gpd.read_file(label_path) # Read shapefile
            if gdf.crs != img_crs: # If CRS mismatch
                gdf = gdf.to_crs(img_crs) # Reproject
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()] # Filter valid geometries
            gdf_cache[cache_key] = gdf # Store in cache

        if gdf.empty: # If no geometries left
            continue # Skip image

        windows = [ # Generate tiling windows
            Window(j, i, tile_size, tile_size) # Create window
            for i in range(0, height, tile_size) # Iterate rows
            for j in range(0, width, tile_size) # Iterate columns
            if i + tile_size <= height and j + tile_size <= width # Ensure within bounds
        ] # End of window generation

        sindex = gdf.sindex # Get spatial index
        image_index = len(image_paths) # Get current image index
        for window in windows: # Iterate through windows
            left, bottom, right, top = window_bounds(window, img_transform) # Get window bounds
            tile_poly = box(left, bottom, right, top) # Create polygon for window
            if list(sindex.query(tile_poly, predicate="intersects")): # Check intersection
                burn_windows.append((image_index, window)) # Add to positive windows
            else: # If no intersection
                unburn_windows.append((image_index, window)) # Add to negative windows

        image_paths.append(image_path) # Append image path
        label_paths.append(label_path) # Append label path
        gdfs_per_image.append(gdf) # Append GDF
        bands_per_image.append(band_indices) # Append bands
        max_channels = max(max_channels, len(band_indices)) # Update max channels

    all_windows = burn_windows + unburn_windows # Combine all windows
    tile_labels = [1] * len(burn_windows) + [0] * len(unburn_windows) # Create labels
    return image_paths, label_paths, bands_per_image, all_windows, tile_labels, max_channels, gdfs_per_image # Return results


def _pad_bands(bands_per_image: Sequence[Sequence[int]], max_channels: int) -> List[List[int]]: # Function to pad band lists
    """Pad shorter band lists by repeating the last band so every entry has ``max_channels`` indices.""" # Docstring for _pad_bands
    padded: List[List[int]] = [] # Initialize padded list
    for b in bands_per_image: # Iterate through band lists
        b = list(b) # Convert to list
        if len(b) < max_channels: # If list is shorter than max
            b = b + [b[-1]] * (max_channels - len(b)) # Pad with last element
        padded.append(b[:max_channels]) # Append truncated/padded list
    return padded # Return padded list


def build_dataloaders( # Function to build DataLoaders
    config: TrainingConfig, # Training configuration
    pairs: Sequence[Tuple[str, str]], # Matched raster/label pairs
    band_layout: Optional[Dict[str, int]] = None, # Optional band layout
    reflectance_scale: float = 10000.0, # Reflectance scale
) -> Tuple[DataLoader, DataLoader, int]: # Return train/val loaders and input channels
    """Build stratified train/validation DataLoaders from matched raster/label pairs.

    When ``band_layout`` is supplied, each tile is augmented with the four
    Sentinel-2 indices (NDVI, NDWI, SAVI, BAIS2) computed by
    :func:`compute_sentinel2_indices`, and the returned channel count reflects
    the added bands so the model can be sized accordingly.

    Returns ``(train_loader, val_loader, model_input_channels)``. Raises
    ``RuntimeError`` when no burned tiles are found (training would be degenerate).
    """ # Docstring for build_dataloaders
    image_paths, label_paths, bands_per_image, windows, tile_labels, max_channels, gdfs_per_image = collect_tile_windows( # Collect tiles
        pairs, config.tile_size # Pass pairs and tile size
    ) # End of collection

    if not windows: # If no windows found
        raise RuntimeError("No tiles collected from the matched pairs.") # Raise error
    if 1 not in tile_labels: # If no burned tiles found
        raise RuntimeError( # Raise error
            "No burned tiles found. Check your shapefiles and spatial overlap logic." # Error message
        ) # End of RuntimeError

    print(f" -> Total dataset size (no tiles discarded): {len(windows)} tiles", flush=True) # Print size

    train_windows, val_windows, _, _ = train_test_split( # Split into train and val
        windows, # Windows list
        tile_labels, # Labels list
        test_size=config.val_split, # Validation split ratio
        random_state=config.random_seed, # Random seed
        stratify=tile_labels, # Stratify by labels
    ) # End of split

    padded_bands = _pad_bands(bands_per_image, max_channels) # Pad band lists

    print(" -> Pre-rasterizing shapefiles to image grids...", flush=True) # Print progress
    packed_masks = [] # Initialize masks
    image_widths = [] # Initialize widths
    for idx, (image_path, label_path) in enumerate(zip(image_paths, label_paths)): # Iterate through pairs
        gdf = gdfs_per_image[idx] # Get GDF
        with rasterio.open(image_path) as src: # Open raster
            out_shape = src.shape # Get shape
            transform = src.transform # Get transform
            width = src.width # Get width
        image_widths.append(width) # Store width

        valid_geoms = gdf.geometry.buffer(0) # Fix geometries
        shapes = [(mapping(geom), 1) for geom in valid_geoms if not geom.is_empty] # Prepare shapes
        if shapes: # If shapes exist
            mask = rasterize( # Rasterize
                shapes, # Shapes
                out_shape=out_shape, # Shape
                transform=transform, # Transform
                fill=0, # Fill
                all_touched=True, # All touched
                dtype=np.uint8, # Dtype
            ) # End of rasterize
        else: # If no shapes
            mask = np.zeros(out_shape, dtype=np.uint8) # Create zero mask
        packed_mask = np.packbits(mask, axis=-1) # Pack bits for efficiency
        packed_masks.append(packed_mask) # Store mask
    print(" -> Pre-rasterization completed successfully.", flush=True) # Print progress

    train_dataset = SegmentationDataset( # Create training dataset
        image_paths, # Image paths
        label_paths, # Label paths
        train_windows, # Train windows
        target_size=config.target_size, # Target size
        bands=padded_bands, # Bands
        normalization_percentiles=config.normalization_percentiles, # Normalization
        band_layout=band_layout, # Layout
        reflectance_scale=reflectance_scale, # Scale
        packed_masks=packed_masks, # Packed masks
        image_widths=image_widths, # Widths
    ) # End of training dataset
    val_dataset = SegmentationDataset( # Create validation dataset
        image_paths, # Image paths
        label_paths, # Label paths
        val_windows, # Val windows
        target_size=config.target_size, # Target size
        bands=padded_bands, # Bands
        normalization_percentiles=config.normalization_percentiles, # Normalization
        band_layout=band_layout, # Layout
        reflectance_scale=reflectance_scale, # Scale
        packed_masks=packed_masks, # Packed masks
        image_widths=image_widths, # Widths
    ) # End of validation dataset

    train_loader = DataLoader( # Create training loader
        train_dataset, # Dataset
        batch_size=config.batch_size, # Batch size
        shuffle=True, # Shuffle enabled
        num_workers=config.num_workers, # Workers
        pin_memory=config.pin_memory, # Pin memory
    ) # End of train loader
    val_loader = DataLoader( # Create validation loader
        val_dataset, # Dataset
        batch_size=config.batch_size, # Batch size
        shuffle=False, # Shuffle disabled
        num_workers=config.num_workers, # Workers
        pin_memory=config.pin_memory, # Pin memory
    ) # End of val loader
    model_input_channels = max_channels + ( # Calculate input channels
        len(SENTINEL2_INDEX_NAMES) if band_layout else 0 # Add indices if layout present
    ) # End of calculation
    return train_loader, val_loader, model_input_channels # Return loaders and channels