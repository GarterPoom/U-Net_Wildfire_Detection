"""Raster file discovery and output‑path generation utilities.

This module provides functions to locate GeoTIFF files on disk and to compute the
output file path for mask and probability products, optionally preserving the
input directory hierarchy.
"""

from __future__ import annotations

import glob  # pattern matching for file discovery
import os  # path manipulation
from pathlib import Path  # convenient path handling
from typing import List  # list type hint


def get_tiff_files(input_path: str | Path, recursive: bool = True) -> List[str]:
    """Return a sorted list of ``.tif`` and ``.tiff`` files under ``input_path``.

    Args:
        input_path: Path (string or :class:`pathlib.Path`) to the directory (or file) to search.
        recursive: If True, descend into sub‑directories; otherwise only the top level is examined.

    Returns:
        A list of absolute file paths sorted alphabetically.
    """
    input_path = str(input_path)  # ensure string for glob patterns
    pattern = "**/*.tif" if recursive else "*.tif"  # set glob pattern based on recursion flag

    # Gather all .tif files (including nested ones if requested)
    files = glob.glob(os.path.join(input_path, pattern), recursive=recursive)

    # Also gather .tiff files using a separate pattern to avoid missing extensions
    files += glob.glob(os.path.join(input_path, pattern.replace(".tif", ".tiff")), recursive=recursive)

    # Filter out anything that is not an actual file and sort for deterministic order
    return sorted(f for f in files if os.path.isfile(f))


def generate_output_path(
    input_path: str,
    input_base_dir: str,
    output_dir: str,
    suffix: str = "_predicted_mask",
    preserve_structure: bool = False,
) -> str:
    """Construct the output file path, optionally mirroring the input directory structure.

    Args:
        input_path: Full path to the source raster.
        input_base_dir: Base directory used for computing relative paths (either the image dir or its parent).
        output_dir: Destination root folder where the result will be written.
        suffix: Suffix appended to the filename before the extension (default ``"_predicted_mask"``).
        preserve_structure: If True, create sub‑folders inside ``output_dir`` that replicate the
            relative location of the input file.

    Returns:
        Full path (as string) for the generated output raster.
    """
    os.makedirs(output_dir, exist_ok=True)  # ensure target directory exists

    base_name = os.path.splitext(os.path.basename(input_path))[0]  # filename without extension

    if preserve_structure:
        # Compute a relative path from the input base to the file's containing folder.
        rel_path = os.path.relpath(os.path.dirname(input_path), input_base_dir)
        output_subdir = os.path.join(output_dir, rel_path)  # build mirrored sub‑folder name
        os.makedirs(output_subdir, exist_ok=True)          # create it if missing
        return os.path.join(output_subdir, f"{base_name}{suffix}.tif")
    # Simple case: write directly into the output root folder.
    return os.path.join(output_dir, f"{base_name}{suffix}.tif")
