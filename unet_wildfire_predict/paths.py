"""Raster file discovery and output-path generation."""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List


def get_tiff_files(input_path: str | Path, recursive: bool = True) -> List[str]:
    """Return a sorted list of ``.tif`` / ``.tiff`` files under ``input_path``."""
    input_path = str(input_path)
    pattern = "**/*.tif" if recursive else "*.tif"
    files = glob.glob(os.path.join(input_path, pattern), recursive=recursive)
    files += glob.glob(os.path.join(input_path, pattern.replace(".tif", ".tiff")), recursive=recursive)
    return sorted(f for f in files if os.path.isfile(f))


def generate_output_path(
    input_path: str,
    input_base_dir: str,
    output_dir: str,
    suffix: str = "_predicted_mask",
    preserve_structure: bool = False,
) -> str:
    """Build the output file path, optionally mirroring the input directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    if preserve_structure:
        rel_path = os.path.relpath(os.path.dirname(input_path), input_base_dir)
        output_subdir = os.path.join(output_dir, rel_path)
        os.makedirs(output_subdir, exist_ok=True)
        return os.path.join(output_subdir, f"{base_name}{suffix}.tif")
    return os.path.join(output_dir, f"{base_name}{suffix}.tif")
