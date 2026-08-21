"""Parsing utilities for the ``--band_layout`` JSON argument.

This module converts a JSON string that maps Sentinel‑2 band names to indices
into a 0‑indexed dictionary expected by downstream functions.
"""

from __future__ import annotations

import json  # standard library JSON parser
from typing import Dict, Optional  # type hints for dict and optional return


def parse_band_layout(raw: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse a JSON band‑layout string into a 0‑indexed dict.

    The function accepts either 0‑indexed or 1‑indexed band positions in the
    source JSON and normalizes to 0‑indexing (the format expected by
    :func:`compute_sentinel2_indices`). A layout is considered 1‑indexed if its
    minimum value is ``>= 1`` and it does not contain ``0``.

    Args:
        raw: JSON string mapping band names (e.g., "B03") to integer indices,
            or ``None`` to indicate no special layout.

    Returns:
        A dictionary with band names as keys and 0‑indexed integers as values,
        or ``None`` if ``raw`` is falsy.
    """
    # If the input is falsy (None or empty string) we return None to signal
    # “no custom band layout”.
    if raw is None:
        return None

    # Parse the JSON text into a Python dictionary; raise ValueError on malformed input.
    layout = json.loads(raw)

    # Validate that the parsed object is a dict; otherwise the input is invalid.
    if not isinstance(layout, dict):
        raise ValueError("--band_layout must be a JSON object mapping band names to indices")

    # Convert all keys and values to strings then integers for robust handling.
    layout = {str(k): int(v) for k, v in layout.items()}

    # Detect 1‑indexed layouts: if the smallest index is >= 1 we shift everything down by one.
    if min(layout.values()) >= 1:
        layout = {k: v - 1 for k, v in layout.items()}

    return layout
