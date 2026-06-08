"""Parsing utilities for the ``--band_layout`` JSON argument."""

from __future__ import annotations

import json
from typing import Dict, Optional


def parse_band_layout(raw: Optional[str]) -> Optional[Dict[str, int]]:
    """Parse a JSON band-layout string into a 0-indexed dict.

    Accepts either 0-indexed or 1-indexed band positions in the source JSON and
    normalizes to 0-indexed (which is what :func:`compute_sentinel2_indices`
    expects). A layout is considered 1-indexed if its minimum value is ``>= 1``
    and it does not contain ``0``.

    Example input: ``'{"B03":2,"B04":3,"B06":5,"B07":6,"B08":7,"B8A":8,"B12":10}'``.
    """
    if raw is None:
        return None
    layout = json.loads(raw)
    if not isinstance(layout, dict):
        raise ValueError("--band_layout must be a JSON object mapping band names to indices")
    layout = {str(k): int(v) for k, v in layout.items()}
    if min(layout.values()) >= 1:
        layout = {k: v - 1 for k, v in layout.items()}
    return layout
