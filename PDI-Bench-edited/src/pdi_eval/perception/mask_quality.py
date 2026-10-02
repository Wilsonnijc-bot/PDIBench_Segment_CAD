"""Small area-only validity guard for table-sized link7 segmentations."""
from __future__ import annotations

import numpy as np


MAX_LINK7_AREA_FRACTION = .25
MASK_QUALITY_METHOD = "link7-frame-area-v1"


def link7_area_validity(masks, max_area_fraction=MAX_LINK7_AREA_FRACTION):
    """Return validity, pixel areas and image-area fractions for [T,H,W] masks.

    Equality fails the cutoff. This only rejects oversized masks; passing is
    not evidence that a segmentation is otherwise correct.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or min(masks.shape) < 1:
        raise ValueError("Link7 masks must have nonempty [T,H,W] dimensions")
    if not 0 < max_area_fraction <= 1:
        raise ValueError("Maximum link7 area fraction must lie in (0, 1]")
    areas = masks.sum(axis=(1, 2))
    fractions = areas / (masks.shape[1] * masks.shape[2])
    return fractions < max_area_fraction, areas, fractions
