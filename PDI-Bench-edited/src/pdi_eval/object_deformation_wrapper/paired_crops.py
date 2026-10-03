"""Remove unequal transparent margins without resampling object pixels."""
from __future__ import annotations

import numpy as np

METHOD = "shared-canvas-native-pixels-v1"


def normalize_pair(current: np.ndarray, reference: np.ndarray):
    """Trim alpha-only margins and center both objects on one canvas size.

    The common canvas makes the browser and DINO shorter-edge resize use the
    same factor for both images. Only integer translations are applied: visible
    RGB, alpha, shape, native pixel area and aspect ratio remain unchanged. This
    does not estimate physical scale or revise the frame-zero correspondence.
    """
    images, roles = [], {}
    for role, image in (("query", current), ("reference", reference)):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError("Expected uint8 H×W×4 RGBA crops")
        yy, xx = np.where(image[:, :, 3] > 0)
        if not len(xx):
            raise ValueError("Cannot normalize an empty visible crop")
        x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)
        trimmed = image[y0:y1, x0:x1].copy()
        trimmed[trimmed[:, :, 3] == 0] = 0
        images.append(trimmed)
        roles[role] = {"source_size_wh": [image.shape[1], image.shape[0]],
                       "content_bbox_xyxy": [x0, y0, x1, y1],
                       "visible_pixel_count": int(len(xx))}
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    result = []
    for role, image in zip(("query", "reference"), images):
        y = (height - image.shape[0]) // 2
        x = (width - image.shape[1]) // 2
        canvas = np.zeros((height, width, 4), np.uint8)
        canvas[y:y+image.shape[0], x:x+image.shape[1]] = image
        roles[role]["content_offset_xy"] = [x, y]
        result.append(canvas)
    return result[0], result[1], {"method": METHOD, "canvas_size_wh": [width, height],
                                  "resampling": "none", "roles": roles}
