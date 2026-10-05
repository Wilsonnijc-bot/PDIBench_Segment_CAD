"""Image-only common support from existing link5 masks and frozen 2D tracks.

Borrows the object-occlusion pipeline's visibility/mask sampling convention.
No geometry, RGB, descriptor, or anomaly score is registered or warped.
"""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

from dataclasses import asdict, dataclass
import warnings

import cv2
import numpy as np

from object.preprocessing.occlusion.occlusion import sample


@dataclass(frozen=True)
class Config:
    erosion_px: int = 2
    visibility_mode: str = 'smaller_view_anchor'
    track_quality_gate: bool = False
    correspondence_px: float = 6.0
    minimum_tracks: int = 8
    minimum_inlier_fraction: float = .6
    minimum_hull_fraction: float = .15
    minimum_common_fraction: float = .10
    minimum_pixels: int = 256
    depth_window: int = 5
    depth_mad_multiplier: float = 8.0
    depth_relative_threshold: float = .05
    depth_minimum_neighbors: int = 8
    minimum_depth_retained_fraction: float = .5

    def validate(self):
        if self.visibility_mode not in ('strict_hulls', 'smaller_view_anchor'):
            raise ValueError('unknown visibility_mode')
        if self.erosion_px not in (2, 3, 4):
            raise ValueError('erosion_px must be 2, 3, or 4 source-image pixels')
        if self.depth_window < 3 or self.depth_window % 2 != 1:
            raise ValueError('depth_window must be odd and >=3')
        if self.minimum_tracks < 3 or self.minimum_pixels < 128:
            raise ValueError('insufficient minimum correspondence/cloud support')
        if self.correspondence_px <= 0 or self.depth_mad_multiplier <= 0 or self.depth_relative_threshold <= 0:
            raise ValueError('thresholds must be positive')
        for name in ('minimum_inlier_fraction', 'minimum_hull_fraction', 'minimum_common_fraction', 'minimum_depth_retained_fraction'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(name + ' must be in (0,1]')
        return self


def erode(mask, pixels):
    # Zero padding erodes pixels at the camera boundary as well as silhouettes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pixels+1, 2*pixels+1))
    return cv2.erode(np.asarray(mask, np.uint8), kernel,
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def select_frames(masks, config):
    raw = masks.sum(axis=(1, 2)).astype(int)
    areas = np.array([erode(m, config.erosion_px).sum() for m in masks], dtype=int)
    valid = areas >= config.minimum_pixels
    base = int(np.flatnonzero(valid)[0]) if valid.any() else None
    count = len(masks)
    edges = np.linspace(.1*count, count, 11)
    selected = []
    for b in range(10):
        candidates = np.flatnonzero(valid & (np.arange(count) >= edges[b]) & (np.arange(count) < edges[b+1]))
        candidates = candidates[candidates != base]
        t = int(candidates[np.argmax(raw[candidates])]) if len(candidates) else None
        selected.append(dict(temporal_bin=b, bin_interval_frames=[float(edges[b]), float(edges[b+1])],
                             test_frame_id=t, raw_mask_area=int(raw[t]) if t is not None else 0,
                             eroded_mask_area=int(areas[t]) if t is not None else 0,
                             failure_reason=None if t is not None else 'no valid saved Link5 mask in temporal bin'))
    return base, selected, raw, areas


def hull_mask(points, shape):
    result = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(result, cv2.convexHull(np.rint(points).astype(np.int32)), 1)
    return result.astype(bool)


def common_visible(reference, test, reference_tracks, test_tracks, reference_visibility, test_visibility, config):
    if reference.shape != test.shape:
        raise ValueError('reference/test masks must share the original source pixel grid')
    eligible = (reference_visibility & test_visibility & sample(reference, reference_tracks)
                & sample(test, test_tracks) & np.isfinite(reference_tracks).all(axis=1)
                & np.isfinite(test_tracks).all(axis=1))
    ids = np.flatnonzero(eligible)
    # Tracks still estimate the image-only mask map. They no longer veto a
    # usable pair by an arbitrary consensus percentage or hull-coverage quota.
    # Two distinct correspondences are the mathematical minimum for similarity.
    needed = config.minimum_tracks if config.track_quality_gate else 2
    if len(ids) < needed:
        raise ValueError(f'only {len(ids)} mutually visible in-mask saved tracks; need {needed} to estimate the mask map')
    a, b = reference_tracks[ids], test_tracks[ids]
    cv2.setRNGSeed(0)
    transform, flags = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
        ransacReprojThreshold=config.correspondence_px, maxIters=3000, confidence=.99, refineIters=10)
    if transform is None or not np.isfinite(transform).all():
        raise ValueError('no stable global 2D similarity correspondence')
    inliers = flags.ravel().astype(bool)
    fraction = float(inliers.mean())
    if config.track_quality_gate and (inliers.sum() < config.minimum_tracks or fraction < config.minimum_inlier_fraction):
        raise ValueError(f'weak 2D correspondence: {inliers.sum()}/{len(ids)} inliers ({fraction:.3f})')
    scale = float(np.linalg.norm(transform[:, 0]))
    if not .25 <= scale <= 4:
        raise ValueError(f'implausible image correspondence scale {scale:.3f}')
    inverse = cv2.invertAffineTransform(transform)
    residual = np.linalg.norm(a @ transform[:, :2].T + transform[:, 2] - b, axis=1)
    # Hull coverage checks whether tracks support a meaningful portion of each
    # mask. The anchored mode allows extrapolation to observed mask edges;
    # neither mode claims dense 3D co-visibility.
    ra = reference & hull_mask(a[inliers], reference.shape)
    tb = test & hull_mask(b[inliers], test.shape)
    fractions = [float(ra.sum()/max(1, reference.sum())), float(tb.sum()/max(1, test.sum()))]
    if config.track_quality_gate and min(fractions) < config.minimum_hull_fraction:
        raise ValueError(f'saved tracks cover too little mask area: reference/test hull fractions {fractions}')
    h, w = reference.shape
    warp = lambda mask, mat: cv2.warpAffine(mask.astype(np.uint8), mat, (w, h),
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if config.visibility_mode == 'strict_hulls':
        rcommon = ra & warp(tb, inverse)
        tcommon = tb & warp(ra, transform)
        rcommon &= warp(tcommon, inverse)
        tcommon &= warp(rcommon, transform)
        anchor = None
    else:
        # Compare areas in the same image scale. Pixel counts alone mistake a
        # camera zoom for greater surface visibility. Track hulls remain a
        # reliability check above, but do not crop reliable mask silhouettes.
        anchor = 'reference' if reference.sum()*scale**2 <= test.sum() else 'test'
        if anchor == 'reference':
            mapped = warp(reference, transform)
            target = mapped & test
            # Prune only anchor pixels whose mapped location is unavailable in
            # the actual destination mask. No repeated hull intersections.
            rcommon = reference & warp(test, inverse)
            tcommon = target
        else:
            mapped = warp(test, inverse)
            target = mapped & reference
            tcommon = test & warp(reference, transform)
            rcommon = target
    counts = [int(rcommon.sum()), int(tcommon.sum())]
    common_fraction = [counts[0]/max(1, reference.sum()), counts[1]/max(1, test.sum())]
    if min(counts) < config.minimum_pixels or min(common_fraction) < config.minimum_common_fraction:
        raise ValueError(f'insufficient symmetric common support: pixels {counts}, fractions {common_fraction}')
    method = ('saved-track-supported symmetric hull intersection; global 2D similarity'
              if anchor is None else 'smaller-visible-view anchored mask mapping; global 2D similarity')
    return rcommon, tcommon, dict(method=method, visibility_mode=config.visibility_mode,
        track_quality_gate=config.track_quality_gate,
        track_quality_policy='count/inlier-percentage/hull-coverage vetoes enabled' if config.track_quality_gate else 'diagnostic only; no count/inlier-percentage/hull-coverage vetoes',
        anchor_role=anchor, reference_area_in_test_pixels=float(reference.sum()*scale**2),
        test_area_in_test_pixels=int(test.sum()), hull_used_for_cropping=anchor is None,
        borrowed_code='object.preprocessing.occlusion.occlusion.sample',
        reference_to_test_affine=transform.tolist(), test_to_reference_affine=inverse.tolist(),
        candidate_track_count=len(ids), inlier_track_count=int(inliers.sum()),
        inlier_fraction=fraction, inlier_track_ids=ids[inliers].tolist(),
        median_inlier_residual_px=float(np.median(residual[inliers])), image_scale=scale,
        hull_fractions=fractions, common_pixel_counts=counts, common_fractions=common_fraction,
        geometry_registration='none; transform used exclusively to intersect original-grid masks',
        limitation='2D tracked support estimates common surface; self-occlusion/view changes are not proven by a dense 3D visibility model')


def depth_filter(depth, mask, config):
    """Reject isolated spikes; do not replace depths or smooth XYZ."""
    valid = np.asarray(mask, bool) & np.isfinite(depth) & (depth > 0)
    half = config.depth_window//2
    local = np.where(valid, depth, np.nan).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(local, half, constant_values=np.nan), (config.depth_window, config.depth_window))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(windows, axis=(-2, -1))
        mad = np.nanmedian(np.abs(windows-median[..., None, None]), axis=(-2, -1))
    threshold = np.maximum(config.depth_relative_threshold*np.abs(median), config.depth_mad_multiplier*1.4826*mad)
    # A genuine region boundary has other similarly deep neighbors. Require the
    # offending center to be supported by <=2 pixels (including itself).
    same_depth = (np.abs(windows-depth[..., None, None]) <= threshold[..., None, None]).sum(axis=(-2, -1))
    enough = np.isfinite(windows).sum(axis=(-2, -1)) >= config.depth_minimum_neighbors
    spikes = valid & enough & (np.abs(depth-median) > threshold) & (same_depth <= 2)
    rejected = np.asarray(mask, bool) & (~valid | spikes)
    final = valid & ~spikes
    info = dict(input_mask_pixel_count=int(mask.sum()), finite_positive_depth_count=int(valid.sum()),
        invalid_depth_pixel_count=int((mask & ~valid).sum()), isolated_spike_pixel_count=int(spikes.sum()),
        rejected_depth_pixel_count=int(rejected.sum()), final_valid_pixel_count=int(final.sum()),
        retained_fraction=float(final.sum()/max(1, mask.sum())), window=config.depth_window,
        threshold=f'max({config.depth_relative_threshold} * local median, {config.depth_mad_multiplier} * 1.4826 * local MAD)',
        maximum_same_depth_neighbors=2, depth_smoothing=False)
    return final, rejected, info


def restore_cleaned_mask(source_mask, rejected_grid, bbox, canvas_hw):
    """Lift rejection flags, not a resampled silhouette, back to source pixels.

    A mask→depth-grid→mask round trip loses boundary pixels even with no depth
    rejection. Keeping the source mask and lifting only actual rejections avoids
    introducing an undocumented second erosion or false rejection overlay.
    """
    x0,y0,x1,y1 = bbox
    height,width = canvas_hw
    rejected = cv2.resize(rejected_grid.astype(np.uint8), (width,height),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    source_rejected = np.zeros_like(source_mask, dtype=bool)
    source_rejected[y0:y1,x0:x1] = rejected[:y1-y0,:x1-x0] & source_mask[y0:y1,x0:x1]
    # MegaSaM can upsample a small crop. Nearest-neighbor downsampling of the
    # rejection image can then miss a single flagged depth cell entirely. Mark
    # the actual source-mask pixel that supplied each flagged native-grid cell.
    yy,xx = np.where(rejected_grid)
    sy = np.minimum((yy*height/rejected_grid.shape[0]).astype(int), y1-y0-1)+y0
    sx = np.minimum((xx*width/rejected_grid.shape[1]).astype(int), x1-x0-1)+x0
    source_rejected[sy,sx] |= source_mask[sy,sx]
    return source_mask & ~source_rejected, source_rejected
