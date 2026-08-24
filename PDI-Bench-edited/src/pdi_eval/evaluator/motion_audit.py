import numpy as np
from typing import Tuple, Optional

from ..geometry.se3 import (
    invert_rigid_transform,
    rigid_transform_valid,
    rotation_angle,
    scale_rigid_increment,
)


FOUNDATION_POSE_DISCONTINUITY_METHOD = "foundationpose-pose-discontinuity-v1"


def _stabilize_derived_rigid_transform(transform: np.ndarray) -> np.ndarray:
    """Project floating-point composition drift back onto SE(3)."""
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("derived transform must be a finite 4x4 matrix")
    left, _, right = np.linalg.svd(value[:3, :3])
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = value[:3, 3]
    return result


def compose_metric_world_link_poses(
    T_W_from_C: np.ndarray,
    T_C_from_L: np.ndarray,
    depth_scale: float,
) -> np.ndarray:
    """Compose object poses after applying the shared metric scale to camera motion."""
    camera_poses = np.asarray(T_W_from_C, dtype=np.float64)
    link_poses = np.asarray(T_C_from_L, dtype=np.float64)
    if camera_poses.shape != link_poses.shape or camera_poses.ndim != 3:
        raise ValueError("camera and link poses must have matching shape (T,4,4)")
    if camera_poses.shape[1:] != (4, 4):
        raise ValueError("camera and link poses must have shape (T,4,4)")
    if not np.isfinite(depth_scale) or depth_scale <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    result = np.full_like(link_poses, np.nan, dtype=np.float64)
    for frame_index, (camera_pose, link_pose) in enumerate(
        zip(camera_poses, link_poses)
    ):
        if not rigid_transform_valid(camera_pose) or not rigid_transform_valid(link_pose):
            continue
        metric_camera_pose = camera_pose.copy()
        metric_camera_pose[:3, 3] *= depth_scale
        result[frame_index] = metric_camera_pose @ link_pose
    return result


def audit_foundation_pose_discontinuity(
    T_W_from_L: np.ndarray,
    frame_times_seconds: np.ndarray,
    pose_valid: np.ndarray,
    link_diameter: float,
    *,
    pose_objective: np.ndarray | None = None,
    pose_source: np.ndarray | None = None,
    translation_rate_threshold: float = 3.0,
    rotation_rate_threshold_degrees: float = 450.0,
    quality_threshold: float = 0.40,
    maximum_gap_factor: float = 2.0,
) -> dict:
    """Measure departure from constant body-frame velocity in SE(3)."""
    poses = np.asarray(T_W_from_L, dtype=np.float64)
    times = np.asarray(frame_times_seconds, dtype=np.float64)
    valid = np.asarray(pose_valid, dtype=bool)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("T_W_from_L must have shape (T,4,4)")
    if times.shape != (len(poses),) or valid.shape != (len(poses),):
        raise ValueError("timestamps and pose_valid must have shape (T,)")
    if len(poses) and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0)):
        raise ValueError("frame timestamps must be finite and strictly increasing")
    if not np.isfinite(link_diameter) or link_diameter <= 0.0:
        raise ValueError("link_diameter must be finite and positive")
    if translation_rate_threshold <= 0.0 or rotation_rate_threshold_degrees <= 0.0:
        raise ValueError("pose-discontinuity thresholds must be positive")
    if not 0.0 < maximum_gap_factor:
        raise ValueError("maximum_gap_factor must be positive")
    quality = (
        np.zeros(len(poses), dtype=np.float64)
        if pose_objective is None
        else np.asarray(pose_objective, dtype=np.float64)
    )
    sources = (
        np.full(len(poses), 2, dtype=np.uint8)
        if pose_source is None
        else np.asarray(pose_source)
    )
    if quality.shape != (len(poses),) or sources.shape != (len(poses),):
        raise ValueError("pose_objective and pose_source must have shape (T,)")

    frame_count = len(poses)
    translation_ratio = np.full(frame_count, np.nan, dtype=np.float64)
    rotation_degrees = np.full(frame_count, np.nan, dtype=np.float64)
    translation_rate = np.full(frame_count, np.nan, dtype=np.float64)
    rotation_rate = np.full(frame_count, np.nan, dtype=np.float64)
    severity = np.full(frame_count, np.nan, dtype=np.float64)
    raw_translation_ratio = np.full(frame_count, np.nan, dtype=np.float64)
    raw_rotation_degrees = np.full(frame_count, np.nan, dtype=np.float64)
    computable = np.zeros(frame_count, dtype=bool)
    high_quality = np.zeros(frame_count, dtype=bool)
    event = np.zeros(frame_count, dtype=bool)
    physical_event = np.zeros(frame_count, dtype=bool)
    classification = np.full(frame_count, "unavailable", dtype="<U32")

    nominal_interval = (
        float(np.median(np.diff(times))) if frame_count > 1 else np.nan
    )
    for frame_index in range(1, frame_count):
        if not valid[frame_index - 1:frame_index + 1].all():
            continue
        if not (
            rigid_transform_valid(poses[frame_index - 1])
            and rigid_transform_valid(poses[frame_index])
        ):
            continue
        raw_delta = (
            invert_rigid_transform(poses[frame_index - 1]) @ poses[frame_index]
        )
        raw_translation_ratio[frame_index] = (
            np.linalg.norm(raw_delta[:3, 3]) / link_diameter
        )
        raw_rotation_degrees[frame_index] = np.degrees(
            rotation_angle(raw_delta[:3, :3])
        )

    for frame_index in range(2, frame_count):
        if not valid[frame_index - 2:frame_index + 1].all():
            classification[frame_index] = "pose_innovation_gap"
            continue
        triplet = poses[frame_index - 2:frame_index + 1]
        if not all(rigid_transform_valid(transform) for transform in triplet):
            classification[frame_index] = "invalid_rigid_transform"
            continue
        previous_interval = times[frame_index - 1] - times[frame_index - 2]
        current_interval = times[frame_index] - times[frame_index - 1]
        if (
            previous_interval > maximum_gap_factor * nominal_interval
            or current_interval > maximum_gap_factor * nominal_interval
        ):
            classification[frame_index] = "pose_innovation_gap"
            continue
        previous_delta = invert_rigid_transform(triplet[0]) @ triplet[1]
        if not rigid_transform_valid(previous_delta):
            previous_delta = _stabilize_derived_rigid_transform(previous_delta)
        predicted = triplet[1] @ scale_rigid_increment(
            previous_delta, current_interval / previous_interval
        )
        if not rigid_transform_valid(predicted):
            predicted = _stabilize_derived_rigid_transform(predicted)
        innovation = invert_rigid_transform(predicted) @ triplet[2]
        if not rigid_transform_valid(innovation):
            innovation = _stabilize_derived_rigid_transform(innovation)
        translation_ratio[frame_index] = (
            float(np.linalg.norm(innovation[:3, 3])) / link_diameter
        )
        rotation_degrees[frame_index] = float(
            np.degrees(rotation_angle(innovation[:3, :3]))
        )
        translation_rate[frame_index] = (
            translation_ratio[frame_index] / current_interval
        )
        rotation_rate[frame_index] = rotation_degrees[frame_index] / current_interval
        severity[frame_index] = max(
            translation_rate[frame_index] / translation_rate_threshold,
            rotation_rate[frame_index] / rotation_rate_threshold_degrees,
        )
        computable[frame_index] = True
        quality_triplet = quality[frame_index - 2:frame_index + 1]
        high_quality[frame_index] = bool(
            np.isfinite(quality_triplet).all()
            and np.all(quality_triplet <= quality_threshold)
        )
        event[frame_index] = bool(severity[frame_index] > 1.0)
        if not event[frame_index]:
            classification[frame_index] = "none"
        elif not high_quality[frame_index]:
            classification[frame_index] = "estimator_discontinuity"
        elif np.any(sources[frame_index - 2:frame_index + 1] != 2):
            classification[frame_index] = "estimator_reset"
        else:
            classification[frame_index] = "motion_discontinuity"
            physical_event[frame_index] = True

    aggregate_selector = computable & high_quality & (classification != "estimator_reset")
    aggregate_values = severity[aggregate_selector]
    return {
        "method": FOUNDATION_POSE_DISCONTINUITY_METHOD,
        "translation_rate_threshold": float(translation_rate_threshold),
        "rotation_rate_threshold_degrees": float(rotation_rate_threshold_degrees),
        "quality_threshold": float(quality_threshold),
        "translation_innovation_ratio": translation_ratio,
        "rotation_innovation_degrees": rotation_degrees,
        "translation_innovation_rate": translation_rate,
        "rotation_innovation_rate_degrees": rotation_rate,
        "severity": severity,
        "raw_translation_ratio": raw_translation_ratio,
        "raw_rotation_degrees": raw_rotation_degrees,
        "computable": computable,
        "high_quality": high_quality,
        "event": event,
        "physical_event": physical_event,
        "classification": classification,
        "valid_innovation_count": int(np.count_nonzero(aggregate_selector)),
        "event_count": int(np.count_nonzero(physical_event)),
        "event_rate": float(np.mean(physical_event[aggregate_selector]))
        if np.any(aggregate_selector)
        else None,
        "severity_median": float(np.median(aggregate_values))
        if len(aggregate_values)
        else None,
        "severity_p95": float(np.percentile(aggregate_values, 95))
        if len(aggregate_values)
        else None,
        "severity_max": float(np.max(aggregate_values))
        if len(aggregate_values)
        else None,
        "pose_discontinuity": bool(np.any(physical_event)),
    }


def _medfilt1d(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """1D median filter without scipy (numpy sliding window)."""
    pad = k // 2
    padded = np.pad(arr, pad, mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(windows, axis=1)


def audit_3d_trajectory_consistency(
    pointmaps: Optional[np.ndarray],
    masks: np.ndarray,
    fps: float = 24.0,
) -> np.ndarray:
    """3D kinematic trajectory audit in world coordinates.

    Samples foreground centroid from MegaSAM world point maps and scores
    motion smoothness (acceleration spikes + direction reversals).

    Compared with classic VP-based methods:
    - No dependence on h_pixel; orthogonal to scale term
    - Robust to camera motion (absolute position in world frame)
    - Works for straight and curved paths

    Args:
        pointmaps: (T, H, W, 3) world-coordinate point map (MegaSAM)
        masks:     (T, H, W) foreground masks (SAM2)
        fps:       frame rate for time normalization (m/s, m/s^2)

    Returns:
        (T-1,) trajectory residual sequence
    """
    if pointmaps is None or masks is None:
        return np.zeros(1)

    T = pointmaps.shape[0]
    if T < 3:
        return np.zeros(max(1, T - 1))

    # Validity: all-zero pointmaps means MegaSAM used fallback
    mask0 = masks[0]
    if mask0.shape[:2] != pointmaps.shape[1:3]:
        import cv2
        mask0 = cv2.resize(mask0.astype(np.uint8), (pointmaps.shape[2], pointmaps.shape[1]),
                           interpolation=cv2.INTER_NEAREST)
    fg_pts0 = pointmaps[0][mask0 > 0]
    if fg_pts0.shape[0] == 0 or np.mean(np.any(fg_pts0 != 0, axis=-1)) < 0.5:
        return np.zeros(T - 1)

    # Step 1: robust 3D centroid
    # - Median instead of mean to reduce depth bleeding at SAM2 mask boundaries
    #   (edges often leak background depth; median resists outliers)
    # - If mask is lost, keep previous centroid to avoid jumping to world origin
    world_traj = np.zeros((T, 3))
    last_valid = np.zeros(3)

    for t in range(T):
        m = masks[t]
        pm = pointmaps[t]
        if m.shape[:2] != pm.shape[:2]:
            import cv2
            m = cv2.resize(m.astype(np.uint8), (pm.shape[1], pm.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        valid_pts = pm[m > 0]
        if len(valid_pts) > 10:
            centroid = np.median(valid_pts, axis=0)
            world_traj[t] = centroid
            last_valid = centroid
        else:
            world_traj[t] = last_valid

    # Step 1.5: light temporal median smoothing for MegaSAM depth flicker
    for i in range(3):
        world_traj[:, i] = _medfilt1d(world_traj[:, i], k=3)

    # Step 2: 3D velocity and acceleration (physical units via dt)
    # dt = 1/fps makes the metric frame-rate invariant
    dt = 1.0 / max(fps, 1.0)
    velocity = np.diff(world_traj, axis=0) / dt      # (T-1, 3), m/s
    speed = np.linalg.norm(velocity, axis=1)          # (T-1,), m/s
    acceleration = np.diff(velocity, axis=0) / dt    # (T-2, 3), m/s^2
    accel_mag = np.linalg.norm(acceleration, axis=1)  # (T-2,)

    # Normalize by global mean speed, not per-frame speed.
    # Per-frame tiny denominators blow up noise on slow frames.
    # Global reference measures acceleration vs typical motion scale in this clip;
    # static clips fall back via global_floor tied to accel magnitude.
    speed_median = float(np.median(speed))
    accel_median = float(np.median(accel_mag))
    global_floor = accel_median * 2.0   # noise floor from accel median
    speed_ref    = max(speed_median, global_floor, 1e-6)

    # Relative acceleration rate (1/s)
    relative_accel_raw = accel_mag / speed_ref          # (T-2,)

    # tanh compress unbounded values to [0, 2), matching angle_penalty scale
    # tanh(x/5)*2: x=5 -> 1.52, x=10 -> 1.93, x>=15 -> ~2
    relative_accel = 2.0 * np.tanh(relative_accel_raw / 5.0)  # (T-2,)

    # Step 3: adjacent velocity cosine (penalize sharp direction reversals)
    v1, v2 = velocity[:-1], velocity[1:]
    n1 = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-9
    n2 = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-9
    cos_angles = np.clip(np.sum((v1 / n1) * (v2 / n2), axis=1), -1.0, 1.0)

    # Near-static: direction penalty is meaningless; use speed_ref gate
    moving = (speed[:-1] > speed_ref * 0.1) & (speed[1:] > speed_ref * 0.1)
    angle_penalty = np.zeros_like(cos_angles)
    angle_penalty[moving] = 1.0 - cos_angles[moving]   # same direction=0, reversal=2

    # Step 4: combine and pad to length (T-1,)
    eps_traj = relative_accel * 0.5 + angle_penalty * 0.5
    return np.insert(eps_traj, 0, eps_traj[0])


def audit_trajectory_consistency(
    h_seq: np.ndarray,
    xy_seq: np.ndarray,
    vanishing_point: Tuple[float, float],
) -> np.ndarray:
    """Generalized perspective trajectory audit (handles lateral motion).

    Auto mode:
    - Forward / oblique: log H-VP homography residual, log(h1/ht) vs log(d1/dt)
    - Lateral translation (VP at infinity): height stability |h(t)-h(0)|/h(0)

    Lateral if VP distance range ratio < 5%.

    Args:
        h_seq:           (T,) SAM2 pixel heights
        xy_seq:          (T, 2) Co-Tracker centroid positions
        vanishing_point: (vx, vy)

    Returns:
        (T-1,) trajectory residuals
    """
    T = len(h_seq)
    if T < 2:
        return np.zeros(1)

    vp = np.array(vanishing_point, dtype=np.float64)
    dist = np.linalg.norm(xy_seq.astype(np.float64) - vp, axis=1)  # (T,)

    dist_range_ratio = float(np.ptp(dist)) / (float(np.mean(dist)) + 1e-6)

    if dist_range_ratio < 0.05:
        # Lateral: depth ~ const, h should stay stable
        h0 = max(float(h_seq[0]), 1e-6)
        errors = np.abs(h_seq[1:] - h0) / h0
        return errors

    # Forward / oblique: log-space H-VP
    log_h = np.log(np.maximum(h_seq, 1e-6))
    log_d = np.log(np.maximum(dist, 1e-6))

    # First 5 frames median baseline to damp early noise
    n_ref = min(5, T)
    h_base = float(np.median(log_h[:n_ref]))
    d_base = float(np.median(log_d[:n_ref]))

    log_h_ratio = log_h - h_base
    log_d_ratio = log_d - d_base

    errors = np.abs(log_h_ratio - log_d_ratio)
    return errors[1:]
