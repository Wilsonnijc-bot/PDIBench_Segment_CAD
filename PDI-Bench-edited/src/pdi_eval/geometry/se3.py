"""Small, validated SE(3) operations used by CAD pose metrics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rigid_transform_valid(
    transform: np.ndarray,
    *,
    atol: float = 1e-6,
) -> bool:
    """Return whether one 4x4 matrix is a finite proper rigid transform."""
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        return False
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=atol, rtol=0.0):
        return False
    rotation = value[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0.0)
    )


def require_rigid_transform(
    transform: np.ndarray,
    *,
    name: str = "transform",
    atol: float = 1e-6,
) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if not rigid_transform_valid(value, atol=atol):
        raise ValueError(f"{name} must be a finite proper 4x4 rigid transform")
    return value


def invert_rigid_transform(transform: np.ndarray) -> np.ndarray:
    value = require_rigid_transform(transform)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = value[:3, :3].T
    inverse[:3, 3] = -value[:3, :3].T @ value[:3, 3]
    return inverse


def rotation_angle(rotation: np.ndarray) -> float:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = np.clip((np.trace(value) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def se3_exp(twist: np.ndarray) -> np.ndarray:
    """Map ``[rho, phi]`` in se(3) to a 4x4 transform."""
    value = np.asarray(twist, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("twist must be a finite six-vector")
    rho = value[:3]
    phi = value[3:]
    theta = float(np.linalg.norm(phi))
    omega = _skew(phi)
    omega2 = omega @ omega
    if theta < 1e-8:
        theta2 = theta * theta
        coefficient_b = 0.5 - theta2 / 24.0
        coefficient_c = 1.0 / 6.0 - theta2 / 120.0
    else:
        coefficient_b = (1.0 - np.cos(theta)) / (theta * theta)
        coefficient_c = (theta - np.sin(theta)) / (theta**3)
    jacobian = np.eye(3) + coefficient_b * omega + coefficient_c * omega2
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(phi).as_matrix()
    transform[:3, 3] = jacobian @ rho
    return transform


def se3_log(transform: np.ndarray) -> np.ndarray:
    """Map a proper 4x4 transform to ``[rho, phi]`` in se(3)."""
    value = require_rigid_transform(transform)
    phi = Rotation.from_matrix(value[:3, :3]).as_rotvec()
    theta = float(np.linalg.norm(phi))
    omega = _skew(phi)
    omega2 = omega @ omega
    if theta < 1e-8:
        inverse_jacobian = np.eye(3) - 0.5 * omega + omega2 / 12.0
    else:
        coefficient = (
            1.0 - 0.5 * theta / np.tan(0.5 * theta)
        ) / (theta * theta)
        inverse_jacobian = np.eye(3) - 0.5 * omega + coefficient * omega2
    rho = inverse_jacobian @ value[:3, 3]
    return np.concatenate((rho, phi))


def scale_rigid_increment(transform: np.ndarray, scale: float) -> np.ndarray:
    if not np.isfinite(scale):
        raise ValueError("increment scale must be finite")
    return se3_exp(se3_log(transform) * float(scale))
