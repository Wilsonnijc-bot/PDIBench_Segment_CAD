__all__ = [
    "audit_scale_consistency",
    "audit_trajectory_consistency",
    "audit_3d_volume_stability",
    "audit_reconstruction",
    "audit_reconstruction_math",
    "audit_reconstruction_mllm",
    "load_from_npz",
    "render_three_views_white_bg",
]


def __getattr__(name):
    if name == "audit_scale_consistency":
        from .scale_audit import audit_scale_consistency

        return audit_scale_consistency
    if name == "audit_trajectory_consistency":
        from .motion_audit import audit_trajectory_consistency

        return audit_trajectory_consistency
    if name == "audit_3d_volume_stability":
        from .volume_audit import audit_3d_volume_stability

        return audit_3d_volume_stability
    reconstruction_names = {
        "audit_reconstruction",
        "audit_reconstruction_math",
        "audit_reconstruction_mllm",
        "load_from_npz",
        "render_three_views_white_bg",
    }
    if name in reconstruction_names:
        from . import reconstruction_audit

        return getattr(reconstruction_audit, name)
    raise AttributeError(name)
