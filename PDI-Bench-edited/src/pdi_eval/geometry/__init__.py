__all__ = ["CameraModel", "ProjectionJudge"]


def __getattr__(name):
    if name == "CameraModel":
        from .camera import CameraModel

        return CameraModel
    if name == "ProjectionJudge":
        from .projection import ProjectionJudge

        return ProjectionJudge
    raise AttributeError(name)
