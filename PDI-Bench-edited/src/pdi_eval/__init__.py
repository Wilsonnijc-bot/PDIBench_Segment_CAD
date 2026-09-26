"""PDI evaluation package with lazy pipeline imports."""

__version__ = "0.2.0"
__author__ = "PDI-Eval Team"

__all__ = ["MultiObjectPDIEvaluationPipeline"]


def __getattr__(name):
    if name == "MultiObjectPDIEvaluationPipeline":
        from .v1.pipeline import MultiObjectPDIEvaluationPipeline

        return MultiObjectPDIEvaluationPipeline
    raise AttributeError(name)
