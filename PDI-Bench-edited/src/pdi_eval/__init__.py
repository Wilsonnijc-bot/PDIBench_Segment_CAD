"""PDI evaluation package with lazy pipeline imports."""

__version__ = "0.2.0"
__author__ = "PDI-Eval Team"

__all__ = ["PDIEvaluationPipeline", "MultiObjectPDIEvaluationPipeline"]


def __getattr__(name):
    if name == "PDIEvaluationPipeline":
        from .pipeline import PDIEvaluationPipeline

        return PDIEvaluationPipeline
    if name == "MultiObjectPDIEvaluationPipeline":
        from .multi_object_pipeline import MultiObjectPDIEvaluationPipeline

        return MultiObjectPDIEvaluationPipeline
    raise AttributeError(name)
