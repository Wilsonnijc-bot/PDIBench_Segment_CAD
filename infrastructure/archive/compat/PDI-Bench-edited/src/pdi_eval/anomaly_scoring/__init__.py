"""Independent reference/query anomaly inference; imports models only on demand."""
from .anomalydino import AnomalyDINOScorer

__all__ = ['AnomalyDINOScorer']
