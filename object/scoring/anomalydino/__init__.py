"""Independent reference/query anomaly inference; imports models only on demand."""
from object.scoring.anomalydino.anomalydino import AnomalyDINOScorer

__all__ = ['AnomalyDINOScorer']
