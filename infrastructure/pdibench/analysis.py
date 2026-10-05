"""Generic binary AUROC over explicit case-level scores and labels, without imputation."""
import csv
import json
import math
from collections import Counter
from pathlib import Path


def read_unique(path, key):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if key not in (reader.fieldnames or []):
            raise ValueError(f'{path}: missing join column {key}')
        result = {}
        for row in reader:
            identity = row[key]
            if not identity or identity in result:
                raise ValueError(f'{path}: empty or duplicate identity {identity!r}')
            result[identity] = row
        return result


def auroc(points):
    positive = [score for score, label in points if label == 1]
    negative = [score for score, label in points if label == 0]
    if not positive or not negative:
        return None
    # Pairwise definition handles ties exactly and avoids fitting a threshold.
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def analyze(scores, labels, *, key='case', score='score', label='label', positive=1.0, negative=0.0, lower_is_anomalous=False):
    if positive == negative:
        raise ValueError('Positive and negative labels must differ')
    scored, labeled = read_unique(scores, key), read_unique(labels, key)
    points, excluded = [], []
    for identity in sorted(set(scored) | set(labeled)):
        reason = None
        if identity not in labeled:
            reason = 'missing_label'
        elif identity not in scored:
            reason = 'missing_score'
        else:
            try:
                value = float(scored[identity][score])
                truth = float(labeled[identity][label])
                if not math.isfinite(value) or not math.isfinite(truth):
                    reason = 'nonfinite'
                elif truth not in (positive, negative):
                    reason = 'nonbinary_label'
                else:
                    points.append((-value if lower_is_anomalous else value, int(truth == positive)))
            except KeyError as exc:
                raise ValueError(f'Missing analysis column: {exc.args[0]}') from exc
            except (TypeError, ValueError):
                reason = 'unavailable_value'
        if reason:
            excluded.append({'case': identity, 'reason': reason})
    return {'method': 'binary-auroc-pairwise-ties-half', 'auroc': auroc(points),
            'score_direction': 'lower' if lower_is_anomalous else 'higher',
            'positive_label': positive, 'negative_label': negative,
            'positive_count': sum(label for _, label in points),
            'negative_count': sum(1-label for _, label in points),
            'excluded_counts': dict(Counter(row['reason'] for row in excluded)), 'excluded': excluded}
