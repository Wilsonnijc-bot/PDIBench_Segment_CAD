"""Do not accept almost-constant scores or a model with clean false alarms."""
import json

from robot.experiments.link5_shape_codebook.train_structural_localization_v2 import acceptance


def cases(ratio=2):
    return [dict(example=example, type=kind, parameters=dict(angle_degrees=35), raw_top80_ratio=ratio,
                 region_vs_outside_ratio=ratio, matched_region_score_ratio=ratio)
            for kind,example in [('shortening','shortening_25pct'),('lengthening','lengthening_25pct'),('bending','bend_35deg_0')]]


def test_acceptance_report_serializes_and_distinguishes_a_strong_candidate():
    result=acceptance(cases(),[dict(split='heldout',false_positive_fraction=.01)],dict(median_point_auroc=.9))
    assert json.loads(json.dumps(result))['status']=='passed'


def test_high_rank_metric_cannot_hide_almost_constant_scores():
    result=acceptance(cases(1.0001),[dict(split='heldout',false_positive_fraction=0)],dict(median_point_auroc=.99))
    assert result['status']=='failed' and not result['checks']['strong_separation_each_family']


def test_clean_false_alarms_fail_even_when_synthetic_separation_is_strong():
    result=acceptance(cases(),[dict(split='heldout',false_positive_fraction=.25)],dict(median_point_auroc=.95))
    assert result['status']=='failed' and not result['checks']['p95_clean_false_positive']
