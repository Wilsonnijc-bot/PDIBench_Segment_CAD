"""CLI: prepare selected crop pairs, run inference, or smoke-test a real backbone."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .anomalydino import AnomalyDINOScorer
from .runner import DEFAULT_EXCLUSIONS, prepare_selected, read_pairs, run_pairs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--crop-root', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--exclude', nargs='*', default=list(DEFAULT_EXCLUSIONS))
    run = commands.add_parser('run')
    run.add_argument('--pairs', type=Path, required=True)
    run.add_argument('--input-root', type=Path, default=Path('.'))
    run.add_argument('--output-root', type=Path, required=True)
    smoke = commands.add_parser('smoke')
    smoke.add_argument('--reference', type=Path, required=True)
    smoke.add_argument('--query', type=Path, required=True)
    smoke.add_argument('--output', type=Path)
    for command in (run, smoke):
        command.add_argument('--model-name', default='dinov2_vits14')
        command.add_argument('--device', default='cuda:0')
        command.add_argument('--resolution', type=int, default=448)
        command.add_argument('--rotation', action=argparse.BooleanOptionalAction, default=True)
        command.add_argument('--masking', action=argparse.BooleanOptionalAction, default=False)
        command.add_argument('--faiss-on-cpu', action=argparse.BooleanOptionalAction, default=True)
        command.add_argument('--dino-repo')
        command.add_argument('--checkpoint', type=Path)
        command.add_argument('--max-cached-references', type=int, default=16)
    args = parser.parse_args()
    if args.command == 'prepare':
        manifest = prepare_selected(args.crop_root, args.exclude)
        write_json(args.output, manifest)
        print(json.dumps({'pair_count': manifest['pair_count'],
                          'video_count': sum(v['status'] == 'ready' for v in manifest['videos'])}))
        return
    scorer = AnomalyDINOScorer(
        model_name=args.model_name, device=args.device, resolution=args.resolution,
        rotation=args.rotation, masking=args.masking, faiss_on_cpu=args.faiss_on_cpu,
        dino_repo=args.dino_repo, checkpoint_path=args.checkpoint,
        max_cached_references=args.max_cached_references)
    if args.command == 'smoke':
        self_result = scorer.score(args.reference, args.reference, return_map=True)
        query_result = scorer.score(args.reference, args.query, return_map=True)
        scores = [self_result['anomaly_score'], query_result['anomaly_score']]
        if not all(isinstance(s, float) and math.isfinite(s) for s in scores):
            raise AssertionError('Scores must be finite floats')
        if scorer.reference_cache_misses != 1 or scorer.reference_cache_hits != 1:
            raise AssertionError('Reference was not reused')
        for result in (self_result, query_result):
            if result['anomaly_map'].ndim != 2:
                raise AssertionError('Expected a dense 2D anomaly map')
        result = dict(status='passed', self_score=scores[0], query_score=scores[1],
                      reference_cache_hits=scorer.reference_cache_hits,
                      reference_cache_misses=scorer.reference_cache_misses,
                      settings=scorer.settings)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    manifest, pairs = read_pairs(args.pairs)
    run_pairs(scorer, pairs, args.input_root, args.output_root, manifest=manifest)


if __name__ == '__main__':
    main()
