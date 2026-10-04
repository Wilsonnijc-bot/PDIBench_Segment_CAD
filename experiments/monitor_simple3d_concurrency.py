"""Measure GPU headroom and scoring throughput without changing the experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=420)
    args = parser.parse_args()
    execution = args.root/'metadata/execution'
    samples = []
    start = time.time()
    while time.time()-start < args.seconds:
        gpu, memory, power = map(float, subprocess.check_output([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits'], text=True).strip().split(','))
        rows = [json.loads(p.read_text()) for p in args.root.glob('cases/*/bin_*/comparison.json')]
        sample = dict(time=time.time(), gpu_utilization_percent=gpu, memory_used_mb=memory,
                      power_watts=power, successful_scores=sum(r['simple3d_status']=='complete' for r in rows),
                      failures=sum(r['simple3d_status']!='complete' for r in rows))
        samples.append(sample)
        with (execution/f'concurrency_{args.workers}_samples.jsonl').open('a') as f:
            f.write(json.dumps(sample)+'\n')
        summary = dict(workers=args.workers, sample_count=len(samples), start=start,
                       last_sample=sample, mean_gpu_utilization=statistics.mean(s['gpu_utilization_percent'] for s in samples),
                       peak_gpu_utilization=max(s['gpu_utilization_percent'] for s in samples),
                       peak_memory_mb=max(s['memory_used_mb'] for s in samples),
                       new_successful_scores=sample['successful_scores']-samples[0]['successful_scores'],
                       elapsed_seconds=time.time()-start)
        temp = execution/f'concurrency_{args.workers}_summary.tmp'
        temp.write_text(json.dumps(summary, indent=2)+'\n')
        temp.replace(execution/f'concurrency_{args.workers}_summary.json')
        time.sleep(2)


if __name__ == '__main__':
    main()
