"""Merge independent coordinator groups into one read-only scientific review."""
import argparse
from pathlib import Path

from infrastructure.deformation_detect.coordinator import now, read, review, write


def merge(plan):
    state = {'schema_version': 1, 'run_id': plan['run_id'], 'status': 'running',
             'updated_at': now(), 'groups': [], 'cases': {}}
    terminal = True
    for group in plan['groups']:
        path = Path(group['output'])/'run.json'
        current = read(path) if path.is_file() else {'status': 'pending', 'cases': {}}
        state['groups'].append({'id': group['id'], 'status': current['status'], 'output': group['output']})
        terminal &= current['status'] in {'complete', 'completed_with_disabled_cases'}
        for name in group['cases']:
            if name in state['cases']:
                raise ValueError('Duplicate case in coordinator groups: '+name)
            state['cases'][name] = current['cases'].get(name, {'status': 'pending', 'stages': {}})
    if terminal:
        state['status'] = 'complete' if all(c['status'] == 'complete' for c in state['cases'].values()) else 'completed_with_disabled_cases'
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    output = Path(plan['output'])
    state = merge(plan)
    write(output/'run.json', state)
    review(output, state)
    print(state['status'], {c['status']: sum(v['status'] == c['status'] for v in state['cases'].values()) for c in state['cases'].values()})


if __name__ == '__main__':
    main()
