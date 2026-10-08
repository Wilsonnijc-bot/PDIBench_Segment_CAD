"""Collect and SHA-verify CPU-only replay display samples from Eris."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from .collect_eris_results import SSH
from .prepare_gpu import sha, write


def collect(remote_root, root):
    command = 'tar -C ' + shlex.quote(remote_root) + ' -cf - replay_context'
    process = subprocess.Popen(SSH + [command], stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
        for member in archive:
            if member.isdir():
                continue
            path = Path(member.name)
            if not member.isfile() or path.parts[0] != 'replay_context' or '..' in path.parts or path.is_absolute():
                raise ValueError('Unexpected replay archive path')
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, destination.open('wb') as target:
                while block := source.read(8 << 20):
                    target.write(block)
    if process.wait(timeout=60):
        raise RuntimeError('Replay context transfer failed')
    completion = json.loads((root / 'replay_context/completion.json').read_text())
    receipts = []
    for case in completion['cases']:
        path = root / 'replay_context' / (case + '.json.gz')
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        if sha(path) != receipt['context_sha256']:
            raise ValueError('Replay context checksum mismatch')
        receipts.append(dict(case=case, bytes=path.stat().st_size, sha256=receipt['context_sha256']))
    write(root / 'metadata/replay_context_collection.json', dict(status='complete', videos=len(receipts), files=receipts))
    print('REPLAY_CONTEXT_TRANSFER_VERIFIED', len(receipts), sum(r['bytes'] for r in receipts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--remote-root', required=True)
    args = parser.parse_args(); collect(args.remote_root, args.root)
