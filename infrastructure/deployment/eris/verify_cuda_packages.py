"""Check every relayed CUDA archive against the Mac's verified transfer manifest."""
import hashlib
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
records = json.loads((directory/'manifest.json').read_text())
for record in records:
    path = directory/record['file']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    assert path.stat().st_size == record['size'] and digest.hexdigest() == record['sha256'], record['file']
print(json.dumps({'status': 'verified', 'cuda_packages': len(records), 'bytes': sum(r['size'] for r in records)}))
