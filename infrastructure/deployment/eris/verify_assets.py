"""Verify staged source and transfer receipts before releasing a GPU job."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace', type=Path, required=True)
parser.add_argument('--metadata', type=Path, required=True)
parser.add_argument('--dependency', type=Path, required=True)
args = parser.parse_args()


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


source = json.loads((args.workspace/'source-stage.json').read_text())['files']
sidecars = list((args.workspace/'documentation/data/references').rglob('._*'))
assert not sidecars, 'macOS AppleDouble sidecars must be excluded from reference transfers'
for relative, expected in source.items():
    assert digest(args.workspace/relative) == expected, f'Staged source changed: {relative}'
selection = json.loads((args.metadata/'selected-videos.json').read_text())
assert len(selection['staging_cases']) == 45 and len(selection['run_cases']) == 10
assert digest(args.workspace/selection['workbook']) == selection['workbook_sha256']
videos = args.metadata.parent/'inputs/videos'
for line in (args.metadata/'video-source-sha256.txt').read_text().splitlines():
    expected, relative = line.split(maxsplit=1)
    assert digest(videos/relative) == expected, f'Video transfer mismatch: {relative}'
for receipt, directory in [('model-source-sha256.txt', args.dependency/'models'), ('mega-source-sha256.txt', args.dependency/'source/infrastructure/vendor/mega_sam'), ('qwen-source-sha256.txt', args.dependency/'models/qwen_model')]:
    for line in (args.metadata/receipt).read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        assert digest(directory/relative) == expected, f'Checkpoint transfer mismatch: {relative}'
manifest = json.loads((args.metadata/'manifest.json').read_text())
assert manifest['robot_links'] == ['link2', 'link7']
assert [case['id'] for case in manifest['cases']] == selection['run_cases']
for resource in manifest['resources'].values():
    for value in resource if isinstance(resource, list) else [resource]:
        assert Path(value).exists(), f'Missing resource: {value}'
for case in manifest['cases']:
    assert Path(case['video']).is_file() and Path(case['prompt_file']).read_text().strip()
for interpreter in manifest['environments'].values():
    assert Path(interpreter).is_file(), f'Missing interpreter: {interpreter}'
for profile in ('sam3', 'geometry', 'models'):
    assert (args.dependency/'metadata'/f'prepare-{profile}.exit').read_text().strip() == '0'
native = (args.dependency/'metadata/native-imports.txt').read_text()
assert native.startswith('NATIVE_IMPORTS_COMPLETE '), 'Missing native extension import receipt'
assert 'droid_backends' in native and 'lietorch_backends' in native
unidepth = args.dependency/'cache/huggingface/hub/models--lpiccinelli--unidepth-v2-vitl14/snapshots/1d0d3c52f60b5164629d279bb9a7546458e6dcc4'
assert (unidepth/'config.json').is_file() and (unidepth/'pytorch_model.bin').is_file()
torchhub = Path(manifest['resources']['mega_sam'])/'torchhub/facebookresearch_dinov2_main'
assert torchhub.resolve() == Path(manifest['resources']['dino_repo']).resolve()
assert Path(manifest['secrets_file']).stat().st_mode & 0o777 == 0o600
record = {'status': 'verified', 'source_files': len(source), 'staged_videos': 45, 'run_cases': selection['run_cases'],
    'checkpoint_transfer_receipts': ['model-source-sha256.txt', 'mega-source-sha256.txt', 'qwen-source-sha256.txt'],
    'note': 'Checkpoint SHA256 verified against source files. Historical masks/tracks/geometry/crops/scores are not inputs.'}
(args.metadata/'assets-verified.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
