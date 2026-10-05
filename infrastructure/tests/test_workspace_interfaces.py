import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from infrastructure.deformation_detect.analysis import analyze, auroc
from infrastructure.deformation_detect.layout import root, manifest


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def test_analysis_reports_missing_and_moderate_without_imputation(tmp_path):
    scores,labels=tmp_path/'scores.csv',tmp_path/'labels.csv'
    write_csv(scores,[{'case':'a','score':2},{'case':'b','score':1},{'case':'c','score':9},{'case':'d','score':''}])
    write_csv(labels,[{'case':'a','label':1},{'case':'b','label':0},{'case':'c','label':0.5},{'case':'d','label':1},{'case':'e','label':0}])
    report=analyze(scores,labels)
    assert report['auroc']==1 and report['positive_count']==report['negative_count']==1
    assert report['excluded_counts']=={'nonbinary_label':1,'unavailable_value':1,'missing_score':1}
    assert analyze(scores,labels,lower_is_anomalous=True)['auroc']==0
    assert auroc([(1,1),(1,0)])==0.5
    assert auroc([(1,1)]) is None


def test_analysis_rejects_ambiguous_duplicate_cases(tmp_path):
    path=tmp_path/'scores.csv';write_csv(path,[{'case':'a','score':1},{'case':'a','score':2}])
    with pytest.raises(ValueError,match='duplicate'):analyze(path,path)


def test_old_imports_use_single_canonical_scoring_function():
    from robot.scoring.metrics import evaluate_object_metrics
    from robot.workflows.pipeline import evaluate_object_metrics as compatibility
    assert compatibility is evaluate_object_metrics


def test_cli_records_exit_and_does_not_overwrite(tmp_path):
    record=tmp_path/'execution'
    command=[sys.executable,'-m','deformation_detect','run','--record',str(record),'links.run','--','--help']
    done=subprocess.run(command,cwd=root(),capture_output=True,text=True)
    assert done.returncode==0,done.stderr
    payload=json.loads((record/'execution.json').read_text())
    assert payload['status']=='complete' and payload['exit_code']==0
    again=subprocess.run(command,cwd=root(),capture_output=True,text=True)
    assert again.returncode!=0
    assert json.loads((record/'execution.json').read_text())==payload


def test_archived_source_aliases_resolve_to_their_owner():
    for row in manifest()['files']:
        assert (root()/row['compat']).is_symlink(),row
        assert (root()/row['compat']).resolve()==(root()/row['new']).resolve(),row


def test_runtime_imports_are_direct_and_archives_are_not_on_pythonpath():
    import ast
    from infrastructure.deformation_detect.layout import environment, interfaces
    forbidden={'pdi_eval','persistent_masking','generation','experiments','scripts'}
    owners=('robot','object','infrastructure/shared','infrastructure/deformation_detect','documentation/publication')
    for owner in owners:
        for path in (root()/owner).rglob('*.py'):
            if {'archive','results','__pycache__'} & set(path.relative_to(root()).parts):continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
                for module in modules:
                    assert module.split('.')[0] not in forbidden,(path,node.lineno,module)
                    assert '.archive.' not in module,(path,node.lineno,module)
    assert not (root()/'infrastructure/compat').exists()
    assert '/compat' not in environment()['PYTHONPATH']
    assert '/archive/' not in environment()['PYTHONPATH']
    for entry in interfaces().values():
        target=root()/entry['module'].replace('.','/')
        assert target.with_suffix('.py').is_file() or (target/'__main__.py').is_file(),entry


def test_canonical_asset_locations_and_coordinator_without_site_packages(tmp_path):
    from infrastructure.deformation_detect.layout import environment
    from infrastructure.shared.replay.rigidity_replay import _SOURCE_PATH
    assert _SOURCE_PATH.with_name('rigidity_replay.html').is_file()
    assert (root()/'infrastructure/shared/replay/assets/plotly.min.js').is_file()
    command=[sys.executable,'-S','-m','infrastructure.deformation_detect','coordinate','--manifest',
             str(root()/'documentation/pipeline/coordinator.example.json'),'--plan']
    result=subprocess.run(command,cwd=tmp_path,env=environment(),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'link2_masks' in json.loads(result.stdout)['stages']


def test_staged_checkout_runs_new_cli_without_old_entry_points(tmp_path):
    from infrastructure.deformation_detect.layout import environment
    snapshot = tmp_path / 'source'
    result = subprocess.run(
        [sys.executable, '-m', 'deformation_detect', 'stage', '--output', str(snapshot)],
        cwd=root(), env=environment(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (snapshot / 'deformation_detect.py').is_file()
    assert not (snapshot / 'documentation/website').exists()
    assert not (snapshot / 'pdibench.py').exists()
    assert not (snapshot / 'infrastructure/pdibench').exists()
    result = subprocess.run(
        [sys.executable, '-S', '-m', 'deformation_detect', 'list'],
        cwd=snapshot, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'objects.score' in result.stdout and 'links.run' in result.stdout
    result = subprocess.run(
        [sys.executable, '-S', '-m', 'pdibench', 'list'],
        cwd=snapshot, capture_output=True, text=True)
    assert result.returncode != 0
