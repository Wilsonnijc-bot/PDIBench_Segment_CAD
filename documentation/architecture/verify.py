"""Verify source ownership, archived aliases and optional immutable artifact baselines."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class NormalizePaths(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=='_SOURCE_PATH':return ast.parse('Path(__file__)',mode='eval').body
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if (isinstance(node.func,ast.Attribute) and node.func.attr=='resolve' and not node.args
            and ast.unparse(node.func.value)=='Path(__file__)'):
            return node.func.value
        # Original MegaSAM uses dirname(__file__) for its checkout-relative root.
        if ast.unparse(node)=='os.path.dirname(__file__)':
            return ast.parse('str(Path(__file__).parent)',mode='eval').body
        return node


def normalized(source):
    source=re.sub(r'\n# Source ownership changed; retain historical resource/provenance paths\.\n.*?        break\n\n','\n',source,flags=re.S)
    return NormalizePaths().visit(ast.parse(source))


def verify(baseline=None, historical_semantics=False):
    manifest=json.loads((ROOT/'documentation/architecture/layout.json').read_text())
    failures=[];checked=Counter()
    for row in manifest['files']:
        old,new=ROOT/row['compat'],ROOT/row['new']
        if not old.is_symlink() or old.resolve()!=new.resolve() or not new.is_file():
            failures.append('source alias: '+row['old']);continue
        checked['source_aliases']+=1
        if new.suffix=='.py':
            ast.parse(new.read_text())
        if not historical_semantics:
            continue
        if new.suffix=='.py':
            before=subprocess.check_output(['git','show',manifest['source_commit']+':'+row['old']],cwd=ROOT,text=True)
            try:
                a,b=normalized(before),normalized(new.read_text())
                for extraction in manifest.get('extractions',[]):
                    if row['old']==extraction['from']:
                        extracted={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((ROOT/extraction['to']).read_text()).body if isinstance(n,ast.FunctionDef)}
                        for n in a.body:
                            if isinstance(n,ast.FunctionDef) and n.name in extraction['symbols']:
                                if ast.dump(n,include_attributes=False)!=extracted.get(n.name):failures.append('changed extracted function: '+n.name)
                        a.body=[n for n in a.body if not (isinstance(n,ast.FunctionDef) and n.name in extraction['symbols'])]
                        b.body=[n for n in b.body if not (isinstance(n,ast.ImportFrom) and n.module=='robot_scoring')]
                if ast.dump(a,include_attributes=False)!=ast.dump(b,include_attributes=False):failures.append('non-path source change: '+row['new'])
                checked['python_semantic_comparisons']+=1
            except SyntaxError as exc:failures.append(f'{new}: {exc}')
        elif row.get('shell_path_tokens'):
            current=re.sub(r'# BEGIN PDI SOURCE LOCATION\n.*?# END PDI SOURCE LOCATION\n','',new.read_text(),flags=re.S)
            current=current.replace('${pdi_source_dir}',row['shell_path_tokens'])
            if hashlib.sha256(current.encode()).hexdigest()!=row['sha256_before']:
                failures.append('non-path shell change: '+row['new'])
            checked['shell_semantic_comparisons']+=1
        elif sha(new)!=row['sha256_before']:
            failures.append('changed non-Python source: '+row['new'])
    for row in manifest['results']+manifest.get('result_subtrees',[]):
        old,new=ROOT/row['old'],ROOT/row['new']
        if not old.is_symlink() or old.resolve()!=new.resolve():failures.append('result alias: '+row['old'])
        checked['result_aliases']+=1
    if baseline:
        frozen=json.loads(Path(baseline).read_text())
        for name,record in frozen['files'].items():
            path=ROOT/name
            if not path.is_file() or path.stat().st_size!=record['bytes'] or sha(path)!=record['sha256']:
                failures.append('artifact changed: '+name)
            checked['artifact_files']+=1
            checked['artifact_bytes']+=record['bytes']
        for name,target in frozen['symlinks'].items():
            # Relative text may change; the logical target must remain identical.
            old=ROOT/name
            original=Path(target) if os.path.isabs(target) else old.parent/target
            if not old.is_symlink() or old.resolve()!=original.resolve():failures.append('artifact symlink target: '+name)
            checked['artifact_symlinks']+=1
    return {'status':'passed' if not failures else 'failed','checked':dict(checked),'failures':failures}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-baseline',type=Path)
    p.add_argument('--historical-semantics',action='store_true',help='Compare against the old Git snapshot, including later intentional changes')
    p.add_argument('--output',type=Path)
    args=p.parse_args();result=verify(args.artifact_baseline, args.historical_semantics)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
