"""Record and remove legacy explicit architecture flags in a staged build copy.

Torch then honors TORCH_CUDA_ARCH_LIST=9.0+PTX. Numerical source is untouched.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
import ast
import os

source, metadata = map(Path, sys.argv[1:])
path = source / "setup.py"
before = path.read_bytes()
text = before.decode()
text = re.sub(r"^\s*['\"]-gencode=arch=compute_\d+,code=(?:sm|compute)_\d+['\"],?\s*$", "", text, flags=re.MULTILINE)
# The upstream file invokes setup twice, which is not a valid modern pip wheel.
# Combine its unchanged extension declarations into one installable distribution.
tree = ast.parse(text)
calls = [node.value for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'setup']
if len(calls) == 2:
    first, second = calls
    primary = next(k for k in first.keywords if k.arg == 'ext_modules')
    secondary = next(k for k in second.keywords if k.arg == 'ext_modules')
    primary.value.elts.extend(secondary.value.elts)
    first.keywords.extend(k for k in second.keywords if k.arg in {'packages', 'package_dir', 'version'})
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Expr) and node.value is second)]
    text = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
path.write_text(text)
record = json.dumps({
    "path": str(path), "original_sha256": hashlib.sha256(before).hexdigest(),
    "build_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "change": "Remove explicit legacy architecture flags; honor TORCH_CUDA_ARCH_LIST=9.0+PTX for H200. Combine the two setup calls for modern pip packaging; extension numerical source and compile optimization flags are unchanged.",
}, indent=2) + "\n"
(metadata / f"native-build-change-{os.environ.get('SLURM_JOB_ID', 'manual')}.json").write_text(record)
(metadata / "native-build-change.json").write_text(record)
