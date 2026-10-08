"""Fetch auxiliary checkpoints at explicit revisions, independently of inference."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

dependency = Path(sys.argv[1]).resolve()
specifications = [
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "qwen_model", None),
    ("lpiccinelli/unidepth-v2-vitl14", "1d0d3c52f60b5164629d279bb9a7546458e6dcc4", "unidepth", ["config.json", "pytorch_model.bin", "model.safetensors"]),
]


def download(specification):
    repo, revision, directory, patterns = specification
    options = {"repo_id": repo, "revision": revision, "max_workers": 4}
    if patterns:
        options["allow_patterns"] = patterns
    # UniDepth expects its standard Hugging Face cache, including pinned revision.
    if directory != "unidepth":
        options["local_dir"] = str(dependency / "models" / directory)
    result = snapshot_download(**options)
    print(json.dumps({"repository": repo, "revision": revision, "path": result}), flush=True)
    return {"repository": repo, "revision": revision, "path": result}


with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(download, specifications))
(dependency / "metadata" / "downloads.json").write_text(json.dumps(results, indent=2) + "\n")
