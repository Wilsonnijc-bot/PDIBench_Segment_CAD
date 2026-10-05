"""A content-cached, one-shot API over the official AnomalyDINO inference code."""
from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


from collections import OrderedDict
import hashlib
import importlib
import importlib.util
import importlib.metadata
import math
from pathlib import Path
import sys
from typing import Protocol


class AnomalyScorer(Protocol):
    def score(self, reference_image, query_image, *, return_map: bool = False) -> dict: ...


def load_upstream(upstream_root: str | Path | None = None):
    root = (Path(upstream_root) if upstream_root else
            (_workspace_root() / 'infrastructure/vendor/AnomalyDINO')).resolve()
    source = root / 'src'
    if not (source / 'detection.py').is_file():
        raise FileNotFoundError(f'Initialize the AnomalyDINO submodule at {root}')
    # The upstream package is named `src`; avoid colliding with other packages.
    namespace = '_pdi_anomalydino_' + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            namespace, source / '__init__.py', submodule_search_locations=[str(source)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = module
        spec.loader.exec_module(module)
    return (importlib.import_module(namespace + '.backbones'),
            importlib.import_module(namespace + '.detection'))


def rgb_array(image):
    """Accept paths, PIL images, or uint8 arrays; composite transparency on black."""
    import numpy as np
    from PIL import Image
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return rgb_array(opened)
    if isinstance(image, Image.Image):
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            rgba = image.convert('RGBA')
            image = Image.alpha_composite(Image.new('RGBA', rgba.size, (0, 0, 0, 255)), rgba)
        return np.array(image.convert('RGB'), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError('Arrays must be uint8 H×W×3 RGB or H×W×4 RGBA')
        if not image.shape[0] or not image.shape[1]:
            raise ValueError('Image is empty')
        if image.shape[2] == 4:
            return rgb_array(Image.fromarray(image))
        return np.ascontiguousarray(image)
    raise TypeError('Expected an image path, PIL image, or uint8 RGB/RGBA array')


class AnomalyDINOScorer:
    """Initialize one DINO backbone; keep a separate bank for each reference.

    Defaults follow upstream's custom-data agnostic_no_mask procedure: eight
    rotations of the one supplied reference, no PCA masking, normalized L2 1NN.
    Instances are intended for sequential inference, not concurrent mutation.
    """

    def __init__(self, *, model_name='dinov2_vits14', device='cuda:0',
                 resolution=448, rotation=True, masking=False,
                 mask_ref_images=False, faiss_on_cpu=True, max_cached_references=16,
                 upstream_root=None, dino_repo=None, checkpoint_path=None):
        if not model_name.startswith('dinov2'):
            raise ValueError('This scorer uses a DINOv2 backbone')
        if resolution < 14 or max_cached_references < 1:
            raise ValueError('Invalid resolution or reference-cache size')
        import torch
        if str(device).startswith('cuda'):
            torch.cuda.set_device(torch.device(device))
        backbones, self._inference = load_upstream(upstream_root)
        if dino_repo is not None or checkpoint_path is not None:
            self.model = backbones.DINOv2Wrapper(
                model_name, device, resolution, dino_repo=dino_repo,
                checkpoint_path=checkpoint_path)
        else:
            self.model = backbones.get_model(model_name, device, resolution)
        self.rotation = rotation
        self.masking = masking
        self.mask_ref_images = mask_ref_images
        self.faiss_on_cpu = faiss_on_cpu
        self.max_cached_references = max_cached_references
        self._cache = OrderedDict()
        self.reference_cache_hits = 0
        self.reference_cache_misses = 0
        self.settings = dict(model_name=model_name, device=str(device), resolution=resolution,
                             rotation=rotation, masking=masking, mask_ref_images=mask_ref_images,
                             faiss_on_cpu=faiss_on_cpu, knn_metric='L2_normalized', knn_neighbors=1,
                             shots=1, alpha_background='black', dino_repo=str(dino_repo) if dino_repo else None,
                             checkpoint_path=str(checkpoint_path) if checkpoint_path else None)
        self.settings['upstream_commit'] = 'b9d1c2648e3a5247437d4d953d907a8f3d994457'
        self.settings['inference_source_sha256'] = {
            name: hashlib.sha256((Path(self._inference.__file__).parent / name).read_bytes()).hexdigest()
            for name in ('backbones.py', 'detection.py', 'post_eval.py', 'utils.py')}
        self.settings['package_versions'] = {
            name: importlib.metadata.version(name)
            for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn', 'scipy')}
        if checkpoint_path is not None:
            self.settings['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()

    def _reference(self, image):
        import numpy as np
        key = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if key in self._cache:
            self.reference_cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key][0]
        features = self._inference.extract_reference(
            self.model, image, self.rotation, self.masking, self.mask_ref_images)[0]
        if not np.isfinite(features).all():
            raise ValueError('Reference features are nonfinite')
        # Includes the GPU resource owner when GPU FAISS is selected.
        bank = self._inference.build_knn_index(features, 'L2_normalized', self.faiss_on_cpu)
        self._cache[key] = bank
        self.reference_cache_misses += 1
        while len(self._cache) > self.max_cached_references:
            self._cache.popitem(last=False)
        return bank[0]

    def precompute_reference(self, reference_image):
        self._reference(rgb_array(reference_image))

    def clear_reference_cache(self):
        self._cache.clear()

    def score(self, reference_image, query_image, *, return_map=False) -> dict:
        reference = rgb_array(reference_image)
        query = rgb_array(query_image)
        bank = self._reference(reference)
        result = self._inference.score_query(
            self.model, query, bank, masking=self.masking,
            knn_metric='L2_normalized', knn_neighbors=1, return_map=return_map)
        if not math.isfinite(result['anomaly_score']):
            raise ValueError('Upstream inference returned a nonfinite score')
        return result
