"""Convert cached HF DINOv2-base weights for the official DINOv2 implementation.

This reverses Hugging Face's official parameter rename and Q/K/V split:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/convert_dinov2_to_hf.py
No training, tensor approximation, or feature implementation is involved.
Requires safetensors; transformers is used only for optional conversion validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from object.scoring.anomalydino.anomalydino import load_upstream, rgb_array
from object.scoring.anomalydino.runner import sha256, write_json


def convert_base(source_root: Path, dino_repo: str, destination: Path, verify_image: Path):
    import torch
    from safetensors.torch import load_file
    from transformers import Dinov2Model
    source_root = Path(source_root)
    config = json.loads((source_root / 'config.json').read_text())
    if (config.get('hidden_size'), config.get('num_hidden_layers'), config.get('patch_size')) != (768, 12, 14):
        raise ValueError('This conversion supports the observed DINOv2-base architecture only')
    if config.get('use_swiglu_ffn', False):
        raise ValueError('Expected the standard DINOv2-base MLP')
    state = load_file(str(source_root / 'model.safetensors'), device='cpu')
    converted = {}
    keys = {
        'embeddings.cls_token': 'cls_token', 'embeddings.mask_token': 'mask_token',
        'embeddings.position_embeddings': 'pos_embed',
        'embeddings.patch_embeddings.projection.weight': 'patch_embed.proj.weight',
        'embeddings.patch_embeddings.projection.bias': 'patch_embed.proj.bias',
        'layernorm.weight': 'norm.weight', 'layernorm.bias': 'norm.bias'}
    for layer in range(12):
        hf = f'encoder.layer.{layer}.'
        original = f'blocks.{layer}.'
        for name in ('norm1', 'norm2', 'mlp.fc1', 'mlp.fc2'):
            for suffix in ('weight', 'bias'):
                keys[hf + name + '.' + suffix] = original + name + '.' + suffix
        for number in (1, 2):
            keys[hf + f'layer_scale{number}.lambda1'] = original + f'ls{number}.gamma'
        for suffix in ('weight', 'bias'):
            keys[hf + 'attention.output.dense.' + suffix] = original + 'attn.proj.' + suffix
            converted[original + 'attn.qkv.' + suffix] = torch.cat([
                state.pop(hf + 'attention.attention.' + name + '.' + suffix)
                for name in ('query', 'key', 'value')], dim=0)
    for hf, original in keys.items():
        converted[original] = state.pop(hf)
    if state:
        raise ValueError(f'Unmapped checkpoint parameters: {sorted(state)}')
    backbone, _ = load_upstream()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    torch.save(converted, temporary)
    wrapper = backbone.DINOv2Wrapper('dinov2_vitb14', 'cuda:0', 448,
                                     dino_repo=dino_repo, checkpoint_path=temporary)
    image = rgb_array(verify_image)
    tensor, grid = wrapper.prepare_image(image)
    tokens = wrapper.extract_features(tensor)
    hf_model = Dinov2Model.from_pretrained(str(source_root), local_files_only=True).eval().cuda()
    with torch.inference_mode():
        hf_tokens = hf_model(tensor.unsqueeze(0).cuda()).last_hidden_state[:, 1:, :].squeeze(0)
    original_tokens = torch.from_numpy(tokens).cuda()
    torch.testing.assert_close(original_tokens, hf_tokens, atol=1e-3, rtol=1e-4)
    report = {'status': 'passed', 'source_checkpoint_sha256': sha256(source_root / 'model.safetensors'),
              'converted_checkpoint_sha256': sha256(temporary), 'model_name': 'dinov2_vitb14',
              'mapped_parameter_count': len(converted), 'strict_state_dict_load': True,
              'token_max_absolute_difference': float((original_tokens - hf_tokens).abs().max().item()),
              'verification_grid': list(grid), 'atol': 1e-3, 'rtol': 1e-4,
              'source_config': config}
    temporary.replace(destination)
    write_json(destination.with_suffix('.conversion.json'), report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--dino-repo', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--verify-image', type=Path, required=True)
    args = parser.parse_args()
    convert_base(args.source_root, args.dino_repo, args.destination, args.verify_image)


if __name__ == '__main__':
    main()
