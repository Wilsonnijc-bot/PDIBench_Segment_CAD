"""Scientific checks: preserve real surface variation, reject stretched support."""
from dataclasses import replace
import numpy as np
import pytest

from robot.preprocessing.depth.link5_depth_filter import Config, filter_depth, xyz_from_depth
from robot.preprocessing.link5_refinement.link5_point_guard import guard_config, REVIEW_SPECS
from robot.preprocessing.link7_persistent.interface.config import role_config


K=np.array([[180,0,60],[0,180,40],[0,0,1]],np.float32)


@pytest.mark.parametrize('shape',['flat','sloped','curved'])
def test_preserves_coherent_surface_and_never_moves_depth(shape):
    y,x=np.indices((80,120));u=(x-60)/180;v=(y-40)/180
    if shape=='flat':depth=np.ones_like(u)
    elif shape=='sloped':depth=1/(1+.6*u+.3*v)
    else:depth=1+.3*np.sin(u*4)+.1*v*v
    depth=depth.astype(np.float32);before=depth.copy();mask=np.ones_like(depth,bool)
    result=filter_depth(depth,mask,K,Config(erosion_pixels=0))
    assert result['valid'].all()
    np.testing.assert_array_equal(depth,before)


def test_rejects_smooth_connected_boundary_tail():
    depth=np.ones((80,120),np.float32)
    # Coherent 5-column stretch: not an isolated image-depth impulse.
    depth[:,115:]=np.array([1.1,1.2,1.3,1.4,1.5])
    result=filter_depth(depth,np.ones_like(depth,bool),K,Config(erosion_pixels=0))
    assert result['rejected'][:,116:].all()
    assert result['valid'][4:-4,:110].all()


def test_erosion_uses_source_pixels_before_depth_grid_alignment():
    source=np.ones((160,240),bool);depth=np.ones((80,120),np.float32)
    result=filter_depth(depth,source,K)
    assert result['mask'].all()
    assert result['valid'][1:-1,1:-1].all()
    assert not result['valid'][0].any()
    assert result['valid'].sum()==78*118


def test_invalid_depth_never_enters_cloud_and_coordinates_are_observed():
    depth=np.ones((80,120),np.float32);depth[15,15]=np.nan;depth[20,20]=0
    result=filter_depth(depth,np.ones_like(depth,bool),K)
    xyz,pixels=xyz_from_depth(depth,result['valid'],K)
    assert np.isfinite(xyz).all()
    np.testing.assert_array_equal(xyz[:,2],depth[pixels[:,0],pixels[:,1]])
    assert result['rejection_reason'][15,15]==1
    assert result['rejection_reason'][20,20]==1


def test_filter_is_invariant_to_scene_scale():
    depth=np.ones((80,120),np.float32);depth[:,115:]=[1.1,1.2,1.3,1.4,1.5]
    a=filter_depth(depth,np.ones_like(depth,bool),K)
    b=filter_depth(depth*8,np.ones_like(depth,bool),K)
    np.testing.assert_array_equal(a['valid'],b['valid'])


def test_guard_settings_are_scoped_and_keep_the_endpoint():
    primary=role_config('vlm2');fallback=role_config('vlm2_malformed_fallback')
    assert guard_config(primary)['reasoning_effort']=='high'
    assert guard_config(fallback)['reasoning_effort']=='xhigh'
    assert guard_config(fallback)['max_completion_tokens']>=65536
    assert guard_config(primary)['timeout_seconds']>=600
    assert guard_config(fallback)['timeout_seconds']>=1200
    assert 'max_completion_tokens' not in guard_config(dict(primary,max_completion_tokens=4096))
    assert 'max_tokens' not in guard_config(dict(fallback,max_tokens=4096))
    assert guard_config(primary)['api_base']==primary['api_base']
    assert fallback['reasoning_effort']=='high'  # other links' defaults unchanged
    assert all(spec['system_prompt'].startswith("You review SAM3 point prompts")
               and 'Think extremely hard' not in spec['system_prompt'] for spec in REVIEW_SPECS.values())


def test_link5_requests_do_not_change_link7_requests(monkeypatch):
    """Inspect real serialized requests without network access or paid calls."""
    import io
    import json
    from PIL import Image
    from robot.preprocessing.link5_refinement.link5_point_guard import _review
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient

    monkeypatch.setenv('PDI_VLM_ROLE_OVERRIDES', '{}')
    monkeypatch.setenv('VLM2_API_KEY', 'test-only-key')
    requests = []

    def reply(request, *, timeout):
        requests.append((request.full_url, json.loads(request.data), timeout))
        return io.BytesIO(json.dumps({
            'choices': [{'message': {'content': 'PASS' if len(requests)==3 else
                '{"decision":"PLACE","positive_points_xy":[[1,1],[12,12],[23,23]]}'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 2, 'completion_tokens': 1},
        }).encode())

    monkeypatch.setattr('urllib.request.urlopen', reply)
    image = Image.new('RGB', (32, 32))
    primary = role_config('vlm2')
    fallback = role_config('vlm2_malformed_fallback')
    before = (dict(primary), dict(fallback))
    original = np.array([[1, 1], [8, 8], [16, 16], [24, 24], [30, 30]])
    records = []
    for config in (primary, fallback):
        decision, _, record, error = _review(
            'positive', [image], 'Link5 review', 32, 32, original,
            configured_model=True, config=config,
        )
        assert decision == 'PLACE' and error is None
        records.append(record)
    link7 = VLMClient('vlm2_sam_prompting', role_config('vlm2')).ask(
        [image], 'Link7 prompt', system_prompt='Link7 system',
    )
    assert requests[0][1]['reasoning_effort'] == 'high'
    assert requests[0][1]['max_tokens'] == 65536 and requests[0][2] == 600
    assert requests[1][1]['reasoning_effort'] == 'xhigh'
    assert requests[1][1]['max_completion_tokens'] == 65536 and requests[1][2] == 1200
    assert 'max_tokens' not in requests[1][1]
    assert requests[2][1]['max_tokens'] == 4096 and requests[2][2] == 240
    assert 'reasoning_effort' not in requests[2][1]
    assert requests[2][1]['messages'][0]['content'] == 'Link7 system'
    assert requests[2][1]['messages'][1]['content'][-1]['text'] == 'Link7 prompt'
    assert all(url == primary['api_base'] + '/chat/completions' for url, _, _ in requests)
    assert (primary, fallback) == before
    assert (role_config('vlm2'), role_config('vlm2_malformed_fallback')) == before
    assert all(record['output_token_limit'] == 65536 and 'elapsed_seconds' in record
               for record in records)
    assert 'output_token_limit' not in link7 and 'elapsed_seconds' not in link7
