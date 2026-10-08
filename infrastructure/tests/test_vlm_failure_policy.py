"""Exercise actual timeout classification and routing without paid requests."""
import urllib.error

import pytest
from PIL import Image

from infrastructure.shared.contracts.vlm_failure import VLMTimeoutError, failure_record, is_request_timeout
from infrastructure.deformation_detect.worker import configure
from robot.preprocessing.link7_persistent.interface.config import role_config
from robot.preprocessing.link7_persistent.vlm_client import VLMClient


@pytest.mark.parametrize('error',[TimeoutError('slow'),urllib.error.URLError(TimeoutError('slow'))])
def test_cloud_timeout_yields_without_nested_retries(monkeypatch,error):
    calls=[]
    def respond(request,timeout):
        calls.append(timeout)
        raise error
    monkeypatch.setenv('VLM2_API_KEY','test-only')
    monkeypatch.setenv('PDI_VLM_TRANSPORT_ATTEMPTS','3')
    monkeypatch.setattr('urllib.request.urlopen',respond)
    with pytest.raises(VLMTimeoutError) as failed:
        VLMClient('test',role_config('vlm2')).ask([Image.new('RGB',(2,2))],'test')
    assert calls==[240]
    assert failure_record(failed.value)['category']=='vlm_timeout'
    assert not is_request_timeout(urllib.error.URLError('name resolution failed'))


def test_worker_retains_primary_after_timeout_and_gives_luna_longer(monkeypatch):
    resources={key:'test-path' for key in ('sam3_checkpoint','sam3_bpe','dino_directory','palm_references','qwen_model','ffmpeg')}
    config={'resources':resources,'environments':{'vlm1':'test-python'},'vlm':{
        'vlm2':{'model':'gemini-3.8-flash','api_base':'https://api.302.ai/v1'},
        'vlm2_malformed_fallback':{'model':'gpt-6-luna','timeout_seconds':240,'api_base':'https://api.302.ai/v1'}}}
    monkeypatch.delenv('PDI_VLM_ROLE_OVERRIDES',raising=False)
    # Configure alters environment; monkeypatch restores it after this test.
    for key in ('PATH','PDI_VLM_ROLE_OVERRIDES','PDI_SAM3_CHECKPOINT','PDI_SAM3_BPE','PDI_DINO_DIRECTORY','PDI_PALM_REFERENCES','PDI_PMASK_QWEN_MODEL','PDI_PMASK_QWEN_PYTHON','FFMPEG_BINARY','PDI_VLM_TRANSPORT_ATTEMPTS'):
        monkeypatch.setenv(key,__import__('os').environ.get(key,''))
    configure({'config':config,'attempt':2,'vlm_route':'primary'})
    assert role_config('vlm2')['model']=='gemini-3.8-flash'
    configure({'config':config,'attempt':3,'vlm_route':'alternate'})
    fallback=role_config('vlm2')
    assert fallback['model']=='gpt-6-luna'
    assert fallback['timeout_seconds']>=600 and fallback['api_base']=='https://api.302.ai/v1'
    config['vlm'].pop('vlm2_malformed_fallback')
    configure({'config':config,'attempt':2,'vlm_route':'alternate'})
    assert role_config('vlm2')['api_base']=='https://api.302.ai/v1'






def test_vlm1_timeout_preserves_successful_frames_for_deferred_resume(tmp_path,monkeypatch):
    from infrastructure.deformation_detect.coordinator import digest,read,write
    from robot.preprocessing.link7_persistent import pipeline
    from robot.workflows.coordination import persistent
    work=tmp_path/'initial/work';work.mkdir(parents=True)
    image=work/'frame.png';Image.new('RGB',(20,20)).save(image)
    calls=[{'frame':i,'crop':str(image),'sha256':digest(image)} for i in range(3)]
    write(work/'provenance.json',{'config':{'vlm1_response_request':'test','vlm1_system_prompt':'test'},'results':{
        'case':{'status':'vlm1_pending','calls':calls,'source_frame_count':4,'reference':{'path':str(image)}}}})
    observed=[]
    class Client:
        def __init__(self,*args):pass
        def ask(self,images,prompt,**kwargs):
            frame=next(i for i in range(3) if f'frame {i}.' in prompt)
            observed.append(frame)
            if observed==[0,1]:raise VLMTimeoutError('no response')
            return {'answer':'good'}
    monkeypatch.setattr('robot.preprocessing.link7_persistent.palm_recovery.parse_palm_response',lambda _: {'state':'normal'})
    monkeypatch.setattr(pipeline,'VLMClient',Client)
    request={'directory':str(tmp_path/'attempt1'),'case':{'id':'case'},'config':{'resources':{'vlm2_examples':[str(image)]*3,'ffmpeg':'unused'}},
             'stage':'vlm1','dependencies':{'link7_initial':str(work.parent)}}
    with pytest.raises(VLMTimeoutError):persistent(request)
    diagnosis=read(tmp_path/'attempt1/work/provenance.json')['results']['case']['diagnoses']
    assert len(diagnosis)==2 and diagnosis[-1]['failure']['category']=='vlm_timeout'
    request.update(directory=str(tmp_path/'attempt2'),previous_attempt=str(tmp_path/'attempt1'))
    assert persistent(request)['native_status']=='no_confirmed_deformation'
    assert observed==[0,1,1,2]


