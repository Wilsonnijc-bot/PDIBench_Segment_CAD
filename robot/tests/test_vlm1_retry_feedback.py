import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from robot.preprocessing.link7_persistent import pipeline


class RetryFeedbackTests(unittest.TestCase):
    def test_retry_keeps_successful_frames_and_reasks_bad_frame_with_same_images(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            paths=[folder/name for name in ('reference.png','bad.png','good.png')]
            for path in paths:Image.new('RGB',(20,16)).save(path)
            calls=[{'frame':29,'crop':str(paths[1]),'sha256':'bad'},
                   {'frame':30,'crop':str(paths[2]),'sha256':'good'}]
            feedback={'error':'Invisible target cannot receive a binary verdict',
                      'answer':'{"state":"deformed","probability":null}'}
            cached={**calls[1],'state':'deformed','raw_response':'accepted','parse_error':''}
            result={'status':'vlm1_pending','calls':calls,'reference':{'path':str(paths[0])},
                    'source_frame_count':93,'diagnoses':[{**calls[0],'retry_feedback':feedback},cached]}
            record={'results':{'case':result},'config':{'vlm1_response_request':'Return JSON.',
                                                       'vlm1_system_prompt':'Assess visibility.'}}
            response={'segment':'palm','state':'unclear','issue':'target_not_visible',
                      'evidence':'The palm is occluded.','severity':'unclear','probability':None}
            client=Mock();client.ask.return_value={'answer':json.dumps(response)}
            args=Mock();args.cases=['case']
            with patch.object(pipeline,'load',return_value=record),patch.object(pipeline,'commit'),\
                 patch.object(pipeline,'VLMClient',return_value=client):
                pipeline.select_frames(args)
            self.assertEqual(client.ask.call_count,1)
            images,prompt=client.ask.call_args.args
            self.assertEqual([image.size for image in images],[(20,16),(20,16)])
            self.assertIn(feedback['error'],prompt)
            self.assertIn('Reassess the SAME two images',prompt)
            self.assertEqual(result['diagnoses'][0]['state'],'unclear')
            self.assertEqual(result['diagnoses'][1],cached)
            self.assertEqual(result['reseeding_frame'],31)


if __name__=='__main__':unittest.main()
