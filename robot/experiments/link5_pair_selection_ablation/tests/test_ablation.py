import ast,math,collections,json,unittest
from pathlib import Path
import numpy as np
from robot.experiments.link5_pair_selection_ablation.balanced_v0 import select_balanced
from robot.experiments.link5_pair_selection_ablation.score import audit_3d_rigidity_cv
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[4]
class Contracts(unittest.TestCase):
    def test_earlier_proposal_exact_parity(self):
        source=ROOT/'robot/experiments/link5_rigidity_review/results/20260929/provenance/pair_selection.py'
        tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select')
        ns=dict(np=np,math=math,collections=collections);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        for case in ['COSMOS2.5_0010','COSMOS3_0010','LVP_ROBOWM_0015','COSMOS2.5_0015','COSMOS3_0015']:
            d=json.loads((source.parent.parent/case/'evidence.json').read_text())
            xyz,xy,_,_,_,expected,_=ns['select'](d);actual,stats=select_balanced(xy,xyz)
            self.assertEqual(expected,actual);self.assertEqual(stats['pair_count'],30)
    def test_rigid_deformed_and_carry_scoring(self):
        xy=np.array([[1,1],[2,1],[3,1],[4,1],[5,1]],float)
        tracks=np.tile(xy,(4,1,1));pm=np.zeros((4,8,8,3));vis=np.ones((4,5));mask=np.ones((4,8,8),bool)
        for t in range(4):
            for i,(x,y) in enumerate(xy.astype(int)):pm[t,y,x]=[i*(2 if t==1 else 1),0,1]
        for i,(x,y) in enumerate(xy.astype(int)):pm[2,y,x,0]=i*i
        vis[3]=0
        pairs=[dict(i=0,j=j) for j in range(1,5)]
        with patch('robot.experiments.link5_pair_selection_ablation.score.select_pairs',return_value=(pairs,{})):
            evidence={};score,h=audit_3d_rigidity_cv(pm,tracks,vis,mask,evidence=evidence,insufficient_policy='raise')
        ratios=np.array([1,2,3,4]);expected=np.median(abs(ratios-np.median(ratios)))/(np.median(ratios)+1e-6)
        self.assertEqual(h[0],0);self.assertEqual(h[1],0);self.assertEqual(h[2],expected);self.assertEqual(h[3],h[2]);self.assertEqual(evidence['carried_frames'],[3]);self.assertEqual(score,np.mean(h[1:]))
        # Coherent scaling is intentionally invisible to MAD/median; minority
        # changes can also be suppressed. Preserve this baseline behavior.
    def test_exclusions(self):
        from robot.experiments.link5_pair_selection_ablation.prepare_manifest import build,EXCLUDED
        m=build(ROOT/'results/link5_shape_codebook/round_four_frame0')
        self.assertEqual(m['primary_count'],40);self.assertEqual(m['additional_count'],1)
        self.assertFalse(set(EXCLUDED)&{e['video_id'] for e in m['entries']})
if __name__=='__main__':unittest.main()
