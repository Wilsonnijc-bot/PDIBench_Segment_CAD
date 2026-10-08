"""Exercise author code with a test-only encoder fixture, never training a detector."""
import unittest
from pathlib import Path
import sys
import tempfile

import torch

from robot.experiments.link5_shape_codebook.upstream import shape_modules,codebook_state,restore_hash_keys,training_module


class AuthorIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.modules,cls.source=shape_modules()

    def test_public_default_attention_dimension_is_invalid(self):
        with self.assertRaises(AssertionError):self.modules['network'].HierarchicalAnomalyNet()

    def test_training_entry_point_cannot_be_shadowed_by_backbone_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'train.py').write_text('raise RuntimeError("foreign backbone training entry point")')
            sys.path.insert(0,directory)
            try:
                module=training_module()
                self.assertEqual(Path(module.__file__).resolve().parent.name,'Shape-Anomaly-Codebook')
                self.assertIs(module.AnomalyLoss,self.modules['losses'].AnomalyLoss)
            finally:sys.path.remove(directory)

    def test_original_modules_support_sparse_observations_and_raw_scores(self):
        torch.manual_seed(5)
        # A minimal fixture verifies the published external_encoder boundary;
        # actual experiment only accepts the strict pretrained Minkowski adapter.
        fixture=torch.nn.Linear(3,32)
        model=self.modules['network'].HierarchicalAnomalyNet(scales=[(8,192),(32,64),(64,32)],
            attention_head_dim=66,external_encoder=fixture,codebook_threshold=.85)
        xyz=torch.randn(1,256,3)*.1
        model.update_codebook(xyz)
        codebook=codebook_state(model)
        self.assertTrue(all(record['size']>0 for record in codebook))
        offset,logits,aux=model(xyz)
        self.assertEqual(offset.shape,(1,256,3));self.assertEqual(logits.shape,(1,256))
        raw=offset.abs().sum(-1)*logits.sigmoid()
        self.assertTrue(torch.isfinite(raw).all())
        paper=self.modules['losses'].offset_to_score(offset,logits)
        self.assertGreaterEqual(float(paper.min()),0);self.assertLessEqual(float(paper.max()),1)
        loss,_=self.modules['losses'].AnomalyLoss()(offset,logits,torch.zeros_like(offset),torch.zeros_like(logits))
        loss.backward();self.assertTrue(torch.isfinite(fixture.weight.grad).all())
        for book in model.codebook.books:book.hash_keys=[set() for _ in book.hash_keys]
        restore_hash_keys(model,codebook)
        self.assertEqual(codebook_state(model),codebook)

    def test_missing_region_augmentation_audit_matches_actual_author_code(self):
        torch.manual_seed(3)
        xyz=torch.randn(256,3)*.05;normals=torch.nn.functional.normalize(torch.randn_like(xyz),dim=-1)
        augmentation=self.modules['augmentation'].NegativeAugmentation()
        for kind in ('holes','angle_displacement','plane_missing'):
            anom,offset,mask=augmentation(xyz,normals,atype=kind,severity=.1)
            torch.testing.assert_close(anom,xyz)
            self.assertGreater(float(offset.abs().sum()),0)
            self.assertGreater(float(mask.sum()),0)


if __name__=='__main__':unittest.main()
