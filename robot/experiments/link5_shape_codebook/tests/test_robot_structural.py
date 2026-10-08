"""Beam geometry and exact supervised-target contracts, independent of detector."""
import math
import unittest

import torch

from robot.experiments.link5_shape_codebook.robot_structural import RobotStructuralAugmentation


class RobotStructuralMath(unittest.TestCase):
    def setUp(self):
        self.u=torch.tensor([1/3,2/3,2/3],dtype=torch.float64)
        self.a=torch.tensor([.2,-.7,1.3],dtype=torch.float64)
        self.v=torch.linalg.cross(self.u,torch.tensor([1.,0,0],dtype=torch.float64));self.v/=self.v.norm()
        self.w=torch.linalg.cross(self.u,self.v)
        self.x=torch.linspace(0,2,31,dtype=torch.float64)
        angle=torch.arange(12,dtype=torch.float64)*2*math.pi/12
        ring=.1*(torch.cos(angle)[:,None]*self.v+torch.sin(angle)[:,None]*self.w)
        self.points=(self.a+self.x[:,None,None]*self.u+ring[None]).reshape(-1,3)
        self.generator=RobotStructuralAugmentation(dict(axis=self.u.tolist(),anchor=self.a.tolist(),length=2.0))

    def test_axial_strain_changes_only_distal_length_and_keeps_correspondence(self):
        for alpha in (.7,.95,1.05,1.3):
            normal=self.points.clone();anomalous,offset,mask=self.generator.axial(normal,alpha,.6)
            torch.testing.assert_close(normal,self.points,rtol=0,atol=0)
            torch.testing.assert_close(offset,normal-anomalous,rtol=0,atol=0)
            before=(normal-self.a)@self.u;after=(anomalous-self.a)@self.u
            expected=.8+alpha*(before-.8)
            torch.testing.assert_close(after[before>.8],expected[before>.8],rtol=0,atol=1e-12)
            torch.testing.assert_close(anomalous[before<.8-1e-10],normal[before<.8-1e-10],rtol=0,atol=0)
            perpendicular=(anomalous-normal)-((anomalous-normal)@self.u)[:,None]*self.u
            self.assertLess(float(perpendicular.abs().max()),1e-12)
            self.assertTrue(torch.equal(mask.bool(),offset.ne(0).any(-1)))

    def test_bending_preserves_each_cross_section_as_a_rigid_plane(self):
        for degrees in (10,35):
            anomalous,offset,mask=self.generator.bend(self.points,degrees,.2,.7)
            normal=self.points.reshape(31,12,3);bent=anomalous.reshape(31,12,3)
            for a,b in zip(normal,bent):
                torch.testing.assert_close(torch.cdist(a,a),torch.cdist(b,b),atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(offset,self.points-anomalous,rtol=0,atol=0)
            proximal=(self.points-self.a)@self.u<.4-1e-10
            torch.testing.assert_close(offset[proximal],torch.zeros_like(offset[proximal]),rtol=0,atol=0)
            # Circular centerline chord length approaches original arclength.
            centers=bent.mean(1)
            length=(centers[1:]-centers[:-1]).norm(dim=-1).sum()
            self.assertLess(abs(float(length)-2),1e-3)

    def test_zero_curvature_is_identity_and_distribution_uses_only_robot_types(self):
        anomalous,offset,mask=self.generator.bend(self.points,0,.2,0)
        torch.testing.assert_close(anomalous,self.points,rtol=0,atol=0)
        self.assertEqual(float(offset.abs().max()),0);self.assertEqual(float(mask.sum()),0)
        torch.manual_seed(7)
        for _ in range(400):self.generator(self.points)
        counts=self.generator.counts
        self.assertEqual(sum(counts.values()),400)
        self.assertTrue(all(value>50 for value in counts.values()))
        self.assertLess(abs((counts['shortening']+counts['lengthening'])/400-.5),.08)


if __name__=='__main__':unittest.main()
