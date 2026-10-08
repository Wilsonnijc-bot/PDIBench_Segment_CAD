"""Add intrinsic length/thickness and centerline features to the learned head.

Input points and the four observed normal buffers stay in the fixed reference
coordinates. Dimensionless descriptors distinguish uniform reconstruction scale
from axial strain; they are features rather than preprocessing normalization.
"""
import torch
from torch import nn

from robot.experiments.link5_shape_codebook.structural_geometry_head import StructuralGeometryDetector


class StructuralInvariantDetector(StructuralGeometryDetector):
    INTRINSIC_BINS=12
    EXTRA_FEATURES=4+3+INTRINSIC_BINS*4

    def __init__(self,native,normal_points,geometry):
        super().__init__(native,normal_points,geometry)
        original=self.geometry_mlp[0]
        self.geometry_mlp[0]=nn.Linear(original.in_features+self.EXTRA_FEATURES,original.out_features,
                                      device=original.weight.device,dtype=original.weight.dtype)

    @torch.no_grad()
    def intrinsic_features(self,points):
        rows=[]
        for xyz in points:
            centered=xyz-xyz.mean(dim=0)
            _,vectors=torch.linalg.eigh(centered.T@centered/len(xyz))
            u=vectors[:,-1]
            if torch.dot(u,self.frame_basis[:,0])<0:u=-u
            v=self.frame_basis[:,1]-u*torch.dot(u,self.frame_basis[:,1]);v=v/v.norm().clamp_min(1e-6)
            w=torch.linalg.cross(u,v)
            local=centered@torch.stack([u,v,w],dim=1)
            lo,hi=torch.quantile(local[:,0],local.new_tensor([.01,.99]));length=(hi-lo).clamp_min(1e-5)
            fraction=(local[:,0]-lo)/length
            membership=(fraction*self.INTRINSIC_BINS).floor().long().clamp(0,self.INTRINSIC_BINS-1)
            centers=[];radii=[];support=[]
            for i in range(self.INTRINSIC_BINS):
                sample=local[membership==i,1:]
                if len(sample):
                    center=sample.median(dim=0).values
                    radius=torch.quantile((sample-center).norm(dim=-1),.9)
                else:center=local.new_zeros(2);radius=local.new_zeros(())
                centers.append(center);radii.append(radius);support.append((membership==i).float().mean())
            centers=torch.stack(centers);radii=torch.stack(radii);support=torch.stack(support)
            # A transverse scale preserves length/thickness differences. It is
            # never used to rescale the observation supplied to FCGF or poses.
            thickness=2*radii[radii>1e-5].median() if (radii>1e-5).any() else local.new_tensor(1e-3)
            thickness=thickness.clamp_min(1e-3)
            relative=(local[:,1:]-centers[membership])/thickness
            point_shape=torch.cat([fraction[:,None],relative,relative.norm(dim=-1,keepdim=True)],dim=-1)
            global_shape=torch.cat([(length/thickness)[None],
                                   ((local[:,1:].amax(0)-local[:,1:].amin(0))/thickness)])
            profile=torch.cat([centers/thickness,radii[:,None]/thickness,support[:,None]],dim=-1).reshape(-1)
            context=torch.cat([global_shape,profile])
            rows.append(torch.cat([point_shape,context[None].expand(len(xyz),-1)],dim=-1))
        result=torch.stack(rows)
        if result.shape[-1]!=self.EXTRA_FEATURES or not torch.isfinite(result).all():
            raise ValueError('finite intrinsic length/thickness features required')
        return result

    def geometry_features(self,points):
        return torch.cat([super().geometry_features(points),self.intrinsic_features(points)],dim=-1)


def upgrade_geometry_model(warm,geometry):
    model=StructuralInvariantDetector(warm.native,[getattr(warm,f'normal_{i}') for i in range(4)],geometry)
    state=warm.state_dict();first=state.pop('geometry_mlp.0.weight');bias=state.pop('geometry_mlp.0.bias')
    missing,unexpected=model.load_state_dict(state,strict=False)
    if set(missing)!={'geometry_mlp.0.weight','geometry_mlp.0.bias'} or unexpected:
        raise ValueError('only the first geometry layer may expand')
    with torch.no_grad():
        target=model.geometry_mlp[0];target.weight.zero_();target.bias.copy_(bias)
        # Original input layout is [FCGF, old shape, native offset/logit]. New
        # intrinsic shape precedes the last four native-prediction columns.
        target.weight[:,:-4-model.EXTRA_FEATURES].copy_(first[:,:-4])
        target.weight[:,-4:].copy_(first[:,-4:])
    return model


def wrap_invariant_model(native,samples,geometry):
    return StructuralInvariantDetector(native,[s[1] for s in samples],geometry).to(samples[0][1].device)
