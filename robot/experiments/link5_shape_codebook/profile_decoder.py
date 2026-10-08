"""Learned shape-profile restoration control conditioned on four normal clouds.

The frozen native book/network remain available in the checkpoint, but this
ablation decodes intrinsic geometry directly. It does not run native retrieval
for its score. Profile reference statistics use only the four original clouds.
No observed point is recentered/rescaled before pipeline/backbone processing.
"""
import torch
from torch import nn
from robot.experiments.link5_shape_codebook.structural_geometry_head import StructuralGeometryDetector


class StructuralProfileDecoder(StructuralGeometryDetector):
    BINS=12
    DESCRIPTOR_SIZE=4+3+BINS*4

    def __init__(self,native,normal_points,geometry):
        super().__init__(native,normal_points,geometry)
        self.geometry_mlp.requires_grad_(False);self.offset_head.requires_grad_(False);self.mask_head.requires_grad_(False)
        profiles=[]
        for points in normal_points:
            descriptor,_,_=self.canonical_features(points[None]);profiles.append(descriptor[0,0,4:])
        self.register_buffer('reference_profiles',torch.stack(profiles))
        self.register_buffer('reference_profile_mean',self.reference_profiles.mean(0))
        self.register_buffer('reference_profile_std',self.reference_profiles.std(0).clamp_min(.05))
        size=self.DESCRIPTOR_SIZE+(self.DESCRIPTOR_SIZE-4)
        self.profile_mlp=nn.Sequential(nn.Linear(size,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,64),nn.GELU())
        self.profile_offset=nn.Linear(64,3);self.profile_mask=nn.Linear(64,1)

    @torch.no_grad()
    def canonical_features(self,points):
        descriptors=[];frames=[];thicknesses=[]
        for xyz in points:
            centered=xyz-xyz.mean(0)
            _,vectors=torch.linalg.eigh(centered.T@centered/len(xyz))
            u=vectors[:,-1]
            if torch.dot(u,self.frame_basis[:,0])<0:u=-u
            v=vectors[:,-2]
            guide=self.frame_basis[:,2]
            if abs(float(torch.dot(v,guide)))<.1:guide=self.frame_basis[:,1]
            if torch.dot(v,guide)<0:v=-v
            w=torch.linalg.cross(u,v);frame=torch.stack([u,v,w],dim=1)
            local=centered@frame
            lo,hi=torch.quantile(local[:,0],local.new_tensor([.01,.99]));length=(hi-lo).clamp_min(1e-5)
            fraction=(local[:,0]-lo)/length
            bins=(fraction*self.BINS).floor().long().clamp(0,self.BINS-1)
            centers=[];radii=[];support=[]
            for b in range(self.BINS):
                sample=local[bins==b,1:]
                if len(sample):
                    center=sample.median(0).values;radius=torch.quantile((sample-center).norm(dim=-1),.9)
                else:center=local.new_zeros(2);radius=local.new_zeros(())
                centers.append(center);radii.append(radius);support.append((bins==b).float().mean())
            centers=torch.stack(centers);radii=torch.stack(radii);support=torch.stack(support)
            thickness=2*radii[radii>1e-5].median() if (radii>1e-5).any() else local.new_tensor(1e-3)
            thickness=thickness.clamp_min(1e-3)
            relative=(local[:,1:]-centers[bins])/thickness
            local_feature=torch.cat([fraction[:,None],relative,relative.norm(dim=-1,keepdim=True)],dim=-1)
            global_feature=torch.cat([(length/thickness)[None],(local[:,1:].amax(0)-local[:,1:].amin(0))/thickness,
                                      torch.cat([centers/thickness,radii[:,None]/thickness,support[:,None]],dim=-1).reshape(-1)])
            descriptors.append(torch.cat([local_feature,global_feature[None].expand(len(xyz),-1)],dim=-1))
            frames.append(frame);thicknesses.append(thickness)
        result=torch.stack(descriptors)
        if result.shape[-1]!=self.DESCRIPTOR_SIZE or not torch.isfinite(result).all():raise ValueError('finite canonical shape descriptors required')
        return result,torch.stack(frames),torch.stack(thicknesses)

    def forward(self,points):
        shape,frame,thickness=self.canonical_features(points)
        if self.reference_exclusion is None:mean,std=self.reference_profile_mean,self.reference_profile_std
        else:
            if self.reference_exclusion not in (0,1,2,3):raise ValueError('invalid reference exclusion')
            profiles=self.reference_profiles[torch.arange(4,device=points.device)!=self.reference_exclusion]
            mean=profiles.mean(0);std=profiles.std(0).clamp_min(.05)
        context=((shape[:,:,4:]-mean)/std).clamp(-10,10)
        learned=self.profile_mlp(torch.cat([shape,context],dim=-1))
        canonical=self.profile_offset(learned)
        offset=torch.bmm(canonical,frame.transpose(1,2))*thickness[:,None,None]
        logits=self.profile_mask(learned).squeeze(-1)
        aux=dict(best_scale=torch.full(points.shape[:2],-1,dtype=torch.long,device=points.device),
                 method='learned intrinsic profile restoration; native scale retrieval bypassed in this control')
        return offset,logits,aux


def wrap_profile_model(native,samples,geometry):
    return StructuralProfileDecoder(native,[s[1] for s in samples],geometry).to(samples[0][1].device)
