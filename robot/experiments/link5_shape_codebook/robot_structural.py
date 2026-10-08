"""Correspondence-preserving axial strain and circular-arc beam bending.

This replaces only the pseudo-anomaly callable, not any author model or loss.
All geometry is in the one fixed-normalized frame0 reference coordinate system.
"""
import math

import torch


DEFAULTS = dict(axial_probability=.5,shortening_alpha=[.70,.95],lengthening_alpha=[1.05,1.30],
    axial_affected_fraction_ranges=[[.40,.60],[.60,.90],[.90,1.0]],
    bend_start_fraction=[0,.5],mild_bend_angle_degrees=[5,15],strong_bend_angle_degrees=[20,45],
    mild_bend_probability=.5,max_region_resamples=8)


class RobotStructuralAugmentation:
    def __init__(self,geometry,settings=None):
        self.geometry=geometry
        self.settings={**DEFAULTS,**(settings or {})}
        self.last_parameters={}
        self.counts={name:0 for name in ('shortening','lengthening','mild_bend','strong_bend')}
        axis=torch.as_tensor(geometry['axis'],dtype=torch.float64)
        if axis.shape!=(3,) or not torch.isfinite(axis).all() or abs(axis.norm().item()-1)>1e-5:
            raise ValueError('structural axis must be a fixed unit vector')
        if geometry['length']<=0:raise ValueError('structural reference length must be positive')
        for key in ('axial_probability','mild_bend_probability'):
            if not 0<=self.settings[key]<=1:raise ValueError(key+' must lie in [0,1]')

    @staticmethod
    def uniform(interval,points):
        return interval[0]+(interval[1]-interval[0])*torch.rand((),device=points.device,dtype=points.dtype).item()

    def frame(self,points):
        if points.ndim!=2 or points.shape[1]!=3 or not torch.isfinite(points).all():
            raise ValueError('structural generator needs finite observed [N,3] points')
        u=points.new_tensor(self.geometry['axis'])
        anchor=points.new_tensor(self.geometry['anchor'])
        q=points-anchor
        x=q@u
        r=q-x[:,None]*u
        return anchor,u,x,r

    @staticmethod
    def targets(normal,anomalous):
        offset=normal-anomalous
        # Actual floating-point modification, not merely membership in a region.
        mask=offset.ne(0).any(dim=-1).to(normal.dtype)
        return anomalous,offset,mask

    def axial(self,points,alpha,affected_fraction):
        if alpha<=0 or not 0<affected_fraction<=1:raise ValueError('invalid axial strain parameters')
        anchor,u,x,r=self.frame(points)
        start=(1-affected_fraction)*self.geometry['length']
        t=(x-start).clamp_min(0)
        # x' = x for x<=s, otherwise s + alpha*(x-s).
        # Cross-sectional r is unchanged; no isotropic scaling or recentering.
        anomalous=points+(alpha-1)*t[:,None]*u
        self.last_parameters=dict(type='shortening' if alpha<1 else 'lengthening',alpha=alpha,
            affected_fraction=affected_fraction,start=start,anchor=self.geometry['anchor'],axis=self.geometry['axis'])
        return self.targets(points,anomalous)

    def bend(self,points,angle_degrees,start_fraction,direction_angle):
        if not 0<=start_fraction<1:raise ValueError('invalid bending start fraction')
        anchor,u,x,r=self.frame(points)
        # Stable transverse basis from a Cartesian axis least parallel to u.
        basis=torch.eye(3,device=points.device,dtype=points.dtype)[u.abs().argmin()]
        e1=torch.linalg.cross(u,basis);e1=e1/e1.norm()
        e2=torch.linalg.cross(u,e1)
        v=math.cos(direction_angle)*e1+math.sin(direction_angle)*e2
        w=torch.linalg.cross(u,v)
        start=start_fraction*self.geometry['length']
        span=(1-start_fraction)*self.geometry['length']
        kappa=math.radians(angle_degrees)/span
        if abs(kappa)<1e-12:
            self.last_parameters=dict(type='bend',angle_degrees=angle_degrees,start=start,kappa=kappa)
            return self.targets(points,points.clone())
        t=(x-start).clamp_min(0);theta=kappa*t
        sine=torch.sin(theta);cosine=torch.cos(theta)
        # Circular arc parametrized by axial arclength t. Use 2*sin(theta/2)^2
        # instead of 1-cos(theta) to avoid cancellation for very mild bends.
        center=anchor+start*u+(sine/kappa)[:,None]*u+(2*torch.sin(theta/2).square()/kappa)[:,None]*v
        # Rodrigues rotation about w: u rotates toward v and every point in
        # each section receives the same rigid rotation R_w(kappa*t).
        rotated=r*cosine[:,None]+torch.linalg.cross(w.expand_as(r),r)*sine[:,None]
        rotated=rotated+w*((r@w)*(1-cosine))[:,None]
        anomalous=torch.where((x>start)[:,None],center+rotated,points)
        self.last_parameters=dict(type='bend',angle_degrees=angle_degrees,start_fraction=start_fraction,
            start=start,kappa=kappa,direction_angle=direction_angle,bend_direction=v.cpu().tolist(),
            rotation_axis=w.cpu().tolist(),anchor=self.geometry['anchor'],axis=self.geometry['axis'])
        return self.targets(points,anomalous)

    def __call__(self,points,normals=None,atype=None,severity=None):
        # Same three-output boundary used by the author's Phase2 loop.
        # Normal estimation remains native; this distribution does not use it.
        if atype is None:
            if torch.rand(()).item()<self.settings['axial_probability']:
                atype='shortening' if torch.rand(()).item()<.5 else 'lengthening'
            else:
                atype='mild_bend' if torch.rand(()).item()<self.settings['mild_bend_probability'] else 'strong_bend'
        for attempt in range(self.settings['max_region_resamples']):
            if atype in ('shortening','lengthening'):
                alpha=self.uniform(self.settings[atype+'_alpha'],points)
                ranges=self.settings['axial_affected_fraction_ranges']
                region=int(torch.randint(len(ranges),(1,)).item())
                result=self.axial(points,alpha,self.uniform(ranges[region],points))
            elif atype in ('mild_bend','strong_bend'):
                angle=self.uniform(self.settings[atype+'_angle_degrees'],points)
                result=self.bend(points,angle,self.uniform(self.settings['bend_start_fraction'],points),
                                 self.uniform([0,2*math.pi],points))
                self.last_parameters['type']=atype
            else:raise ValueError('unknown robot structural family: '+str(atype))
            if result[2].any():
                self.counts[atype]+=1;self.last_parameters['region_resamples']=attempt
                return result
        raise ValueError('no observed points in sampled structural deformation regions; do not hallucinate missing surfaces')
