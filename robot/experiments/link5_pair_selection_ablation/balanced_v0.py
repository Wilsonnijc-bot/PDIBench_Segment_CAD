"""Earlier balanced proposal, ported from the audited prototype.
Same thresholds, objective, order and tie-breaks. Missing quota slots are reported,
not asserted or silently filled. xyz=None is explicitly a 2D-only preview.
xy uses the tracker/pointmap pixel grid; xyz uses its world coordinate system.
"""
import math
import numpy as np

def select_balanced(xy, xyz=None):
    xy=np.asarray(xy,float)
    if len(xy)<5 or not np.isfinite(xy).all():
        raise ValueError('At least five finite frame-zero points required')
    if xyz is not None:
        xyz=np.asarray(xyz,float)
        if xyz.shape!=(len(xy),3) or not np.isfinite(xyz).all():
            raise ValueError('xyz must contain finite world positions for each point')
    axis=np.linalg.svd(xy-xy.mean(0))[2][0]
    if axis[0]<0:axis=-axis
    rotation=np.array([axis,[-axis[1],axis[0]]]).T; uv=(xy-xy.mean(0))@rotation
    length=np.ptp(uv[:,0]);width=np.ptp(uv[:,1]);lo=uv[:,0].min()
    # Baseline floor is relative to this reconstruction, not a claim of metric units.
    distances=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1) if xyz is not None else np.full((len(xy),len(xy)),np.inf)
    floor=.02*np.max(distances) if xyz is not None else 0.
    candidates=[]
    for i in range(len(xy)):
        for j in range(i+1,len(xy)):
            delta=abs(uv[j]-uv[i]);dist=float(np.linalg.norm(delta));angle=float(np.degrees(np.arctan2(delta[1],delta[0])))
            if dist<max(6,.30*width) or distances[i,j]<floor:continue
            mid=(uv[i]+uv[j])/2;band=min(4,int((mid[0]-lo)/length*5))
            kind=None
            if angle>=55:kind='transverse'
            elif 20<=angle<55 and delta[0]<=.30*length:kind='diagonal'
            elif angle<20 and delta[0]>=.45*length:kind='longitudinal'
            if kind is None:continue
            candidates.append(dict(i=i,j=j,kind=kind,band=band,angle=angle,distance_2d=dist,baseline_3d=float(distances[i,j]),mid=mid.tolist()))
    degree=np.zeros(len(xy),int);selected=[];deficits=[]
    # Spread local-pair midpoints: two transverse and two diagonals per section.
    requests=[(kind,b) for repeat in range(2) for b in [2,1,3,0,4] for kind in ['transverse','diagonal']]
    requests += [('longitudinal',None)]*10
    for kind,band in requests:
        options=[p for p in candidates if p['kind']==kind and (band is None or p['band']==band) and degree[p['i']]<3 and degree[p['j']]<3 and p not in selected]
        if not options:
            deficits.append(dict(kind=kind,band=band));continue
        def objective(p):
            i,j=p['i'],p['j']; midpoint=np.array(p['mid'])
            same=[q for q in selected if q['kind']==kind]
            # Reward a new spatial measurement, not a longer baseline indefinitely.
            spread=min([np.linalg.norm((midpoint-np.array(q['mid']))/[length,width]) for q in same],default=1.)
            reuse=(degree[i]+degree[j])/6
            target=.75*length if kind=='longitudinal' else .7*width if kind=='transverse' else width
            preferred_length=math.exp(-abs(math.log(p['distance_2d']/target)))
            preferred_angle=(p['angle']-55)/35 if kind=='transverse' else 1-abs(p['angle']-37.5)/17.5 if kind=='diagonal' else 1-p['angle']/20
            return 2*spread+.35*preferred_length+.20*preferred_angle-.8*reuse
        best=max(options,key=lambda p:(objective(p),-p['i'],-p['j']));selected.append(best);degree[[best['i'],best['j']]]+=1
    return selected, dict(pair_count=len(selected),distinct_endpoints=int((degree>0).sum()),max_degree=int(degree.max()),deficits=deficits,depth_baseline_gate_applied=xyz is not None)
