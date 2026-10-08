"""CPU-only frame-zero pair geometry experiment on the retained current mask cache.

No tracker, geometry model, VLM, SAM, later frame, or historical track is used.
The query candidates use the existing CoTracker sampler's pure CPU methods.
This module is a proposed pair selector, not a validated rigidity detector.
"""
from __future__ import annotations
import ast
import collections
import hashlib
import json
import logging
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
CACHE = REPO / 'results/link5_shape_codebook/round_four_frame0'
OUT = CACHE / 'pair_initialization_frame0'
CONFIG = dict(queries=100, max_dimension=880, erosion_source_px=2,
              profile_bins=101, transition_half_width=.14, max_degree=3,
              transition_upper=6, transition_lower=6, transition_width=4,
              global_long=6, other_width=4, other_diagonal=4)


def cpu_sampler():
    """Load only the five unchanged NumPy/OpenCV methods, without importing Torch."""
    source=REPO/'infrastructure/shared/inference/cotracker_core.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CoTrackerCore')
    names={'_sample_region_queries','_spatially_balance_queries','_sift_sample_queries',
           '_shi_tomasi_sample_queries','_grid_sample_queries'}
    body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(body)==5
    node=ast.ClassDef(name='CPUSampler',bases=[],keywords=[],body=body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    namespace={'np':np,'cv2':cv2,'pdi_logger':logging.getLogger('pair-init')}
    exec(compile(module,str(source),'exec'),namespace)
    return namespace['CPUSampler'](),hashlib.sha256(source.read_bytes()).hexdigest()


def mask_geometry(mask):
    """A width-transition prior, invariant to translation/scale and long-axis sign.

    Choose the broad side by sustained excess width beyond the central neck,
    not by screen-left/right or a per-video annotated coordinate.
    """
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));component=labels==largest
    yy,xx=np.where(component);pixels=np.c_[xx,yy].astype(float);origin=pixels.mean(0)
    _,_,axes=np.linalg.svd(pixels-origin,full_matrices=False);axis=axes[0]
    if axis[0]<0:axis=-axis
    basis=np.stack([axis,[-axis[1],axis[0]]],axis=1)
    uv=(pixels-origin)@basis;umin=uv[:,0].min();length=np.ptp(uv[:,0]);s=(uv[:,0]-umin)/length
    grid=np.linspace(0,1,CONFIG['profile_bins']);low=[];high=[]
    for center in grid:
        vals=uv[abs(s-center)<=.015,1]
        low.append(float(np.quantile(vals,.025)) if len(vals) else np.nan)
        high.append(float(np.quantile(vals,.975)) if len(vals) else np.nan)
    low=np.array(low);high=np.array(high)
    for array in [low,high]:
        valid=np.isfinite(array);array[:]=np.interp(grid,grid[valid],array[valid])
    width=high-low
    # Fixed finite kernels, source-space relative coordinates.
    smooth=np.convolve(np.pad(width,(3,3),mode='edge'),[1,2,3,4,3,2,1],mode='valid')/16
    allowed=(grid>=.25)&(grid<=.70);minimum=smooth[allowed].min()
    near=allowed&(smooth<=minimum*1.06);neck=float(np.median(grid[near]))
    left=(grid>=.08)&(grid<neck);right=(grid>neck)&(grid<=.92)
    excess=np.maximum(smooth-minimum,0)
    l=float(excess[left].sum());r=float(excess[right].sum());side=1 if r>=l else -1
    gain=np.interp(np.clip(grid+.06,0,1),grid,smooth)-np.interp(np.clip(grid-.06,0,1),grid,smooth)
    search=(grid>=.18)&(grid<=.82)&((grid-neck)*side>=.035)
    ids=np.flatnonzero(search);transition=float(grid[ids[np.argmax(side*gain[ids])]])
    delta=float(np.interp(transition+.06*side,grid,smooth)-np.interp(transition-.06*side,grid,smooth))
    support=cv2.erode(component.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    flags=[]
    if max(l,r)/max(min(l,r),1e-6)<1.2:flags.append('ambiguous_broad_side')
    if delta/minimum<.10:flags.append('weak_width_transition')
    if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():flags.append('mask_touches_image_edge')
    if component.sum()/mask.sum()<.95:flags.append('substantial_secondary_components')
    return dict(origin=origin,basis=basis,umin=umin,length=length,grid=grid,low=low,high=high,
                width=width,smooth_width=smooth,neck=neck,transition=transition,broad_side=side,
                transition_gain_fraction=delta/minimum,broad_side_ratio=max(l,r)/max(min(l,r),1e-6),
                component_fraction=float(component.sum()/mask.sum()),support=support,flags=flags)


def initialize_queries(rgb,geometry,sampler):
    height,width=rgb.shape[:2];scale=min(1,CONFIG['max_dimension']/max(height,width))
    small=cv2.resize(rgb,(int(width*scale),int(height*scale)))
    mask=cv2.resize(geometry['support'].astype(np.uint8),(small.shape[1],small.shape[0]),interpolation=cv2.INTER_NEAREST)
    queries=sampler._sample_region_queries(cv2.cvtColor(small,cv2.COLOR_RGB2GRAY),mask,CONFIG['queries'],'link5')
    points=queries[:,1:]*[width/small.shape[1],height/small.shape[0]]
    # Resize rounding can move a candidate across the support boundary. Reject it,
    # rather than claiming it is on the raw mask. No manually moved points.
    pix=np.rint(points).astype(int);ok=geometry['support'][pix[:,1],pix[:,0]]
    return points[ok],np.flatnonzero(ok),int((~ok).sum())


def choose_pairs(points,g):
    uv=(points-g['origin'])@g['basis'];s=(uv[:,0]-g['umin'])/g['length']
    low=np.interp(s,g['grid'],g['low']);high=np.interp(s,g['grid'],g['high'])
    transverse=(uv[:,1]-(low+high)/2)/np.maximum((high-low)/2,1)
    center=g['transition'];half=CONFIG['transition_half_width'];pairs=[]
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            du=abs(s[i]-s[j]);dist=float(np.linalg.norm(points[i]-points[j]))
            local_width=float(np.interp((s[i]+s[j])/2,g['grid'],g['smooth_width']))
            if dist<max(8,.18*local_width):continue
            crosses=min(s[i],s[j])<center-.015 and max(s[i],s[j])>center+.015
            in_zone=abs(s[i]-center)<=half and abs(s[j]-center)<=half
            dt=abs(transverse[i]-transverse[j]);kind=None
            if crosses and in_zone and .06<=du<=.28 and transverse[i]<-.12 and transverse[j]<-.12:kind='transition_upper'
            elif crosses and in_zone and .06<=du<=.28 and transverse[i]>.12 and transverse[j]>.12:kind='transition_lower'
            elif abs((s[i]+s[j])/2-center)<=half and du<=.075 and transverse[i]*transverse[j]<0 and dt>=.65:kind='transition_width'
            elif du>=.50:kind='global_long'
            elif abs((s[i]+s[j])/2-center)>half and du<=.09 and transverse[i]*transverse[j]<0 and dt>=.60:kind='other_width'
            elif abs((s[i]+s[j])/2-center)>half and .07<=du<=.27 and dt>=.35:kind='other_diagonal'
            if kind:
                pairs.append(dict(i=i,j=j,kind=kind,span=du,mid_s=float((s[i]+s[j])/2),
                                  mid_v=float((transverse[i]+transverse[j])/2),distance_px=dist))
    degree=np.zeros(len(points),int);selected=[];deficits={}
    for kind in ['transition_upper','transition_lower','transition_width','other_width','other_diagonal','global_long']:
        for k in range(CONFIG[kind]):
            options=[p for p in pairs if p['kind']==kind and degree[p['i']]<CONFIG['max_degree'] and degree[p['j']]<CONFIG['max_degree'] and p not in selected]
            if not options:break
            def score(p):
                group=[q for q in selected if q['kind']==kind];reuse=(degree[p['i']]+degree[p['j']])/6
                novelty=min([np.hypot((p['mid_s']-q['mid_s'])/(half if kind.startswith('transition') else 1), (p['mid_v']-q['mid_v'])/2) for q in group],default=1)
                span_target=.70 if kind=='global_long' else .14 if kind in ['transition_upper','transition_lower','other_diagonal'] else .025
                baseline_preference=np.exp(-abs(p['span']-span_target)/.12)
                return 2*novelty-.6*reuse+.30*baseline_preference
            best=max(options,key=lambda p:(round(float(score(p)),12),-p['i'],-p['j']));selected.append(best);degree[[best['i'],best['j']]]+=1
        actual=sum(p['kind']==kind for p in selected)
        if actual!=CONFIG[kind]:deficits[kind]=CONFIG[kind]-actual
    return selected,dict(pair_count=len(selected),distinct_endpoints=int((degree>0).sum()),max_degree=int(degree.max()),
                         transition_pairs=sum(p['kind'].startswith('transition') for p in selected),deficits=deficits),s,transverse


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value


def run():
    import sys
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    sampler,sampler_hash=cpu_sampler();manifest=json.loads((CACHE/'mask_cache_manifest.json').read_text())
    source_paths=json.loads((OUT/'metadata/source_video_paths.json').read_text());results=[]
    def process(entry):
        name=entry['video_id'];folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
        mp=CACHE/entry['mask'];assert hashlib.sha256(mp.read_bytes()).hexdigest()==entry['mask_sha256']
        # Index zero only is consumed from the compressed all-frame cache.
        with np.load(mp,allow_pickle=False) as ar:mask=ar['object_masks'][0,0].astype(bool)
        cv2.imwrite(str(folder/'mask.png'),mask.astype(np.uint8)*255)
        source_path=source_paths.get(name)
        if source_path:
            assert hashlib.sha256((REPO/source_path).read_bytes()).hexdigest()==entry['source_video_sha256']
            capture=cv2.VideoCapture(str(REPO/source_path));ok,bgr=capture.read();capture.release();assert ok
            cv2.imwrite(str(folder/'frame0.png'),bgr)
        elif (folder/'frame0.png').exists():bgr=cv2.imread(str(folder/'frame0.png'))
        else:
            return dict(case=name,status='source_frame_missing',mask=str(mp))
        assert bgr.shape[:2]==mask.shape
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);g=mask_geometry(mask);points,ids,dropped=initialize_queries(rgb,g,sampler)
        pairs,stats,s,v=choose_pairs(points,g)
        # Repeat pair selection from the same inputs to check deterministic ordering.
        again,again_stats,_,_=choose_pairs(points,g);assert pairs==again and stats==again_stats
        rounded=np.rint(points).astype(int);assert mask[rounded[:,1],rounded[:,0]].all()
        assert len({(p['i'],p['j']) for p in pairs})==len(pairs);assert stats['max_degree']<=3
        result=dict(case=name,status='complete' if len(pairs)==30 else 'insufficient_geometric_support',
                    source_video_sha256=entry['source_video_sha256'],mask_path=entry['mask'],mask_sha256=entry['mask_sha256'],
                    raw_mask_frame=0,query_count=len(points),resize_boundary_rejections=dropped,query_points_xy=points,
                    point_ids=ids,pairs=pairs,stats=stats,geometry={k:val for k,val in g.items() if k!='support'},
                    config=CONFIG,sampler_source_sha256=sampler_hash,note='New CPU frame-zero query/pair initialization, not historical CoTracker tracks; no depth or temporal visibility validation.')
        (folder/'pairs.json').write_text(json.dumps(jsonable(result),indent=2))
        print(name,len(points),len(pairs),'transition',round(g['transition'],2),'side',g['broad_side'],'flags',g['flags'],stats['deficits'],flush=True)
        return jsonable(result)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:results=list(pool.map(process,manifest['entries']))
    (OUT/'metadata/results.json').write_text(json.dumps(results,indent=2))
    (OUT/'metadata/execution.json').write_text(json.dumps(dict(workers=WORKERS,seconds=time.monotonic()-started,
        cases=len(results),complete=sum(r['status']=='complete' for r in results),gpu_inference=False,
        scope='frame-zero mask geometry, CPU feature queries and pair selection only'),indent=2))
    print('FRAME0_PAIR_TEST_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=REPO);parser.add_argument('--cache',type=Path,default=CACHE)
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--source-root',type=Path);parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();REPO=args.repo.resolve();CACHE=args.cache.resolve();OUT=args.output.resolve();WORKERS=args.workers
    (OUT/'metadata').mkdir(parents=True,exist_ok=True)
    if args.source_root:
        entries=json.loads((CACHE/'mask_cache_manifest.json').read_text())['entries'];paths={}
        for entry in entries:
            generator,number=entry['video_id'].rsplit('_',1);generator='LVP' if generator=='LVP_ROBOWM' else generator
            paths[entry['video_id']]=str(args.source_root/generator/(number+'.mp4'))
        (OUT/'metadata/source_video_paths.json').write_text(json.dumps(paths,indent=2))
    run()
