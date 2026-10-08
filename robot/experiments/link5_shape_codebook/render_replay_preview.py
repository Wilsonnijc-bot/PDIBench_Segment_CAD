"""One scientific preview of a completed matched triplet; no model inference."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from .common import normalize,raw_metrics,read


def run(config,number,mode,fraction):
    root=Path(config['output']);full=root/'full_video';reference=read(root/'link5_normalization.json')
    cases=[g+'_'+number for g in ('LVP_ROBOWM','COSMOS2.5','COSMOS3')]
    data=[];limit=0.
    for case in cases:
        folder=full/'cases'/case;summary=read(folder/f'video_{mode}.json');rows=read(folder/f'frames_{mode}.json')
        frame=round(fraction*(summary['expected_frame_count']-1));observation=folder/'observations'/f'frame_{frame:05d}'
        suffix='' if mode=='paper_original' else '_robot_structural'
        with np.load(observation/'input.npz',allow_pickle=False) as a:rgb=a['rgb'].copy();pixels=a['pixels_yx'].copy()
        with np.load(observation/f'anomaly{suffix}.npz',allow_pickle=False) as a:
            points=a['points_normalized'].copy();scores=a['raw_point_scores'].copy();ids=a['sampled_input_indices'].copy()
        assert np.isclose(raw_metrics(scores)['raw_mean_top80'],rows[frame]['raw_mean_top80'])
        with np.load(folder/'observations/frame_00000/input.npz',allow_pickle=False) as a:normal=normalize(a['xyz_camera'],reference)
        limit=max(limit,float(np.quantile(scores,.95)))
        data.append((case,summary,rows,frame,rgb,pixels[ids],points,scores,normal))
    fig=plt.figure(figsize=(16,11),facecolor='#f6f7f8');norm=Normalize(0,max(limit,1e-6));scatter=None
    for col,(case,summary,rows,frame,rgb,pixels,points,scores,normal) in enumerate(data):
        ax=fig.add_subplot(3,3,col+1);ax.imshow(rgb);ax.scatter(pixels[:,1],pixels[:,0],c=scores,cmap='viridis',norm=norm,s=1,alpha=.65,rasterized=True)
        ax.set_title(case+'\nFrame '+str(frame)+' / '+str(summary['expected_frame_count']-1)+' · raw top80 '+format(rows[frame]['raw_mean_top80'],'.5g'),fontsize=11);ax.axis('off')
        ax=fig.add_subplot(3,3,col+4,projection='3d');rng=np.random.default_rng(0)
        take=rng.choice(len(points),min(3500,len(points)),replace=False);ref_ids=rng.choice(len(normal),min(2000,len(normal)),replace=False)
        ax.scatter(*normal[ref_ids].T,c='#999fa4',s=1,alpha=.25,rasterized=True)
        scatter=ax.scatter(*points[take].T,c=scores[take],cmap='viridis',norm=norm,s=2,alpha=.8,rasterized=True)
        ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),zlim=(-1.5,1.5));ax.set_box_aspect((1,1,1));ax.view_init(elev=16,azim=-65)
        ax.set_title('Canonicalized cloud · frame0 reference in gray',fontsize=10);ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z');ax.tick_params(labelsize=7)
        ax=fig.add_subplot(3,3,col+7);x=[r['frame_id'] for r in rows];y=[r.get('raw_mean_top80',np.nan) for r in rows]
        ax.plot(x,y,color='#167a8c',lw=1.5);ax.axvline(frame,color='#d78c32',ls='--',lw=1)
        ax.set_title('Every-frame sum: '+format(summary['sum_all_frame_raw_mean_top80'],'.6g')+' · '+str(summary['scored_frame_count'])+'/'+str(summary['expected_frame_count'])+' frames',fontsize=11)
        ax.set_xlabel('Original frame number');ax.set_ylabel('Raw frame top80');ax.grid(alpha=.2)
    fig.suptitle('Link5 full-video replay · matched number '+number+' · '+mode,fontsize=17)
    fig.text(.5,.015,'One shared raw color scale. Every source frame contributes to the video sum, including frame0. Exploratory detector: synthetic-sensitivity checks failed.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.955));fig.colorbar(scatter,ax=[fig.axes[1],fig.axes[4],fig.axes[7]],fraction=.018,pad=.01,label='Raw point score')
    dest=full/'replay/preview.png';dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=130);plt.close(fig)
    print('LINK5_REPLAY_PREVIEW_RENDERED',dest,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--number',default='0001');parser.add_argument('--mode',choices=('paper_original','robot_structural'),default='robot_structural')
    parser.add_argument('--fraction',type=float,default=.5);args=parser.parse_args();run(read(args.config),args.number,args.mode,args.fraction)


if __name__=='__main__':main()
