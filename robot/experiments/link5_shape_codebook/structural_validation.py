"""Fixed normal-reference longitudinal geometry, minimal examples, and numeric gates."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .common import normalize,observed_sample,read,sha,write, split_path
from .robot_structural import DEFAULTS,RobotStructuralAugmentation


def fit_geometry(output):
    split=read(split_path(output))
    normalization=read(output/'link5_normalization.json')
    sampled=[]
    for row in split['train']:
        if sha(row['input_path'])!=row['input_sha256']:raise ValueError('normal geometry input changed')
        with np.load(row['input_path'],allow_pickle=False) as archive:
            sampled.append(observed_sample(normalize(archive['xyz_camera'],normalization),2000)[0])
    pooled=np.concatenate(sampled)
    # Robust geometry estimation only; no training/inference points are removed,
    # centered or individually rescaled by this diagnostic fit.
    low,high=np.quantile(pooled,[.01,.99],axis=0)
    clean=pooled[((pooled>=low)&(pooled<=high)).all(1)]
    center=np.median(clean,axis=0)
    values,vectors=np.linalg.eigh(np.cov((clean-center).T))
    axis=vectors[:,-1]
    if axis[np.argmax(abs(axis))]<0:axis=-axis
    projection=(clean-center)@axis
    start,end=np.quantile(projection,[.01,.99])
    anchor=center+start*axis
    geometry=dict(axis=axis.tolist(),anchor=anchor.tolist(),length=float(end-start),
        source='fixed pooled frame0 reference PCA longitudinal axis; robust endpoint1/99 percentiles; no per-cloud normalization',
        longitudinal_to_transverse_variance_ratio=float(values[-1]/max(values[-2],1e-12)),
        principal_variances=values[::-1].tolist(),normal_reference_count=len(split['train']),
        split_sha256=sha(split_path(output)),normalization_sha256=sha(output/'link5_normalization.json'))
    if geometry['longitudinal_to_transverse_variance_ratio']<1.5:
        raise ValueError('normal observations do not establish a clear longitudinal axis; inspect geometry before structural training')
    write(output/'augmentation/robot_structural_geometry.json',geometry)
    return geometry


def ply(path,points):
    header=f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'
    with path.open('w') as stream:
        stream.write(header);np.savetxt(stream,points,fmt='%.9g')


def check_example(normal,result,parameters,geometry):
    anomalous,offset,mask=result
    anchor,u,x,r=RobotStructuralAugmentation(geometry).frame(normal)
    unaffected=mask==0
    record=dict(point_count=len(normal),affected_point_count=int(mask.sum()),
        exact_target_max_error=float((offset-(normal-anomalous)).abs().max()),
        unaffected_offset_max=float(offset[unaffected].abs().max()) if unaffected.any() else 0.0,
        input_correspondence_preserved=anomalous.shape==normal.shape,
        parameters=parameters)
    if parameters['type'] in ('shortening','lengthening'):
        changed=anomalous-anchor
        transverse=changed-(changed@u)[:,None]*u
    else:
        t=(x-parameters['start']).clamp_min(0)
        theta=t*parameters['kappa'];v=normal.new_tensor(parameters['bend_direction'])
        center=anchor+parameters['start']*u+(torch.sin(theta)/parameters['kappa'])[:,None]*u
        center=center+(2*torch.sin(theta/2).square()/parameters['kappa'])[:,None]*v
        center=torch.where((x>parameters['start'])[:,None],center,anchor+x[:,None]*u)
        transverse=anomalous-center
    record['cross_section_radial_norm_max_error']=float((transverse.norm(dim=-1)-r.norm(dim=-1)).abs().max())
    record['passed']=record['exact_target_max_error']<=1e-7 and record['unaffected_offset_max']<=1e-7 and record['cross_section_radial_norm_max_error']<=1e-5 and record['affected_point_count']>0
    return record


def run(config):
    output=Path(config['output']);geometry=fit_geometry(output)
    settings={**DEFAULTS,**config.get('robot_structural',{})}
    split=read(split_path(output));normalization=read(output/'link5_normalization.json')
    # Use a real normal training cloud with broad observed longitudinal coverage.
    def extent(row):
        with np.load(row['input_path'],allow_pickle=False) as archive:
            return np.ptp(archive['xyz_camera']@np.asarray(geometry['axis']))
    row=max(split['train'],key=extent)
    with np.load(row['input_path'],allow_pickle=False) as archive:
        points=observed_sample(normalize(archive['xyz_camera'],normalization),10000)[0]
    normal=torch.from_numpy(points)
    generator=RobotStructuralAugmentation(geometry,settings)
    destination=output/'augmentation/robot_structural_examples';destination.mkdir(parents=True,exist_ok=True)
    ply(destination/'normal.ply',points);np.save(destination/'normal.npy',points,allow_pickle=False)
    examples={};checks=[]
    for name,parameters in [('shortened',dict(alpha=.8,affected_fraction=.8)),('lengthened',dict(alpha=1.2,affected_fraction=.8)),
                            ('mild_bend',dict(angle_degrees=10,start_fraction=.2,direction_angle=.5)),
                            ('strong_bend',dict(angle_degrees=35,start_fraction=.2,direction_angle=.5))]:
        result=generator.axial(normal,**parameters) if 'alpha' in parameters else generator.bend(normal,**parameters)
        numeric=check_example(normal,result,dict(generator.last_parameters),geometry)
        checks.append(dict(example=name,**numeric));examples[name]=result[0].numpy()
        ply(destination/(name+'.ply'),examples[name])
        np.savez_compressed(destination/(name+'.npz'),normal=points,anomalous=examples[name],gt_offset=result[1].numpy(),gt_mask=result[2].numpy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,anomalous in examples.items():
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for axis,(x,y) in zip(axes,[(0,1),(0,2),(1,2)]):
            axis.scatter(points[::5,x],points[::5,y],s=2,alpha=.3,color='gray',label='normal')
            axis.scatter(anomalous[::5,x],anomalous[::5,y],s=2,alpha=.4,color='tab:red',label=name)
            axis.set_aspect('equal',adjustable='datalim');axis.set_xlabel('XYZ'[x]);axis.set_ylabel('XYZ'[y])
        axes[-1].legend();fig.suptitle(name+' — same observed points and correspondence, fixed normalization')
        fig.tight_layout();fig.savefig(destination/(name+'_comparison.png'),dpi=160);plt.close(fig)
    receipt=dict(status='passed' if all(row['passed'] for row in checks) else 'failed',checks=checks,
        geometry=geometry,settings=settings,source_video=row['video_id'],source_input_sha256=row['input_sha256'],
        generated_missing_surfaces=False,offset_definition='P_normal - P_anomaly',
        augmentation_source_sha256=sha(Path(__file__).with_name('robot_structural.py')))
    write(output/'augmentation/robot_structural_validation.json',receipt)
    if receipt['status']!='passed':raise ValueError('structural augmentation numeric validation failed; full training is disabled')
    print('LINK5_ROBOT_STRUCTURAL_AUGMENTATION_VALIDATED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();run(read(args.config))


if __name__=='__main__':main()
