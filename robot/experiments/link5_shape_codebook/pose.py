"""Independent FoundationPose registration, original mesh convention and QC."""
import argparse
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np

from .common import ROOT, MODES, mode_folder, check_se3, read, sha, transform_points, write, split_path
from .upstream import POSE_REVISION, source_record


class Link5Pose:
    def __init__(self,config):
        import torch
        import trimesh
        source=ROOT/'infrastructure/vendor/FoundationPose'
        self.provenance=source_record(source,POSE_REVISION)
        self.provenance['weights_sha256']={name:sha(source/'weights'/name/'model_best.pth') for name in
            ('2023-10-28-18-33-37','2024-01-11-20-02-45')}
        sys.path.insert(0,str(source))
        from estimater import FoundationPose
        from Utils import nvdiffrast_render
        self.render=nvdiffrast_render
        cad=Path(config['cad_path'])
        loaded=trimesh.load(str(cad),process=False)
        self.mesh=loaded.to_geometry() if isinstance(loaded,trimesh.Scene) else loaded
        self.mesh.vertices=np.asarray(self.mesh.vertices)*float(config.get('cad_unit_scale',1.0))
        self.provenance.update(cad_sha256=sha(cad),cad_unit_scale=config.get('cad_unit_scale',1.0))
        # Native renderer expects SimpleMaterial.image; current trimesh loads
        # this DAE as PBRMaterial. Convert appearance only using trimesh's own
        # adapter, preserving all CAD vertices, faces and texture coordinates.
        material=getattr(self.mesh.visual,'material',None)
        if material is not None and not hasattr(material,'image'):
            vertices=self.mesh.vertices.copy();faces=self.mesh.faces.copy()
            self.mesh.visual.material=material.to_simple()
            if not np.array_equal(vertices,self.mesh.vertices) or not np.array_equal(faces,self.mesh.faces):
                raise ValueError('CAD material compatibility conversion changed geometry')
            self.provenance['CAD_material_compatibility']='trimesh PBRMaterial.to_simple(); vertices/faces/UV unchanged'
        # Verify actual source rather than inferring transform convention.
        text=(source/'estimater.py').read_text()
        if 'best_pose = poses[0]@self.get_tf_to_centered_mesh()' not in text or 'return best_pose.data.cpu().numpy()' not in text:
            raise ValueError('FoundationPose register convention changed; review before alignment')
        self.provenance['transform_convention']='register returns T_camera<-original_mesh; original center offset already composed'
        self.provenance['native_pose_depth_preprocessing']='FoundationPose register applies radius2 erosion/bilateral filtering internally for pose estimation only; scored camera cloud remains the exact Link5-filtered input'
        self.provenance['pose_depth_input']='full aligned MegaSAM CVD depth, finite positive samples; raw mask as supplied; Link5 erosion/density filtering applies only to detector cloud'
        self.estimator=FoundationPose(model_pts=self.mesh.vertices,model_normals=self.mesh.vertex_normals,
                                      mesh=self.mesh,debug=0,debug_dir=config['debug_directory'])
        # Native compute_mesh_diameter returns np.float64. Current torch.tensor
        # infers float64 from that scalar inside the crop-offset list, while
        # camera intrinsics/poses are float32. Keep the same diameter value and
        # express it as a Python float for the native float32 crop pipeline.
        self.estimator.diameter=float(self.estimator.diameter)
        self.provenance['native_scalar_compatibility']='NumPy mesh diameter -> Python float; native float32 crop offsets, same geometry/value'
        self.registration_iterations=int(config.get('registration_iterations',5))
        if self.registration_iterations<1:raise ValueError('native registration needs at least one refinement iteration')
        self.provenance['registration_iterations']=self.registration_iterations
        self.mask_association_fallback=bool(config.get('mask_association_fallback',False))
        self.provenance['mask_association_fallback']=self.mask_association_fallback
        self.provenance['mask_association_policy']='if native top1 IoU<0.1 and native bank maxIoU>=0.5 with gain>=0.4, select existing maxIoU hypothesis; otherwise preserve native top1; no refit'

    def estimate(self,input_path,output):
        import torch
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        with np.load(input_path,allow_pickle=False) as archive:
            rgb=archive['rgb'].copy();depth=archive['depth'].copy();k=archive['K'].copy()
            mask=archive['mask'].copy();valid=archive['valid'].copy();points=archive['xyz_camera'].copy()
        status=dict(status='failed',valid_point_count=len(points),mask_area=int(mask.sum()),
                    registration='independent; no track_one/ICP/nonrigid registration',input_sha256=sha(input_path),provenance=self.provenance)
        try:
            # No temporal pose initializer: native register generates new hypotheses.
            self.estimator.pose_last=None
            if hasattr(self.estimator,'scores'):del self.estimator.scores
            # Official run_demo passes full aligned scene depth separately from
            # the object mask. Keeping neighboring scene geometry is important
            # when repeated robot-link appearances occur in the RGB crop.
            pose_depth=np.where(np.isfinite(depth)&(depth>0),depth,0).astype(np.float32)
            t=check_se3(self.estimator.register(K=k,rgb=rgb,depth=pose_depth,
                                              ob_mask=mask.astype(np.uint8),iteration=self.registration_iterations))
            h,w=depth.shape
            # Render the ORIGINAL mesh with the returned original-mesh pose.
            _,render_depth,_=self.render(K=k,H=h,W=w,ob_in_cams=torch.as_tensor(t,dtype=torch.float32,device='cuda')[None],
                                        mesh=self.mesh,glctx=self.estimator.glctx,extra={})
            silhouette=render_depth[0].cpu().numpy()>0
            intersection=(silhouette&mask).sum();union=(silhouette|mask).sum()
            iou=float(intersection/max(1,union))
            native_t=t.copy();native_iou=iou;selected_rank=0;association_used=False;bank_best_iou=None
            if self.mask_association_fallback and iou<.1 and hasattr(self.estimator,'scores') and hasattr(self.estimator,'poses'):
                candidates,ious=self.candidate_mask_qc(k,mask)
                candidate_rank=int(ious.argmax().item());bank_best_iou=float(ious[candidate_rank])
                if bank_best_iou>=.5 and bank_best_iou-iou>=.4:
                    t=check_se3(candidates[candidate_rank].cpu().numpy())
                    selected_rank=candidate_rank;association_used=True
                    _,render_depth,_=self.render(K=k,H=h,W=w,ob_in_cams=torch.as_tensor(t,dtype=torch.float32,device='cuda')[None],
                        mesh=self.mesh,glctx=self.estimator.glctx,extra={})
                    silhouette=render_depth[0].cpu().numpy()>0
                    iou=float((silhouette&mask).sum()/max(1,(silhouette|mask).sum()))
            preview=rgb.copy();preview[silhouette]=(preview[silhouette]*.5+np.array([30,200,255])*.5).astype(np.uint8)
            contours,_=cv2.findContours(silhouette.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(preview,contours,-1,(0,255,255),1)
            cv2.imwrite(str(output/'cad_projection.png'),preview[...,::-1])
            cv2.imwrite(str(output/'cad_mask.png'),silhouette.astype(np.uint8)*255)
            cad_camera=transform_points(self.mesh.vertices,t)
            observed_center=points.mean(0)
            distance=float(np.linalg.norm(cad_camera.mean(0)-observed_center))
            diameter=float(np.linalg.norm(np.ptp(cad_camera,axis=0)))
            # Weak fit is descriptive QC, never an anomaly-dependent exclusion.
            catastrophic=not hasattr(self.estimator,'scores') or not silhouette.any() or np.mean(cad_camera[:,2]>0)<.9 or distance>2*diameter
            confidence=None
            if hasattr(self.estimator,'scores'):
                confidence=float(torch.as_tensor(self.estimator.scores).reshape(-1)[selected_rank].item())
            status.update(status='catastrophic' if catastrophic else 'ok',T_camera_from_link5=t.tolist(),
                          pose_rank_score=confidence,pose_rank_score_is_calibrated_confidence=False,cad_mask_iou=iou,
                          cad_centroid_distance=distance,weak_fit=iou<.1,
                          native_top1_T_camera_from_link5=native_t.tolist(),native_top1_cad_mask_iou=native_iou,
                          selected_native_sorted_rank=selected_rank,mask_association_fallback_used=association_used,
                          native_bank_best_mask_iou=bank_best_iou,
                          catastrophic_policy='native fallback/no scores, no CAD projection, CAD behind camera, or centroid >2 CAD diameters')
        except Exception as exc:
            status.update(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            traceback.print_exc()
        write(output/'pose.json',status)
        return status

    def candidate_mask_qc(self,k,mask):
        """Project native registered hypotheses; no pose fitting or refinement."""
        import torch
        poses=self.estimator.poses@self.estimator.get_tf_to_centered_mesh()
        observed=torch.as_tensor(mask,device='cuda',dtype=torch.bool)
        ious=[]
        for start in range(0,len(poses),16):
            _,depth,_=self.render(K=k,H=mask.shape[0],W=mask.shape[1],ob_in_cams=poses[start:start+16],
                mesh=self.mesh,glctx=self.estimator.glctx,extra={})
            silhouette=depth>0
            intersection=(silhouette&observed).sum(dim=(1,2))
            union=(silhouette|observed).sum(dim=(1,2)).clamp_min(1)
            ious.append(intersection.float()/union)
        return poses,torch.cat(ious)


def reference(config):
    output=Path(config['output']);split=read(split_path(output))
    if read(output/'alignment/audit.json')['status']!='passed' or split['alignment_audit_sha256']!=sha(output/'alignment/audit.json'):
        raise ValueError('fixed reference pose requires the current passing frame0 alignment audit')
    # Cleanest naturally aligned medoid was first selected by geometry audit.
    row=split['train'][0]
    if sha(row['input_path'])!=row['input_sha256']:raise ValueError('normal reference observation changed after selection')
    if (output/'link5_normalization.json').exists():
        record=read(output/'link5_normalization.json')
        if record['reference_video']!=row['video_id'] or record['reference_input_sha256']!=sha(row['input_path']) or record['cad_sha256']!=sha(config['pose']['cad_path']):
            raise ValueError('fixed reference normalization provenance changed; do not overwrite it')
        check_se3(record['T_ref'])
        print('LINK5_FIXED_REFERENCE_NORMALIZATION_RETAINED',flush=True)
        return
    adapter=Link5Pose({**config['pose'],'debug_directory':str(output/'metadata/foundationpose-debug')})
    if (output/'reference_pose/pose.json').exists():
        previous=read(output/'reference_pose/pose.json')
        if previous['status']!='ok':
            history=output/'metadata/reference_pose_attempts.json'
            attempts=read(history) if history.exists() else []
            attempts.append(previous);write(history,attempts)
    record=adapter.estimate(row['input_path'],output/'reference_pose')
    reference_passed=record['status']=='ok' and record['cad_mask_iou']>=.25
    write(output/'reference_pose/reference_qc.json',dict(status='passed' if reference_passed else 'failed',
        pose_status=record['status'],cad_mask_iou=record.get('cad_mask_iou'),minimum_reference_iou=.25,
        reference_video=row['video_id'],reference_input_sha256=row['input_sha256'],
        policy='clean normal reference gate only; runtime weak fits remain scoreable'))
    if not reference_passed:
        raise ValueError('clean normal T_ref failed CAD projection QC; fixed normalization cannot be defined')
    vertices=transform_points(adapter.mesh.vertices,record['T_camera_from_link5'])
    center=(vertices.min(0)+vertices.max(0))/2
    scale=float(np.linalg.norm(vertices-center,axis=-1).max())
    write(output/'link5_normalization.json',dict(fixed_center=center.tolist(),fixed_scale=scale,
        T_ref=record['T_camera_from_link5'],reference_video=row['video_id'],reference_frame=0,
        reference_input_sha256=row['input_sha256'],cad_sha256=adapter.provenance['cad_sha256'],
        coordinate_system='unmodified frame0 camera',definition='transformed CAD bounding-box center and max CAD vertex radius',
        cad_use='fixed center/scale and rigid poses only; never normal anomaly templates',upstream_per_sample_normalization=False))
    print('LINK5_REFERENCE_NORMALIZATION_COMPLETE',flush=True)


def runtime(config,mode):
    output=Path(config['output'])
    receipt=read(mode_folder(output,'sanity',mode)/'checks.json')
    if receipt['status']!='passed' or receipt['checkpoint_sha256']!=sha(mode_folder(output,'training',mode)/'link5/final.pt'):
        raise ValueError('full runtime pose estimation requires the passing detector sanity gate')
    if receipt['normalization_sha256']!=sha(output/'link5_normalization.json') or receipt['split_sha256']!=sha(split_path(output)):
        raise ValueError('runtime pose gate provenance changed')
    adapter=Link5Pose({**config['pose'],'debug_directory':str(output/'metadata/foundationpose-debug')})
    records=[]
    for case in config['cases']:
        for row in read(output/'cases'/case['id']/'observations.json'):
            if row['frame_id']==0:continue
            folder=Path(row['input_path']).parent/'pose'
            if (folder/'pose.json').is_file():
                record=read(folder/'pose.json')
                if record['input_sha256']!=sha(row['input_path']) or record['provenance']!=adapter.provenance:
                    raise ValueError('owned independent pose cache provenance changed')
            else:
                # Reuse immutable CAD/CNN weights and renderer only. Each call
                # clears temporal state and invokes native register, never track.
                record=adapter.estimate(row['input_path'],folder)
            records.append(dict(video_id=row['video_id'],frame_id=row['frame_id'],status=record['status']))
            print('LINK5_INDEPENDENT_RUNTIME_POSE',row['video_id'],row['frame_id'],record['status'],flush=True)
    write(output/'metadata/runtime-pose-exits.json',records)
    print('LINK5_INDEPENDENT_RUNTIME_POSES_FINISHED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--reference',action='store_true')
    parser.add_argument('--runtime',action='store_true')
    parser.add_argument('--augmentation-mode',choices=MODES,default='paper_original')
    args=parser.parse_args();config=read(args.config)
    if args.reference:reference(config)
    elif args.runtime:runtime(config,args.augmentation_mode)
    else:
        adapter=Link5Pose({**config['pose'],'debug_directory':str(Path(config['output'])/'metadata/foundationpose-debug')})
        status=adapter.estimate(args.input,args.output)
        if status['status']!='ok':raise RuntimeError('FoundationPose failure: '+status['status'])


if __name__=='__main__':main()
