"""Create one isolated ERIS trial from the current staged source and frozen45."""
import argparse
from pathlib import Path
import shutil
import subprocess

from .common import ROOT, read, sha, write, FOUR_REFERENCE_IDS
from .robot_structural import DEFAULTS
from robot.preprocessing.depth.link5_depth_filter import Config, settings


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deployment',type=Path,required=True)
    parser.add_argument('--workspace-name',default='workspace-link5-shape-codebook-four-frame0-20261006')
    args=parser.parse_args();base=args.deployment;dependency=base.parent/'dependency'
    workspace=base/args.workspace_name
    if not workspace.exists():shutil.copytree(base/'workspace-link5-20261005',workspace,symlinks=True,
        ignore=shutil.ignore_patterns('results','__pycache__','*.pyc'))
    # The package must already have been extracted to invoke this module. Never
    # re-extract an older snapshot over subsequently verified adapter fixes.
    cfg=read(base/'metadata/manifest-group-a-link5-ready.json')
    cfg['run_id']='link5-shape-codebook-four-frame0-20261006'
    owned_output=workspace/'robot/experiments/link5_shape_codebook/results'
    legacy_output=workspace/'results/link5_shape_codebook'
    owned_output.mkdir(parents=True,exist_ok=True)
    legacy_output.parent.mkdir(exist_ok=True)
    if legacy_output.is_symlink():
        if legacy_output.resolve()!=owned_output.resolve():raise ValueError('Result alias points to a different owner')
    elif legacy_output.exists():
        raise ValueError('Legacy result directory exists; preserve it before staging a fresh run')
    else:
        legacy_output.symlink_to('../robot/experiments/link5_shape_codebook/results',target_is_directory=True)
    cfg['output']=str(owned_output)
    selected=read(base/'metadata/selected-videos.json')['staging_cases']
    cfg['cases']=[dict(id=c['id'],video=str(base/'inputs/videos'/c['video_relative']),
                       video_sha256=sha(base/'inputs/videos'/c['video_relative'])) for c in selected]
    cfg['robot_links']=['link5']
    cfg['normal_reference']=dict(video_ids=list(FOUR_REFERENCE_IDS),frame_id=0,
        construction='four exact observed clouds; pooled union for replay; four separate normal training samples',
        later_frames_allowed=False)
    cfg['robot_structural']=DEFAULTS
    cfg['mask_policy']=dict(erosion_pixels=Config().erosion_pixels,simple3d_mask_mapping=False,source_mask='raw guarded mask retained; detector support filtered',depth_filter='ray_normalized_3d_knn_density_v1')
    cfg['depth_filter']=settings(Config())
    cfg.setdefault('vlm',{}).setdefault('vlm2',{}).update(model='gemini-3.8-flash',reasoning_effort='high',max_tokens=65536,timeout_seconds=600)
    cfg['vlm'].setdefault('vlm2_malformed_fallback',{}).update(model='gpt-6-luna',reasoning_effort='xhigh',max_completion_tokens=65536,timeout_seconds=1200,temperature=None)
    cfg.setdefault('policy',{})['vlm_stage_timeout_seconds']=7200
    cfg['execution']=dict(workers=5,gpu_slots=1,gpu_assignment='shared')
    cfg['preparation']=dict(workers_per_gpu=5)
    def replace(value):
        if isinstance(value,str):return value.replace(str(base/'workspace-link5-20261005'),str(workspace))
        if isinstance(value,list):return [replace(v) for v in value]
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        return value
    cfg=replace(cfg)
    positive_reference=workspace/'documentation/data/references/link5_guard/positive_three_points_reference.png'
    source_reference=ROOT/'documentation/data/references/link5_guard/positive_three_points_reference.png'
    if source_reference.resolve()!=positive_reference.resolve():
        positive_reference.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_reference,positive_reference)
    if not positive_reference.is_file():raise FileNotFoundError(positive_reference)
    cfg['resources']['link5_positive_guard_reference']=str(positive_reference)
    cfg['pose']=dict(cad_path=str(dependency/'models/franka_fer/link5.dae'),cad_unit_scale=1.0,
        registration_iterations=5,mask_association_fallback=True)
    cfg['environments']['foundationpose']=str(dependency/'env/foundationpose/bin/python')
    cfg['environments']['shape_codebook']=str(dependency/'env/shape-codebook/bin/python')
    cfg['backbone']=dict(source_root=str(dependency/'source/FCGF'),module='model.resunet',class_name='ResUNetBN2C',
        source_revision='499813ea35f57d885cbc3458cd74fc57dae860c7',
        kwargs=dict(in_channels=1,out_channels=32,D=3,normalize_feature=True,conv1_kernel_size=7,bn_momentum=.05),voxel_size=2/256,
        checkpoint=str(dependency/'models/shape_codebook/fcgf-3dmatch-32feat.pth'),checkpoint_sha256=None,state_key='state_dict',
        checkpoint_url='https://huggingface.co/chrischoy/FCGF/resolve/main/2019-08-19_06-17-41.pth',
        provenance='User-authorized official FCGF 3DMatch pretrained32-D MinkowskiEngine ResUNetBN2C substitute for paper MinkUNet34C; no PointNet substitute')
    cfg['sanity']=dict(normal_video_ids=['LVP_ROBOWM_0001','LVP_ROBOWM_0005','LVP_ROBOWM_0010'],
        normal_video_policy='provisional LVP candidates for rigid-pose QC; only frame0 normality is user-confirmed, later-frame normality is not independently established',
        rigid_score_relative_tolerance=.02,synthetic_deformation_min_ratio=1.2,
        heldout_normal_p95_vs_train_p95_max_ratio=3,pose_alignment_median_diagonal=.10,pose_alignment_p95_diagonal=.30)
    path=owned_output/'config.json';write(path,cfg)
    write(owned_output/'metadata/staging.json',dict(status='staged',workspace=str(workspace),
        source_archive_sha256=sha(base/'metadata/link5-shape-source.tar.gz'),config_sha256=sha(path),
        videos=len(cfg['cases']),normal_reference_video_ids=list(FOUR_REFERENCE_IDS),normal_reference_frame_id=0,
        mask_policy='fresh three-positive VLM guard and SAM; no historical masks or geometry'))
    print(path)


if __name__=='__main__':main()
