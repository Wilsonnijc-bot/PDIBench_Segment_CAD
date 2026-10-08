"""One small result index, including failed gates and unavailable outputs."""
import argparse
from pathlib import Path

from .common import read, split_path


def run(output):
    def optional(path):
        file=output/path
        return read(file) if file.is_file() else None
    config=read(output/'config.json')
    completion=optional('completion.json')
    alignment=optional('alignment/audit.json')
    poses=optional('sanity/pose_checks.json')
    reference_qc=optional('reference_pose/reference_qc.json') or optional('reference_pose/pose.json')
    checks=optional('sanity/checks.json')
    trained=optional('training/completion.json')
    structural_trained=optional('training_robot_structural/completion.json')
    structural_checks=optional('sanity_robot_structural/checks.json')
    structural_completion=optional('completion_robot_structural.json')
    full_video_config=optional('full_video/config.json')
    full_video_completion=optional('full_video/completion.json')
    scope=optional('full_video/metadata/evaluation_scope.json')
    backbone=optional('metadata/pretrained-backbone.json')
    split=read(split_path(output)) if split_path(output).exists() else None
    count=len(split['train']) if split else len(config.get('normal_reference',{}).get('video_ids',[])) or 20
    videos=optional('scores/videos.json') or []
    scores={row['video_id']:row for row in videos}
    train_ids={row['video_id'] for row in split['train']} if split else set()
    rows=[]
    for case in config['cases']:
        status=optional(f"cases/{case['id']}/status.json") or dict(status='pending')
        observation=optional(f"cases/{case['id']}/observations/frame_00000/observation.json")
        rows.append((case,status,observation))
    finished=sum(status['status']=='complete' for _,status,_ in rows)
    state='preparing'
    failure=None
    for name,receipt in [('frame0 alignment',alignment),('reference pose',reference_qc),('runtime pose alignment',poses),('detector sanity',checks)]:
        if receipt and receipt['status'] in ('failed','catastrophic'):failure=name
    if failure:state='gated: '+failure+' failed'
    elif completion:state=completion['status']
    elif trained:state='trained; evaluation gates pending'
    elif alignment and alignment['status']=='passed':state='frame0 alignment passed; pose/training pending'
    if full_video_config:state='full-video exploratory evaluation '+(full_video_completion['status'] if full_video_completion else 'running')
    lines=['# Link5 deformation trial','',f'**Status:** {state}. Fresh preparation: {finished}/{len(config["cases"])} videos.','',
           f'Normal detectors use {count} frame0 observations in their original camera coordinates. The remaining {len(config["cases"])-count} videos are held out. Later frames are filtered test observations, aligned with independent FoundationPose and shared fixed normalization; they never enter normal training.','',
           'The detector calls the [authors’ implementation](https://github.com/alexandor91/Shape-Anomaly-Codebook) directly. It is an audited trial, not a paper-exact reproduction. The released code differs from the [paper](https://arxiv.org/html/2604.03972v1); see the method audit and actual backbone below.','']
    links=[('Full-video interactive replay','full_video/replay/index.html'),
           (f'Frame0 prompts, masks and {count} references','full_video/reference_replay/index.html'),
           ('Local guard cost comparison','refinement/COST_COMPARISON.md'),
           ('Local depth-filter provenance','refinement/metadata/depth_summary.json'),
           ('Human forearm AB correlations','full_video/analysis/forearm_correlation.md'),
           ('Full-video structural matched-generator sums','full_video/scores/robot_structural/matched_generators.csv'),
           ('Full-video original matched-generator sums','full_video/scores/paper_original/matched_generators.csv'),
           ('Frame0 overlay','alignment/frame0_overlay.png'),('Geometry and scale audit','alignment/audit.json'),
           ('Exact normal-reference split',str(split_path(output).relative_to(output))),('Fixed normalization','link5_normalization.json'),
           ('Reference CAD projection','reference_pose/cad_projection.png'),('Runtime pose overlay','sanity/runtime_pose_overlay.png'),
           ('Reference pose quality gate','reference_pose/reference_qc.json'),
           ('Pose checks before training','sanity/pose_checks.json'),('Effective training configuration','training/effective_config.json'),
           ('Training loss curves','training/loss_curves.png'),('Detector checkpoint','training/link5/final.pt'),
           ('Method deviations','metadata/paper_audit.json'),('Required sanity checks','sanity/checks.json'),
           ('All frame raw metrics','scores/frames.csv'),('Video aggregates','scores/videos.csv'),
           ('Visibility correlations','scores/visibility_correlations.json')]
    links.extend([('Structural augmentation numeric checks','augmentation/robot_structural_validation.json'),
        ('Shortening comparison','augmentation/robot_structural_examples/shortened_comparison.png'),
        ('Lengthening comparison','augmentation/robot_structural_examples/lengthened_comparison.png'),
        ('Mild bending comparison','augmentation/robot_structural_examples/mild_bend_comparison.png'),
        ('Strong bending comparison','augmentation/robot_structural_examples/strong_bend_comparison.png'),
        ('Structural effective configuration','training_robot_structural/effective_config.json'),
        ('Structural loss curves','training_robot_structural/loss_curves.png'),
        ('Structural checkpoint','training_robot_structural/link5/final.pt'),
        ('Structural detector sanity','sanity_robot_structural/checks.json'),
        ('Structural raw frame metrics','scores_robot_structural/frames.csv'),
        ('Structural video aggregates','scores_robot_structural/videos.csv')])
    lines.extend(f'- [{label}]({path})' for label,path in links if (output/path).is_file())
    erosion=next((o['mask_erosion_pixels'] for _,_,o in rows if o and 'mask_erosion_pixels' in o),
        config.get('mask_policy',{}).get('erosion_pixels',0))
    filter_description=('The Link5 ray-normalized 3D density filter rejects sparsely supported observed depth samples.' if erosion else
        'The previous Simple3D local median/MAD filter rejects isolated depth samples.')
    lines.extend(['',f'Raw guarded masks are retained. Detector support uses {erosion} original-image pixels of erosion; no Simple3D cropping or mask mapping. '+filter_description,'',
        f'The separate `robot_structural` run changes only the Phase2 pseudo-anomaly distribution. Both modes share the exact {count} frame0 normal references, split, normalization, frozen pretrained backbone and native normal-only codebook construction/update procedure.'])
    if structural_trained or structural_checks or structural_completion:
        lines.append('Structural run: '+('complete' if structural_completion else
            'gated: sanity failed' if structural_checks and structural_checks['status']=='failed' else
            'trained; evaluation pending' if structural_trained else 'pending')+'.')
    if backbone:
        lines.extend(['',f"Actual backbone: `{backbone['class_name']}`, pretrained FCGF3DMatch32-D; checkpoint SHA256 `{backbone['checkpoint_sha256']}`. This architecture/checkpoint differs from paper MinkUNet34C."])
    if full_video_config:
        if scope:lines.extend(['',f"Current evaluation scope: **{scope['selected_video_count']} full videos**, ten matched video numbers across three generators. Original45-video launch configuration is preserved; the user requested stopping the allocation after this30-video cohort completed. Only selected complete video sums enter the forearm AB correlation analysis.",''])
        lines.extend(['','The user requested full-video exploratory testing after the sensitivity failures were disclosed. Those failed receipts remain unchanged. Every decoded frame is retained; the primary video score is the sum of every raw_mean_top80 frame score, including frame0. The ten-frame testing configuration belongs only to Simple3D. Missing frames are not zeros: the complete sum is unavailable if any score is missing; a separately labeled available-frame subtotal remains recorded. Both checkpoints score identical clouds and independent FoundationPose poses.',''])
    if failure and not full_video_config:
        lines.extend(['','The failed gate prevents full evaluation. Missing scores below are unavailable results, not zeros.'])
    lines.extend(['','The primary frame score is **raw_mean_top80**, before any per-cloud minmax normalization. Each anomaly archive also retains the author-style point map, raw values, predicted offsets, validity logits, observed-point indices and aligned coordinates. Full-video scores sum every frame including frame0.','',
                  '| Video | Split | Preparation | Frame0 points | Frame0 mask | Raw max / median / top3 |',
                  '|---|---|---|---:|---|---|'])
    for case,status,observation in rows:
        video_id=case['id'];folder=f'cases/{video_id}'
        membership=('train' if video_id in train_ids else 'held out') if split else 'pending'
        points=f"[{observation['valid_point_count']}]({folder}/observations/frame_00000/input.npz)" if observation else '—'
        mask=f'[{observation["mask_area"]}]({folder}/observations/frame_00000/mask.png)' if observation else '—'
        score=scores.get(video_id)
        metrics=' / '.join(f'{score[key]:.6g}' if score[key] is not None else '—' for key in
            ('max_sampled_frame_score','median_sampled_frame_score','mean_top3_sampled_frame_scores')) if score else '—'
        preparation=f'[{status["status"]}]({folder}/status.json)' if (output/folder/'status.json').is_file() else status['status']
        lines.append(f'| {video_id} | {membership} | {preparation} | {points} | {mask} | {metrics} |')
    lines.extend(['','Raw masks and their current-runtime guard provenance remain in each case’s `masking/` folder. `observations/` separates exact RGB/depth/cloud inputs from CAD projection and prediction outputs. Native scratch and build files are omitted from the review export; preserved source hashes and manifests identify the run.',''])
    (output/'README.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    run(Path(read(args.config)['output']))


if __name__=='__main__':main()
