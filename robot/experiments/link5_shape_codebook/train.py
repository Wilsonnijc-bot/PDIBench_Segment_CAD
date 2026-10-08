"""Call the author's training entry point with dataset/backbone/logging adapters."""
import argparse
import csv
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import yaml

from .common import ROOT, MODES, mode_folder, normalize, observed_sample, read, sha, write, split_path, reference_ids, normal_manifest_path
from .upstream import build_model, codebook_state, shape_modules,training_module


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--augmentation-mode',choices=MODES,default='paper_original')
    args=parser.parse_args();experiment=read(args.config);output=Path(experiment['output']);mode=args.augmentation_mode
    split=read(split_path(output));normalization=read(output/'link5_normalization.json')
    ids=reference_ids(experiment)
    expected_count=len(ids) if ids else 20
    train_ids={r['video_id'] for r in split['train']}
    expected_test=0 if ids and split.get('reference_only') else len(experiment['cases'])-expected_count
    if len(split['train'])!=expected_count or len(split['test'])!=expected_test or train_ids & {r['video_id'] for r in split['test']} or any(r['frame_id']!=0 for r in split['train']):
        raise ValueError('normal training requires a disjoint frame0-only split')
    if ids and train_ids!=set(ids):raise ValueError('training references differ from the four requested videos')
    for row in split['train']:
        if row['input_sha256']!=sha(row['input_path']):raise ValueError('training reference input changed')
    if split['alignment_audit_sha256']!=sha(output/'alignment/audit.json'):raise ValueError('training alignment receipt changed')
    if read(output/'alignment/audit.json')['status']!='passed':raise ValueError('normal alignment gate failed')
    pose_receipt=read(output/'sanity/pose_checks.json')
    if pose_receipt['status']!='passed' or pose_receipt['normalization_sha256']!=sha(output/'link5_normalization.json'):
        raise ValueError('normal runtime pose alignment must pass before detector training')
    if ids and ({r['video_id'] for r in pose_receipt['pose_checks']}!=set(ids) or any(r['frame_id']!=0 or not r['passed'] for r in pose_receipt['pose_checks'])):
        raise ValueError('four normal frame0 pose checks required')
    training=mode_folder(output,'training',mode);training.mkdir(parents=True,exist_ok=True)
    if (training/'link5/final.pt').exists():raise ValueError('this one detector has already been trained')
    modules,source=shape_modules()
    cfg=yaml.safe_load((ROOT/'infrastructure/vendor/Shape-Anomaly-Codebook/config.yaml').read_text())
    cfg['data'].update(normalize=False,num_points=10000)
    cfg['patches']['scales']=[[8,192],[32,64],[64,32]]
    cfg['model']['attention_head_dim']=66  # Native RoPE requires divisibility by6.
    cfg['codebook'].update(threshold=.85,hash_grid=256)
    cfg['augmentation']['severities']=[.01,.1]
    cfg['train'].update(epochs=1500,lr=.001,optimizer='adam',batch_size=1)
    cfg['loss'].update(lambda_sim=.5,lambda_bce=.5)
    cfg['backbone']=experiment['backbone']
    cfg['augmentation_mode']=mode
    geometry=None
    if mode=='robot_structural':
        from .robot_structural import DEFAULTS,RobotStructuralAugmentation
        validation=read(output/'augmentation/robot_structural_validation.json')
        if validation['status']!='passed':raise ValueError('structural augmentation numeric gate failed')
        geometry=read(output/'augmentation/robot_structural_geometry.json')
        if geometry['split_sha256']!=sha(split_path(output)) or geometry['normalization_sha256']!=sha(output/'link5_normalization.json'):
            raise ValueError('structural augmentation uses a different normal reference')
        if validation['augmentation_source_sha256']!=sha(Path(__file__).with_name('robot_structural.py')):
            raise ValueError('structural augmentation changed after validation')
        cfg['augmentation']['types']=['shortening','lengthening','mild_bend','strong_bend']
        cfg['augmentation']['compatibility_unused_paper_severities']=True
        cfg['augmentation']['robot_structural']={**DEFAULTS,**experiment.get('robot_structural',{})}
        cfg['augmentation']['structural_geometry']=geometry

    # The paired experiment may differ only in its pseudo-anomaly distribution.
    other_mode='robot_structural' if mode=='paper_original' else 'paper_original'
    other_training=mode_folder(output,'training',other_mode)
    other_config=other_training/'effective_config.json'
    control={key:value for key,value in cfg.items() if key not in ('augmentation','augmentation_mode')}
    if other_config.exists():
        other=read(other_config)
        if control!={key:value for key,value in other.items() if key not in ('augmentation','augmentation_mode')}:
            raise ValueError('paired runs differ outside the pseudo-anomaly configuration')
        if read(normal_manifest_path(other_training))!=split['train']:
            raise ValueError('paired runs do not use the same exact normal normal observations')

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    write(training/('normal_manifest.json' if ids else 'train20_manifest.json'),split['train'])
    write(training/'pipeline_config.json',experiment)
    write(training/'effective_config.json',cfg)
    write(training/'paired_control.json',dict(non_augmentation_configuration=control,
        split_sha256=sha(split_path(output)),normalization_sha256=sha(output/'link5_normalization.json'),
        compared_existing_other_run=other_config.exists(),only_intended_difference='Phase2 pseudo-anomaly distribution'))
    write(training/'upstream_sources.json',source)
    adapter_sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    write(training/'adapter_sources.json',adapter_sources)
    write(training/'command.json',dict(interpreter=sys.executable,module='robot.experiments.link5_shape_codebook.train',
        arguments=['--config',str(args.config),'--augmentation-mode',mode],native_seed=0,backbone_frozen=True))
    shutil.copy2(ROOT/'robot/experiments/link5_shape_codebook/paper_audit.json',training/'paper_audit.json')
    model,_,_=build_model(cfg) # Fail before training if actual pretrained encoder is unavailable.
    data_root=output/'normal_training_data'
    dataset=data_root/'link5/train/normal';dataset.mkdir(parents=True,exist_ok=True)
    expected={row['video_id']+'.npy' for row in split['train']}
    if set(p.name for p in dataset.glob('*.npy'))-expected:raise ValueError('training folder contains observations outside the exact normal manifest')
    for row in split['train']:
        if sha(row['input_path'])!=row['input_sha256']:raise ValueError('training observation identity changed')
        with np.load(row['input_path'],allow_pickle=False) as archive:points=normalize(archive['xyz_camera'],normalization)
        path=dataset/(row['video_id']+'.npy')
        if path.exists():
            if not np.array_equal(np.load(path),points):raise ValueError('shared normal training data changed')
        else:
            temporary=path.with_suffix(f'.{os.getpid()}.tmp')
            with temporary.open('wb') as stream:np.save(stream,points,allow_pickle=False)
            temporary.replace(path)

    # Probe the actual author model/loss on the largest real training cloud.
    # No optimizer update: reset codebook and gradients before native training.
    largest=max(split['train'],key=lambda row:row['valid_point_count'])
    xyz=torch.from_numpy(observed_sample(np.load(dataset/(largest['video_id']+'.npy')),10000)[0]).cuda()[None]
    model.cuda()
    augmentation=(RobotStructuralAugmentation(geometry,cfg['augmentation']['robot_structural']) if mode=='robot_structural'
        else modules['augmentation'].NegativeAugmentation(modules['augmentation'].AugConfig(severities=(.01,.1))))
    probe_loss=modules['losses'].AnomalyLoss(lambda_sim=.5,lambda_bce=.5)
    elapsed=[]
    for index in range(3):
        torch.cuda.synchronize();started=time.monotonic()
        normals=modules['augmentation'].estimate_normals(xyz[0])
        anomalous,offset,mask=augmentation(xyz[0],normals,atype='shortening' if mode=='robot_structural' else 'sink',severity=.01)
        model.update_codebook(xyz)
        predicted,logits,_=model(anomalous[None])
        loss,parts=probe_loss(predicted,logits,offset[None],mask[None])
        if not torch.isfinite(loss):raise ValueError('actual native training startup loss is nonfinite')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('actual native training startup gradient is nonfinite')
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize();elapsed.append(time.monotonic()-started)
    write(training/'preflight.json',dict(status='passed',actual_pretrained_backbone=model.encoder.provenance,
        video_id=largest['video_id'],point_count=int(xyz.shape[1]),native_loss=parts,
        full_native_batch_seconds=elapsed,estimated_native_training_seconds=float(np.mean(elapsed[1:])*1500*expected_count),
        optimizer_updates=0,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()))
    print('LINK5_ACTUAL_NATIVE_TRAINING_PREFLIGHT_PASSED',elapsed,flush=True)
    model.codebook.reset();model.zero_grad(set_to_none=True)
    del xyz,normals,anomalous,offset,mask,predicted,logits,loss
    effective=training/'effective_config.yaml';effective.write_text(yaml.safe_dump(cfg))
    # The original dataset is used with normalize=False. Replace its sampling
    # boundary only: points below10000 remain observed and are never duplicated.
    def sample(points,n,rng):
        return points[rng.choice(len(points),min(n,len(points)),replace=False)]
    modules['dataset'].uniform_sample=sample
    author_train=training_module()
    if Path(author_train.__file__).resolve().parent!=(ROOT/'infrastructure/vendor/Shape-Anomaly-Codebook').resolve():
        raise RuntimeError('training entry point is not the pinned author code')
    author_train.build_model=lambda ignored:model
    original_loader=author_train.DataLoader
    execution=experiment.get('training_execution',{})
    if execution.get('loader')=='native_equivalent_single_process':
        from .training_loader import NativeEquivalentLoader
        benchmark=read(output/'metadata/training_loader_benchmark.json')
        if benchmark['status']!='passed' or benchmark['source_sha256']!=sha(Path(__file__).with_name('training_loader.py')):
            raise ValueError('optimized loader requires matching native-equivalence benchmark')
        author_train.DataLoader=NativeEquivalentLoader
    write(training/'execution_policy.json',dict(loader=execution.get('loader','native_two_workers'),
        batch_size=1,epochs=1500,normal_clouds=expected_count,
        sampler_and_optimizer_unchanged=True,source_sha256=sha(Path(__file__).with_name('training_loader.py')) if execution.get('loader')=='native_equivalent_single_process' else None))
    original_augmentation=author_train.NegativeAugmentation
    structural_generators=[]
    if mode=='robot_structural':
        def factory(original_aug_config):
            generator=RobotStructuralAugmentation(geometry,cfg['augmentation']['robot_structural'])
            structural_generators.append(generator)
            return generator
        author_train.NegativeAugmentation=factory
    # Capture native Phase1 construction without altering any update or loop.
    normal_updates=0
    native_update=model.update_codebook
    def recorded_update(points):
        nonlocal normal_updates
        native_update(points);normal_updates+=1
        if normal_updates==expected_count*cfg['train']['codebook_warmup_epochs']:
            initial=dict(normal_codebook=model.codebook.state_dict(),hash_state=codebook_state(model),
                split_sha256=sha(split_path(output)),normalization_sha256=sha(output/'link5_normalization.json'))
            temporary=training/f'phase1_normal_codebook.{os.getpid()}.tmp'
            original_save(initial,temporary)
            temporary.replace(training/'phase1_normal_codebook.pt')
            write(training/'phase1_normal_codebook.json',dict(only_normal_inputs=True,normal_observations=expected_count,
                statistics=initial['hash_state'],split_sha256=initial['split_sha256'],normalization_sha256=initial['normalization_sha256']))
            other_snapshot=other_training/'phase1_normal_codebook.pt'
            if other_snapshot.exists():
                other_initial=torch.load(other_snapshot,map_location='cpu',weights_only=False)
                tensors_match=all(torch.allclose(value.detach().cpu(),other_initial['normal_codebook'][key],rtol=1e-5,atol=1e-6)
                    for key,value in initial['normal_codebook'].items())
                write(training/'phase1_paired_comparison.json',dict(tensors_match_within_tolerance=tensors_match,
                    hash_state_identical=initial['hash_state']==other_initial['hash_state'],other_snapshot_sha256=sha(other_snapshot),
                    rtol=1e-5,atol=1e-6,normal_inputs_only=True))
    model.update_codebook=recorded_update
    batches=[]
    original_loss=author_train.AnomalyLoss
    class RecordedLoss(original_loss):
        def forward(self,*values,**kwargs):
            loss,parts=super().forward(*values,**kwargs)
            if not torch.isfinite(loss):raise ValueError('nonfinite original training loss')
            batches.append({**parts,'total':float(loss.detach())})
            if len(batches)%expected_count==0:
                epoch=len(batches)//expected_count
                means={key:float(np.mean([r[key] for r in batches[-expected_count:]])) for key in batches[-1]}
                path=training/'losses.csv'
                exists=path.exists()
                with path.open('a',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['epoch',*means])
                    if not exists:writer.writeheader()
                    writer.writerow(dict(epoch=epoch,**means))
            return loss,parts
    author_train.AnomalyLoss=RecordedLoss
    original_adam=torch.optim.Adam
    optimizers=[]
    def adam(*values,**kwargs):
        optimizer=original_adam(*values,**kwargs);optimizers.append(optimizer);return optimizer
    torch.optim.Adam=adam
    original_save=torch.save
    def save(payload,path,*save_args,**save_kwargs):
        payload={**payload,'codebook_hash_state':codebook_state(model),'normalization':normalization,
                 'split_sha256':sha(split_path(output)),'normalization_sha256':sha(output/'link5_normalization.json'),
                 'paper_exact':False,'upstream_sources':source,'actual_backbone':model.encoder.provenance,'torch_rng_state':torch.get_rng_state(),
                 'cuda_rng_state':torch.cuda.get_rng_state_all(),'optimizer':optimizers[0].state_dict(),
                 'augmentation_mode':mode,'augmentation_statistics':structural_generators[0].counts if structural_generators else None}
        payload['adapter_sources_sha256']=adapter_sources
        payload['runtime_pose_config']=experiment['pose']
        original_save(payload,path,*save_args,**save_kwargs)
        write(training/'codebook_statistics.json',payload['codebook_hash_state'])
    torch.save=save
    old_argv=sys.argv
    sys.argv=['train.py','--config',str(effective),'--class_name','link5','--data_root',str(data_root),
              '--out_dir',str(training),'--device','cuda','--seed','0']
    try:
        # Preserve the original warmup, augmentation, codebook refresh, Adam
        # optimization and training loss; do not replace them with local logic.
        author_train.main()
    finally:
        torch.save=original_save;torch.optim.Adam=original_adam;author_train.NegativeAugmentation=original_augmentation;author_train.DataLoader=original_loader;model.update_codebook=native_update;sys.argv=old_argv
    final=training/'link5/epoch_1500.pt'
    checkpoint=torch.load(final,map_location='cpu',weights_only=False)
    if checkpoint['epoch']!=1500 or len(batches)!=1500*expected_count:raise ValueError('incomplete native1500-epoch training')
    shutil.copy2(final,training/'link5/final.pt')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=np.genfromtxt(training/'losses.csv',delimiter=',',names=True)
    fig,ax=plt.subplots(figsize=(9,5))
    for key in ('L_dist','L_sim','L_bce','total'):ax.plot(rows['epoch'],rows[key],label=key)
    ax.legend();ax.set_xlabel('Epoch');fig.tight_layout();fig.savefig(training/'loss_curves.png',dpi=180);plt.close(fig)
    write(training/'completion.json',dict(status='complete',epoch=1500,checkpoint_sha256=sha(training/'link5/final.pt'),
        actual_backbone=experiment['backbone'],backbone_frozen=True,augmentation_mode=mode,
        augmentation_statistics=structural_generators[0].counts if structural_generators else None,
        paper_exact=False,author_training_entry_point=author_train.__file__))
    print('LINK5_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
