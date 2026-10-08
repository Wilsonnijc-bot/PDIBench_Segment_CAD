"""Train an intrinsic-profile restoration control using the same four normals."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import torch

from robot.experiments.link5_shape_codebook.benign_variation_v4 import DEFAULTS as BENIGN, paired_variation
from robot.experiments.link5_shape_codebook.common import ROOT, read, sha, split_path, write
from robot.experiments.link5_shape_codebook.localization_objective import paired_loss
from robot.experiments.link5_shape_codebook.robot_structural import RobotStructuralAugmentation
from robot.experiments.link5_shape_codebook.structural_checkpoint import load_checkpoint
from robot.experiments.link5_shape_codebook.train_localization import generate, load_points
from robot.experiments.link5_shape_codebook.train_structural_localization_v2 import TYPES, evaluate, setup
from robot.experiments.link5_shape_codebook.upstream import codebook_state, restore_hash_keys


def run(config_path, warm_path, epochs, resume, variant):
    config=read(config_path);root=Path(config['output'])
    root,split,norm,heldout,geometry,distribution=setup(config,root/'guides/manifest.json',root/'guides/structural_distribution.json')
    destination=root/'structural_localization/profile_shape_v6'/variant;destination.mkdir(parents=True,exist_ok=True)
    lock=(destination/'.train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (destination/'completion.json').exists():
        receipt=read(destination/'completion.json')
        if receipt['status']=='complete' and sha(destination/'final.pt')==receipt['checkpoint_sha256']:
            print('LINK5_BENIGN_FINETUNE_ALREADY_COMPLETE',flush=True);return
        raise ValueError('previous completion identity mismatch')
    latest=destination/'latest.pt'
    if (destination/'losses.csv').exists() and (not resume or not latest.exists()):
        raise ValueError('preserve previous attempt; resume only a saved coherent checkpoint')
    if torch.cuda.device_count()!=1 or float(torch.ones(4,device='cuda').sum())!=4:raise ValueError('one allocated GPU required')
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    model,warm=load_checkpoint(config,warm_path)
    if warm['variant']!='structural_geometry_head':raise ValueError('geometry head warm start required')
    from robot.experiments.link5_shape_codebook.profile_decoder import wrap_profile_model
    samples=[(row,*load_points(row,norm)) for row in split['train']]
    model=wrap_profile_model(model.native,samples,geometry).cuda()
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    model.native.requires_grad_(False)
    frozen_native={name:tensor.detach().cpu().clone() for name,tensor in model.native.state_dict().items()}
    generator=RobotStructuralAugmentation(geometry,distribution['settings'])
    samples=[(row,*load_points(row,norm)) for row in split['train']]
    source_names=['train_profile_v6.py','benign_variation_v4.py','structural_geometry_head.py','profile_decoder.py','robot_structural.py',
                  'localization_objective.py','train_localization.py','train_structural_localization_v2.py']
    sources={name:sha(ROOT/'robot/experiments/link5_shape_codebook'/name) for name in source_names}
    seed_offset=warm['optimizer_updates']
    policy=dict(variant=variant,epochs=epochs,optimizer='Adam',learning_rate=.001,batch_size=1,seed=0,
                augmentation_seed_offset=seed_offset,learning_rate_schedule='0.001;0.0005 after1000;0.00025 after1500',
                normal_manifest=split['train'],normal_split_sha256=sha(split_path(root)),normalization_sha256=sha(root/'link5_normalization.json'),
                augmentation_types=TYPES,augmentation_distribution=distribution,benign_training_variation=BENIGN,
                warm_start_checkpoint=str(warm_path),warm_start_checkpoint_sha256=sha(warm_path),warm_start_epoch=warm['epoch'],
                warm_start_optimizer_updates=warm['optimizer_updates'],pretrained_backbone_frozen=True,native_network_frozen=True,
                normal_codebook_frozen=True,normal_codebook_updates_during_finetune=0,geometry_head_trainable=False,profile_head_trainable=True,
                adapter_sources=sources,detector_config=warm['config'],architecture='learned intrinsic profile restoration/probability; canonical offsets mapped to unchanged world inputs',
                intrinsic_shape_features=True,profile_reference_sources=4,native_network_bypassed_for_scoring=True,
                native_checkpoint_retained=True,profile_head_initialized_fresh=True,
                optimization_inputs='clean and structural copies derived only from the four frame0s; same benign transform/noise for both members',
                stored_observations_modified=False,observed_guides_used_for_optimization=False,
                objective='same paired clean-zero/region-balanced restoration+BCE+raw-score margin as preceding structural trial',
                calibration_policy='99.5%normal-only quantile; include independently seeded benign four-reference copies, source-excluded nearest geometry context',
                research_acceptance_thresholds=warm['policy'].get('acceptance_thresholds',warm['policy'].get('research_acceptance_thresholds')))
    if (destination/'training_policy.json').exists():
        if read(destination/'training_policy.json')!=policy:raise ValueError('resume policy/source identity changed')
    else:write(destination/'training_policy.json',policy)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    stream=np.random.default_rng(0);updates=0;start_epoch=0;elapsed_before=0.;counts={};examples=[]
    if (destination/'training_examples/manifest.json').exists():examples=read(destination/'training_examples/manifest.json')['examples']
    if resume and latest.exists():
        payload=torch.load(latest,map_location='cpu',weights_only=False)
        if payload['policy']!=policy:raise ValueError('resume checkpoint differs from frozen policy')
        model.load_state_dict(payload['model'],strict=True);restore_hash_keys(model,payload['codebook_hash_state'])
        optimizer.load_state_dict(payload['optimizer']);stream.bit_generator.state=payload['loader_rng_state']
        torch.random.set_rng_state(payload['torch_cpu_rng']);torch.cuda.set_rng_state_all(payload['torch_cuda_rng'])
        updates=payload['optimizer_updates'];start_epoch=payload['epoch'];counts=payload['augmentation_counts'];elapsed_before=payload['elapsed_seconds']
        rows=list(csv.DictReader((destination/'losses.csv').open()))
        retained=[r for r in rows if int(r['epoch'])<=start_epoch]
        if len(retained)!=start_epoch:raise ValueError('resume loss history incomplete')
        if len(rows)>len(retained):
            backup=destination/f"partial_losses_job_{os.environ.get('SLURM_JOB_ID','local')}.csv"
            backup.write_text((destination/'losses.csv').read_text())
            with (destination/'losses.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(retained[0]));writer.writeheader();writer.writerows(retained)
        print('LINK5_BENIGN_FINETUNE_RESUMED',start_epoch,updates,flush=True)
    # Probe the actual warm head and paired benign objective without changing
    # optimizer/book/RNG state. Keep the new stage's inherited native weights.
    xyz=samples[0][1];model.reference_exclusion=0
    with torch.no_grad():
        native_offset,native_logits,_=model.native(xyz[None])
        if not torch.isfinite(native_offset).all() or not torch.isfinite(native_logits).all():raise ValueError('actual frozen native forward failed')
    rng_cpu=torch.random.get_rng_state();rng_cuda=torch.cuda.get_rng_state_all()
    anomaly,offset,mask=generate(generator,xyz,None,'shortening',None,99999)
    clean,anomaly,offset,mask,benign=paired_variation(xyz,anomaly,mask,77777)
    co,cl,_=model(clean[None]);ao,al,_=model(anomaly[None]);probe,parts=paired_loss(co,cl,ao,al,offset[None],mask[None])
    if not torch.isfinite(probe):raise ValueError('warm paired preflight loss nonfinite')
    probe.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('warm paired preflight gradient nonfinite')
    model.zero_grad(set_to_none=True);torch.random.set_rng_state(rng_cpu);torch.cuda.set_rng_state_all(rng_cuda)
    write(destination/f"preflight_job_{os.environ.get('SLURM_JOB_ID','local')}.json",dict(status='passed',optimizer_updates=0,resumed_epoch=start_epoch,
          loss=parts,backbone=model.encoder.provenance,device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda))
    del anomaly,offset,mask,clean,co,cl,ao,al,probe
    print('LINK5_BENIGN_FINETUNE_PREFLIGHT_PASSED',flush=True)
    model.train();model.native.eval();started=time.monotonic();(destination/'training_examples').mkdir(exist_ok=True);(destination/'predictions').mkdir(exist_ok=True)
    epoch=start_epoch
    def payload():
        return dict(model=model.state_dict(),codebook_hash_state=codebook_state(model),config=warm['config'],variant=variant,epoch=epoch,
                    optimizer=optimizer.state_dict(),policy=policy,normalization=norm,augmentation_counts=counts,optimizer_updates=updates,
                    loader_rng_state=stream.bit_generator.state,torch_cpu_rng=torch.random.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all(),
                    elapsed_seconds=elapsed_before+time.monotonic()-started)
    def checkpoint(path):
        temporary=path.with_suffix(f'.{os.getpid()}.tmp');torch.save(payload(),temporary);temporary.replace(path)
    receipt=None
    with (destination/'losses.csv').open('a' if start_epoch else 'w',newline='') as log:
        writer=csv.DictWriter(log,fieldnames=['epoch','offset','mask','ranking','total','seconds'])
        if not start_epoch:writer.writeheader()
        for epoch in range(start_epoch+1,epochs+1):
            model.native.eval()
            rate=.00025 if epoch>1500 else .0005 if epoch>1000 else .001
            for group in optimizer.param_groups:group['lr']=rate
            parts=[]
            for sample_index in stream.permutation(4):
                row,xyz,indices=samples[int(sample_index)];model.reference_exclusion=int(sample_index)
                kind=TYPES[updates%4];seed=100000+seed_offset+updates
                anomaly,offset,mask=generate(generator,xyz,None,kind,None,seed)
                parameters=dict(generator.last_parameters)
                clean,anomaly,offset,mask,benign=paired_variation(xyz,anomaly,mask,200000+seed_offset+updates)
                name=row['video_id']+'_'+kind;archive=destination/'training_examples'/(name+'.npz')
                if not archive.exists():
                    np.savez_compressed(archive,source_normal=xyz.cpu().numpy(),normal=clean.cpu().numpy(),anomalous=anomaly.cpu().numpy(),
                                        gt_offset=offset.cpu().numpy(),gt_mask=mask.cpu().numpy(),sampled_input_indices=indices)
                    examples.append(dict(video_id=row['video_id'],frame_id=0,type=kind,parameters=parameters,benign_parameters=benign,seed=seed,
                                         optimizer_update=updates+1,source_input_sha256=row['input_sha256'],archive=archive.name,archive_sha256=sha(archive)))
                    write(destination/'training_examples/manifest.json',dict(status='actual saved optimizer inputs',examples=examples))
                co,cl,_=model(clean[None]);ao,al,_=model(anomaly[None]);loss,values=paired_loss(co,cl,ao,al,offset[None],mask[None])
                if not torch.isfinite(loss):raise ValueError('nonfinite finetune loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite finetune gradient')
                optimizer.step();parts.append(values);updates+=1;counts[kind]=counts.get(kind,0)+1
            means={k:float(np.mean([v[k] for v in parts])) for k in parts[0]}
            writer.writerow(dict(epoch=epoch,**means,seconds=elapsed_before+time.monotonic()-started));log.flush()
            if epoch<=3 or epoch%50==0:print('BENIGN_FINETUNE_TRAIN',epoch,means,flush=True)
            if epoch%25==0 or epoch==epochs:checkpoint(latest)
            if epoch in (300,1000,1500,epochs):
                checkpoint(destination/f'epoch_{epoch:04d}.pt')
                receipt=evaluate(model,split,norm,heldout,generator,distribution,destination,epoch,epoch==epochs)
    if receipt is None:
        # A resumed allocation may find the final weights already saved while
        # the preceding allocation ended during final evaluation.
        checkpoint(destination/f'epoch_{epochs:04d}.pt')
        receipt=evaluate(model,split,norm,heldout,generator,distribution,destination,epoch,True)
    final=destination/'final.pt';final.write_bytes((destination/f'epoch_{epochs:04d}.pt').read_bytes())
    for name,tensor in model.native.state_dict().items():
        torch.testing.assert_close(tensor.detach().cpu(),frozen_native[name],rtol=0,atol=0)
    setup(config,root/'guides/manifest.json',root/'guides/structural_distribution.json')
    write(destination/'completion.json',dict(status='complete',epoch=epochs,optimizer_updates=updates,checkpoint_sha256=sha(final),
          warm_start_checkpoint_sha256=sha(warm_path),warm_start_epoch=warm['epoch'],warm_start_optimizer_updates=warm['optimizer_updates'],
          elapsed_seconds=elapsed_before+time.monotonic()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
          evaluation_summary=receipt['summary'],research_acceptance=receipt['acceptance'],normal_codebook_updates=0,
          original_full_video_gate_passed=False,normal_split_sha256=sha(split_path(root)),normalization_sha256=sha(root/'link5_normalization.json')))
    print('LINK5_BENIGN_FINETUNE_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--warm-checkpoint',type=Path,required=True);parser.add_argument('--epochs',type=int,default=600)
    parser.add_argument('--variant',choices=['structural_profile_decoder'],default='structural_profile_decoder')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();torch.set_num_threads(6)
    if args.epochs<1:raise ValueError('positive epochs required')
    run(args.config,args.warm_checkpoint,args.epochs,args.resume,args.variant)
