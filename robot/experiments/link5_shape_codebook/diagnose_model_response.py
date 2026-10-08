"""Read-only trace of a completed checkpoint on saved validation inputs.

Uses the real pretrained sparse encoder and native detector, without updating
weights, codebooks, masks, geometry, configuration or validation gates.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from robot.experiments.link5_shape_codebook.common import mode_folder, read, sha, write, raw_metrics
from robot.experiments.link5_shape_codebook.score import load_detector


def stats(value):
    x=value.detach().float().reshape(-1)
    return dict(minimum=float(x.min()),mean=float(x.mean()),maximum=float(x.max()),std=float(x.std(unbiased=False)))


@torch.no_grad()
def run(config_path,mode):
    config=read(config_path);root=Path(config['output']);destination=root/'diagnostics/model_response'
    destination.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('one allocated CUDA GPU required')
    probe=torch.ones(4,device='cuda');assert float(probe.sum())==4
    model,modules,norm=load_detector(root,mode)
    training=mode_folder(root,'training',mode);sanity=mode_folder(root,'sanity',mode)
    checks=read(sanity/'checks.json');traces=[];current={}
    def encoder_hook(module,inputs,output):
        current['encoder']=dict(feature_norm=stats(output.norm(dim=-1)),point_channel_std_mean=float(output.std(dim=1,unbiased=False).mean()),zero_feature_fraction=float((output.norm(dim=-1)<1e-8).float().mean()))
    original=model._patch_features
    def patch_features(infos,features):
        result=original(infos,features);scales=[]
        for book,info,pf in zip(model.codebook.books,infos,result):
            geom=model.patch_encoder(info['centroid']);q=F.normalize(pf,dim=-1)
            size=int(book.size.item());bank=book.features[:size];cos=q@bank.T
            best,ids=cos.max(dim=-1);hist=torch.bincount(ids.reshape(-1),minlength=size)
            scales.append(dict(patches=pf.shape[1],book_entries=size,patch_centroid_norm=stats(info['centroid'].norm(dim=-1)),
                geometry_feature_norm=stats(geom.norm(dim=-1)),nearest_similarity=stats(best),retrieved_entry_counts=hist.tolist(),
                dominant_retrieval_fraction=float(hist.max()/hist.sum()),patch_feature_channel_std_mean=float(pf.std(dim=1,unbiased=False).mean())))
        current['scales']=scales;return result
    def attention_pre(module,inputs):
        templates=inputs[2][0];unique=torch.unique(templates,dim=0)
        current['attention_templates']=dict(unique_retrieved_templates=len(unique),patch_count=len(templates),template_channel_std_mean=float(templates.std(dim=0,unbiased=False).mean()))
    def attention_hook(module,inputs,output):
        current['attention_output']=dict(norm=stats(output.norm(dim=-1)),point_channel_std_mean=float(output.std(dim=1,unbiased=False).mean()))
    def modulation_pre(module,inputs):
        delta=1-(F.normalize(inputs[1],dim=-1)*F.normalize(inputs[2],dim=-1)).sum(dim=-1)
        current['normal_patch_discrepancy']=stats(delta)
    def modulation_hook(module,inputs,output):
        current['modulation_output']=dict(norm=stats(output.norm(dim=-1)),point_channel_std_mean=float(output.std(dim=1,unbiased=False).mean()))
    handles=[model.encoder.register_forward_hook(encoder_hook),model.cross_attn.register_forward_pre_hook(attention_pre),
             model.cross_attn.register_forward_hook(attention_hook),model.modulation.register_forward_pre_hook(modulation_pre),
             model.modulation.register_forward_hook(modulation_hook)]
    model._patch_features=patch_features
    video=checks['heldout_normals'][0]['video_id'];normal_path=sanity/(video+'_heldout_frame0.npz')
    inputs=[dict(name='heldout_normal',path=normal_path,key='points_normalized',score_key='raw_point_scores')]
    for row in checks['synthetic_deformations']:
        inputs.append(dict(name=row['type']+'_'+row['parameter'],path=sanity/f"synthetic_{row['type']}_{row['parameter']}.npz",key='anomalous',score_key='raw_scores'))
    for item in inputs:
        current={}
        with np.load(item['path'],allow_pickle=False) as a:points=a[item['key']].copy();saved=a[item['score_key']].copy()
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        offset,logits,aux=model(torch.from_numpy(points).cuda()[None])
        raw=(offset.abs().sum(-1)*logits.sigmoid())[0].cpu().numpy()
        np.testing.assert_allclose(raw,saved,rtol=2e-5,atol=1e-7)
        current.update(name=item['name'],source_sha256=sha(item['path']),source_path=str(item['path']),point_count=len(points),
            saved_scores_reproduced=True,raw_metrics=raw_metrics(raw),selected_scale=int(aux['best_scale'][0]),scale_scores=aux['scale_scores'][0].tolist(),
            predicted_offset_mean=offset.mean(dim=1)[0].tolist(),predicted_offset_std=offset.std(dim=1,unbiased=False)[0].tolist(),
            predicted_validity_probability=stats(logits.sigmoid()),raw_score_distribution=stats(torch.from_numpy(raw)))
        traces.append(current);print('RESPONSE_TRACED',mode,item['name'],current['attention_templates'],flush=True)
    for handle in handles:handle.remove()
    model._patch_features=original
    receipt=dict(status='complete',mode=mode,allocation_id=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('HOSTNAME'),
        gpu=torch.cuda.get_device_name(),checkpoint_sha256=sha(training/'link5/final.pt'),config_sha256=sha(config_path),
        diagnostic_source_sha256=sha(__file__),model_training=False,codebook_updated=False,source_video_id=video,frame_id=0,traces=traces)
    write(destination/(mode+'.json'),receipt)
    print('LINK5_MODEL_RESPONSE_DIAGNOSTIC_COMPLETE',mode,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--augmentation-mode',choices=('paper_original','robot_structural'),required=True)
    args=parser.parse_args();run(args.config,args.augmentation_mode)
