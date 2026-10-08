"""Protect one variant from duplicate launches while independent variants share a GPU."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

from .common import MODES, ROOT, mode_folder, read, sha, split_path, write


def run(config_path,mode):
    config=read(config_path);output=Path(config['output'])
    training=mode_folder(output,'training',mode);training.mkdir(parents=True,exist_ok=True)
    with (training/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        done=training/'completion.json';stage=training/'stage_receipt.json'
        if done.exists():
            receipt=read(done);control=read(training/'paired_control.json')
            if receipt['status']!='complete' or receipt['epoch']!=1500 or receipt['checkpoint_sha256']!=sha(training/'link5/final.pt'):
                raise ValueError('completed training artifact changed')
            if control['split_sha256']!=sha(split_path(output)) or control['normalization_sha256']!=sha(output/'link5_normalization.json') or read(training/'normal_manifest.json')!=read(split_path(output))['train']:
                raise ValueError('completed training reference changed')
            print('LINK5_COMPLETED_TRAINING_RETAINED',mode,flush=True)
            return 0
        if stage.exists() and read(stage).get('status')=='failed':
            raise RuntimeError('previous training launch failed; inspect stage receipt before retrying')
        write(stage,dict(status='running',augmentation_mode=mode))
        result=subprocess.run([sys.executable,'-u','-m','robot.experiments.link5_shape_codebook.train',
                               '--config',str(config_path),'--augmentation-mode',mode],cwd=ROOT)
        write(stage,dict(status='complete' if result.returncode==0 else 'failed',augmentation_mode=mode,exit_status=result.returncode))
        return result.returncode


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--augmentation-mode',choices=MODES,required=True)
    args=parser.parse_args();sys.exit(run(args.config,args.augmentation_mode))
