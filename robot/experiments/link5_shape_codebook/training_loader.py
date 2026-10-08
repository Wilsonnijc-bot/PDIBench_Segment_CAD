"""Remove worker startup for four tiny clouds while preserving native sampling."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import time
import socket

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import read, sha, write


class WorkerEquivalentDataset(Dataset):
    def __init__(self,dataset,workers=2):
        self.dataset=dataset;self.workers=workers
        self.reset()

    def reset(self):
        self.position=0;self.generators=[]
        for _ in range(self.workers):
            generator=np.random.default_rng(0)
            generator.bit_generator.state=copy.deepcopy(self.dataset.rng.bit_generator.state)
            self.generators.append(generator)

    def __len__(self):return len(self.dataset)

    def __getitem__(self,index):
        parent=self.dataset.rng
        self.dataset.rng=self.generators[self.position%self.workers];self.position+=1
        try:return self.dataset[index]
        finally:self.dataset.rng=parent


class NativeEquivalentLoader(DataLoader):
    """Match batch1/two nonpersistent workers' round-robin NumPy RNG streams."""
    def __init__(self,dataset,*args,**kwargs):
        if args or kwargs.get('num_workers')!=2 or kwargs.get('batch_size')!=1 or kwargs.get('persistent_workers',False):
            raise ValueError('optimized loader only supports the audited native batch1/two-worker contract')
        if kwargs.get('drop_last',False) or 'worker_init_fn' in kwargs or kwargs.get('generator') is not None:
            raise ValueError('custom worker/RNG settings require separate equivalence verification')
        self.proxy=WorkerEquivalentDataset(dataset)
        kwargs.pop('multiprocessing_context',None)
        kwargs={**kwargs,'num_workers':0}
        super().__init__(self.proxy,**kwargs)

    def __iter__(self):
        self.proxy.reset()
        return super().__iter__()


def measure(dataset_factory,epochs=10,loader_kwargs=None):
    kwargs=dict(batch_size=1,shuffle=True,num_workers=2,drop_last=False)
    kwargs.update(loader_kwargs or {})
    results=[]
    for factory in (DataLoader,NativeEquivalentLoader):
        torch.manual_seed(0)
        dataset=dataset_factory();loader=factory(dataset,**kwargs)
        records=[];started=time.perf_counter()
        for _ in range(epochs):
            for batch in loader:
                records.append((batch['name'],hashlib.sha256(batch['xyz'].numpy().tobytes()).hexdigest(),torch.rand(7).tolist()))
        results.append(dict(seconds=time.perf_counter()-started,records=records,torch_rng_state=torch.get_rng_state()))
    equivalent=results[0]['records']==results[1]['records'] and torch.equal(results[0]['torch_rng_state'],results[1]['torch_rng_state'])
    return dict(status='passed' if equivalent else 'failed',epochs=epochs,batches=len(results[0]['records']),
        point_order_names_and_torch_rng_identical=equivalent,native_loader_seconds=results[0]['seconds'],
        optimized_loader_seconds=results[1]['seconds'],host_loading_speedup=results[0]['seconds']/results[1]['seconds'],
        scope='host loading only; does not establish end-to-end training speedup',source_sha256=sha(__file__))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=read(args.config);output=Path(config['output'])
    from .upstream import shape_modules
    modules,_=shape_modules();dataset_module=modules['dataset']
    dataset_module.uniform_sample=lambda points,n,rng:points[rng.choice(len(points),min(n,len(points)),replace=False)]
    factory=lambda:dataset_module.PointCloudAnomalyDataset(root=output/'normal_training_data',class_name='link5',split='train_normal',num_points=10000,normalize=False,seed=0)
    receipt=measure(factory)
    receipt.update(host=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),torch_version=torch.__version__)
    write(output/'metadata/training_loader_benchmark.json',receipt)
    if receipt['status']!='passed':raise RuntimeError('optimized training loader changes native data/RNG behavior')
    print('LINK5_NATIVE_EQUIVALENT_LOADER_VERIFIED',receipt,flush=True)


if __name__=='__main__':main()
