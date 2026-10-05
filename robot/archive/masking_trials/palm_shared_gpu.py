"""Bound only this experiment's GPU/CPU usage; never signal other GPU workers."""
import argparse
import os
import runpy
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--qwen',action='store_true');p.add_argument('module');p.add_argument('args',nargs=argparse.REMAINDER)
    args=p.parse_args();required=22500 if args.qwen else 15000
    os.nice(10)
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free>=required:break
        print(f'Waiting for GPU headroom: free={free} MiB, required={required} MiB',flush=True)
        time.sleep(10)
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.90 if args.qwen else .55)
    print(f'GPU admission: free={free} MiB; per-process allocation fraction={.90 if args.qwen else .55}; full GPU weights, no offload',flush=True)
    sys.argv=[args.module,*args.args];runpy.run_module(args.module,run_name='__main__')


if __name__=='__main__':main()
