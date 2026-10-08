"""Read-only stage/process telemetry; tolerate unavailable kernel PSI counters."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    try:return Path(path).read_text().strip()
    except (OSError,PermissionError):return 'not exposed'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root-pid',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('a') as stream:
        while Path('/proc/'+str(a.root_pid)).exists():
            text=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,ppid=,rss=,pcpu=,stat=,args='],text=True)
            rows=[]
            for line in text.splitlines():
                parts=line.strip().split(None,5)
                if len(parts)==6:
                    rows.append(dict(pid=int(parts[0]),ppid=int(parts[1]),rss_kib=int(parts[2]),cpu_percent=float(parts[3]),state=parts[4],command=parts[5]))
            descendants={a.root_pid}
            for _ in range(20):
                expanded=descendants|{r['pid'] for r in rows if r['ppid'] in descendants}
                if expanded==descendants:break
                descendants=expanded
            processes=[r for r in rows if r['pid'] in descendants]
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total,utilization.gpu,utilization.memory','--format=csv,noheader,nounits'],text=True).strip()
            # Node memory and process-tree RSS are distinct; RSS sums shared pages.
            row=dict(unix_time=time.time(),gpu_csv=gpu,process_tree_rss_kib=sum(r['rss_kib'] for r in processes),
                summed_process_cpu_percent=sum(r['cpu_percent'] for r in processes),processes=processes,
                load_average=os.getloadavg(),host_memory=read('/proc/meminfo'),
                pressure={name:read('/proc/pressure/'+name) for name in ('cpu','memory','io')},
                allocation_cgroups=read('/proc/'+str(a.root_pid)+'/cgroup'),
                process_io={str(r['pid']):read('/proc/'+str(r['pid'])+'/io') for r in processes})
            stream.write(json.dumps(row)+'\n');stream.flush();time.sleep(5)


if __name__=='__main__':main()
