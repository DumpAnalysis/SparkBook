"""Two logical CPU nodes on one host; launched under lab.py's process group."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
with socket.socket() as s:
    s.bind(("127.0.0.1",0));port=s.getsockname()[1]
# The bind-to-launch race is recorded rather than hidden; rerun in a new
# directory if another process takes this ephemeral port.
processes=[];handles=[];records=[]
try:
    for node in range(2):
        stdout=(a.out/f"node-{node}-stdout.txt").open("w")
        stderr=(a.out/f"node-{node}-stderr.txt").open("w")
        handles.extend([stdout,stderr])
        cmd=[sys.executable,"-m","torch.distributed.run","--nnodes=2",
             "--nproc-per-node=1",f"--node-rank={node}","--master-addr=127.0.0.1",
             f"--master-port={port}","--max-restarts=0",str(Path(__file__).with_name("worker.py")),
             "train","--out",str(a.out)]
        processes.append(subprocess.Popen(cmd,stdout=stdout,stderr=stderr))
        records.append(dict(node_rank=node,argv=cmd))
    deadline=time.monotonic()+75
    for proc,record in zip(processes,records):
        record["returncode"]=proc.wait(timeout=max(.1,deadline-time.monotonic()))
    if any(r["returncode"] for r in records):raise RuntimeError("A logical node failed")
finally:
    for proc in processes:
        if proc.poll() is None:proc.terminate()
    for h in handles:h.close()
    (a.out/"nodes.json").write_text(json.dumps(records,indent=2)+"\n")
