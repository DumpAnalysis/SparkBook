"""Capture complete commands and combined output in a fresh directory."""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(__file__).resolve().parent.parent;out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    def rel(path):return os.path.relpath(path,root)
    jobs=[]
    for n in (1,2,4):jobs.append((f'0{len(jobs)+1}-shared{n}',
        ['sharing/service.py','--workers',str(n),'--batch','1'],0))
    jobs += [('04-batched',['sharing/service.py','--workers','4','--batch','4','--profile'],0),
             ('05-replicas',['sharing/service.py','--mode','replicas','--workers','4','--batch','1'],0),
             ('06-training',['sharing/train.py'],0),
             ('07-faults',['sharing/faults.py'],0),
             ('08-plot',['sharing/plot.py','--run',rel(out/'04-batched')],0),
             ('09-cuda',['sharing/service.py','--device','cuda','--workers','2','--batch','2'],None),
             ('10-streams',['sharing/streams.py'],None)]
    alltext=[];summary=[]
    import torch,numpy,matplotlib
    env=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),
             torch=torch.__version__,numpy=numpy.__version__,matplotlib=matplotlib.__version__,
             cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available())
    (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    for name,argv,want in jobs:
        argv=argv+['--out',rel(out/name)]
        cmd=[sys.executable]+argv
        proc=subprocess.Popen(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,start_new_session=True,
                              env=os.environ|{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        try:output,_=proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);output,_=proc.communicate()
            output+='\nOUTER DEADLINE 120 seconds\n'
        text='$ python '+shlex.join(argv)+'\n'+output+f'[exit {proc.returncode}]\n'
        (out/(name+'.txt')).write_text(text);alltext.append(text)
        summary.append(dict(name=name,exitcode=proc.returncode))
        print(text,end='',flush=True)
        if want is not None and proc.returncode!=want:raise RuntimeError(name+' failed')
        if want is None and proc.returncode not in (0,2):raise RuntimeError(name+' failed')
    (out/'transcripts.txt').write_text('\n'.join(alltext))
    (out/'runs.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
