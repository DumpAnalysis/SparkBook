"""Capture the chapter's complete driver transcripts in a new directory."""
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import signal
import sys

def capture_command(argv, timeout=210):
    """Retain partial output and bound this owned process group on timeout."""
    proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,start_new_session=True)
    expired=False
    try:stdout,stderr=proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        expired=True
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:stdout,stderr=proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            stdout,stderr=proc.communicate()
    r=subprocess.CompletedProcess(argv,124 if expired else proc.returncode,stdout,stderr)
    r.timed_out=expired
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    commands=[
     ("00-probe","lab.py",["probe"],0),
     ("01-train2","pipe_lab.py",["train"],0),
     ("02-train4","pipe_lab.py",["train","--workers","4"],0),
     ("03-ddp","lab.py",["ddp"],None),
     ("04-slow","pipe_lab.py",["slow"],0),
     ("05-plot","lab.py",["plot","--source",str(a.out/"04-slow")],0),
     ("06-crash","pipe_lab.py",["crash"],0),
     ("07-mismatch","pipe_lab.py",["mismatch"],0),
     ("08-timeout","pipe_lab.py",["timeout"],0),
     ("09-checkpoint","pipe_lab.py",["checkpoint"],0),
     ("10-shard","pipe_lab.py",["shard"],0),
     ("11-gpu-ddp","lab.py",["gpu-ddp"],None),
     ("12-nodes","lab.py",["nodes"],None),
    ]
    records=[];transcripts=[]
    for name,script,args,expected in commands:
        display=["python",f"distributed/{script}",*args,"--out",str(a.out/name)]
        argv=[sys.executable,*display[1:]]
        r=capture_command(argv)
        text="$ "+shlex.join(display)+"\n"+r.stdout
        if r.stderr:text+="[stderr]\n"+r.stderr
        if r.timed_out:text+="[outer capture timeout]\n"
        text+=f"[exit {r.returncode}]\n"
        (a.out/f"{name}.txt").write_text(text)
        transcripts.append(text)
        records.append(dict(name=name,argv=argv,returncode=r.returncode,expected=expected,timed_out=r.timed_out))
        print(name,"exit",r.returncode,flush=True)
        (a.out/"transcripts.txt").write_text("\n".join(transcripts))
        (a.out/"commands.json").write_text(json.dumps(records,indent=2)+"\n")
        if expected is not None and r.returncode!=expected:raise RuntimeError(name)
    (a.out/"transcripts.txt").write_text("\n".join(transcripts))
    (a.out/"environment.json").write_text(json.dumps(dict(python=sys.version,
        executable=sys.executable,machine=platform.machine(),kernel=platform.release(),
        cwd=os.getcwd(),pythonpath=os.environ.get("PYTHONPATH"),commands=records),indent=2)+"\n")

if __name__=="__main__":main()
