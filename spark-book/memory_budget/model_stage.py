"""Worker for the supervised 51-layer Qwen experiment in guarded.py."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from windows_watch import GIB, GuardStop, guest_available, utc, validate

ROOT=Path(__file__).resolve().parents[1]
PROFILE=json.loads(Path(__file__).with_name('qwen32-profile.json').read_text())
sys.path.insert(0,str(ROOT))
import spark_lab


def save(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def fingerprint(path, record):
    if not path.is_file() or path.stat().st_size!=record['bytes']:
        raise GuardStop('Missing file or wrong size: '+str(path))
    with path.open('rb') as f:
        actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=record['sha256']:
        raise GuardStop('Checksum mismatch: '+str(path))


def model_inputs():
    model_dir=ROOT/'models/32b-q8_0'
    for item in PROFILE['shards']:
        fingerprint(model_dir/item['name'],item)
    runtime=[(ROOT/f['path'],f) for f in PROFILE['runtime']]
    for path,record in runtime: fingerprint(path,record)
    server=next(path for path,_ in runtime if path.name=='llama-server')
    libraries=sorted({str(path.parent) for path,_ in runtime if '.so' in path.name})
    return model_dir/PROFILE['shards'][0]['name'],server,os.pathsep.join(libraries)


def admit(cuda_free,dedicated_free,guest,host):
    plan=PROFILE['placement']
    shared_allowance=max(0,plan['GPU_planning_bytes']-dedicated_free)
    tests={'cuda_free':cuda_free>=plan['GPU_planning_bytes'],
           'guest_available':guest>=plan['CPU_planning_bytes'],
           'host_available':host['available_bytes']>=shared_allowance+3*GIB,
           'host_commit_headroom':host['commit_limit_bytes']-host['committed_bytes']>=4*GIB}
    return tests


def health(base, process):
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        if process.poll() is not None:raise RuntimeError('Native server exited during loading')
        reply=spark_lab.http(base,'/health',timeout=1)
        if reply['status']==200 and isinstance(reply['body'],dict) and reply['body'].get('status')=='ok':return
        time.sleep(.25)
    raise RuntimeError('Native server did not report ready within 600 seconds')


def run(out):
    model,exe,libpath=model_inputs()
    # A short-lived process queries CUDA so its context is released before the native server starts.
    query = ("import json,torch; print(json.dumps([torch.cuda.device_count(),"
             "*torch.cuda.mem_get_info(0)]))")
    try:
        count,cuda_free,cuda_total=json.loads(subprocess.check_output([sys.executable,'-c',query],text=True,timeout=30))
    except (OSError,subprocess.SubprocessError,ValueError) as exc:
        raise GuardStop('CUDA budget query failed in the active framework environment') from exc
    if count!=1: raise GuardStop('This profile requires one visible CUDA device')
    smi=Path('/usr/lib/wsl/lib/nvidia-smi')
    dedicated=int(subprocess.check_output([str(smi),'--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,timeout=5).strip())*2**20
    guest=guest_available()
    host=json.loads((out/'host.jsonl').read_text(encoding='utf-8-sig').splitlines()[-1])
    validate(host,2*GIB,time.time())
    checks=admit(cuda_free,dedicated,guest,host)
    save(out/'admission.json',dict(utc=utc(),checks=checks,cuda_free=cuda_free,cuda_total=cuda_total,
         dedicated_free=dedicated,guest_available=guest,host=host,plan=PROFILE['placement']))
    if not all(checks.values()):raise GuardStop('Placement admission failed: '+str(checks))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    argv=[str(exe),'-m',str(model),'--alias','local-model','--host','127.0.0.1','--port',str(port),
          '-c','2048','-np','1','--fit','off','--no-context-shift','-b','512','-ub','128',
          '-ctk','f16','-ctv','f16','-ngl','51']
    env=dict(os.environ,LD_LIBRARY_PATH=libpath+(':'+os.environ['LD_LIBRARY_PATH'] if os.environ.get('LD_LIBRARY_PATH') else ''),LLAMA_ARG_LOG_VERBOSITY='5')
    prompt=(ROOT/'prompt.txt').read_text()
    native_log=out/'native-stderr.txt'
    labels=['first','second','warmup','benchmark-1','benchmark-2','benchmark-3','pressure']
    replies=[]
    with native_log.open('xb') as err,(out/'native-stdout.txt').open('xb') as stdout:
        process=subprocess.Popen(argv,stdout=stdout,stderr=err,cwd=ROOT,env=env)
        try:
            save(out/'server.json',{'argv':argv,'pid':process.pid,'started_utc':utc()})
            health(base,process)
            log=native_log.read_text(errors='replace')
            found=re.search(r'offloaded\s+(\d+)/(\d+)\s+layers to GPU',log,re.I)
            if not found or tuple(map(int,found.groups()))!=(51,65):
                raise RuntimeError('Expected observed placement of 51/65 layers; inspect native-stderr.txt')
            for label in labels:
                if process.poll() is not None:raise RuntimeError('Owned server exited before '+label)
                start=utc();answer=spark_lab.request(base,prompt,128);end=utc()
                save(out/(label+'.json'),dict(started_utc=start,ended_utc=end,**answer))
                if answer['status']!=200 or not spark_lab.answer(answer)[0]:raise RuntimeError('No completed answer for '+label)
                replies.append(dict(label=label,started_utc=start,ended_utc=end,elapsed_s=answer['elapsed_s'],
                                    finish_reason=spark_lab.answer(answer)[1]))
                save(out/'requests.json',{'placement':list(map(int,found.groups())),'requests':replies})
        finally:
            if process.poll() is None: process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill();process.wait(timeout=5)
            save(out/'native-exit.json',{'pid':process.pid,'exit':process.returncode,'ended_utc':utc()})
    return 0


if __name__=='__main__':
    def interrupted(signum, frame):
        raise GuardStop('Model worker stopped by its supervisor')
    signal.signal(signal.SIGTERM, interrupted)
    try: sys.exit(run(Path(sys.argv[1])))
    except GuardStop as exc:
        print(str(exc),file=sys.stderr);sys.exit(3)
