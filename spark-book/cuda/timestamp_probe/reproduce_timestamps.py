"""Standalone bounded GPU/CUPTI diagnostic. Writes only a new output directory."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shlex, signal, subprocess, sys, time
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
parser=argparse.ArgumentParser()
parser.add_argument('--out',required=True,type=Path)
parser.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda-13.4'))
parser.add_argument('--arch',default='sm_121')
args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
source=Path(__file__).with_name('timestamp_matrix.cu');exe=out/'timestamp_matrix'
env=dict(os.environ,CUDA_CACHE_DISABLE='1');captures=[]
def run(name,argv):
    directory=out/name;directory.mkdir();argv=list(map(str,argv));start=utc();tick=time.perf_counter()
    with (directory/'stdout.txt').open('wb') as stdout,(directory/'stderr.txt').open('wb') as stderr:
        proc=subprocess.Popen(argv,cwd=out,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        timeout=False
        try:rc=proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
    r={'argv':argv,'command_shell':shlex.join(argv),'cwd':str(out),'started_utc':start,'ended_utc':utc(),'elapsed_seconds':time.perf_counter()-tick,'exit_code':rc,'timed_out':timeout,'source_sha256':sha(source),'runner_sha256':sha(__file__),'environment':{k:env.get(k) for k in ['CUDA_CACHE_DISABLE','LD_LIBRARY_PATH','LD_PRELOAD','CUDA_VISIBLE_DEVICES']},'stdout_sha256':sha(directory/'stdout.txt'),'stderr_sha256':sha(directory/'stderr.txt')}
    (directory/'metadata.json').write_text(json.dumps(r,indent=2)+'\n');captures.append(r);print(json.dumps(r),flush=True)
    return rc
cupti=args.cuda/'extras/CUPTI'
rc=run('compile',[args.cuda/'bin/nvcc','-arch='+args.arch,'-I'+str(cupti/'include'),source,'-L'+str(cupti/'lib64'),'-Xlinker','-rpath','-Xlinker',cupti/'lib64','-lcupti','-o',exe])
if rc:raise SystemExit(rc)
results={mode:run(mode,[exe,mode]) for mode in ['delay','hes-delay']}
(out/'ACCEPTANCE.json').write_text(json.dumps({'utc':utc(),'status':'PASSED' if all(x==0 for x in results.values()) else 'FAILED','execution':'GPU','results':results,'meaning':'Each mode needs numerical correctness, at least2 timed kernels with start>0 and end>start, and zero invalid kernels. Exit2 means timeline unavailable. A CUDA-event duration is not a CUPTI timeline.','executable_sha256':sha(exe)},indent=2)+'\n')
raise SystemExit(0 if all(x==0 for x in results.values()) else 2)
