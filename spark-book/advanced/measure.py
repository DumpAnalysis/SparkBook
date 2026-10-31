"""Collect bounded WSL observations or emit an explicitly unmeasured template."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
from reference import save_new


def observation(argv):
    try:
        r=subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {'argv':argv,'returncode':r.returncode,
                'stdout':r.stdout[:32768],'stderr':r.stderr[:8192],
                'truncated':len(r.stdout)>32768 or len(r.stderr)>8192}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'argv':argv,'returncode':None,'error':str(e)}


def template():
    return {'schema':1,'status':'UNMEASURED',
      'reported':{'installed_memory_GB':64,'gpu_approx_GB':30,
                  'qualification':'Owner reports; decimal/binary units not reconciled'},
      'measured':{'utc':None,'platform':None,'commands':[]},
      'experiments':{'inference':[],'offload':[],'training':[],
                     'kernels':[],'graphs':[],'tool_support':[]}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--template',action='store_true')
    p.add_argument('--torch',action='store_true')
    p.add_argument('--out',required=True)
    a=p.parse_args();r=template()
    if not a.template:
        release=platform.release()
        if 'microsoft' not in release.lower():
            raise SystemExit('WSL2 collection required; use --template elsewhere')
        commands=[['uname','-a'],['free','-b'],
          ['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used',
           '--format=csv,noheader,nounits'],['nvcc','--version']]
        if a.torch:
            commands.append([sys.executable,'-c',
              'import json,torch; print(json.dumps(dict(version=str(torch.__version__),'
              'cuda=torch.version.cuda,available=torch.cuda.is_available(),'
              'capability=torch.cuda.get_device_capability() if torch.cuda.is_available() else None,'
              'free_total=torch.cuda.mem_get_info() if torch.cuda.is_available() else None)))'])
        r['status']='COLLECTED; inspect each return code'
        r['measured']={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      'platform':platform.platform(),
                      'commands':[observation(c) for c in commands]}
    save_new(a.out,r)
    print('status='+r['status'])
    print('benchmark_results='+str(sum(map(len,r['experiments'].values()))))
    print('saved='+a.out)


if __name__=='__main__':
    main()
