"""Read-only inventory; a management query is not CUDA qualification."""
import argparse, importlib.util, json, os, platform, shlex, shutil, subprocess, sys, sysconfig
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--require-wsl',action='store_true')
p.add_argument('--check-headers',action='store_true',help='preprocess Python.h for this interpreter')
p.add_argument('--check-ninja',action='store_true',help='require Ninja on the active PATH')
a=p.parse_args()
release=platform.release();wsl='microsoft' in release.lower()
gpu=shutil.which('nvidia-smi')
if not gpu and Path('/usr/lib/wsl/lib/nvidia-smi').is_file():gpu='/usr/lib/wsl/lib/nvidia-smi'
record={'origin':'NEW READER CPU INVENTORY','python':sys.version,'executable':sys.executable,
        'architecture':platform.machine(),'kernel':release,'wsl':wsl,
        'virtual_environment':sys.prefix!=sys.base_prefix,'cwd':str(Path.cwd()),
        'disk_free_bytes':shutil.disk_usage(Path.cwd()).free,
        'torch_importable':importlib.util.find_spec('torch') is not None,
        'tools':{n:shutil.which(n) for n in ['bash','nvcc','cmake','ninja','docker','kubectl','nsys','ncu']},
        'cuda_computation':'NOT TESTED','packages_installed':False,'settings_changed':False}
if gpu:
 try:
  c=subprocess.run([gpu],capture_output=True,text=True,timeout=15)
  record['gpu_query']={'argv':[gpu],'exit_code':c.returncode,'stdout':c.stdout,'stderr':c.stderr}
 except subprocess.TimeoutExpired:record['gpu_query']={'status':'TIMED OUT'}
else:record['gpu_query']={'status':'UNAVAILABLE'}
passed = not a.require_wsl or wsl
if a.check_headers:
    directories = sorted({sysconfig.get_path('include'),sysconfig.get_path('platinclude')})
    compiler = shlex.split(os.environ.get('CC') or sysconfig.get_config_var('CC') or 'cc')
    command = compiler + ['-I'+path for path in directories] + ['-E','-x','c','-','-o',os.devnull]
    try:
        result = subprocess.run(command,input='#include <Python.h>\n',text=True,capture_output=True,timeout=20)
        record['python_headers'] = dict(command=command, include=directories, exit=result.returncode, stderr=result.stderr.strip())
        passed = passed and result.returncode == 0
    except (OSError,subprocess.TimeoutExpired) as error:
        record['python_headers'] = dict(command=command, error=str(error))
        passed = False
if a.check_ninja:
    passed = passed and bool(record['tools']['ninja'])
print(json.dumps(record,indent=2))
raise SystemExit(0 if passed else 2)
