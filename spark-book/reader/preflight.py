"""Read-only inventory; a management query is not CUDA qualification."""
import argparse, importlib.util, json, os, platform, shutil, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--require-wsl',action='store_true');a=p.parse_args()
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
print(json.dumps(record,indent=2));raise SystemExit(2 if a.require_wsl and not wsl else 0)
