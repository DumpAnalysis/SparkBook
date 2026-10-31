"""Run all CPU controls with complete stdout, stderr, commands and source hashes."""
import argparse,json,os,platform,subprocess,sys
from pathlib import Path
from common import fresh,sha,write_json
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parent; jobs=[]
out=Path(a.out).resolve()
if not out.is_relative_to(root.parent):
    p.error('--out must be inside spark-book')
out=fresh(out).resolve()
os.chdir(root.parent)
for n in range(235,251):
    job=out/str(n);cmd=['python3','runtimes/lab.py',str(n),'--out',str((job/'artifacts').relative_to(root.parent))]
    r=subprocess.run(cmd,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')})
    job.mkdir(exist_ok=True)
    (job/'stdout.txt').write_text(r.stdout);(job/'stderr.txt').write_text(r.stderr)
    (job/'transcript.txt').write_text('$ '+' '.join(cmd)+'\n'+r.stdout+r.stderr+'[exit '+str(r.returncode)+']\n')
    jobs.append({'exercise':n,'exit':r.returncode,'command':cmd});print(n,'PASS' if r.returncode==0 else 'FAIL',flush=True)
write_json(out/'result.json',{'scope':'CPU controls only; no learned model or WSL2 GPU execution','python':platform.python_version(),'machine':platform.machine(),'jobs':jobs,'passed':sum(j['exit']==0 for j in jobs)})
write_json(out/'source-hashes.json',{p.name:sha(p) for p in sorted(root.glob('*.py'))})
sys.exit(any(j['exit']!=0 for j in jobs))
