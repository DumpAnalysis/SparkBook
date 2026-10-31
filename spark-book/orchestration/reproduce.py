"""Run all new CPU controls with exact commands, outputs and failure preservation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False)
    jobs=[('environment','environment.py',[]),
          ('langchain','langchain_lab.py',['control']),('langchain-faults','langchain_lab.py',['faults']),
          ('http','client_contract.py',[]),('graph-restart','graph_control.py',['restart']),
          ('graph-faults','graph_control.py',['faults']),('graph-agent','graph_control.py',['agent']),
          ('langsmith','langsmith_lab.py',['control']),('evaluation','langsmith_lab.py',['evaluate']),
          ('langsmith-plan','langsmith_lab.py',['plan']),('pydantic','pydantic_lab.py',['control']),
          ('pydantic-budget','pydantic_lab.py',['budget']),('jev','jev_lab.py',['control']),
          ('calibration','jev_lab.py',['calibration']),('jev-plan','jev_lab.py',['plan'])]
    records=[]
    for name,script,args in jobs:
        folder=root/name;result=folder/('details' if script=='graph_control.py' else 'result.json')
        command=[sys.executable,'catalogs/capture.py','--out',str(folder),'--',sys.executable,'orchestration/'+script,*args,'--out',str(result)]
        cp=subprocess.run(command,timeout=120)
        records.append({'name':name,'exit':cp.returncode})
        if cp.returncode:
            (root/'result.json').write_text(json.dumps({'status':'FAILED','jobs':records},indent=2)+'\n')
            raise SystemExit(cp.returncode)
    sources={str(x.name):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(Path('orchestration').glob('*.py'))}
    (root/'source-hashes.json').write_text(json.dumps(sources,indent=2)+'\n')
    (root/'result.json').write_text(json.dumps({'status':'PASS','jobs':records,'scope':'15 CPU command sessions; fixtures, real framework APIs and loopback HTTP; no learned model or GPU'},indent=2)+'\n')
    print('PASS all 15 CPU command sessions; no hosted service or GPU inference executed')

if __name__=='__main__':main()
