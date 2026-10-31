"""Drive separate Python processes to demonstrate real checkpoint recovery."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from common import no_network,save

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['restart','faults','agent']);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False);db=root/'checkpoint.sqlite'
    commands=[]
    def call(action,extra=()):
        out=root/(action+'-'+str(len(commands))+'.json')
        cmd=[sys.executable,'orchestration/graph_lab.py',action,'--db',str(db),'--out',str(out),*extra]
        cp=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        commands.append({'command':cmd,'output':cp.stdout,'returncode':cp.returncode})
        print(cp.stdout,end='')
        if cp.returncode:raise RuntimeError('child failed')
        return json.loads(out.read_text())
    if a.mode in ['restart','faults']:
        start=call('start');assert start['status']=='paused'
        result=call('resume',['--approve','yes']);assert result['approved'] is True
        if a.mode=='faults':call('faults')
        else:print('PASS start and resume ran in separate Python processes')
    else:
        from graph_lab import step
        from langchain_lab import ScriptedChat
        with no_network():
            step(db,'agent-case','start',False,'SIMULATED-AGENT')
            result=step(db,'agent-case','resume',True,'SIMULATED-AGENT',ScriptedChat(),root/'messages.json')
            assert result['verdict'].startswith('SCRIPTED:')
        print('PASS approved graph node invoked real LangChain loop with scripted model')
        print('PASS no model planning or GPU inference claimed')
        save(root/'result.json',result)
    save(root/'commands.json',commands)

if __name__=='__main__':main()
