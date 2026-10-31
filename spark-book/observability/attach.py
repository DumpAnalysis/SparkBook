"""Add or remove a local, already qualified inference metrics endpoint."""
import argparse,json
from pathlib import Path
from client import get
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--kind',choices=['vllm','triton'],required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--remove',action='store_true');a=p.parse_args()
 if not 1024<=a.port<=65535:p.error('port must be 1024..65535')
 path=a.run_dir/'inference-targets.json';current=json.loads(path.read_text())
 current=[x for x in current if x.get('labels',{}).get('engine')!=a.kind]
 if not a.remove:
  text=get(f'http://127.0.0.1:{a.port}/metrics');prefix='vllm:' if a.kind=='vllm' else 'nv_inference_'
  if not any(x.startswith(prefix) for x in text.splitlines()):p.error('Expected metrics absent: preserve raw endpoint and qualify this server before attaching')
  (a.run_dir/(a.kind+'-metrics-before.txt')).write_text(text)
  current.append({'targets':[f'127.0.0.1:{a.port}'],'labels':{'engine':a.kind}})
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(current,indent=2)+'\n');tmp.replace(path)
 print(('Removed ' if a.remove else 'Attached ')+a.kind+'; file discovery refreshes within 5 seconds')
if __name__=='__main__':main()
