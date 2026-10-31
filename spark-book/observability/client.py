"""Small local-only API client; query output can be captured without credentials."""
import argparse,base64,json,time
from pathlib import Path
from urllib.request import build_opener,ProxyHandler,Request
from urllib.parse import urlencode
OPENER=build_opener(ProxyHandler({}))
def get(url,password=None):
 req=Request(url)
 if password:req.add_header('Authorization','Basic '+base64.b64encode(('admin:'+password).encode()).decode())
 with OPENER.open(req,timeout=5) as r:return r.read().decode()
def query(expr):return json.loads(get('http://127.0.0.1:9090/api/v1/query?'+urlencode({'query':expr})))
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
 q=sub.add_parser('query');q.add_argument('expression')
 sub.add_parser('targets');sub.add_parser('alerts');sub.add_parser('metrics')
 g=sub.add_parser('grafana');g.add_argument('--run-dir',type=Path,required=True);g.add_argument('--provision-only',action='store_true')
 w=sub.add_parser('work');w.add_argument('--count',type=int,default=20);w.add_argument('--delay-ms',type=int,default=10);w.add_argument('--errors',type=int,default=2)
 a=p.parse_args()
 if a.cmd=='query':obj=query(a.expression)
 elif a.cmd=='targets':obj=json.loads(get('http://127.0.0.1:9090/api/v1/targets'))
 elif a.cmd=='alerts':obj=json.loads(get('http://127.0.0.1:9090/api/v1/alerts'))
 elif a.cmd=='metrics':print(get('http://127.0.0.1:9400/metrics'),end='');return
 elif a.cmd=='grafana':
  password=(a.run_dir/'grafana-password.txt').read_text().strip()
  health=json.loads(get('http://127.0.0.1:3000/api/health'))
  ds=json.loads(get('http://127.0.0.1:3000/api/datasources/uid/lab-prometheus',password))
  d=json.loads(get('http://127.0.0.1:3000/api/dashboards/uid/wsl-gpu-lab',password))
  obj={'database':health['database'],'version':health['version'],'datasource_uid':ds['uid'],'datasource_url':ds['url'],'dashboard_uid':d['dashboard']['uid'],'panel_count':len(d['dashboard']['panels']),'scope':'provisioning only; query backend not qualified'}
  if a.provision_only:print(json.dumps(obj,indent=2));return
  check=json.loads(get('http://127.0.0.1:3000/api/datasources/uid/lab-prometheus/health',password))
  assert check['status']=='OK',check
  result=json.loads(get('http://127.0.0.1:3000/api/datasources/proxy/uid/lab-prometheus/api/v1/query?query=lab_gpu_memory_used_bytes',password))
  assert result['status']=='success' and result['data']['result'],result
  obj={'database':health['database'],'version':health['version'],'datasource_uid':ds['uid'],'datasource_url':ds['url'],'datasource_health':check['status'],'proxied_query_series':len(result['data']['result']),'dashboard_uid':d['dashboard']['uid'],'panel_count':len(d['dashboard']['panels'])}
 elif a.cmd=='work':
  from urllib.error import HTTPError
  if not 1<=a.count<=200 or not 0<=a.errors<=a.count or not 0<=a.delay_ms<=500:p.error('bounded count/delay/errors required')
  counts={'200':0,'503':0}
  for i in range(a.count):
   try:get('http://127.0.0.1:8089/work?'+urlencode({'delay_ms':a.delay_ms,'fail':int(i<a.errors)}));counts['200']+=1
   except HTTPError as e:
    if e.code!=503:raise
    counts['503']+=1
  obj={'evidence':'SYNTHETIC CPU','responses':counts,'delay_ms':a.delay_ms}
 print(json.dumps(obj,indent=2))
if __name__=='__main__':main()
