"""Foreground local laboratory. Ctrl-C stops all child processes. Run without sudo."""
import argparse,json,os,secrets,signal,subprocess,sys,time,datetime,hashlib
from pathlib import Path
from configure import configure,HERE
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tools',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True)
 m=p.add_mutually_exclusive_group(required=True);m.add_argument('--fixture',type=Path);m.add_argument('--live',action='store_true');p.add_argument('--gpu',default='0');a=p.parse_args()
 a.run_dir.mkdir(parents=True,exist_ok=False);root=configure(a.run_dir);tools=json.loads(a.tools.read_text());password=secrets.token_urlsafe(24)
 with open(root/'grafana-password.txt','x',opener=lambda path,flags:os.open(path,flags,0o600)) as f:f.write(password+'\n')
 env=dict(os.environ,GF_SECURITY_ADMIN_PASSWORD=password,GF_SECURITY_ADMIN_USER='admin',GF_PATHS_BUNDLED_PLUGINS=str(Path(tools['grafana_home'])/'data/plugins-bundled'))
 exporter=[sys.executable,str(HERE/'exporter.py'),'--log',str(root/'logs/gpu.jsonl')]
 exporter+=['--live','--gpu',a.gpu] if a.live else ['--fixture',str(a.fixture.resolve())]
 commands={
 'exporter':exporter,
 'workload':[sys.executable,str(HERE/'workload.py')],
 'prometheus':[tools['prometheus'],'--config.file='+str(root/'prometheus.yml'),'--storage.tsdb.path='+str(root/'data-prometheus'),'--storage.tsdb.retention.time=24h','--storage.tsdb.retention.size=500MB','--web.listen-address=127.0.0.1:9090'],
 'grafana':[tools['grafana'],'server','--homepath',tools['grafana_home'],'--config',str(root/'grafana.ini')]}
 children={};logs=[];stopping=False;records={}
 def stop(*args):
  nonlocal stopping
  stopping=True
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 try:
  for name,command in commands.items():
   log=(root/'logs'/f'{name}.log').open('w');logs.append(log);records[name]={'argv':command,'cwd':os.getcwd(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'start_epoch':time.time(),'environment_note':'Grafana password read privately; never recorded'};children[name]=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env if name=='grafana' else None)
  (root/'processes.json').write_text(json.dumps({n:p.pid for n,p in children.items()},indent=2)+'\n')
  print('Stack started; readiness is checked separately.',flush=True)
  print('Prometheus http://127.0.0.1:9090 | Grafana http://127.0.0.1:3000',flush=True)
  print('Grafana user admin; password file '+str(root/'grafana-password.txt'),flush=True)
  print('Evidence mode '+('WSL LIVE' if a.live else 'SIMULATED GPU + SYNTHETIC CPU'),flush=True)
  while not stopping:
   failed=[(n,p.returncode) for n,p in children.items() if p.poll() is not None]
   if failed:raise RuntimeError('Child exited; inspect logs: '+str(failed))
   time.sleep(.2)
 finally:
  for p in children.values():
   if p.poll() is None:p.terminate()
  for p in children.values():
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for name,child in children.items():records[name].update(exit_code=child.returncode,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-records[name]['start_epoch'])
  (root/'child-commands.json').write_text(json.dumps(records,indent=2))
  for f in logs:f.close()
  print('Stack stopped; data and logs retained.',flush=True)
if __name__=='__main__':main()
