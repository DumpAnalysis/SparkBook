"""Qualify a native monitoring stack with explicit simulated GPU fields."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path
from fixture import write
from client import get,query
HERE=Path(__file__).resolve().parent
def wait_for(name,fn,timeout=65):
 end=time.monotonic()+timeout;last=None
 while time.monotonic()<end:
  try:
   value=fn()
   if value:print('PASS '+name,flush=True);return value
  except Exception as e:last=str(e)
  time.sleep(.5)
 raise RuntimeError(name+' timed out: '+str(last))
def values(expr):return query(expr)['data']['result']
def value_is(expr,expected):
 r=values(expr);return bool(r) and abs(float(r[0]['value'][1])-expected)<1e-6
def alert(name,state='firing'):
 return any(x['labels']['alertname']==name and x['state']==state for x in json.loads(get('http://127.0.0.1:9090/api/v1/alerts'))['data']['alerts'])
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tools',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--allow-unavailable-grafana-backend',action='store_true');a=p.parse_args()
 a.evidence.mkdir(parents=True,exist_ok=True);a.run_dir=a.run_dir.resolve();fixture=a.run_dir.with_suffix('.fixture.json');write(fixture,'baseline')
 tools=json.loads(a.tools.read_text());log=(a.evidence/'stack-output.txt').open('w')
 proc=subprocess.Popen([sys.executable,str(HERE/'stack.py'),'--tools',str(a.tools.resolve()),'--run-dir',str(a.run_dir),'--fixture',str(fixture)],stdout=log,stderr=subprocess.STDOUT)
 def capture(name,obj):(a.evidence/name).write_text(json.dumps(obj,indent=2)+'\n')
 try:
  wait_for('Prometheus ready',lambda:get('http://127.0.0.1:9090/-/ready'))
  subprocess.run([tools['promtool'],'check','config',str(a.run_dir/'prometheus.yml')],check=True)
  wait_for('GPU target up',lambda:value_is('up{job="gpu"}',1))
  wait_for('SIMULATED busy ratio = 0.70',lambda:value_is('lab_gpu_busy_ratio',.7))
  wait_for('SIMULATED memory ratio = 0.40',lambda:value_is('lab:gpu_memory_used:ratio',.4))
  capture('baseline-query.json',query('lab:gpu_memory_used:ratio'));capture('targets.json',json.loads(get('http://127.0.0.1:9090/api/v1/targets')))
  def grafana():
   r=subprocess.run([sys.executable,str(HERE/'client.py'),'grafana','--run-dir',str(a.run_dir),'--provision-only'],capture_output=True,text=True)
   if r.returncode:
    (a.evidence/'grafana-last-error.txt').write_text(r.stderr);return None
   obj=json.loads(r.stdout);capture('grafana.json',obj);return obj
  wait_for('Grafana health, datasource and 13-panel dashboard',grafana)
  backend=subprocess.run([sys.executable,str(HERE/'client.py'),'grafana','--run-dir',str(a.run_dir)],capture_output=True,text=True)
  (a.evidence/'grafana-backend.txt').write_text(backend.stdout+backend.stderr+'\n[exit '+str(backend.returncode)+']\n')
  if backend.returncode:
   print('UNAVAILABLE Grafana query backend; see grafana-backend.txt and logs/grafana.log',flush=True)
   if not a.allow_unavailable_grafana_backend:raise RuntimeError('Grafana query backend qualification failed')
  else:print('PASS Grafana query backend and proxied PromQL',flush=True)
  write(fixture,'missing');wait_for('missing busy field = availability 0',lambda:value_is('lab_gpu_field_available{field="utilization.gpu"}',0))
  wait_for('missing busy gauge omitted',lambda:not values('lab_gpu_busy_ratio'))
  capture('missing-query.json',query('lab_gpu_busy_ratio'))
  wait_for('unavailable counter alert firing while up=1',lambda:alert('GpuBusyCounterUnavailable') and value_is('up{job="gpu"}',1))
  capture('unavailable-alert.json',json.loads(get('http://127.0.0.1:9090/api/v1/alerts')))
  write(fixture,'zero');wait_for('valid zero busy gauge exists',lambda:value_is('lab_gpu_busy_ratio',0))
  wait_for('unavailable counter alert resolved',lambda:not alert('GpuBusyCounterUnavailable'))
  write(fixture,'pressure');wait_for('memory pressure pending',lambda:alert('GpuMemoryPressure','pending'))
  wait_for('memory pressure firing after hold',lambda:alert('GpuMemoryPressure'))
  capture('pressure-alert.json',json.loads(get('http://127.0.0.1:9090/api/v1/alerts')))
  write(fixture,'baseline');wait_for('memory pressure resolved',lambda:not alert('GpuMemoryPressure'))
  subprocess.run([sys.executable,str(HERE/'client.py'),'work','--count','20','--delay-ms','20','--errors','2'],check=True)
  wait_for('synthetic request count 20',lambda:value_is('sum(lab_demo_requests_total)',20))
  wait_for('synthetic p95 finite and positive',lambda:bool(values('histogram_quantile(0.95, sum by(le)(rate(lab_demo_request_duration_seconds_bucket[1m]))) > 0')))
  capture('latency-query.json',query('histogram_quantile(0.95, sum by(le)(rate(lab_demo_request_duration_seconds_bucket[1m])))'))
  capture('result.json',{'evidence':'Native monitoring stack; GPU values SIMULATED','prometheus_pipeline':'passed','grafana_provisioning':'passed','grafana_query_backend':'unavailable' if backend.returncode else 'passed','device_workload':'not part of this fixture test'})
  print('PASS Prometheus pipeline and Grafana provisioning controls',flush=True)
  print('Grafana query backend: '+('UNAVAILABLE' if backend.returncode else 'PASS'),flush=True)
  print('GPU measurements in this integration use the simulated fixture.',flush=True)
 finally:
  proc.terminate()
  try:proc.wait(timeout=25)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
  log.close()
  if (a.run_dir/'logs').exists():shutil.copytree(a.run_dir/'logs',a.evidence/'logs',dirs_exist_ok=True)
  if (a.run_dir/'grafana-password.txt').exists():(a.run_dir/'grafana-password.txt').unlink()
if __name__=='__main__':main()
