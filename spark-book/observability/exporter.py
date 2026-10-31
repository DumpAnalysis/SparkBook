"""Loopback-only teaching exporter. No driver installation or state-changing GPU calls."""
import argparse, copy, csv, io, json, logging, math, re, shutil, signal, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Query field: metric name, SI multiplier, lower bound, upper bound.
FIELDS={
 'utilization.gpu':('lab_gpu_busy_ratio',.01,0,100),
 'utilization.memory':('lab_gpu_memory_busy_ratio',.01,0,100),
 'memory.used':('lab_gpu_memory_used_bytes',1048576,0,None),
 'memory.free':('lab_gpu_memory_free_bytes',1048576,0,None),
 'memory.total':('lab_gpu_memory_total_bytes',1048576,0,None),
 'temperature.gpu':('lab_gpu_temperature_celsius',1,-40,200),
 'power.draw':('lab_gpu_power_watts',1,0,None),
 'power.limit':('lab_gpu_power_limit_watts',1,0,None),
 'clocks.current.sm':('lab_gpu_sm_clock_hertz',1000000,0,None),
 'clocks.current.memory':('lab_gpu_memory_clock_hertz',1000000,0,None)}

def parse(field,text):
 rows=list(csv.reader(io.StringIO(text.strip())))
 if len(rows)!=1 or len(rows[0])!=1:return None,'shape_error'
 raw=rows[0][0].strip()
 if raw.lower() in ('n/a','[n/a]','not supported','[not supported]',''):return None,'unavailable'
 try:
  n=float(raw);_,scale,lo,hi=FIELDS[field]
  if not math.isfinite(n) or n<lo or hi is not None and n>hi:raise ValueError()
  return n*scale,'ok'
 except ValueError:return None,'parse_error'

def call(argv):
 try:
  p=subprocess.run(argv,capture_output=True,text=True,timeout=1)
  return {'argv':argv,'exit':p.returncode,'stdout':p.stdout[:16384],'stderr':p.stderr[:16384]}
 except (OSError,subprocess.TimeoutExpired) as e:
  return {'argv':argv,'exit':None,'stdout':'','stderr':str(e),'error':'query_error'}

def poll_live(smi,gpu):
 values={};raw=[]
 for field in FIELDS:
  r=call([smi,'-i',gpu,'--query-gpu='+field,'--format=csv,noheader,nounits']);raw.append(r)
  value,status=parse(field,r['stdout']) if r['exit']==0 else (None,'query_error')
  values[field]={'value':value,'status':status}
 return values,raw

def poll_fixture(path):
 try:
  if path.stat().st_size>8192:raise ValueError('fixture too large')
  obj=json.loads(path.read_text());values={}
  if obj.get('evidence')!='SIMULATED':raise ValueError('fixture evidence must be SIMULATED')
  for field in FIELDS:
   value,status=parse(field,str(obj.get('fields',{}).get(field,'N/A')))
   values[field]={'value':value,'status':status}
  return values,[{'fixture':str(path),'evidence':'SIMULATED','case':obj.get('case')}]
 except (OSError,ValueError,TypeError,AttributeError) as e:
  return {f:{'value':None,'status':'query_error'} for f in FIELDS},[{'fixture_error':str(e)}]

def render(snapshot,now=None,stale_after=60):
 now=time.time() if now is None else now
 source=snapshot['source'];gpu=snapshot['gpu'];labels=f'gpu="{gpu}",source="{source}"'
 fresh=snapshot['time']>0 and now-snapshot['time']<=stale_after
 lines=['# HELP lab_exporter_fixture One for simulated observations, zero for WSL queries.',
 '# TYPE lab_exporter_fixture gauge',f'lab_exporter_fixture {int(source=="fixture")}',
 '# HELP lab_exporter_last_poll_timestamp_seconds Completion time of the most recent poll, successful or failed.',
 '# TYPE lab_exporter_last_poll_timestamp_seconds gauge',f'lab_exporter_last_poll_timestamp_seconds {snapshot["time"]}',
 '# HELP lab_exporter_poll_duration_seconds Wall duration of the last collector poll.',
 '# TYPE lab_exporter_poll_duration_seconds gauge',f'lab_exporter_poll_duration_seconds {snapshot["duration"]}',
 '# HELP lab_gpu_field_available One only for a valid fresh field; zero does not mean zero GPU activity.',
 '# TYPE lab_gpu_field_available gauge',
 '# HELP lab_gpu_field_status Current bounded status, one active status per field.',
 '# TYPE lab_gpu_field_status gauge']
 for field,(metric,_,_,_) in FIELDS.items():
  item=snapshot['fields'].get(field,{'status':'unavailable','value':None})
  status=item['status'] if fresh else 'stale';ok=status=='ok'
  lines += [f'lab_gpu_field_available{{{labels},field="{field}"}} {int(ok)}',
            f'lab_gpu_field_status{{{labels},field="{field}",status="{status}"}} 1']
  if ok:
   lines += [f'# HELP {metric} Fresh {field} observation converted to the metric unit; source identifies simulation.',
             f'# TYPE {metric} gauge',f'{metric}{{{labels}}} {item["value"]}']
 return '\n'.join(lines)+'\n'

def main():
 p=argparse.ArgumentParser(description=__doc__)
 mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--live',action='store_true');mode.add_argument('--fixture',type=Path)
 p.add_argument('--gpu',default='0');p.add_argument('--smi');p.add_argument('--port',type=int,default=9400)
 p.add_argument('--interval',type=float,default=5);p.add_argument('--log',type=Path,required=True)
 a=p.parse_args()
 if not .2<=a.interval<=30:p.error('interval must be .2..30 seconds')
 smi=a.smi or shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
 source='wsl' if a.live else 'fixture';gpu='fixture-gpu'
 if a.live:
  if 'microsoft' not in Path('/proc/sys/kernel/osrelease').read_text().lower():
   p.error('The live WSL provider requires a WSL kernel; this environment is not qualified.')
  ident=call([smi,'-i',a.gpu,'--query-gpu=uuid','--format=csv,noheader,nounits'])
  gpu=ident['stdout'].strip()
  if ident['exit']!=0 or not re.fullmatch(r'GPU-[A-Za-z0-9-]{8,80}',gpu):
   p.error('Cannot qualify one GPU UUID from mapped WSL nvidia-smi. Keep the failure; do not substitute a fixture for live evidence.')
 a.log.parent.mkdir(parents=True,exist_ok=True)
 logger=logging.getLogger('gpu-exporter');logger.setLevel(logging.INFO)
 handler=RotatingFileHandler(a.log,maxBytes=1048576,backupCount=3);logger.addHandler(handler)
 lock=threading.Lock();stop=threading.Event()
 state={'source':source,'gpu':gpu,'time':0.,'duration':0.,'fields':{}}
 def polling():
  while not stop.is_set():
   start=time.monotonic()
   values,raw=poll_live(smi,gpu) if a.live else poll_fixture(a.fixture)
   stamp=time.time();duration=time.monotonic()-start
   logger.info(json.dumps({'time':stamp,'source':source,'gpu':gpu,'raw':raw,'fields':values},allow_nan=False))
   with lock:state.update(time=stamp,duration=duration,fields=values)
   stop.wait(max(0,a.interval-duration))
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_GET(self):
   if self.path not in ('/metrics','/healthz'):self.send_error(404);return
   with lock:snapshot=copy.deepcopy(state)
   data=(render(snapshot) if self.path=='/metrics' else 'exporter alive\n').encode()
   self.send_response(200);self.send_header('Content-Type','text/plain; version=0.0.4; charset=utf-8')
   self.send_header('Content-Length',str(len(data)));self.end_headers()
   try:self.wfile.write(data)
   except (BrokenPipeError,ConnectionResetError):pass
 class Server(HTTPServer):
  def get_request(self):
   sock,addr=super().get_request();sock.settimeout(2);return sock,addr
 server=Server(('127.0.0.1',a.port),Handler)
 def end(*args):
  stop.set();threading.Thread(target=server.shutdown,daemon=True).start()
 signal.signal(signal.SIGINT,end);signal.signal(signal.SIGTERM,end)
 thread=threading.Thread(target=polling,daemon=True);thread.start()
 print(f'exporter source={source} gpu={gpu} port={a.port}',flush=True)
 try:server.serve_forever(poll_interval=.1)
 finally:stop.set();server.server_close();thread.join(timeout=15);handler.close()
if __name__=='__main__':main()
