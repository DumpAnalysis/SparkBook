"""Synthetic CPU HTTP work. Histogram examples are NOT GPU/model benchmarks."""
import argparse,math,signal,threading,time
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import urlparse,parse_qs
BUCKETS=(.005,.01,.025,.05,.1,.25,.5,1.0)
class Metrics:
 def __init__(self):self.count={'200':0,'503':0};self.buckets=[0]*len(BUCKETS);self.total=0.;self.n=0
 def observe(self,status,elapsed):
  self.count[str(status)]+=1;self.n+=1;self.total+=elapsed
  for i,b in enumerate(BUCKETS):self.buckets[i]+=int(elapsed<=b)
 def render(self):
  lines=['# HELP lab_demo_requests_total Synthetic CPU HTTP responses, not model throughput.','# TYPE lab_demo_requests_total counter']
  for status,n in self.count.items():lines.append(f'lab_demo_requests_total{{source="synthetic",status="{status}"}} {n}')
  lines+=['# HELP lab_demo_request_duration_seconds Synthetic CPU handler duration.','# TYPE lab_demo_request_duration_seconds histogram']
  for b,n in zip(BUCKETS,self.buckets):lines.append(f'lab_demo_request_duration_seconds_bucket{{source="synthetic",le="{b}"}} {n}')
  lines += [f'lab_demo_request_duration_seconds_bucket{{source="synthetic",le="+Inf"}} {self.n}',f'lab_demo_request_duration_seconds_count{{source="synthetic"}} {self.n}',f'lab_demo_request_duration_seconds_sum{{source="synthetic"}} {self.total}']
  return '\n'.join(lines)+'\n'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8089);a=p.parse_args();metrics=Metrics()
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_GET(self):
   u=urlparse(self.path);status=200
   if u.path=='/metrics':body=metrics.render().encode()
   elif u.path=='/work':
    try:
     q=parse_qs(u.query);delay=float(q.get('delay_ms',['10'])[0]);fail=q.get('fail',['0'])[0]
     if not math.isfinite(delay) or not 0<=delay<=500 or fail not in ('0','1'):raise ValueError()
    except ValueError:self.send_error(400);return
    start=time.monotonic();time.sleep(delay/1000);status=503 if fail=='1' else 200
    metrics.observe(status,time.monotonic()-start);body=b'SYNTHETIC CPU work\n'
   else:self.send_error(404);return
   self.send_response(status);self.send_header('Content-Type','text/plain; version=0.0.4');self.send_header('Content-Length',str(len(body)));self.end_headers()
   try:self.wfile.write(body)
   except (BrokenPipeError,ConnectionResetError):pass
 class Server(HTTPServer):
  def get_request(self):
   sock,addr=super().get_request();sock.settimeout(2);return sock,addr
 server=Server(('127.0.0.1',a.port),Handler)
 def stop(*args):threading.Thread(target=server.shutdown,daemon=True).start()
 signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 print('synthetic CPU workload port='+str(a.port),flush=True)
 try:server.serve_forever(.1)
 finally:server.server_close()
if __name__=='__main__':main()
