"""Local teaching HTTP service. Bounded admission; one inference executor.

Not an Internet-facing production server. Default binding is loopback.
No authentication, TLS, request deduplication or kernel cancellation API.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json,os,re,signal,sys,threading,time,traceback,uuid
from infer import MODEL,REVISION,tokenize,encode

LOCK=threading.Lock()
STATE={'phase':'loading','inflight':0,'counts':Counter(),'infer_seconds':0.,'queue_seconds':0.}
ADMISSION=threading.BoundedSemaphore(2)
EXEC=ThreadPoolExecutor(max_workers=1,thread_name_prefix='model-owner')
DELAY=float(os.environ.get('LAB_INFER_DELAY_MS','0'))/1000

def log(record):
    with LOCK:print(json.dumps(record,separators=(',',':')),flush=True)

def compute(ids,accepted):
    start=time.perf_counter_ns()
    if DELAY:time.sleep(DELAY)
    values=encode(ids).tolist()
    end=time.perf_counter_ns()
    return values,start,end

class Server(ThreadingHTTPServer):
    daemon_threads=False
    block_on_close=True
    request_queue_size=16
    slots=threading.BoundedSemaphore(16)
    def get_request(self):
        sock,addr=super().get_request();sock.settimeout(5);return sock,addr
    def process_request(self,request,address):
        if not self.slots.acquire(False):
            request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
            self.shutdown_request(request);return
        try:super().process_request(request,address)
        except BaseException:self.slots.release();raise
    def process_request_thread(self,request,address):
        try:super().process_request_thread(request,address)
        finally:self.slots.release()

class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,*a):pass
    def reply(self,status,body,content_type='application/json',extra=None):
        raw=body.encode() if isinstance(body,str) else json.dumps(body,separators=(',',':')).encode()
        self.send_response(status);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(raw)));self.send_header('Connection','close')
        self.send_header('X-Request-ID',self.rid)
        for k,v in (extra or {}).items():self.send_header(k,v)
        self.end_headers();self.close_connection=True
        try:self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError):pass
        with LOCK:STATE['counts'][(self.route,status)]+=1
        log(dict(event='http',request_id=self.rid,route=self.route,status=status,
                 model_revision=REVISION,elapsed_ms=(time.perf_counter_ns()-self.begin)/1e6))
    def begin_request(self):
        self.begin=time.perf_counter_ns()
        raw=self.headers.get('X-Request-ID','')
        self.rid=raw if re.fullmatch(r'[A-Za-z0-9_-]{1,64}',raw) else uuid.uuid4().hex
        self.route=self.path if self.path in ['/livez','/readyz','/v1/models','/metrics','/v1/embeddings'] else 'other'
    def do_GET(self):
        self.begin_request()
        with LOCK:phase=STATE['phase']
        if self.path=='/livez':return self.reply(200,{'status':'alive'})
        if self.path=='/readyz':return self.reply(200 if phase=='ready' else 503,{'status':phase})
        if self.path=='/v1/models':return self.reply(200,{'data':[{'id':MODEL,'revision':REVISION,'engine':'numpy','device':'cpu'}]})
        if self.path=='/metrics':
            with LOCK:
                lines=[f'lab_inflight {STATE["inflight"]}',f'lab_ready {int(phase=="ready")}',
                       f'lab_inference_seconds_total {STATE["infer_seconds"]}',
                       f'lab_queue_seconds_total {STATE["queue_seconds"]}']
                lines += [f'lab_http_requests_total{{route="{r}",status="{s}"}} {n}' for (r,s),n in sorted(STATE['counts'].items())]
            return self.reply(200,'\n'.join(lines)+'\n','text/plain; version=0.0.4')
        return self.reply(404,{'error':'unknown route'})
    def do_POST(self):
        self.begin_request()
        if self.path!='/v1/embeddings':return self.reply(404,{'error':'unknown route'})
        if self.headers.get('Transfer-Encoding'):return self.reply(400,{'error':'chunked request bodies are not supported'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.reply(415,{'error':'application/json required'})
        try:
            size=int(self.headers.get('Content-Length','-1'))
            if size<0:return self.reply(411,{'error':'Content-Length required'})
            if size>4096:return self.reply(413,{'error':'body limit 4096 bytes'})
            raw=self.rfile.read(size)
            if len(raw)!=size:raise ValueError('incomplete body')
            body=json.loads(raw)
            if not isinstance(body,dict):raise ValueError('JSON object required')
            if body.get('model')!=MODEL:return self.reply(404,{'error':'unknown model'})
            texts=body.get('input');texts=[texts] if isinstance(texts,str) else texts
            if not isinstance(texts,list) or not 1<=len(texts)<=4:raise ValueError('one to four inputs required')
            ids=[tokenize(t) for t in texts]
        except (ValueError,TypeError,UnicodeError) as e:return self.reply(400,{'error':str(e)})
        with LOCK:
            if STATE['phase']!='ready':admitted=False;status=503
            else:
                admitted=ADMISSION.acquire(False);status=429
                if admitted:STATE['inflight']+=1
        if not admitted:return self.reply(status,{'error':'not ready' if status==503 else 'capacity exhausted'},extra={'Retry-After':'1'})
        accepted=time.perf_counter_ns()
        try:
            values,start,end=EXEC.submit(compute,ids,accepted).result()
            with LOCK:
                STATE['infer_seconds']+=(end-start)/1e9;STATE['queue_seconds']+=(start-accepted)/1e9
            log(dict(event='inference',request_id=self.rid,accepted_ns=accepted,start_ns=start,end_ns=end,
                     queue_ms=(start-accepted)/1e6,inference_ms=(end-start)/1e6,batch=len(ids),
                     model_revision=REVISION,engine='numpy',device='cpu'))
            return self.reply(200,{'object':'list','model':MODEL,'revision':REVISION,
                   'data':[{'object':'embedding','index':i,'embedding':v} for i,v in enumerate(values)]})
        except Exception:
            log(dict(event='inference_error',traceback=traceback.format_exc()))
            return self.reply(500,{'error':'inference failed'})
        finally:
            with LOCK:STATE['inflight']-=1
            ADMISSION.release()

def main():
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8088)
    a=p.parse_args();server=Server((a.host,a.port),Handler);stopping=threading.Event()
    def load():
        time.sleep(float(os.environ.get('LAB_LOAD_DELAY_MS','0'))/1000)
        try:
            if os.environ.get('LAB_FAIL_LOAD')=='1':raise RuntimeError('injected load failure')
            encode([tokenize('warmup')])
            with LOCK:
                if STATE['phase']=='loading':STATE['phase']='ready'
        except Exception:
            with LOCK:STATE['phase']='failed'
            log(dict(event='load_failed',traceback=traceback.format_exc()))
    def stop(*_):
        if stopping.is_set():return
        stopping.set()
        with LOCK:STATE['phase']='draining'
        log({'event':'draining'})
        # Stop accepting new sockets; already admitted requests finish.
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    threading.Thread(target=load,daemon=True).start()
    log(dict(event='listening',host=a.host,port=server.server_port,pid=os.getpid(),model_revision=REVISION))
    server.serve_forever(poll_interval=.05)
    server.server_close();EXEC.shutdown(wait=True)
    log({'event':'stopped','inflight':STATE['inflight']})

if __name__=='__main__':main()
