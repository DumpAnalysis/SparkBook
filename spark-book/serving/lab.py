"""Execute real loopback HTTP controls and retain complete server evidence."""
import argparse,json,os,platform,shutil,subprocess,sys,time,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from client import request,parse_sse
from infer import encode,tokenize,REVISION
from core import oracle

ROOT=Path(__file__).resolve().parent
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

class Session:
    def __init__(self,out,**env):
        self.out=out;out.mkdir();self.log=open(out/'server.jsonl','w')
        self.p=subprocess.Popen([sys.executable,str(ROOT/'server.py'),'--port','0'],
             stdout=self.log,stderr=subprocess.STDOUT,env=os.environ|env)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            lines=(out/'server.jsonl').read_text().splitlines()
            first=next((json.loads(s) for s in lines if s.startswith('{') and json.loads(s).get('event')=='listening'),None)
            if first:self.base='http://127.0.0.1:'+str(first['port']);break
            if self.p.poll() is not None:raise RuntimeError('server failed; inspect log')
            time.sleep(.02)
        else:self.close();raise TimeoutError('listener startup')
    def ready(self):
        for _ in range(100):
            r=request(self.base,'/readyz')
            if r['status']==200:return r
            time.sleep(.02)
        raise TimeoutError('readiness')
    def close(self,signal=True):
        if signal and self.p.poll() is None:self.p.terminate()
        try:self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:self.p.kill();self.p.wait();raise
        finally:self.log.close()

def contract(out):
    s=Session(out/'server');rows=[]
    try:
        s.ready()
        for path,body,rid in [('/v1/models',None,'models'),('/v1/embeddings',{'model':'lab-embedding','input':['alpha','beta']},'case-1'),
          ('/v1/embeddings',{'model':'wrong','input':'alpha'},'wrong-model'),
          ('/v1/embeddings',{'model':'lab-embedding','input':'x'*17},'too-long'),
          ('/v1/embeddings',{'model':'lab-embedding','input':'x'*5000},'body-large')]:
            rows.append(request(s.base,path,body,rid))
        expected=oracle([tokenize('alpha'),tokenize('beta')])
        actual=np.asarray([x['embedding'] for x in json.loads(rows[1]['body'])['data']])
        error=float(np.max(np.abs(actual-expected)))
        assert [r['status'] for r in rows]==[200,200,404,400,413]
        assert rows[1]['headers']['X-Request-ID']=='case-1' and error<=1e-6
        save(out/'responses.json',rows);save(out/'result.json',dict(status='PASS',statuses=[r['status'] for r in rows],max_abs_error=error,revision=REVISION))
        print('PASS live HTTP contract statuses=200,200,404,400,413')
        print(f'embeddings=2 dimension=64 max_abs_error={error:.3e}')
        print('request_id=case-1 preserved; full bodies in responses.json')
    finally:s.close()

def lifecycle(out):
    s=Session(out/'loading',LAB_LOAD_DELAY_MS='400',LAB_INFER_DELAY_MS='200')
    try:
        live=request(s.base,'/livez');early=request(s.base,'/readyz');late=s.ready()
        assert (live['status'],early['status'],late['status'])==(200,503,200)
        with ThreadPoolExecutor(1) as pool:
            f=pool.submit(request,s.base,'/v1/embeddings',{'model':'lab-embedding','input':'alpha'},'drain-1')
            for _ in range(100):
                if 'lab_inflight 1' in request(s.base,'/metrics')['body']:break
                time.sleep(.005)
            else:raise AssertionError('request never admitted')
            s.p.terminate();completed=f.result(timeout=10)
        s.close(signal=False);assert completed['status']==200 and s.p.returncode==0
    finally:
        if not s.log.closed:s.close()
    failed=Session(out/'failed',LAB_FAIL_LOAD='1')
    try:
        for _ in range(100):
            bad=request(failed.base,'/readyz')
            if json.loads(bad['body']).get('status')=='failed':break
            time.sleep(.02)
        assert bad['status']==503
        failure_live=request(failed.base,'/livez');assert failure_live['status']==200
    finally:failed.close()
    save(out/'result.json',dict(status='PASS',liveness=live,loading=early,ready=late,
         drained=completed,exitcode=s.p.returncode,failed_readiness=bad,failed_liveness=failure_live))
    print('PASS loading live=200 ready=503; after warmup ready=200')
    print('injected load failure live=200 ready=503')
    print('SIGTERM admitted response=200 process_exit=0')

def load(out):
    s=Session(out/'server',LAB_INFER_DELAY_MS='100');barrier=threading.Barrier(8)
    try:
        s.ready()
        def one(i):barrier.wait(timeout=10);return request(s.base,'/v1/embeddings',{'model':'lab-embedding','input':'alpha'},f'load-{i}')
        with ThreadPoolExecutor(8) as pool:rows=list(pool.map(one,range(8)))
        metrics=request(s.base,'/metrics');save(out/'responses.json',rows)
        (out/'metrics.txt').write_text(metrics['body'])
        statuses={str(s):sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
        assert statuses=={'200':2,'429':6},statuses
        assert all(r['headers'].get('Retry-After')=='1' for r in rows if r['status']==429)
        save(out/'result.json',dict(status='PASS',status_counts=statuses,delay_ms=100,scope='injected CPU delay'))
        print('PASS simultaneous_requests=8 accepted=2 rejected_429=6')
        print('injected_CPU_delay_ms=100 Retry-After=1')
        print('full responses, metrics and owner phase logs retained')
    finally:s.close()

def sse(out):
    raw=b': keepalive\n\ndata: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: {"choices":[{"delta":{"content":"GPU"}}]}\n\ndata: {"choices":[{"delta":{"content":" memory"}}]}\n\ndata: [DONE]\n\n'
    events=list(parse_sse(raw.splitlines(keepends=True)))
    text=''.join(c.get('delta',{}).get('content','') for e in events if e!='[DONE]' for c in json.loads(e).get('choices',[]))
    assert text=='GPU memory' and events[-1]=='[DONE]'
    (out/'fixture.sse').write_bytes(raw);save(out/'result.json',dict(status='PASS',text=text,events=events,scope='parser fixture, no LLM execution'))
    print('PASS SSE parser ignores heartbeat and role-only event')
    print('content=GPU memory done=True; no LLM execution')

def qualify(out):
    names=['docker','kubectl','k3s','helm','nvidia-smi','vllm','tritonserver'];rows=[]
    for name in names:
        path=shutil.which(name);rows.append(dict(tool=name,path=path))
        print(name+'='+('PRESENT' if path else 'UNAVAILABLE'))
    save(out/'result.json',dict(platform=platform.platform(),machine=platform.machine(),tools=rows,
                              scope='executable inventory only; no deployment attempted'))
    print('No container, cluster, vLLM or Triton deployment was executed')

def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['contract','lifecycle','load','sse','qualify']);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);globals()[a.case](out)
if __name__=='__main__':main()
