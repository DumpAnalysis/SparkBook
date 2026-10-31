"""Dependency-free HTTP client with complete response and SSE retention."""
import argparse,json,time,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import statistics

def request(base,path,body=None,rid='lab',timeout=30):
    raw=None if body is None else json.dumps(body).encode()
    req=urllib.request.Request(base.rstrip('/')+path,raw,
        {'Content-Type':'application/json','X-Request-ID':rid})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start=time.perf_counter_ns()
    try:
        with opener.open(req,timeout=timeout) as r:status=r.status;headers=dict(r.headers);data=r.read().decode()
    except urllib.error.HTTPError as r:status=r.code;headers=dict(r.headers);data=r.read().decode()
    except Exception as e:status=0;headers={};data=repr(e)
    return dict(request_id=rid,status=status,headers=headers,body=data,elapsed_ms=(time.perf_counter_ns()-start)/1e6)

def parse_sse(lines):
    data=[]
    for raw in lines:
        line=raw.decode('utf-8').rstrip('\r\n') if isinstance(raw,bytes) else raw.rstrip('\r\n')
        if not line:
            if data:yield '\n'.join(data);data=[]
        elif line.startswith('data:'):data.append(line[5:].lstrip(' '))
    if data:yield '\n'.join(data)

def stream(base,model,out):
    body={'model':model,'messages':[{'role':'user','content':'Explain GPU memory in one sentence.'}],
          'max_tokens':64,'temperature':0,'stream':True}
    req=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    raw=[];events=[];first=None;content=[];done=False;start=time.perf_counter_ns()
    def lines(response):
        for line in response:
            raw.append(line);yield line
    try:
        with opener.open(req,timeout=120) as response:
            for data in parse_sse(lines(response)):
                now=time.perf_counter_ns()
                if data=='[DONE]':done=True;break
                item=json.loads(data);events.append({'received_ns':now,'data':item})
                if 'error' in item:raise RuntimeError(str(item['error']))
                for choice in item.get('choices',[]):
                    text=choice.get('delta',{}).get('content')
                    if text:
                        if first is None:first=now
                        content.append(text)
        result=dict(status='PASS' if done and first is not None else 'INCOMPLETE',done=done,
            first_content_ms=None if first is None else (first-start)/1e6,
            total_ms=(time.perf_counter_ns()-start)/1e6,text=''.join(content),events=events)
    except urllib.error.HTTPError as e:
        raw.append(e.read());result=dict(status='FAILED',http_status=e.code,error=repr(e),events=events)
    except Exception as e:result=dict(status='FAILED',error=repr(e),events=events)
    (out/'stream.sse').write_bytes(b''.join(raw));(out/'stream.json').write_text(json.dumps(result,indent=2))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['embed','load','stream','triton','triton-load','chat-load'])
    p.add_argument('--base',default='http://127.0.0.1:8088');p.add_argument('--model',default='lab-embedding')
    p.add_argument('--requests',type=int,default=16);p.add_argument('--concurrency',type=int,default=4)
    p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if a.mode=='stream':
        result=stream(a.base,a.model,out);print(json.dumps({k:v for k,v in result.items() if k!='events'},indent=2))
        if result['status']!='PASS':raise SystemExit(2)
        return
    if not 1<=a.requests<=1024 or not 1<=a.concurrency<=16:p.error('requests 1..1024; concurrency 1..16')
    if a.mode in ['triton','triton-load']:
        from infer import tokenize
        body={'inputs':[{'name':'TOKENS','shape':[2,16],'datatype':'INT64','data':[tokenize('alpha'),tokenize('beta')]}],
              'outputs':[{'name':'EMBEDDINGS'}]};path='/v2/models/'+a.model+'/infer'
    elif a.mode=='chat-load':
        body={'model':a.model,'messages':[{'role':'user','content':'Explain GPU memory in one sentence.'}],
              'max_tokens':64,'temperature':0,'stream':False};path='/v1/chat/completions'
    else:body={'model':a.model,'input':['alpha','beta']};path='/v1/embeddings'
    loading=a.mode in ['load','triton-load','chat-load']
    n=a.requests if loading else 1;start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency if loading else 1) as pool:
        results=list(pool.map(lambda i:request(a.base,path,body,f'client-{i}'),range(n)))
    elapsed=time.perf_counter()-start
    (out/'responses.json').write_text(json.dumps(results,indent=2))
    counts={str(s):sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})}
    lat=sorted(r['elapsed_ms'] for r in results if r['status']==200)
    report=dict(requests=n,status_counts=counts,successful=len(lat),elapsed_seconds=elapsed,
       successful_requests_per_second=len(lat)/elapsed,
       success_latency_p50_ms=statistics.median(lat) if lat else None,
       success_latency_p95_ms=lat[max(0,__import__('math').ceil(.95*len(lat))-1)] if lat else None)
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not loading and any(r['status']!=200 for r in results):raise SystemExit(2)

if __name__=='__main__':main()
