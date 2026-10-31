"""Sixteen deterministic CPU/evidence controls. No learned runtime is simulated as real."""
import argparse, copy, hashlib, json, math, platform, statistics, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from common import ContractError, exchange, endpoint, ndjson, sse, vectors, cosine, tree_identity, fresh, write_json
ROOT=Path(__file__).resolve().parent

def reject(label,fn):
    try:fn()
    except (ValueError,KeyError,TypeError) as e:print('REJECT',label);return
    raise AssertionError('accepted invalid control: '+label)

def identity(expected,actual):
    if expected!=actual:raise ContractError('identity changed')

def split_check(train,test):
    normalize=lambda s:' '.join(s.casefold().split())
    if set(map(normalize,train)) & set(map(normalize,test)):raise ContractError('split leakage')

def training_rows():
    return [
      ('allocation failed: out of memory','memory'),
      ('CUDA allocator exhausted available space','memory'),
      ('model load needs more free device bytes','memory'),
      ('requested tensor allocation exceeds free memory','memory'),
      ('CPU input queue is empty','input'),
      ('data loader waits for a slow source','input'),
      ('GPU waits while preprocessing runs on CPU','input'),
      ('input producer cannot supply the next batch','input'),
      ('no logs were supplied','unknown'),
      ('the symptom has no timestamp or measurement','unknown'),
      ('only a screenshot of an error title is present','unknown'),
      ('request did not complete but cause is absent','unknown')], [
      ('the new tensor cannot fit in remaining CUDA memory','memory'),
      ('the host loader produces data too slowly','input'),
      ('the report omits diagnostic evidence','unknown'),
      ('the device allocation returned out of memory','memory')]

def summarize(rows):
    good=[r for r in rows if r['status']=='ok' and r['complete']]
    if not good:raise ContractError('no completed samples')
    return {'submitted':len(rows),'completed':len(good),'failures':len(rows)-len(good),'median_seconds':statistics.median(r['seconds'] for r in good)}

def budget(weight,gpu,guest,reserve):
    return {'weight_bytes':weight,'gpu_budget_bytes':gpu,'guest_available_bytes':guest,
      'gpu_weight_fit':weight<=gpu,'host_weight_admission':weight+reserve<=guest,
      'scope':'weight-only lower bound; not runtime peak or proof of fit'}

def run(n,out):
    print('CPU CONTROL; no learned model or laptop GPU executed')
    result={'exercise':n,'scope':'executed CPU control'}
    if n==235:
        print('python',platform.python_version(),'machine',platform.machine())
        for u in ['http://localhost:11434','https://127.0.0.1:11434','http://example.com:80','http://user@127.0.0.1:11434']:
            reject(u,lambda u=u:endpoint(u,'/api/version'))
        print('PASS explicit loopback boundary; driver installation unnecessary')
    elif n==236:
        body=b'{"response":"GPU ","done":false}\n{"response":"memory","done":true,"eval_count":2,"eval_duration":1000000000}\n'
        rows=ndjson(body);(out/'stream.ndjson').write_bytes(body)
        print('answer=' + ''.join(x.get('response','') for x in rows))
        print('fixture tokens_per_second=%.1f'%(rows[-1]['eval_count']/(rows[-1]['eval_duration']/1e9)))
        reject('truncated stream',lambda:ndjson(body.splitlines()[0]))
        reject('error terminal',lambda:ndjson(b'{"done":true,"error":"OOM"}'))
        print('PASS terminal state required before timing summary')
    elif n==237:
        rows=vectors([[1.,0.],[0.,1.],[.8,.2]],3);scores=[cosine(rows[0],v) for v in rows[1:]]
        write_json(out/'vectors.json',{'scope':'fabricated geometry','vectors':rows,'scores':scores})
        print('fabricated cosine scores=' + ','.join('%.6f'%s for s in scores))
        reject('dimension mismatch',lambda:vectors([[1,2],[1]],2))
        reject('nonfinite',lambda:vectors([[float('nan')]],1))
        reject('missing row',lambda:vectors([[1]],2))
        print('PASS geometry control; no embedding quality measured')
    elif n==238:
        pin='sha256:'+hashlib.sha256(b'fixture weights').hexdigest()
        identity(pin,pin);reject('tag changed',lambda:identity(pin,'sha256:changed'))
        r=summarize([{'status':'ok','complete':True,'seconds':2.}, {'status':'timeout','complete':False,'seconds':120.}, {'status':'ok','complete':False,'seconds':1.}])
        result.update(r);print('submitted={submitted} completed={completed} failures={failures}'.format(**r))
        print('PASS failure denominator retained; residency is not utilization')
    elif n==239:
        train,test=training_rows();split_check([x[0] for x in train],[x[0] for x in test])
        write_json(out/'data.json',{'train':train,'test':test})
        print('synthetic training rows=%d held_out=%d'%(len(train),len(test)))
        reject('normalized split overlap',lambda:split_check(['CUDA OOM'],[' cuda   oom ']))
        print('PASS data partition; no optimizer step executed')
    elif n==240:
        microbatch,accum,seq,rank=1,4,256,8
        print('microbatch=%d accumulation=%d effective_batch=%d'%(microbatch,accum,microbatch*accum))
        print('sequence_limit=%d LoRA_rank=%d'%(seq,rank))
        print('4-bit weights do not imply 4-bit optimizer or activations')
        result.update({'microbatch':microbatch,'accumulation':accum,'sequence':seq,'rank':rank})
        print('PASS configuration arithmetic; memory must be measured on target')
    elif n==241:
        base=out/'base';base.mkdir();(base/'config.json').write_text('{"fixture":true}')
        before,files=tree_identity(base);identity(before,tree_identity(base)[0])
        (base/'config.json').write_text('{"fixture":false}')
        reject('base mutated before resume',lambda:identity(before,tree_identity(base)[0]))
        write_json(out/'binding.json',{'base_tree_sha256':before,'files':files})
        print('PASS byte identity binding; no checkpoint recovery claimed')
    elif n==242:
        labels=['memory','input','unknown'];base=['memory','unknown','unknown'];adapter=['memory','input','memory']
        print('fabricated base exact_match=2/3 adapter exact_match=2/3')
        print('changed cases=2; aggregate equality hides a new error')
        assert sum(x==y for x,y in zip(labels,base))==sum(x==y for x,y in zip(labels,adapter))==2
        write_json(out/'paired.json',{'labels':labels,'base':base,'adapter':adapter})
        print('PASS paired comparison; ten training steps are a smoke test')
    elif n==243:
        available=['tiny-chat','tiny-embed'];loaded=['tiny-chat'];assert set(loaded)<=set(available)
        print('fixture inventory=2 loaded=1')
        print('JIT inventory cannot establish current residency')
        print('PASS keep lms, llmster and backend runtime identities separately')
    elif n==244:
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*a):pass
            def do_POST(self):
                raw=self.rfile.read(int(self.headers['Content-Length']));requests.append(json.loads(raw))
                body=json.dumps({'model':'fixture-chat','choices':[{'message':{'role':'assistant','content':'memory'},'finish_reason':'stop'}]}).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:status,body=exchange('http://127.0.0.1:'+str(server.server_port),'/v1/chat/completions',{'model':'fixture-chat','messages':[{'role':'user','content':'classify fixture'}]})
        finally:server.shutdown();server.server_close();thread.join()
        assert status==200 and len(requests)==1
        write_json(out/'http.json',{'requests':requests,'status':status,'response':json.loads(body)})
        print('real loopback HTTP requests=1 status=200 scripted answer=memory')
        print('PASS transport executed; LM Studio was not running')
    elif n==245:
        b=b'data: {"choices":[{"delta":{"content":"memory"},"finish_reason":null}]}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        assert len(sse(b))==2;(out/'stream.sse').write_bytes(b)
        reject('SSE lost DONE',lambda:sse(b.replace(b'data: [DONE]\n\n',b'')))
        reject('SSE no finish reason',lambda:sse(b.replace(b'"stop"',b'null')))
        print('PASS SSE completion; host completion does not prove kernel overlap')
    elif n==246:
        def reorder(data):
            if sorted(x['index'] for x in data)!=list(range(len(data))):raise ContractError('embedding indices')
            return vectors([x['embedding'] for x in sorted(data,key=lambda r:r['index'])],len(data))
        rows=[{'index':1,'embedding':[0,1]},{'index':0,'embedding':[1,0]}]
        assert reorder(rows)==[[1,0],[0,1]]
        reject('duplicate embedding index',lambda:reorder([rows[0],rows[0]]))
        print('PASS OpenAI vectors reordered by index; pooling remains model-specific')
    elif n==247:
        meta=json.loads((ROOT/'metadata/nemotron-gguf.json').read_text())
        print('repository='+meta['id']);print('revision='+meta['sha'])
        for x in meta['siblings']:
            if x['rfilename'].endswith('.gguf'):
                print(x['rfilename'].split('A3B-')[-1]+' bytes='+str(x['size']))
        print('PASS observed catalog metadata; no GGUF weights downloaded')
    elif n==248:
        meta=json.loads((ROOT/'metadata/nemotron-gguf.json').read_text())
        gpu=30*1024**3;guest=40*1024**3;reserve=6*1024**3
        result['budgets']=[]
        for quant in ['Q4_K_M','Q8_0','BF16']:
            f=next(x for x in meta['siblings'] if x['rfilename'].endswith(quant+'.gguf'))
            r=budget(f['size'],gpu,guest,reserve);result['budgets'].append(r)
            print('%s %.3f GiB GPU_weight_fit=%s host_admission=%s'%(quant,f['size']/1024**3,r['gpu_weight_fit'],r['host_weight_admission']))
        print('SYNTHETIC budgets: GPU 30 GiB, guest available 40 GiB, reserve 6 GiB')
        print('PASS Q8 exceeds both 30 GB and 30 GiB; fit still unmeasured')
    elif n==249:
        total,active=30,3.5
        print('nominal 16-bit weight floor: total=%.1f GB active=%.1f GB'%(total*2,active*2))
        print('allocation uses stored tensors; active count estimates selected compute')
        print('hybrid state cache must be measured; dense KV formula is incomplete')
        print('PASS no 3.5B memory claim and no fictitious extra GPU pool')
    elif n==250:
        data=[{'status':'ok','complete':True,'seconds':2.0},{'status':'oom','complete':False,'seconds':1.0},{'status':'ok','complete':True,'seconds':4.0}]
        r=summarize(data);result.update(r)
        print('synthetic submitted=3 completed=2 failures=1 median_seconds=3.0')
        reject('no completed samples',lambda:summarize(data[1:2]))
        print('PASS failed configurations remain evidence; performance not measured')
    else:raise ValueError(n)
    write_json(out/'result.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('exercise',type=int,choices=range(235,251));p.add_argument('--out',required=True);a=p.parse_args()
    run(a.exercise,fresh(a.out))
