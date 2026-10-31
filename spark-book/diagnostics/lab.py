"""Bounded diagnostic controls. Run from the spark-book directory."""
import argparse,json,time,math,statistics,threading,subprocess,sys,gc,weakref,faulthandler
from pathlib import Path

def save(out,name,obj): (out/name).write_text(json.dumps(obj,indent=2)+'\n')
def numerical(dtype,fault=False,device='cpu'):
 import torch
 ids=torch.tensor([[1,2,0],[3,1,0]],device=device)
 table=torch.arange(20,device=device,dtype=torch.float64).reshape(5,4)/13
 hidden=table.to(dtype)[ids];mask=ids.ne(0).unsqueeze(-1)
 pooled=hidden.mean(1) if fault else (hidden*mask).sum(1)/mask.sum(1)
 w=torch.tensor([[.1,.4],[.3,-.2],[.5,.6],[-.7,.8]],dtype=dtype,device=device)
 return {'ids':ids,'hidden':hidden,'pooled':pooled,'logits':pooled@w}
def compare(ref,got,atol=.01,rtol=.01):
 import torch
 rows=[]
 for key in ref:
  a,b=ref[key].double(),got[key].double()
  if a.shape!=b.shape:raise ValueError('shape mismatch')
  finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
  ok=finite and bool(torch.allclose(b,a,atol=atol,rtol=rtol))
  if key=='ids':ok=bool(torch.equal(a,b))
  rows.append((key,float((a-b).abs().max()) if finite else None,ok))
 return rows
def numerics(a,out):
 import torch
 ref=numerical(torch.float64,device=a.device);records=[]
 for dtype in [torch.float32,torch.bfloat16]:
  got=numerical(dtype,a.fault,a.device);rows=compare(ref,got)
  print(str(dtype)+' atol=0.01 rtol=0.01')
  for name,error,ok in rows:print(f'{name} max_abs={error:.6f} pass={ok}')
  print('first_failure='+next((n for n,e,ok in rows if not ok),'none'))
  print('argmax_agrees='+str(bool(torch.equal(ref['logits'].argmax(-1),got['logits'].argmax(-1)))))
  records.append({'dtype':str(dtype),'stages':rows})
 save(out,'result.json',records)
def pointwise(x):
 import torch
 return torch.relu(x*1.25+.5)
def broken(x):
 import torch
 torch._dynamo.graph_break()
 return x+x.sum().item()
def compile_lab(a,out):
 import torch
 torch.set_num_threads(1);graphs=[]
 def backend(gm,inputs):
  graphs.append(gm.code);(out/f'graph-{len(graphs)}.py').write_text(gm.code);return gm.forward
 if a.mode=='break':
  try:torch.compile(broken,backend=backend,fullgraph=True)(torch.ones(8,device=a.device))
  except Exception as e:
   (out/'expected-error.txt').write_text(str(e));print('fullgraph_rejected='+type(e).__name__)
  else:raise AssertionError('Expected explicit graph break rejection')
  return
 fn=torch.compile(pointwise,backend='inductor' if a.mode=='inductor' else backend,dynamic=a.mode=='dynamic',fullgraph=True)
 sizes=[4096]*6 if a.mode=='inductor' else [8,12,8,16,12];times=[]
 for n in sizes:
  x=torch.linspace(-1,1,n,device=a.device)
  if a.device=='cuda':torch.cuda.synchronize()
  start=time.perf_counter();y=fn(x)
  if a.device=='cuda':torch.cuda.synchronize()
  times.append((time.perf_counter()-start)*1000);torch.testing.assert_close(y,pointwise(x))
 print('input_sizes='+str(sizes));print('outputs_close=True')
 if a.mode=='inductor':print('inductor first_ms=%.3f later_median_ms=%.3f'%(times[0],statistics.median(times[1:])))
 else:print('captured_graphs='+str(len(graphs))+' backend=FX_eager_execution')
 save(out,'result.json',{'sizes':sizes,'ms':times,'graphs':len(graphs) if a.mode!='inductor' else None})
def pipeline(a,out):
 # Threaded bounded prefetch control. DataLoader process workers are a target extension.
 import torch,queue
 torch.set_num_threads(1);q=queue.Queue(maxsize=2);cancel=threading.Event();rows=[]
 delay=.005 if a.mode!='ready' else 0
 def produce():
  for i in range(12):
   if cancel.is_set():return
   time.sleep(delay*4);batch=torch.full((4,128),i/12)
   while not cancel.is_set():
    try:q.put(batch,timeout=.1);break
    except queue.Full:pass
 def batches():
  for i in range(12):
   if a.mode=='prefetch':yield q.get(timeout=5)
   else:time.sleep(delay*4);yield torch.full((4,128),i/12)
 w=torch.eye(128,device=a.device);worker=None
 if a.mode=='prefetch':worker=threading.Thread(target=produce);worker.start()
 t0=time.perf_counter();wait=work=0.;it=iter(batches())
 try:
  for i in range(12):
   t=time.perf_counter();batch=next(it);wait+=time.perf_counter()-t
   t=time.perf_counter();y=batch.to(a.device)@w
   if a.device=='cuda':torch.cuda.synchronize()
   # Fixed host-side consumer work permits visible pipeline overlap.
   time.sleep(.01);work+=time.perf_counter()-t
   assert bool(torch.isfinite(y).all())
 finally:
  cancel.set()
  if worker:worker.join(timeout=2)
 total=time.perf_counter()-t0
 print('mode='+a.mode+' batches=12 records=48 queue_capacity=2')
 print('wait_ms=%.3f work_ms=%.3f total_ms=%.3f'%(wait*1000,work*1000,total*1000))
 print('consumer includes 10ms controlled host work per batch')
 save(out,'result.json',{'wait_s':wait,'work_s':work,'total_s':total,'mode':a.mode,'device':a.device})
def roof_models(n=1024):return [('vector_add',n,12*n),('fma_split',2*n,24*n),('fma_fused',2*n,16*n),('matmul_ideal',2*n**3,12*n*n)]
def roofline(a,out):
 import numpy as np
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 rows=[];print('ILLUSTRATIVE ceilings 100 GB/s and 10 TFLOP/s')
 for n,f,b in roof_models():
  ai=f/b;y=min(10,ai*.1);print('%s AI=%.4f bound=%.4f TFLOP/s'%(n,ai,y));rows.append({'name':n,'ai':ai,'bound':y})
 fig,ax=plt.subplots(figsize=(6.3,3.8));x=np.logspace(-2,3,400);ax.loglog(x,np.minimum(10,x*.1),color='black')
 for i,r in enumerate(rows):ax.scatter(r['ai'],r['bound'],marker=['o','x','s','D'][i],label=r['name'].replace('_',' '))
 ax.legend(loc='upper left',fontsize=8);ax.set(xlabel='Arithmetic intensity (FLOP / byte)',ylabel='Upper bound (TFLOP/s)',title='Hypothetical roofline, not laptop measurements');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'roofline.png',dpi=180);plt.close(fig);save(out,'result.json',rows)
def memory(a,out):
 import torch
 refs=[];held=[];rows=[]
 if a.snapshot:torch.cuda.memory._record_memory_history(max_entries=10000)
 def report(step):
  gc.collect();row={'step':step,'lab_live_bytes':sum(r().numel()*r().element_size() for r in refs if r() is not None)}
  if a.device=='cuda':torch.cuda.synchronize();row.update(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
  rows.append(row);print(' '.join(str(k)+'='+str(v) for k,v in row.items()))
 try:
  for i in range(8):
   n=(i%3+1)*65536 if a.mode=='vary' else 65536;x=torch.ones(n,device=a.device);refs.append(weakref.ref(x))
   if a.mode=='retain':held.append(x)
   del x
   if i in [0,3,7]:report(str(i+1))
  if a.snapshot:torch.cuda.memory._dump_snapshot(str(out/'before-clear.pickle'))
  held.clear();report('clear')
  if a.device=='cuda':torch.cuda.empty_cache();report('empty_cache')
  if a.snapshot:torch.cuda.memory._dump_snapshot(str(out/'after-clear.pickle'))
 finally:
  if a.snapshot:torch.cuda.memory._record_memory_history(enabled=None)
 save(out,'result.json',rows)
def hang(a,out):
 if a.child:
  print('stage=loaded',flush=True);event=threading.Event()
  if a.mode=='repaired':event.set()
  faulthandler.dump_traceback_later(.3,repeat=False);print('stage=wait_started',flush=True);event.wait();faulthandler.cancel_dump_traceback_later();print('stage=completed',flush=True);return
 child=subprocess.Popen([sys.executable,__file__,'hang','--mode',a.mode,'--child'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);expired=False
 try:stdout,stderr=child.communicate(timeout=1.5)
 except subprocess.TimeoutExpired:
  expired=True;child.terminate()
  try:stdout,stderr=child.communicate(timeout=2)
  except subprocess.TimeoutExpired:child.kill();stdout,stderr=child.communicate()
 (out/'stdout.txt').write_text(stdout);(out/'stacks.txt').write_text(stderr);print(stdout,end='');print('timed_out='+str(expired));print('stack_has_wait='+str('in wait' in stderr));print('child_returncode='+str(child.returncode));assert expired==(a.mode=='blocked')
def request(a,out):
 from opentelemetry import trace
 from opentelemetry.sdk.trace import TracerProvider;from opentelemetry.sdk.trace.sampling import ALWAYS_ON
 from opentelemetry.sdk.trace.export import SpanExporter,SpanExportResult,SimpleSpanProcessor
 from opentelemetry.trace import Status,StatusCode
 rows=[];logs=[]
 class Exporter(SpanExporter):
  def export(self,spans):
   for s in spans:rows.append({'name':s.name,'trace_id':format(s.context.trace_id,'032x'),'span_id':format(s.context.span_id,'016x'),'parent_id':format(s.parent.span_id,'016x') if s.parent else None,'start_ns':s.start_time,'end_ns':s.end_time,'status':s.status.status_code.name,'attributes':dict(s.attributes),'events':[e.name for e in s.events]})
   return SpanExportResult.SUCCESS
 provider=TracerProvider(sampler=ALWAYS_ON);provider.add_span_processor(SimpleSpanProcessor(Exporter()));tracer=provider.get_tracer('spark-book')
 def log(event):
  c=trace.get_current_span().get_span_context();logs.append({'event':event,'trace_id':format(c.trace_id,'032x'),'span_id':format(c.span_id,'016x')})
 with tracer.start_as_current_span('request') as root:
  root.set_attribute('lab.fixture',True);root.set_attribute('lab.expected_spans',7 if a.mode=='retry' else 6);log('start')
  with tracer.start_as_current_span('embedding'):time.sleep(.005)
  with tracer.start_as_current_span('retrieval'):time.sleep(.08 if a.mode=='slow' else .005)
  with tracer.start_as_current_span('generation'):time.sleep(.01)
  for attempt in range(1,3):
   with tracer.start_as_current_span('tool') as s:
    s.set_attribute('lab.attempt',attempt)
    if a.mode=='retry' and attempt==1:s.set_status(Status(StatusCode.ERROR,'injected timeout'));s.add_event('controlled_timeout');log('tool_failed');continue
    time.sleep(.005);log('tool_ok');break
  with tracer.start_as_current_span('validation') as s:s.set_attribute('lab.accepted',True)
  log('completed')
 provider.shutdown()
 for name,data in [('spans',rows),('logs',logs)]:(out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in data))
 children=[r for r in rows if r['parent_id']];slow=max(children,key=lambda r:r['end_ns']-r['start_ns']);print('fixture=True mode='+a.mode+' spans='+str(len(rows)));print('error_spans='+str(sum(r['status']=='ERROR' for r in rows)));print('correlated_logs='+str(len(logs))+' longest_child='+slow['name'])
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 ordered=sorted(rows,key=lambda r:r['start_ns']);t0=ordered[0]['start_ns'];fig,ax=plt.subplots(figsize=(6.3,3.6))
 for i,r in enumerate(ordered):ax.barh(i,(r['end_ns']-r['start_ns'])/1e6,left=(r['start_ns']-t0)/1e6,color='#a33d3d' if r['status']=='ERROR' else '#34678c')
 ax.set_yticks(range(len(rows)),[r['name'] for r in ordered]);ax.invert_yaxis();ax.set(xlabel='Elapsed host time (ms)',title='Executed synthetic request spans');fig.tight_layout();fig.savefig(out/'spans.png',dpi=180);plt.close(fig)
STAGES=['retrieval','assembly','truncation','generation','validation']
def quality_run(fault=None,repair=None):
 def bad(s):return fault==s and repair!=s
 docs=['unrelated'] if bad('retrieval') else ['answer=42'];context='' if bad('assembly') else '|'.join(docs);context=context[:4] if bad('truncation') else context;answer='7' if bad('generation') else ('42' if 'answer=42' in context else 'unknown');accepted=False if bad('validation') else answer=='42'
 return {'docs':docs,'context':context,'answer':answer,'accepted':accepted,'pass':answer=='42' and accepted}
def quality(a,out):
 print('DETERMINISTIC FIXTURE no learned model');rows=[]
 for fault in STAGES:
  base=quality_run(fault);repairs=[s for s in STAGES if quality_run(fault,s)['pass']];print('fault=%s pass=%s repairs=%s'%(fault,base['pass'],','.join(repairs)));rows.append({'fault':fault,'baseline':base,'repairs':repairs})
 save(out,'result.json',rows)
def fails(records,version=3):return version in [3,5] and 'PAD' in records and 'TRIGGER' in records
def reduce_records(records,predicate):
 if not predicate(records):raise ValueError('initial input must fail')
 result=list(records);n=2
 while len(result)>=2:
  size=math.ceil(len(result)/n);found=False
  for start in range(0,len(result),size):
   candidate=result[:start]+result[start+size:]
   if predicate(candidate):result=candidate;n=max(2,n-1);found=True;break
  if not found:
   if n>=len(result):break
   n=min(len(result),n*2)
 return result
def reduce_lab(a,out):
 calls=[]
 def oracle(x):v=fails(x);calls.append({'input':x,'fails':v});return v
 result=reduce_records(['a','PAD','b','c','TRIGGER','d','e','f'],oracle);minimal=all(not fails(result[:i]+result[i+1:]) for i in range(len(result)));scan=[fails(result,v) for v in range(1,6)]
 print('initial=8 reduced='+str(len(result)));print('reproducer='+repr(result));print('one_minimal='+str(minimal));print('versions_1_to_5='+str(scan));print('first_observed_failure=3');save(out,'result.json',{'records':result,'one_minimal':minimal,'scan':scan,'calls':calls})
def sample_audit(rows):
 issues=[];seen=set();last=-math.inf
 for r in rows:
  if r['id'] in seen:issues.append('duplicate_sample')
  seen.add(r['id'])
  if r['time']<last:issues.append('backward_time')
  last=r['time']
  if r['age']>2:issues.append('stale_reading')
  if r['unit']!='MiB':issues.append('unit_mismatch')
 return sorted(set(issues))
def trace_audit(rows,logs):
 issues=[];ids={r['span_id'] for r in rows};roots=[r for r in rows if r['parent_id'] is None]
 if len(ids)!=len(rows):issues.append('duplicate_span')
 if len(roots)!=1:issues.append('root_count')
 if any(r['parent_id'] and r['parent_id'] not in ids for r in rows):issues.append('missing_parent')
 if roots:
  root=roots[0]
  if root['attributes'].get('lab.expected_spans')!=len(rows):issues.append('missing_or_extra_span')
  if any(r['trace_id']!=root['trace_id'] for r in rows):issues.append('trace_mismatch')
 pairs={(r['trace_id'],r['span_id']) for r in rows}
 if any((r['trace_id'],r['span_id']) not in pairs for r in logs):issues.append('uncorrelated_log')
 return sorted(set(issues))
def weighted_mean(groups):
 total=sum(n for n,m in groups)
 if total<=0:raise ValueError('no observations')
 return sum(n*m for n,m in groups)/total
def audit(a,out):
 import copy
 source=Path(a.trace);rows=[json.loads(s) for s in (source/'spans.jsonl').read_text().splitlines()];logs=[json.loads(s) for s in (source/'logs.jsonl').read_text().splitlines()];print('healthy_trace='+str(trace_audit(rows,logs)))
 damaged=[copy.deepcopy(r) for r in rows if r['name']!='retrieval'];badlogs=copy.deepcopy(logs);badlogs[0]['trace_id']='0'*32;print('damaged_trace='+str(trace_audit(damaged,badlogs)))
 good=[{'id':i,'time':i,'age':0,'unit':'MiB'} for i in range(4)];bad=copy.deepcopy(good);bad.append(copy.deepcopy(bad[-1]));bad[1]['age']=4;bad[2]['unit']='bytes';bad[-1]['time']=1
 print('healthy_samples='+str(sample_audit(good)));print('damaged_samples='+str(sample_audit(bad)));print('mean_of_means_ms=55.0 weighted_mean_ms=%.1f'%weighted_mean([(1,100),(9,10)]));save(out,'damaged-spans.json',damaged);save(out,'damaged-logs.json',badlogs);save(out,'damaged-samples.json',bad)
def main():
 p=argparse.ArgumentParser();p.add_argument('lab',choices=['numerics','compile','pipeline','roofline','memory','hang','request','quality','reduce','audit']);p.add_argument('--mode',default='normal');p.add_argument('--out');p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--fault',action='store_true');p.add_argument('--snapshot',action='store_true');p.add_argument('--child',action='store_true');p.add_argument('--trace');a=p.parse_args()
 modes={'compile':['static','dynamic','break','inductor'],'pipeline':['ready','delay','prefetch'],'memory':['retain','release','vary'],'hang':['blocked','repaired'],'request':['normal','slow','retry']}
 if a.lab in modes and a.mode not in modes[a.lab]:p.error('invalid mode for '+a.lab)
 if a.child and a.lab!='hang':p.error('--child only for hang')
 if a.snapshot and (a.lab!='memory' or a.device!='cuda'):p.error('--snapshot requires memory CUDA')
 if a.lab=='audit' and not a.trace:p.error('--trace required')
 if a.child:hang(a,None);return
 if not a.out:p.error('--out required and must be new')
 if a.lab in ['numerics','compile','pipeline','memory']:
  import torch
  if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; no CPU substitution')
  print('torch='+torch.__version__+' device='+a.device)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 {'numerics':numerics,'compile':compile_lab,'pipeline':pipeline,'roofline':roofline,'memory':memory,'hang':hang,'request':request,'quality':quality,'reduce':reduce_lab,'audit':audit}[a.lab](a,out)
if __name__=='__main__':main()
