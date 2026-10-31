"""Check live HTTP evidence and deployment invariants, not cluster behavior."""
import argparse,json
from pathlib import Path
import yaml
from deploy import documents
from client import parse_sse

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',default='serving/evidence');a=p.parse_args();e=Path(a.evidence);passed=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        passed.append(name);print('PASS '+name)
    read=lambda f:json.loads((e/f).read_text())
    c=read('01-contract/result.json');rows=read('01-contract/responses.json')
    check('live HTTP contract and errors',c['statuses']==[200,200,404,400,413])
    check('vectors agree with prior independent oracle',c['max_abs_error']<1e-6)
    check('request ID echoed',rows[1]['headers']['X-Request-ID']=='case-1')
    l=read('02-lifecycle/result.json')
    check('loading is live but not ready',l['liveness']['status']==200 and l['loading']['status']==503 and l['ready']['status']==200)
    check('load failure remains unready',l['failed_readiness']['status']==503 and l['failed_liveness']['status']==200)
    check('admitted request drains before exit',l['drained']['status']==200 and l['exitcode']==0)
    load=read('03-load/result.json');rr=read('03-load/responses.json')
    check('bounded load preserves all outcomes',load['status_counts']=={'200':2,'429':6} and len(rr)==8)
    check('overload retry signal',all(r['headers'].get('Retry-After')=='1' for r in rr if r['status']==429))
    metrics=(e/'03-load/metrics.txt').read_text()
    check('metrics match observed response counts','route="/v1/embeddings",status="200"} 2' in metrics and 'status="429"} 6' in metrics)
    records=[json.loads(s) for s in (e/'03-load/server/server.jsonl').read_text().splitlines()]
    phases=[r for r in records if r['event']=='inference']
    check('single-owner intervals do not overlap',len(phases)==2 and phases[0]['end_ns']<=phases[1]['start_ns'] and all(r['accepted_ns']<=r['start_ns']<=r['end_ns'] for r in phases))
    s=read('04-sse/result.json')
    check('SSE role and heartbeat excluded',s['text']=='GPU memory' and s['events'][-1]=='[DONE]')
    check('truncated SSE does not invent done',list(parse_sse([b'data: {"x":1}\n']))==['{"x":1}'])
    b=read('07-backend/result.json');o=read('09-onnx/result.json')
    check('backend splits unequal request batches',b['request_batch_sizes']==[1,2] and b['invalid_shape']=='rejected' and b['max_abs_error']<1e-6)
    check('serialized ONNX agrees on CPU',o['max_abs_error']<1e-6)
    docs=list(yaml.safe_load_all((e/'fixture.yaml').read_text()));spec=docs[1]['spec']['template']['spec']
    check('CPU deployment has no GPU claim','nvidia.com/gpu' not in spec['containers'][0]['resources']['limits'] and 'runtimeClassName' not in spec)
    check('CPU service is internal ClusterIP',docs[2]['spec']['type']=='ClusterIP')
    fake='example.invalid/lab@sha256:'+'a'*64
    gpu=documents('vllm',fake,'/srv/spark-models');g=gpu[1]['spec'];pod=g['template']['spec']
    check('GPU rollout uses one whole device and Recreate',g['replicas']==1 and g['strategy']['type']=='Recreate' and pod['containers'][0]['resources']['limits']['nvidia.com/gpu']==1 and pod['runtimeClassName']=='nvidia')
    check('model volume read only and no service token',pod['containers'][0]['volumeMounts'][0]['readOnly'] and not pod['automountServiceAccountToken'])
    rejected=False
    try:documents('vllm','image:mutable','/models')
    except ValueError:rejected=True
    check('mutable external image refused',rejected)
    c=pod['containers'][0]
    check('startup gate and deliberate liveness policy','startupProbe' in c and 'readinessProbe' in c and 'livenessProbe' not in c)
    print(f'{len(passed)} checks passed; no Docker, Kubernetes, vLLM or Triton deployment tested')
if __name__=='__main__':main()
