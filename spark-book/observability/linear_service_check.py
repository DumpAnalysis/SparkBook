"""Run the recorded 32-request numerical check before and during WSL GPU monitoring."""
import argparse
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import re
import statistics
import time

from trial_stack import require_free_ports, running_stack, series, wait_for

def trial(request, base, dest):
    dest.mkdir(exist_ok=False)
    endpoint='/v2/models/lab_linear/infer'
    def body(shape,dtype,values):
        return {'inputs':[{'name':'X','shape':shape,'datatype':dtype,'data':values}],'outputs':[{'name':'Y'}]}
    data=body([1,8],'FP32',[0.0]*8)
    started=datetime.now(timezone.utc).isoformat()
    warmup=request(base,endpoint,data,'warmup')
    if warmup['status']!=200:
        (dest/'responses.json').write_text(json.dumps({'warmup':warmup},indent=2)+'\n')
        raise RuntimeError('Warm-up failed. Inspect responses.json.')
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(lambda i:request(base,endpoint,data,f'measured-{i}'),range(32)))
    elapsed=time.perf_counter()-start
    faults=[('wrong-width',[1,7],'FP32',7),('wrong-dtype',[1,8],'FP64',8),('above-max-batch',[9,8],'FP32',72)]
    negatives=[request(base,endpoint,body(shape,dtype,[0.0]*n),name) for name,shape,dtype,n in faults]
    (dest/'responses.json').write_text(json.dumps({'warmup':warmup,'measured':rows,'negative':negatives},indent=2)+'\n')
    errors=[]
    for row in rows:
        if row['status']!=200:raise ValueError('Unexpected inference status. Inspect responses.json.')
        tensor=json.loads(row['body'])['outputs'][0]
        if tensor['shape']!=[1,4] or len(tensor['data'])!=4:raise ValueError('Unexpected shape. Inspect responses.json.')
        differences=[abs(a-b) for a,b in zip(tensor['data'],[.1,0,.3,.01])]
        if not all(x<1e-5 for x in differences):raise ValueError('Numerical check failed. Inspect responses.json.')
        errors.append(max(differences))
    rejected=[r['status'] for r in negatives]
    if rejected!=[400]*3:raise ValueError('Expected three HTTP 400 rejections. Inspect responses.json.')
    latency=sorted(r['elapsed_ms'] for r in rows)
    result={'outer_started_utc':started,'outer_ended_utc':datetime.now(timezone.utc).isoformat(),'requests':32,'concurrency':4,'successful':32,'elapsed_s':elapsed,'requests_per_second':32/elapsed,
            'latency_p50_ms':statistics.median(latency),'latency_p95_ms':latency[30],'max_abs_error':max(errors),
            'negative_rejections':rejected,'warmup_excluded':1}
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dest.name,json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools',required=True,type=Path)
    parser.add_argument('--base',default='http://127.0.0.1:8013')
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if not re.fullmatch(r'http://127\.0\.0\.1:[0-9]+',args.base):parser.error('use an IPv4 loopback HTTP endpoint')
    require_free_ports()
    spec=importlib.util.spec_from_file_location('serving_client',Path(__file__).resolve().parents[1]/'serving/client.py')
    client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
    dest=args.out.resolve();dest.mkdir(parents=True,exist_ok=False)
    for i in range(3):trial(client.request,args.base,dest/f'unmonitored-{i}')
    with running_stack(args.tools,dest,['--live','--gpu','0']) as owner:
        def live_ready():
            return any(float(r['value'][1])==1 for r in series('up{job="gpu"}')) and series('lab_gpu_field_available{source="wsl"}')
        wait_for(owner,live_ready,'live WSL exporter and scrape')
        for i in range(3):trial(client.request,args.base,dest/f'monitored-{i}')
        (dest/'live-query.json').write_text(json.dumps(series('lab_gpu_field_available{source="wsl"}'),indent=2)+'\n')


if __name__=='__main__':
    main()
