"""Check supplied or newly captured evidence against independent controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from core import tokens,oracle


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',default=str(Path(__file__).parent/'evidence'))
    args=p.parse_args();e=Path(args.evidence);checks=[]
    def read(name):return json.loads((e/name).read_text())
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name);print('PASS '+name)
    runs=[read(f'{n}/result.json') for n in ['01-shared1','02-shared2','03-shared4','04-batched','05-replicas']]
    check('one model owner with one two and four clients',
          [r['owners'] for r in runs[:3]]==[1,1,1] and [r['workers'] for r in runs[:3]]==[1,2,4])
    check('fixed sixteen request workload in every service run',all(r['requests']==16 and r['status']=='PASS' for r in runs))
    observed=[]
    for folder in ['01-shared1','02-shared2','03-shared4','04-batched','05-replicas']:
        rows=[]
        for f in (e/folder).glob('client-*.json'):rows+=json.loads(f.read_text())
        rows.sort(key=lambda x:x['id']);observed.append(rows)
    check('identities cover each request exactly once',all([r['id'] for r in rows]==list(range(16)) for rows in observed))
    expected=oracle([tokens(i) for i in range(16)])
    check('vectors match independent numpy oracle',all(np.max(np.abs(np.array([r['value'] for r in rows])-expected))<1e-6 for rows in observed))
    wrong=expected.copy();wrong[[0,1]]=wrong[[1,0]]
    check('swapped response counterexample is rejected',float(np.max(np.abs(wrong-expected)))>1e-3)
    check('pending admission is bounded by client count',all(1<=r['max_pending']<=r['workers'] for r in runs))
    check('four model replicas account for four weight copies',runs[4]['owners']==4 and runs[4]['model_bytes']==4*runs[3]['model_bytes'])
    batches=read('04-batched/batches.json')
    check('batching combines distinct clients with complete coverage',
          any(len(b['ids'])>1 for b in batches) and
          all(len(set(i%4 for i in b['ids']))==len(b['ids']) for b in batches) and
          sorted(i for b in batches for i in b['ids'])==list(range(16)))
    reqs=read('04-batched/requests.json')
    check('host time ordering and durations are valid',
          all(r['sent_ns']<=r['accepted_ns']<=r['start_ns']<=r['end_ns'] for r in reqs) and
          all(r['received_ns']>=r['sent_ns'] for rows in observed for r in rows))
    tr=read('06-training/result.json')
    import torch
    ck=torch.load(e/'06-training/training-checkpoint.pt',map_location='cpu',weights_only=True)
    check('single optimizer agrees with serial training and checkpoint counters',
          tr['training_model_owners']==1 and len({r['pid'] for r in tr['preparers']})==2 and
          tr['max_abs_error']<=1e-5 and tr['loss_max_abs_error']<=1e-6 and ck['next_step']==4 and
          all(int(s['step'].item())==4 for s in ck['optimizer']['state'].values()))
    fault=read('07-faults/result.json')
    check('overload cancellation expiry and owner loss retain distinct outcomes',
          fault['queue_events']==[[1,'ACCEPTED'],[2,'ACCEPTED'],[3,'BUSY'],[1,'CANCELLED'],[3,'ACCEPTED'],[2,'EXPIRED'],[3,'DISPATCH'],[3,'NOT_PENDING']] and
          fault['failed']['exitcode']==17 and fault['failed']['response']['outcome']=='UNKNOWN' and
          fault['recovered']['exitcode']==0 and fault['recovered']['response']['id']==7)
    trace=read('08-plot/timeline.json')['traceEvents']
    check('trace preserves all client waits and owner batches',len(trace)==16+len(batches) and all(t['dur']>=0 for t in trace))
    print(f'{len(checks)} checks passed; GPU execution is not implied')


if __name__=='__main__':main()
