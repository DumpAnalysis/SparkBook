"""Compare host-phase durations, refusing unmatched workload definitions."""
import argparse,json,statistics
from pathlib import Path

def read(paths):
    runs=[]
    for name in paths:
        p=Path(name);m=json.loads((p/'metadata.json').read_text());s=json.loads((p/'session.json').read_text());e=json.loads((p/'events.json').read_text())
        if not s['success'] or m.get('fixture'):raise ValueError('requires successful actual sessions')
        runs.append((m,e))
    return runs

def compare(a,b):
    runs=a+b;plan=runs[0][0]['plan']
    if any(m['plan']!=plan for m,e in runs):raise ValueError('plans differ; use an identical plan for the overhead control')
    if any(not m['without_sampling'] for m,e in a) or any(m['without_sampling'] for m,e in b):raise ValueError('baseline must disable sampling; observed group must enable it')
    for i,p in enumerate(plan):
        base=statistics.median(e[i]['end_s']-e[i]['start_s'] for m,e in a)
        observed=statistics.median(e[i]['end_s']-e[i]['start_s'] for m,e in b)
        print(f"{p['label']}: baseline_median_s={base:.6f} observed_median_s={observed:.6f} change_percent={100*(observed/base-1):.3f}")
    print('Host-command duration comparison; not isolated collector overhead or a GPU benchmark.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',nargs='+',required=True);p.add_argument('--observed',nargs='+',required=True);a=p.parse_args();compare(read(a.baseline),read(a.observed))
