"""Plot host observations; never label these intervals as GPU kernels."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from core import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    src=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    result=json.loads((src/'result.json').read_text())
    batches=json.loads((src/'batches.json').read_text())
    clients=[json.loads(f.read_text()) for f in sorted(src.glob('client-*.json'))]
    base=min(r['sent_ns'] for rows in clients for r in rows)
    events=[];fig,ax=plt.subplots(figsize=(7.5,3.4))
    for i,rows in enumerate(clients):
        for r in rows:
            x=(r['sent_ns']-base)/1e6;w=(r['received_ns']-r['sent_ns'])/1e6
            ax.broken_barh([(x,w)],(i-.3,.6),facecolors='#b46b2b')
            events.append(dict(name='client_response_wait',ph='X',pid=r['client'],tid=0,
                               ts=(r['sent_ns']-base)/1000,dur=w*1000,
                               args={'id':r['id']}))
    n=len(clients)
    for b in batches:
        x=(b['start_ns']-base)/1e6;w=(b['end_ns']-b['start_ns'])/1e6
        ax.broken_barh([(x,w)],(n+b['owner']-.3,.6),facecolors='#285e88')
        events.append(dict(name='owner_host_service',ph='X',pid=n+b['owner'],tid=0,
                           ts=x*1000,dur=w*1000,args={'request_ids':b['ids']}))
    ax.set_yticks(range(n+result['owners']),
                  [f'Client {i}' for i in range(n)]+[f'Owner {i}' for i in range(result['owners'])])
    ax.set_xlabel('Host elapsed time (ms)');ax.set_title(f"Service with {n} clients and {result['owners']} owners ({result['device']})",fontsize=12)
    ax.grid(axis='x',alpha=.18)
    ax.legend(handles=[Patch(color='#b46b2b',label='Client response wait'),
                       Patch(color='#285e88',label='Owner host service')],
              loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False,fontsize=9)
    fig.tight_layout();fig.savefig(out/'timeline.png',dpi=190,bbox_inches='tight');plt.close(fig)
    write(out/'timeline.json',{'traceEvents':events,'displayTimeUnit':'ms',
                              'metadata':{'clock':'one host perf_counter_ns','device':result['device']}})
    print(f'PASS host timeline spans={len(events)} device={result["device"]}')
    print('outputs=timeline.png,timeline.json; these are not GPU kernel intervals')


if __name__=='__main__':main()
