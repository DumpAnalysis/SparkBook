"""Render recorded telemetry without filling missing values or gaps."""
import argparse, csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from monitor import FIELDS, dump

def finite(x): return isinstance(x, (int,float)) and math.isfinite(x)

def load(root):
    meta=json.loads((root/'metadata.json').read_text())
    rows=[json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines() if x.strip()]
    events=json.loads((root/'events.json').read_text())
    t=[r['t_s'] for r in rows]
    if not t or any(not finite(x) for x in t) or any(b<=a for a,b in zip(t,t[1:])):
        raise ValueError('samples require finite strictly increasing timestamps')
    for e in events:
        if not finite(e['start_s']) or not finite(e['end_s']) or e['end_s']<e['start_s']:
            raise ValueError('invalid phase interval')
    return meta, rows, events

def series(rows, field):
    return [r['gpu'][field]['value'] if r['gpu'][field]['status']=='ok' and finite(r['gpu'][field]['value']) else float('nan') for r in rows]

def break_gaps(t, y, limit):
    tx=[]; yy=[]
    for i,(x,v) in enumerate(zip(t,y)):
        if i and x-t[i-1]>limit: tx.append((x+t[i-1])/2); yy.append(float('nan'))
        tx.append(x); yy.append(v)
    return tx,yy

def summarize(meta, rows, events):
    result={'evidence':meta['evidence'], 'sample_count':len(rows),
            'observed_interval_s':{'min':None,'median':None,'max':None},'fields':{},'phases':[]}
    dt=np.diff([r['t_s'] for r in rows])
    if len(dt):result['observed_interval_s']={k:float(v) for k,v in zip(['min','median','max'],[min(dt),np.median(dt),max(dt)])}
    for f in FIELDS:
        y=series(rows,f); ok=[v for v in y if math.isfinite(v)]
        statuses={s:sum(r['gpu'][f]['status']==s for r in rows) for s in sorted({r['gpu'][f]['status'] for r in rows})}
        result['fields'][f]={'unit':FIELDS[f][0],'valid':len(ok),'total':len(rows),'statuses':statuses,
                             'sample_min':min(ok) if ok else None,'sample_median':float(np.median(ok)) if ok else None,
                             'sample_max':max(ok) if ok else None}
    for e in events:
        # Only wholly enclosed query spans, not midpoint-only matches.
        subset=[r for r in rows if r['start_s']>=e['start_s'] and r['end_s']<=e['end_s']]
        result['phases'].append({'label':e['label'],'host_seconds':e['end_s']-e['start_s'],
                                'returncode':e['returncode'],'contained_query_spans':len(subset),
                                'sample_scope':'poll spans; hardware averaging windows may cross phase boundaries'})
    return result

def render(root, out):
    meta,rows,events=load(root)
    out.mkdir(parents=True,exist_ok=False)
    report=summarize(meta,rows,events);dump(out/'summary.json',report)
    with (out/'samples.csv').open('w',newline='') as fh:
        cols=['seq','utc','start_s','end_s','t_s','query_ms']
        cols += [s for f in FIELDS for s in [f,f+':status']]
        cols += ['guest:'+k+':MiB' for k in ['MemTotal','MemAvailable','SwapTotal','SwapFree']]
        w=csv.DictWriter(fh,fieldnames=cols);w.writeheader()
        for r in rows:
            row={k:r[k] for k in cols[:6]}
            for f in FIELDS: row[f]=r['gpu'][f]['value'];row[f+':status']=r['gpu'][f]['status']
            for k,v in r['guest_MiB'].items():row['guest:'+k+':MiB']=v
            w.writerow(row)
    t=[r['t_s'] for r in rows]; limit=2.5*meta['interval_s']
    colors=['#375f8a','#8b5d24','#6b5b86','#35715e','#7c4e62','#65743b']
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'font.family':'DejaVu Sans'})
    def canvas(title,n):
        fig,axes=plt.subplots(n+1,1,sharex=True,figsize=(7.2,7),gridspec_kw={'height_ratios':[1]*n+[.7]},layout='constrained')
        fixture='ILLUSTRATIVE GPU VALUES' if meta.get('fixture') else ('CPU CONTROL — GPU DISABLED' if meta.get('no_gpu') else 'LOCAL OBSERVATIONS — CHECK FIELD STATUS')
        if meta.get('gpu_provider_scope','').startswith('Windows host'):
            fixture += ' — WINDOWS HOST GPU PROVIDER'
        fig.suptitle(title+'\n'+fixture,fontsize=12)
        for ax in axes[:-1]:
            ax.grid(alpha=.22)
            for i,e in enumerate(events):ax.axvspan(e['start_s'],e['end_s'],color=colors[i%len(colors)],alpha=.07)
        lane=axes[-1]
        for i,e in enumerate(events):
            lane.broken_barh([(e['start_s'],e['end_s']-e['start_s'])],(i-.32,.64),facecolors=colors[i%len(colors)])
        lane.set_yticks(range(len(events)),[e['label'] for e in events],fontsize=8)
        lane.set_ylim(-.6,len(events)-.4);lane.set_xlabel('Seconds from session origin (host monotonic clock)')
        lane.set_title('Host command phases; not GPU completion markers',fontsize=9)
        lane.set_xlim(min(0,t[0]),max(t[-1],max(e['end_s'] for e in events)))
        return fig,axes
    def draw(ax,ys,labels,ylabel):
        any_valid=False
        for j,(y,label) in enumerate(zip(ys,labels)):
            xx,yy=break_gaps(t,y,limit);count=sum(math.isfinite(z) for z in y);any_valid |= bool(count)
            ax.plot(xx,yy,marker='.',markersize=3,linewidth=1,color=colors[j],label=f'{label} [{count}/{len(y)}]')
        ax.set_ylabel(ylabel);ax.legend(loc='upper left',ncol=2,framealpha=.9)
        if not any_valid:ax.text(.5,.4,'Unavailable — no numeric observations',ha='center',transform=ax.transAxes)
    fig,ax=canvas('GPU activity and memory',3)
    draw(ax[0],[series(rows,f) for f in ['utilization.gpu','utilization.memory']],['GPU busy','Memory busy'],'Percent')
    ax[0].set_ylim(-5,105)
    draw(ax[1],[[v/1024 for v in series(rows,f)] for f in ['memory.used','memory.free']],['Reported used','Reported free'],'GiB')
    draw(ax[2],[[r['guest_MiB'][f]/1024 if finite(r['guest_MiB'][f]) else float('nan') for r in rows] for f in ['MemTotal','MemAvailable']],['WSL total','WSL available'],'GiB')
    fig.savefig(out/'activity.png',dpi=180);plt.close(fig)
    fig,ax=canvas('Temperature power clocks and collection',4)
    draw(ax[0],[series(rows,'temperature.gpu')],['GPU temperature'],'C')
    draw(ax[1],[series(rows,f) for f in ['power.draw','power.limit']],['Reported draw','Reported limit'],'W')
    draw(ax[2],[series(rows,f) for f in ['clocks.current.sm','clocks.current.memory']],['SM clock','Memory clock'],'MHz')
    draw(ax[3],[[r['query_ms'] for r in rows]],['Query span'],'ms')
    fig.savefig(out/'environment.png',dpi=180);plt.close(fig)
    print('evidence='+meta['evidence'])
    print(f'samples={len(rows)} phases={len(events)}')
    for f in FIELDS:
        v=report['fields'][f];print(f"{f} valid={v['valid']}/{v['total']} median={v['sample_median']}")
    print('saved=activity.png,environment.png,samples.csv,summary.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    render(Path(a.run),Path(a.out))
