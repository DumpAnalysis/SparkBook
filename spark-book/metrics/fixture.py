"""Deterministic invented GPU observations for teaching and renderer tests."""
import argparse,json
from pathlib import Path
from monitor import FIELDS,dump

def create(out):
    out.mkdir(parents=True,exist_ok=False)
    meta={'schema':1,'fixture':True,'evidence':'ILLUSTRATIVE deterministic data; no GPU was observed',
          'interval_s':1,'fields':{k:{'unit':v[0]} for k,v in FIELDS.items()},'gpu_selector':'fictional GPU',
          'origin_ns':0,'utc_origin':'not a hardware capture'}
    dump(out/'metadata.json',meta)
    events=[{'label':s,'start_s':a,'end_s':b,'returncode':0,'scope':'invented host phase'} for s,a,b in [('idle',0,5),('load',5,10),('inference',10,20),('embedding',20,30),('training',30,40),('cooldown',40,45)]]
    dump(out/'events.json',events)
    with (out/'samples.jsonl').open('w') as fh:
        for i in range(46):
            # Remove a range entirely to exercise a visible acquisition gap.
            if i in {24,25,26}:continue
            busy=0 if i<5 or i>=40 else 30 if i<10 else 72 if i<20 else 88 if i<30 else 96
            used=1024 if i<5 else 8192 if i<20 else 4096 if i<30 else 12288 if i<40 else 2048
            vals=[busy,busy*.6,used,31232-used,31232,42+min(i,35)*.65,12+busy*.55,90,400+busy*12,3200]
            gpu={f:{'value':float(v),'status':'ok'} for f,v in zip(FIELDS,vals)}
            gpu['clocks.current.memory']={'value':None,'status':'unavailable'}
            if 15<=i<=17:gpu['power.draw']={'value':None,'status':'unavailable'}
            if i==12:gpu['utilization.gpu']={'value':None,'status':'query_error'}
            row={'seq':i,'utc':'ILLUSTRATIVE','t_s':i+.02,'start_s':float(i),'end_s':i+.04,'query_ms':40.,'gpu':gpu,
                 'guest_MiB':{'MemTotal':16384.,'MemAvailable':12000.-used*.2,'SwapTotal':4096.,'SwapFree':4096.}}
            fh.write(json.dumps(row)+'\n')
    print('evidence='+meta['evidence']);print('samples=43 phases=6');print('saved='+str(out))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();create(Path(a.out))
