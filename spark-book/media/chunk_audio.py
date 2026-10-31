"""Make bounded, nonoverlapping WAV chunks with original sample offsets.

This is an inspectable baseline. Words crossing boundaries may be damaged;
overlap-aware timestamp reconciliation is a separate evaluation problem.
"""
import argparse,json
from pathlib import Path
from contracts import read_audio,write_audio
def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=int,default=20);a=p.parse_args()
    if not 1<=a.seconds<=30:p.error('chunk duration must be 1..30 seconds')
    rate,x,source=read_audio(a.input,16000,max_seconds=120);a.out.mkdir(parents=True,exist_ok=False);rows=[];n=a.seconds*rate
    for i,start in enumerate(range(0,len(x),n)):
        stop=min(start+n,len(x));name=f'{i:03d}.wav';write_audio(a.out/name,x[start:stop],rate)
        rows.append(dict(file=name,start_sample=start,end_sample=stop,start_s=start/rate,end_s=stop/rate))
    (a.out/'chunks.json').write_text(json.dumps(dict(source=source,sample_rate=rate,chunks=rows),indent=2));print('chunks',len(rows),'frames',len(x))
if __name__=='__main__':main()
