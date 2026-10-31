"""Execute bounded GPU cases; preserve every command, status, and full console."""
import argparse,json,pathlib,subprocess

def cases(group):
    sizes=[1,31,32,33,255,256,257,1003,1048579]
    groups={
        'vector':[[mode,str(n),str(b)]+(['2'] if mode=='stride' else [])
                  for n in sizes for b in [128,256,512] for mode in ['add','stride']],
        'transpose':[['transpose',v,str(r),str(c)] for r,c in [(1,1),(1,65),(37,65),(32,32),(33,31),(65,37)] for v in ['naive','tile']],
        'reduce':[['reduce',str(n),str(b)] for n in sizes for b in [128,256,512]],
        'hist':[['hist',str(n)] for n in [1,16,1003]],
        'matmul':[['matmul',v,str(m),str(k),str(n)] for m,k,n in [(1,1,1),(17,23,19),(16,16,16),(31,33,29)] for v in ['naive','tile']],
        'rows':[[mode,'3',str(width)] for width in [1,31,769,4096] for mode in ['rms','softmax']],
        'fused':[['fused','1003',str(b)] for b in [128,256,512]],
    }
    if group=='all':return [c for values in groups.values() for c in values]
    return groups[group]

def main():
    p=argparse.ArgumentParser();p.add_argument('binary',type=pathlib.Path)
    p.add_argument('--group',choices=['all','vector','transpose','reduce','hist','matmul','rows','fused'],default='all')
    p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    if a.out.exists():raise SystemExit('Use a new report path')
    binary=a.binary.resolve(strict=True);a.out.parent.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in cases(a.group):
        command=[str(binary),*case]
        result=subprocess.run(command,capture_output=True,text=True,check=False)
        record={'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
        records.append(record)
        a.out.write_text(json.dumps({'scope':'actual target executions','group':a.group,'cases':records},indent=2))
        if result.returncode or 'verified=true' not in result.stdout:
            raise SystemExit('Case failed; inspect '+str(a.out))
    print(f'group={a.group} passed={len(records)}')
    print('saved='+str(a.out))
if __name__=='__main__':main()
