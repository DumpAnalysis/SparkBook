"""Validate and explain event-timing CSV records without hardware claims."""
import argparse,csv,math,pathlib

FIELDS=['mode','n','block','repeats','total_ms','mean_ms','logical_bytes','effective_GB_s']
def read(path):
    with path.open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS:raise ValueError('unexpected CSV columns')
        records=list(reader)
    if len(records)!=2 or [r['mode'] for r in records]!=['split','fused']:
        raise ValueError('expected split and fused rows')
    for r in records:
        for key in ['n','block','repeats']:r[key]=int(r[key])
        for key in ['total_ms','mean_ms','logical_bytes','effective_GB_s']:r[key]=float(r[key])
        if any(not math.isfinite(r[k]) or r[k]<=0 for k in FIELDS[1:]):
            raise ValueError('positive finite values required')
        expected=r['n']*(20 if r['mode']=='split' else 12)
        for got,want in [(r['mean_ms'],r['total_ms']/r['repeats']),
                         (r['logical_bytes'],expected),
                         (r['effective_GB_s'],expected/(r['mean_ms']*1e6))]:
            if not math.isclose(got,want,rel_tol=1e-7):raise ValueError('inconsistent timing arithmetic')
    if any(records[0][k]!=records[1][k] for k in ['n','block','repeats']):
        raise ValueError('workload identities differ')
    return records

def fixture(path):
    with path.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(FIELDS)
        for mode,ms,bytes_per_value in [('split',20,20),('fused',12,12)]:
            w.writerow([mode,100000,256,100,ms,ms/100,bytes_per_value*100000,10])

def main():
    p=argparse.ArgumentParser();p.add_argument('file',type=pathlib.Path)
    p.add_argument('--fixture',action='store_true');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.fixture:
        a.file.parent.mkdir(parents=True,exist_ok=True);fixture(a.file)
        print('evidence=SYNTHETIC timing arithmetic; no GPU measurement')
    rows=read(a.file)
    print('records=2 arithmetic_valid=true')
    if not a.check_only:
        for row in rows:print(f"{row['mode']} mean_ms={row['mean_ms']:.6f} effective_GB_s={row['effective_GB_s']:.6f}")
        print(f"split_over_fused={rows[0]['mean_ms']/rows[1]['mean_ms']:.6f}")
        print('Effective bandwidth uses logical bytes; it is not a DRAM counter.')
if __name__=='__main__':main()
