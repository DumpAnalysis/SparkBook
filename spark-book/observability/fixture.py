"""Explicitly simulated GPU readings for pipeline and alert exercises."""
import argparse,json
from pathlib import Path

def fixture(case):
 fields={'utilization.gpu':70,'utilization.memory':35,'memory.used':12288,
 'memory.free':18432,'memory.total':30720,'temperature.gpu':62,
 'power.draw':85,'power.limit':120,'clocks.current.sm':1500,'clocks.current.memory':4000}
 if case=='pressure':fields.update({'memory.used':29184,'memory.free':1536})
 if case=='missing':fields['utilization.gpu']='N/A'
 if case=='malformed':fields['utilization.gpu']='not-a-number'
 if case=='zero':fields['utilization.gpu']=0
 return {'evidence':'SIMULATED','case':case,'fields':fields}
def write(path,case):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(fixture(case),indent=2)+'\n');tmp.replace(path)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--case',choices=['baseline','pressure','missing','malformed','zero'],default='baseline');a=p.parse_args()
 write(a.out,a.case);print('SIMULATED fixture case='+a.case+' path='+a.out)
