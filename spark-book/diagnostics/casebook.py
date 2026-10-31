import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['patterns','case','quality','reproducer','report']);p.add_argument('--id',choices=['C01','C02','C03','C04','C05','C06'],default='C01');p.add_argument('--answer',action='store_true');p.add_argument('--source');p.add_argument('--out');a=p.parse_args()
 if a.action=='patterns':
  rows=json.loads((ROOT/'patterns.json').read_text())
  for i,r in enumerate(rows,1):print('%02d %s'%(i,r['name']))
  print('patterns='+str(len(rows)))
 elif a.action=='case':
  rows=json.loads((ROOT/'blind'/('answers.json' if a.answer else 'cases.json')).read_text());r=next(x for x in rows if x['id']==a.id);print(r['id'])
  if a.answer:
   for k in ['pattern','next_test','explanation']:print(k+': '+r[k])
  else:
   for s in r['evidence']:print(s)
   print('Write two hypotheses, a discriminating test and a boundary.')
 else:
  if not a.source:p.error('--source required')
  rows=json.loads(Path(a.source).read_text())
  if a.action=='quality':
   for r in rows:print(r['fault']+': '+json.dumps(r['baseline'],sort_keys=True))
  elif a.action=='reproducer':
   print('records='+repr(rows['records']));print('one_minimal='+str(rows['one_minimal']));print('scan='+str(rows['scan']));print('oracle_calls='+str(len(rows['calls'])))
  else:
   if not a.out:p.error('--out required')
   actual=[r['lab_live_bytes'] for r in rows];assert actual==[262144,1048576,2097152,0]
   report={'pattern':'Retained Tensor','observation':actual,'alternative':'changing request sizes','test':'fixed sizes then clear only the owner','result':'live lab bytes return to zero','repair':'release unnecessary list ownership','regression':'release mode has zero live lab bytes','boundary':'CPU lab-owned tensors, not GPU allocator or process RSS'}
   with Path(a.out).open('x') as f:json.dump(report,f,indent=2)
   print('verified_live_bytes='+str(actual));print('report='+a.out);print('boundary='+report['boundary'])
if __name__=='__main__':main()
