"""Compare a retained V2 response with the fixture's numerical reference."""
import argparse,json
from pathlib import Path
import numpy as np
from infer import encode,tokenize
p=argparse.ArgumentParser();p.add_argument('responses');a=p.parse_args()
rows=json.loads(Path(a.responses).read_text())
for row in rows:
    assert row['status']==200,row
    body=json.loads(row['body']);tensor=next(t for t in body['outputs'] if t['name']=='EMBEDDINGS')
    assert tensor['shape']==[2,64] and tensor['datatype']=='FP32',tensor
    actual=np.asarray(tensor['data'],dtype=np.float32).reshape(tensor['shape'])
    error=float(np.max(np.abs(actual-encode([tokenize('alpha'),tokenize('beta')]))))
    assert error<=1e-5,error
print(f'PASS Triton response vectors rows={len(rows)} tolerance=1e-5')
