import argparse,json
from pathlib import Path
import numpy as np
from common import normalize,save_json,sha_file
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.input);x=normalize(np.load(root/'vectors.npy',allow_pickle=False));rows=json.loads((root/'rows.json').read_text())
if len(rows)!=len(x):raise ValueError('row vector mismatch')
s=x@x.T;different=np.array([[a['family']!=b['family'] for b in rows] for a in rows]);e=np.linalg.svd(x-x.mean(0),compute_uv=False);energy=e**2
report=dict(rows=len(x),dimension=x.shape[1],vectors_sha256=sha_file(root/'vectors.npy'),mean_coordinate_variance=float(x.var(0).mean()),unrelated_mean_cosine=float(s[different].mean()),unrelated_max_cosine=float(s[different].max()),first_pc_fraction=float(energy[0]/energy.sum()) if energy.sum()>1e-20 else None)
save_json(a.out,report)
print(json.dumps(report,sort_keys=True,indent=2))
