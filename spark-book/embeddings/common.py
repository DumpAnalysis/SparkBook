import json, hashlib
from pathlib import Path
import numpy as np

def normalize(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all(): raise ValueError('finite 2D vectors required')
    n=np.linalg.norm(x,axis=1,keepdims=True)
    if np.any(n<=1e-12): raise ValueError('zero vector')
    return x/n

def pool(hidden,mask):
    h=np.asarray(hidden,float);m=np.asarray(mask,float)
    if h.ndim!=3 or m.shape!=h.shape[:2] or not np.isin(m,[0,1]).all(): raise ValueError('shape or mask')
    if np.any(m.sum(1)==0): raise ValueError('empty sequence')
    return (h*m[:,:,None]).sum(1)/m.sum(1)[:,None]

def metrics(q,d,relevant):
    q=normalize(q);d=normalize(d)
    if len(q)!=len(relevant) or any(not r or min(r)<0 or max(r)>=len(d) for r in relevant): raise ValueError('relevance')
    order=np.argsort(-(q@d.T),axis=1,kind='stable')
    recall1=[];recall3=[];rr=[]
    for rank,rel in zip(order,relevant):
        recall1.append(len(set(rank[:1])&set(rel))/len(rel))
        recall3.append(len(set(rank[:3])&set(rel))/len(rel))
        rr.append(1/next(i+1 for i,j in enumerate(rank) if j in rel))
    return dict(recall_at_1=float(np.mean(recall1)),recall_at_3=float(np.mean(recall3)),mrr=float(np.mean(rr)))

def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
