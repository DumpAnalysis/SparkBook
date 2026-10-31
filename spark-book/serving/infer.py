"""Fixed offline encoder shared by the HTTP and Triton teaching paths."""
import hashlib
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sharing'))
from core import weights

MODEL='lab-embedding'
E,W,B=weights()
REVISION=hashlib.sha256(E.tobytes()+W.tobytes()+B.tobytes()).hexdigest()

def tokenize(text):
    if not isinstance(text,str):raise ValueError('input must contain strings')
    data=list(text.encode('utf-8'))
    if not 1<=len(data)<=16:raise ValueError('fixture accepts 1..16 UTF-8 bytes per text')
    return data+[0]*(16-len(data))

def encode(ids):
    x=np.asarray(ids)
    if x.ndim!=2 or x.shape[1]!=16 or x.dtype.kind not in 'iu' or np.any(x<0) or np.any(x>255):
        raise ValueError('TOKENS must be integers in 0..255 with shape [N,16]')
    y=np.tanh(E[x].mean(1)@W.T+B)
    return (y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)).astype('float32')
