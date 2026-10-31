"""Exact local search; the manifest binds model bytes to corpus vectors."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from common import normalize,save_json,sha_file
from encode import load_encoder
from data import records,digest

def fingerprint(root):
    h=hashlib.sha256()
    for p in sorted(Path(root).rglob('*')):
        if p.is_file() and p.suffix in {'.json','.txt','.safetensors','.bin'} and '.cache' not in p.parts:
            h.update(str(p.relative_to(root)).encode())
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','query']);p.add_argument('--model',required=True);p.add_argument('--index',required=True);p.add_argument('--device',default='cpu');p.add_argument('--text');a=p.parse_args()
    root=Path(a.index);fp=fingerprint(a.model)
    if a.mode=='build':
        model=load_encoder(a.model,a.device);model.max_seq_length=128;rs=records()
        x=model.encode([r['document'] for r in rs],normalize_embeddings=True,show_progress_bar=False)
        root.mkdir(parents=True,exist_ok=True);np.save(root/'documents.npy',x,allow_pickle=False);save_json(root/'documents.json',rs)
        save_json(root/'manifest.json',dict(model_sha256=fp,data_sha256=digest(),vectors_sha256=sha_file(root/'documents.npy'),documents_sha256=sha_file(root/'documents.json'),max_seq_length=128,dimension=x.shape[1],normalization='L2'))
        print('indexed',len(rs),'documents');return
    manifest=json.loads((root/'manifest.json').read_text())
    if fp!=manifest['model_sha256']:raise ValueError('model changed; rebuild index')
    if any(sha_file(root/f)!=manifest[k] for f,k in [('documents.npy','vectors_sha256'),('documents.json','documents_sha256')]):raise ValueError('index files changed')
    if not a.text:raise ValueError('--text required')
    model=load_encoder(a.model,a.device);model.max_seq_length=manifest['max_seq_length']
    q=model.encode([a.text],normalize_embeddings=True,show_progress_bar=False);d=normalize(np.load(root/'documents.npy',allow_pickle=False));docs=json.loads((root/'documents.json').read_text());scores=(normalize(q)@d.T)[0]
    for i in np.argsort(-scores,kind='stable')[:3]:print(docs[i]['id'],'cosine=%.4f'%scores[i],docs[i]['document'])
if __name__=='__main__':main()
