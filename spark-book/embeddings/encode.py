import argparse,json,time,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer,models
from data import rows,digest
from common import save_json,normalize,metrics,sha_file

def load_encoder(path,device,bert=False):
    if bert:
        backbone=models.Transformer(path,max_seq_length=128,model_args={'add_pooling_layer':False},tokenizer_args={'local_files_only':True})
        return SentenceTransformer(modules=[backbone,models.Pooling(backbone.get_word_embedding_dimension(),pooling_mode='mean'),models.Normalize()],device=device)
    return SentenceTransformer(path,device=device,local_files_only=True,trust_remote_code=False,model_kwargs={'add_pooling_layer':False})

def evaluate(model,split='dev'):
    rs=rows();docs=[r for r in rs if r['kind']=='document'];queries=[r for r in rs if r['kind']!='document' and r['split']==split]
    d=model.encode([r['text'] for r in docs],batch_size=8,normalize_embeddings=True,show_progress_bar=False)
    q=model.encode([r['text'] for r in queries],batch_size=8,normalize_embeddings=True,show_progress_bar=False)
    rel=[{i for i,r in enumerate(docs) if r['family']==qr['family']} for qr in queries]
    return metrics(q,d,rel)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--bert',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--out',default='embeddings/runs/encoded');p.add_argument('--batch',type=int,default=8);a=p.parse_args()
    if a.batch<1:raise ValueError('batch must be positive')
    torch.manual_seed(17);torch.set_num_threads(2)
    model=load_encoder(a.model,a.device,a.bert);model.max_seq_length=min(model.max_seq_length,128)
    rs=rows();texts=[r['text'] for r in rs]
    counts=[len(model.tokenizer(t,truncation=False)['input_ids']) for t in texts]
    if a.device.startswith('cuda'):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    x=model.encode(texts,batch_size=a.batch,normalize_embeddings=True,show_progress_bar=False).astype('float32')
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    normalize(x)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);np.save(out/'vectors.npy',x,allow_pickle=False);save_json(out/'rows.json',rs)
    report=dict(model=str(Path(a.model).resolve()),device=a.device,shape=list(x.shape),dataset_sha256=digest(),vectors_sha256=sha_file(out/'vectors.npy'),max_seq_length=model.max_seq_length,truncated=sum(n>model.max_seq_length for n in counts),seconds=elapsed,versions={k:importlib.metadata.version(k) for k in ['torch','transformers','sentence-transformers','numpy']},dev=evaluate(model),test=evaluate(model,'test'),peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device.startswith('cuda') else None)
    save_json(out/'report.json',report)
    print('shape',list(x.shape),'truncated',report['truncated']);print('dev',json.dumps(report['dev'],sort_keys=True));print('test',json.dumps(report['test'],sort_keys=True));print('saved',a.out)
if __name__=='__main__':main()
