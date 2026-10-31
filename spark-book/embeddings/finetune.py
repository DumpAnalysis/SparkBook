"""Explicit family-aware contrastive loop; no Trainer dependency."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import losses,SentenceTransformer
from data import records,digest
from common import save_json
from encode import load_encoder,evaluate

def batches(rs,batch_size,epoch,wrong=False):
    # Each family occurs once per pass; variants run in separate passes.
    for variant in range(2):
        order=list(range(len(rs)));random.Random(17+epoch*2+variant).shuffle(order)
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size]
            if len(indices)<2:raise ValueError('batch of one has no negatives')
            yield [rs[i]['queries'][variant] for i in indices],[rs[(i+1)%len(rs)]['document'] if wrong else rs[i]['document'] for i in indices]

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--bert',action='store_true');p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch',type=int,default=4);p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--wrong-pairs',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(17);random.seed(17)
    rs=[r for r in records() if r['split']=='train']
    if a.epochs<1 or not 2<=a.batch<=len(rs):raise ValueError('positive epochs; batch from 2 to train family count')
    if len(rs)%a.batch==1:raise ValueError('last batch would have one example')
    model=load_encoder(a.model,a.device,a.bert);model.max_seq_length=128
    initial=evaluate(model);print('initial_dev',json.dumps(initial,sort_keys=True))
    loss_fn=losses.MultipleNegativesRankingLoss(model,scale=20.0)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    best=initial['mrr'];out=Path(a.out);out.mkdir(parents=True,exist_ok=True);model.save_pretrained(str(out));history=[]
    for epoch in range(a.epochs):
        model.train();values=[]
        for q,d in batches(rs,a.batch,epoch,a.wrong_pairs):
            features=[{k:v.to(a.device) for k,v in model.tokenize(texts).items()} for texts in [q,d]]
            opt.zero_grad(set_to_none=True);loss=loss_fn(features,None)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite contrastive loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();values.append(float(loss.detach()))
        score=evaluate(model);history.append(dict(epoch=epoch+1,mean_loss=float(np.mean(values)),dev=score));print('epoch',epoch+1,'mean_loss=%.4f'%np.mean(values),'dev_mrr=%.4f'%score['mrr'])
        if score['mrr']>best:
            best=score['mrr'];model.save_pretrained(str(out))
    selected=load_encoder(str(out),a.device)
    probe=['GPU memory allocation failed.','Embedding vectors require validation.']
    expected=selected.encode(probe,normalize_embeddings=True,show_progress_bar=False)
    reloaded=load_encoder(str(out),a.device)
    actual=reloaded.encode(probe,normalize_embeddings=True,show_progress_bar=False)
    error=float(np.max(np.abs(expected-actual)))
    report=dict(initial_dev=initial,selected_dev=evaluate(selected),test=evaluate(selected,'test'),reload_max_abs=error,dataset_sha256=digest(),model=a.model,wrong_pairs=a.wrong_pairs,device=a.device,batch=a.batch,history=history)
    save_json(out/'finetune-report.json',report)
    print('selected_dev',json.dumps(report['selected_dev'],sort_keys=True));print('test',json.dumps(report['test'],sort_keys=True));print('reload_max_abs=%.8f'%error);print('saved',a.out)
if __name__=='__main__':main()
