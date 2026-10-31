"""A bounded Whisper adaptation experiment: train the final decoder block only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from contracts import read_audio,digest,word_error
from assets import verify
def dataset(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not 2<=len(rows)<=32:raise ValueError('need 2..32 examples')
    seen={};splits=set()
    for r in rows:
        if r.get('split') not in ('train','dev') or not isinstance(r.get('text'),str) or not r['text'].strip():raise ValueError('invalid split/text')
        p=(Path(path).parent/r['audio']).resolve()
        if not p.is_relative_to(Path(path).parent.resolve()):raise ValueError('audio path leaves dataset directory')
        h=digest(p)
        if h in seen:raise ValueError('duplicate audio content, including across splits')
        seen[h]=r['split'];splits.add(r['split']);r.update(path=p,sha256=h)
    if splits!={'train','dev'}:raise ValueError('both train and dev are required')
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=4);a=p.parse_args()
    if not 1<=a.steps<=20:p.error('steps must be 1..20')
    rows=dataset(a.data);verify(a.model,'whisper')
    if not torch.cuda.is_available():raise RuntimeError('CUDA target required')
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    a.out.mkdir(parents=True,exist_ok=False);torch.manual_seed(1610)
    proc=WhisperProcessor.from_pretrained(a.model,local_files_only=True,language='en',task='transcribe')
    m=WhisperForConditionalGeneration.from_pretrained(a.model,local_files_only=True,use_safetensors=True,dtype=torch.float32).to('cuda')
    m.requires_grad_(False);m.model.decoder.layers[-1].requires_grad_(True)
    trainable=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(trainable,lr=1e-5)
    frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad}
    prepared=[]
    for r in rows:
        sr,x,_=read_audio(r['path'],16000,max_seconds=30)
        feat=proc(x,sampling_rate=sr,return_tensors='pt',return_attention_mask=True)
        labels=proc.tokenizer(r['text'],return_tensors='pt').input_ids
        if labels[0,0]==m.config.decoder_start_token_id:labels=labels[:,1:]
        if labels.shape[1]>128:raise ValueError('training label token budget exceeded')
        prepared.append((r,feat.to('cuda'),labels.to('cuda')))
    def evaluate():
        m.eval();scores=[]
        with torch.inference_mode():
            for r,f,y in prepared:
                if r['split']!='dev':continue
                ids=m.generate(**f,language='en',task='transcribe',max_new_tokens=128,do_sample=False)
                text=proc.batch_decode(ids,skip_special_tokens=True)[0]
                scores.append(dict(sha256=r['sha256'],reference=r['text'],hypothesis=text,**word_error(r['text'],text)))
        return scores
    before=evaluate();losses=[];training=[v for v in prepared if v[0]['split']=='train'];m.train()
    for step in range(a.steps):
        r,f,y=training[step%len(training)];opt.zero_grad(set_to_none=True)
        loss=m(input_features=f['input_features'],labels=y,use_cache=False).loss
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,1,error_if_nonfinite=True);opt.step()
        losses.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm)))
    for n,p in m.named_parameters():
        if n in frozen and not torch.equal(p.detach().cpu(),frozen[n]):raise AssertionError('frozen weight changed: '+n)
    after=evaluate();m.save_pretrained(a.out/'model',safe_serialization=True);proc.save_pretrained(a.out/'model')
    torch.save(opt.state_dict(),a.out/'optimizer.pt')
    result=dict(trainable_parameters=sum(p.numel() for p in trainable),frozen_weights_unchanged=True,losses=losses,before=before,after=after)
    (a.out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
