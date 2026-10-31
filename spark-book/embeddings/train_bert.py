"""Small BERT masked language model: scratch or local continued pretraining."""
import argparse,json,math,random
from pathlib import Path
import torch
from transformers import BertConfig,BertForMaskedLM,PreTrainedTokenizerFast,AutoTokenizer
from tokenizers import Tokenizer,models,trainers,normalizers,pre_tokenizers,processors,decoders
from data import rows,digest
from common import save_json

def train_tokenizer(texts,out):
    t=Tokenizer(models.WordPiece(unk_token='[UNK]'))
    t.normalizer=normalizers.BertNormalizer(lowercase=True)
    t.pre_tokenizer=pre_tokenizers.BertPreTokenizer()
    t.decoder=decoders.WordPiece(prefix='##')
    t.train_from_iterator(texts,trainers.WordPieceTrainer(vocab_size=512,min_frequency=1,special_tokens=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]'],show_progress=False))
    t.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',pair='[CLS] $A [SEP] $B:1 [SEP]:1',special_tokens=[('[CLS]',t.token_to_id('[CLS]')),('[SEP]',t.token_to_id('[SEP]'))])
    tok=PreTrainedTokenizerFast(tokenizer_object=t,unk_token='[UNK]',pad_token='[PAD]',cls_token='[CLS]',sep_token='[SEP]',mask_token='[MASK]',model_max_length=128)
    tok.save_pretrained(out);return tok

def mask_batch(batch,tok,generator):
    ids=batch['input_ids'].clone();labels=ids.clone()
    eligible=batch['attention_mask'].bool()
    for special in tok.all_special_ids:eligible &= ids.ne(special)
    selected=(torch.rand(ids.shape,generator=generator)<.15)&eligible
    # A small teaching batch must contain at least one target per nonempty row.
    for i in range(len(ids)):
        if not selected[i].any():
            candidates=eligible[i].nonzero().flatten()
            if len(candidates):selected[i,candidates[torch.randint(len(candidates),(1,),generator=generator)]]=True
    if not selected.any():raise ValueError('no eligible MLM targets')
    labels[~selected]=-100
    roll=torch.rand(ids.shape,generator=generator)
    ids[selected&(roll<.8)]=tok.mask_token_id
    random_ids=torch.randint(len(tok),ids.shape,generator=generator)
    ids[selected&(roll>=.8)&(roll<.9)]=random_ids[selected&(roll>=.8)&(roll<.9)]
    return {**batch,'input_ids':ids,'labels':labels}

def evaluate(model,masked,device):
    model.eval();total=0.;count=0;correct=0
    with torch.no_grad():
        for b in masked:
            b={k:v.to(device) for k,v in b.items()};o=model(**b);sel=b['labels'].ne(-100);n=int(sel.sum());total+=float(o.loss)*n;count+=n;correct+=int((o.logits.argmax(-1)[sel]==b['labels'][sel]).sum())
    return dict(loss=total/count,masked_accuracy=correct/count,targets=count)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--init');p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int,default=60);p.add_argument('--batch',type=int,default=8);p.add_argument('--lr',type=float,default=5e-4);a=p.parse_args()
    if a.steps<1 or a.batch<1:raise ValueError('positive steps and batch required')
    torch.set_num_threads(2);torch.manual_seed(17);random.seed(17)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rs=rows();texts=[r['text'] for r in rs if r['split']=='train'];dev=[r['text'] for r in rs if r['split']=='dev'];test=[r['text'] for r in rs if r['split']=='test']
    if a.init:
        tok=AutoTokenizer.from_pretrained(a.init,local_files_only=True);model=BertForMaskedLM.from_pretrained(a.init,local_files_only=True)
    else:
        tok=train_tokenizer(texts,out)
        model=BertForMaskedLM(BertConfig(vocab_size=len(tok),hidden_size=128,num_hidden_layers=2,num_attention_heads=4,intermediate_size=256,max_position_embeddings=128,pad_token_id=tok.pad_token_id))
    model.to(a.device)
    def encode(ts):return tok(ts,padding=True,truncation=True,max_length=128,return_tensors='pt',return_token_type_ids=True)
    def fixed(ts,seed):
        g=torch.Generator().manual_seed(seed)
        return [mask_batch(encode(ts[i:i+a.batch]),tok,g) for i in range(0,len(ts),a.batch)]
    devmask=fixed(dev,101);testmask=fixed(test,202)
    initial=evaluate(model,devmask,a.device);print('initial_dev',json.dumps(initial,sort_keys=True))
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
    g=torch.Generator().manual_seed(303);best=initial['loss'];logs=[]
    model.save_pretrained(out,safe_serialization=True);tok.save_pretrained(out)
    for step in range(1,a.steps+1):
        idx=torch.randint(len(texts),(a.batch,),generator=g).tolist()
        b={k:v.to(a.device) for k,v in mask_batch(encode([texts[i] for i in idx]),tok,g).items()}
        model.train();opt.zero_grad(set_to_none=True);loss=model(**b).loss
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step%20==0 or step==a.steps:
            value=evaluate(model,devmask,a.device);line=dict(step=step,train_loss=float(loss.detach()),dev=value);logs.append(line);print('step',step,'train_loss=%.4f'%float(loss.detach()),'dev_loss=%.4f'%value['loss'])
            if value['loss']<best:
                best=value['loss'];model.save_pretrained(out,safe_serialization=True);tok.save_pretrained(out)
    selected=BertForMaskedLM.from_pretrained(out,local_files_only=True).to(a.device)
    final=evaluate(selected,devmask,a.device);test_result=evaluate(selected,testmask,a.device)
    selected.eval();sample={k:v.to(a.device) for k,v in devmask[0].items()};expected=selected(**sample).logits.detach()
    reloaded=BertForMaskedLM.from_pretrained(out,local_files_only=True).to(a.device).eval()
    with torch.no_grad():error=float((reloaded(**sample).logits-expected).abs().max())
    report=dict(mode='continued' if a.init else 'scratch',init=a.init,device=a.device,dataset_sha256=digest(),parameters=sum(p.numel() for p in selected.parameters()),vocabulary=len(tok),initial_dev=initial,selected_dev=final,test=test_result,reload_max_abs=error,logs=logs,steps=a.steps,seed=17)
    save_json(out/'training-report.json',report)
    print('selected_dev',json.dumps(final,sort_keys=True));print('test',json.dumps(test_result,sort_keys=True));print('reload_max_abs=%.8f'%error);print('saved',a.out)
if __name__=='__main__':main()
