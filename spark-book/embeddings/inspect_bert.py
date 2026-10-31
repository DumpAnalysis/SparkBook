import argparse,json
import torch
from transformers import AutoTokenizer,BertForMaskedLM
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--text',default='The GPU [MASK] grows with token count.');a=p.parse_args()
torch.set_num_threads(2)
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);model=BertForMaskedLM.from_pretrained(a.model,local_files_only=True).eval()
b=tok(a.text,return_tensors='pt');pos=(b['input_ids'][0]==tok.mask_token_id).nonzero().flatten()
if len(pos)!=1:raise ValueError('provide exactly one [MASK]')
with torch.no_grad():scores=model(**b).logits[0,pos.item()].softmax(-1);v,i=scores.topk(5)
print('tokens',tok.convert_ids_to_tokens(b['input_ids'][0].tolist()))
print('parameters',sum(p.numel() for p in model.parameters()),'vocabulary',len(tok))
for s,t in zip(v,i):print(tok.convert_ids_to_tokens(t.item()),'probability=%.6f'%float(s))
