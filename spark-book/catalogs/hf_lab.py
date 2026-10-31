"""Pinned public Hub metadata, selective downloads and bounded offline controls."""
import argparse,csv,json,os,re,socket,tempfile
from pathlib import Path
from unittest.mock import patch
from common import new_dir,save,public_json,file_inventory,verify_inventory
LOCK=json.loads((Path(__file__).parent/'lock.json').read_text())

def selected(kind):
    pin=LOCK['hf_'+kind]
    if not re.fullmatch('[0-9a-f]{40}',pin['revision']):raise ValueError('full commit required')
    return pin

def info(out):
    out=new_dir(out);summary=[]
    for kind in ['model','dataset']:
        pin=selected(kind);url=f"https://huggingface.co/api/{kind}s/{pin['repo_id']}/revision/{pin['revision']}"
        data,raw,receipt=public_json(url)
        if data['sha']!=pin['revision']:raise ValueError('Hub revision mismatch')
        (out/(kind+'.json')).write_bytes(raw);save(out/(kind+'-http.json'),receipt)
        row={'type':kind,'repo_id':pin['repo_id'],'revision':data['sha'],'files':len(data['siblings']),'gated':data.get('gated',False),'license':(data.get('cardData') or {}).get('license','UNSPECIFIED')};summary.append(row)
        for k,v in row.items():print(f'{k}: {v}')
    save(out/'summary.json',summary)
    print('Public metadata received; license metadata alone does not establish permitted use')

def snapshot(kind,dest,record):
    from huggingface_hub import snapshot_download
    out=new_dir(record);target=new_dir(dest);pin=selected(kind)
    snapshot_download(repo_id=pin['repo_id'],repo_type=kind,revision=pin['revision'],allow_patterns=pin['files'],local_dir=target,token=False,max_workers=2)
    files=file_inventory(target,pin['files'])
    receipt={'repo_type':kind,'repo_id':pin['repo_id'],'revision':pin['revision'],'files':files,'publisher_license':'UNSPECIFIED in pinned metadata','weights_in_companion':False}
    save(out/'receipt.json',receipt);save(target/'book-receipt.json',receipt)
    print(f"repo_type: {kind}\nrepo_id: {pin['repo_id']}\nrevision: {pin['revision']}\nselected_files: {len(files)}\nselected_bytes: {sum(x['bytes'] for x in files)}")
    print(f"receipt_sha256: {__import__('hashlib').sha256((out/'receipt.json').read_bytes()).hexdigest()}")
    print('Selected files acquired and hashed; alternate weight formats were not requested')

def offline(snapshot_dir,out,device):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    import torch
    from transformers import AutoModel,AutoTokenizer
    out=new_dir(out);source=Path(snapshot_dir);receipt=json.loads((source/'book-receipt.json').read_text())
    if receipt['repo_type']!='model' or receipt['revision']!=selected('model')['revision']:raise ValueError('unexpected model receipt')
    verified=verify_inventory(source,receipt['files'])
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    torch.manual_seed(19);torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as empty_cache,patch('socket.socket.connect',side_effect=RuntimeError('network attempted in offline control')) as network:
        tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False,token=False,cache_dir=empty_cache)
        model,loading=AutoModel.from_pretrained(source,local_files_only=True,trust_remote_code=False,use_safetensors=True,token=False,cache_dir=empty_cache,output_loading_info=True)
        if loading['missing_keys'] or loading['mismatched_keys'] or loading.get('error_msgs'):raise ValueError('incomplete checkpoint load; see native loader messages')
        model=model.eval().to(device)
        batch=tokenizer(['Catalog identity precedes execution.','We retain the observed revision.'],padding=True,truncation=True,max_length=64,return_tensors='pt').to(device)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            x=model(**batch).last_hidden_state;y=model(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1)
            pooled=(x*mask).sum(1)/mask.sum(1)
        difference=float((x-y).abs().max().cpu());finite=bool(torch.isfinite(x).all().cpu())
        if not finite or difference!=0:raise AssertionError('offline repeat control failed')
        attempts=network.call_count
    result={'device':device,'torch':torch.__version__,'revision':receipt['revision'],'verified_files':verified,'model_class':type(model).__name__,'parameters':sum(p.numel() for p in model.parameters()),'input_shape':list(batch['input_ids'].shape),'hidden_shape':list(x.shape),'pooled_shape':list(pooled.shape),'finite':finite,'repeat_max_abs_error':difference,'python_socket_connect_attempts':attempts,'missing_keys':loading['missing_keys'],'unexpected_keys':loading['unexpected_keys'],'mismatched_keys':loading['mismatched_keys'],'trained_semantic_quality_claim':False}
    if device=='cuda':result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    save(out/'result.json',result);save(out/'pooled.json',pooled.cpu().tolist())
    for k in ['device','verified_files','model_class','parameters','input_shape','hidden_shape','pooled_shape','finite','repeat_max_abs_error','python_socket_connect_attempts']:print(f'{k}: {result[k]}')
    print('Random test checkpoint only; successful loading is not semantic model quality')

def audit_dataset(snapshot_dir,out):
    out=new_dir(out);source=Path(snapshot_dir);receipt=json.loads((source/'book-receipt.json').read_text())
    if receipt['repo_type']!='dataset' or receipt['revision']!=selected('dataset')['revision']:raise ValueError('unexpected dataset receipt')
    verified=verify_inventory(source,receipt['files']);splits={}
    for name in ['train','test']:
        with (source/'data'/f'{name}.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        if not rows or not {'id','review','star'}.issubset(rows[0]):raise ValueError('unexpected CSV schema')
        if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate ID inside split')
        splits[name]=rows
    ids=lambda rows:{r['id'] for r in rows}
    texts=lambda rows:{' '.join(r['review'].casefold().split()) for r in rows}
    overlap=ids(splits['train']) & ids(splits['test']);text_overlap=texts(splits['train'])&texts(splits['test'])
    injected=splits['test']+[splits['train'][0]]
    detected=len(ids(splits['train'])&ids(injected))
    if overlap or text_overlap or detected!=1:raise AssertionError('split control differs from pinned fixture')
    result={'revision':receipt['revision'],'verified_files':verified,'train_rows':len(splits['train']),'test_rows':len(splits['test']),'id_overlap':len(overlap),'normalized_text_overlap':len(text_overlap),'injected_overlap_detected':detected,'training_readiness':'NOT ESTABLISHED: card provenance and license incomplete'}
    save(out/'audit.json',result)
    for k,v in result.items():print(f'{k}: {v}')
    print('CSV/identity control only; no training and no semantic deduplication')

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('info');q.add_argument('--out',required=True)
    q=sub.add_parser('snapshot');q.add_argument('--kind',choices=['model','dataset'],required=True);q.add_argument('--dest',required=True);q.add_argument('--record',required=True)
    q=sub.add_parser('offline');q.add_argument('--snapshot',required=True);q.add_argument('--out',required=True);q.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    q=sub.add_parser('dataset');q.add_argument('--snapshot',required=True);q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='info':info(a.out)
    elif a.command=='snapshot':snapshot(a.kind,a.dest,a.record)
    elif a.command=='offline':offline(a.snapshot,a.out,a.device)
    else:audit_dataset(a.snapshot,a.out)
if __name__=='__main__':main()
