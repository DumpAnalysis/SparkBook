"""Explicit download of a pinned catalog artifact. No download in plan mode."""
import argparse,json,urllib.request,os,shutil
from pathlib import Path
from common import sha,write_json,ContractError
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('kind',choices=['Q4_K_M','Q8_0','BF16','training-base','chat-gguf']);p.add_argument('--out',required=True);p.add_argument('--download',action='store_true');a=p.parse_args()
meta=json.loads((ROOT/'metadata'/('qwen-base.json' if a.kind=='training-base' else ('qwen-gguf.json' if a.kind=='chat-gguf' else 'nemotron-gguf.json'))).read_text())
if a.kind=='training-base':
    selected=[x for x in meta['siblings'] if x['rfilename'].endswith(('.json','.safetensors','.txt','.model')) or x['rfilename'] in ('LICENSE','README.md')]
elif a.kind=='chat-gguf':selected=[x for x in meta['siblings'] if x['rfilename'].endswith('-q4_k_m.gguf')]
else:selected=[x for x in meta['siblings'] if x['rfilename'].endswith('-'+a.kind+'.gguf')]
assert selected
out=Path(a.out)
plan={'repository':meta['id'],'revision':meta['sha'],'bytes':sum(x.get('size',0) for x in selected),'files':selected,'download_requested':a.download}
print('repository',plan['repository']);print('revision',plan['revision']);print('selected files',len(selected),'bytes',plan['bytes'])
if not a.download:print('PLAN ONLY; no network transfer');raise SystemExit(0)
if out.exists():raise ContractError('output exists; retain it and choose a fresh directory')
out.mkdir(parents=True)
if shutil.disk_usage(out).free<plan['bytes']+1024**3:raise ContractError('insufficient disk free with 1 GiB margin')
write_json(out/'acquisition-plan.json',plan)
# HTTPS artifact redirects to the Hub storage service are expected for these pinned public files.
for item in selected:
    name=item['rfilename'];rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:raise ContractError('invalid catalog path')
    dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);partial=dest.with_name(dest.name+'.partial')
    u='https://huggingface.co/'+meta['id']+'/resolve/'+meta['sha']+'/'+name
    with urllib.request.urlopen(u,timeout=120) as src,open(partial,'xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
    if partial.stat().st_size!=item['size']:raise ContractError('byte size mismatch '+name)
    if 'lfs' in item:
        if sha(partial)!=item['lfs']['sha256']:raise ContractError('LFS SHA256 mismatch '+name)
    else:
        import hashlib
        data=partial.read_bytes();git=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if git!=item['blobId']:raise ContractError('git blob SHA1 mismatch '+name)
    partial.rename(dest);print('VERIFIED',name,flush=True)
write_json(out/'acquisition-receipt.json',{'repository':meta['id'],'revision':meta['sha'],'files':{x['rfilename']:sha(out/x['rfilename']) for x in selected}})
