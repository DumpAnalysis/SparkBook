"""Select the recorded native Linux release archive; download only on request."""
import argparse,json,platform,urllib.request,shutil
from pathlib import Path
from common import sha,ContractError,write_json
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--download',action='store_true');a=p.parse_args()
machine=platform.machine();arch={'aarch64':'arm64','arm64':'arm64','x86_64':'amd64','AMD64':'amd64'}.get(machine)
if not arch:raise ContractError('unqualified architecture '+machine)
m=json.loads((Path(__file__).parent/'metadata/ollama-release.json').read_text());asset=next(x for x in m['assets'] if x['name']=='ollama-linux-'+arch+'.tar.zst')
print('release',m['tag_name'],'native architecture',arch);print('file',asset['name']);print('bytes',asset['size']);print('digest',asset['digest'])
if not a.download:print('PLAN ONLY; no archive downloaded');raise SystemExit(0)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);dest=out/asset['name'];partial=dest.with_name(dest.name+'.partial')
if dest.exists() or partial.exists():raise ContractError('archive or partial already exists')
with urllib.request.urlopen(asset['browser_download_url'],timeout=120) as src,open(partial,'xb') as dst:shutil.copyfileobj(src,dst,1024*1024)
if partial.stat().st_size!=asset['size'] or 'sha256:'+sha(partial)!=asset['digest']:raise ContractError('archive verification failed')
partial.rename(dest);write_json(out/'ollama-receipt.json',{'tag':m['tag_name'],'asset':asset['name'],'bytes':asset['size'],'sha256':sha(dest)})
print('VERIFIED',dest)
