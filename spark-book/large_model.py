"""Plan or download the pinned 32B Q8 GGUF. Does not launch inference."""
import argparse
import json
import pathlib
import shutil
from setup_wsl import fetch, get_json

REPO='Qwen/Qwen2.5-32B-Instruct-GGUF'
REV='a15e3cc10f8bbb2c0af6f8f1f34a32e3b060c09d'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--download',action='store_true')
    a=p.parse_args()
    root=pathlib.Path(__file__).resolve().parent
    hf='https://huggingface.co'
    meta=get_json(f'{hf}/api/models/{REPO}/revision/{REV}?blobs=true')
    parts=sorted([r for r in meta['siblings'] if
        r['rfilename'].startswith('qwen2.5-32b-instruct-q8_0-')
        and r['rfilename'].endswith('.gguf')],key=lambda r:r['rfilename'])
    if len(parts)!=9: raise SystemExit('Expected nine shards')
    size=sum(r['size'] for r in parts)
    print('model='+REPO)
    print('shards='+str(len(parts)))
    print(f'weights_gib={size/2**30:.2f}')
    print('Directory: models/32b-q8_0')
    if not a.download:
        print('Plan only; use --download after checking disk and memory.')
        return
    dest=root/'models/32b-q8_0';dest.mkdir(parents=True,exist_ok=True)
    remaining=sum(r['size'] for r in parts if not (dest/r['rfilename']).exists())
    if shutil.disk_usage(dest).free < remaining+2*2**30:
        raise SystemExit('Insufficient free disk for remaining files plus margin')
    records=[]
    for item in parts:
        name=item['rfilename'];path=dest/name
        sha=fetch(f'{hf}/{REPO}/resolve/{REV}/{name}',path)
        if path.stat().st_size!=item['size']: raise SystemExit('Shard size mismatch')
        expected=item.get('lfs',{}).get('sha256')
        if expected and sha!=expected: raise SystemExit('Shard digest mismatch')
        with path.open('rb') as f:
            if f.read(4)!=b'GGUF': raise SystemExit('Not a GGUF file')
        records.append({'name':name,'bytes':item['size'],'sha256':sha})
    (root/'runs').mkdir(exist_ok=True)
    (root/'runs/large-model-manifest.json').write_text(json.dumps(
        {'repository':REPO,'revision':REV,'records':records},indent=2))
    print('manifest=runs/large-model-manifest.json')
    print('Download complete; placement and capacity still need testing.')


if __name__=='__main__':
    main()
