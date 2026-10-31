"""Download one explicitly selected, revision-pinned model and retain file hashes."""
import argparse,json
from pathlib import Path
from contracts import digest
LOCK=Path(__file__).with_name('models.lock.json')
def verify(root,key):
    spec=json.loads(LOCK.read_text())[key];root=Path(root);r=json.loads((root/'receipt.json').read_text())
    if r['repo']!=spec['repo'] or r['revision']!=spec['revision']:raise ValueError('model identity differs from lock')
    for name,sha in r['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file() or digest(p)!=sha:raise ValueError('model file changed: '+name)
    if not any(n.endswith('.safetensors') for n in r['files']):raise ValueError('missing model weights')
    return spec
def main():
    p=argparse.ArgumentParser();p.add_argument('key',choices=list(json.loads(LOCK.read_text())));p.add_argument('--root',type=Path,default=Path('models/media'));p.add_argument('--verify',action='store_true');a=p.parse_args()
    dst=a.root/a.key
    if a.verify:
        verify(dst,a.key);print('VERIFIED',a.key);return
    from huggingface_hub import snapshot_download
    spec=json.loads(LOCK.read_text())[a.key]
    if dst.exists():raise FileExistsError('use --verify or a new model root; existing directory retained')
    dst.mkdir(parents=True)
    snapshot_download(spec['repo'],revision=spec['revision'],local_dir=dst,allow_patterns=spec['allow_patterns'])
    files={str(f.relative_to(dst)):digest(f) for f in sorted(dst.rglob('*')) if f.is_file() and '.cache' not in f.parts}
    r=dict(repo=spec['repo'],revision=spec['revision'],files=files)
    (dst/'receipt.json').write_text(json.dumps(r,indent=2));verify(dst,a.key)
    print(json.dumps(dict(key=a.key,revision=spec['revision'],files=len(files),bytes=sum((dst/f).stat().st_size for f in files)),sort_keys=True))
if __name__=='__main__':main()
