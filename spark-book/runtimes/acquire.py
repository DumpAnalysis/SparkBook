"""Plan, download or resume a pinned catalog artifact."""
import argparse
import json
import http.client
import os
from pathlib import Path
import shutil
import tempfile
import urllib.parse

from pinned_download import fetch_pinned, locked_directory

ROOT = Path(__file__).resolve().parent
METADATA = {'training-base':'qwen-base.json', 'chat-gguf':'qwen-gguf.json'}


def select(kind):
    metadata = json.loads((ROOT/'metadata'/METADATA.get(kind,'nemotron-gguf.json')).read_text())
    if kind == 'training-base':
        files = [x for x in metadata['siblings'] if x['rfilename'].endswith(('.json','.safetensors','.txt','.model')) or x['rfilename'] in ('LICENSE','README.md')]
    else:
        ending = '-q4_k_m.gguf' if kind == 'chat-gguf' else '-'+kind+'.gguf'
        files = [x for x in metadata['siblings'] if x['rfilename'].endswith(ending)]
    if not files:
        raise ValueError('No pinned artifacts match the requested selection')
    return dict(repository=metadata['id'], revision=metadata['sha'], bytes=sum(x['size'] for x in files), files=files)


def write_json(path, data):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, encoding='utf-8') as stream:
        temp = Path(stream.name)
        try:
            json.dump(data, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def acquire(kind, out, download):
    plan = select(kind)
    print(f"repository {plan['repository']}\nrevision {plan['revision']}\nselected files {len(plan['files'])} bytes {plan['bytes']}")
    out.mkdir(parents=True, exist_ok=True)
    with locked_directory(out):
        plan_path = out/'acquisition-plan.json'
        if plan_path.is_symlink():
            raise ValueError('Invalid plan path')
        if plan_path.exists():
            saved = json.loads(plan_path.read_text())
            if any(saved.get(k) != v for k,v in plan.items()):
                raise ValueError('The directory belongs to a different pinned selection. Choose a new path.')
            if download and not saved.get('download_requested'):
                write_json(plan_path, dict(plan, download_requested=True))
        elif any(out.iterdir()):
            raise ValueError('A nonempty output directory requires its original acquisition plan')
        else:
            write_json(plan_path, dict(plan, download_requested=download))
        if not download:
            print('Offline plan saved to', plan_path)
            return
        remaining = 0
        selected = []
        for entry in plan['files']:
            relative = Path(entry['rfilename'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid catalog path')
            path = out/relative
            if not path.resolve().is_relative_to(out.resolve()):
                raise ValueError(f'Catalog path leaves the output directory: {relative}')
            partial = path.with_name(path.name+'.partial')
            if not path.exists():
                remaining += max(0,entry['size']-(partial.stat().st_size if partial.is_file() else 0))
            selected.append((path, entry))
        if remaining and shutil.disk_usage(out).free < remaining + 1024**3:
            raise ValueError('Insufficient free disk for remaining bytes and the 1 GiB margin')
        hashes = {}
        for path, entry in selected:
            path.parent.mkdir(parents=True,exist_ok=True)
            name = entry['rfilename']
            url = f"https://huggingface.co/{plan['repository']}/resolve/{plan['revision']}/{urllib.parse.quote(name)}"
            lfs = entry.get('lfs')
            hashes[name] = fetch_pinned(url, path, entry['size'], sha256=lfs['sha256'] if lfs else None, git_blob=None if lfs else entry['blobId'])
            print('VERIFIED', name, flush=True)
        receipt = dict(repository=plan['repository'], revision=plan['revision'], files=hashes)
        write_json(out/'acquisition-receipt.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=['Q4_K_M','Q8_0','BF16','training-base','chat-gguf'])
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    try:
        acquire(args.kind, args.out, args.download)
    except (OSError, ValueError, http.client.HTTPException) as error:
        parser.exit(1, str(error)+'\n')
