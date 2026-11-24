"""Inspect the pinned 32B Q8 model offline, or verify and download its nine GGUF shards."""
import argparse
import json
from pathlib import Path
import shutil

from setup_wsl import fetch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='download and hash the selected files')
    args = parser.parse_args()
    profile = json.loads((ROOT / 'memory_budget/qwen32-profile.json').read_text())
    parts = profile['shards']
    size = sum(item['bytes'] for item in parts)
    print(f"model={profile['repository']}\nrevision={profile['revision']}\nshards={len(parts)}")
    print(f'weights_bytes={size}\nweights_gib={size / 2**30:.2f}\nDirectory: models/32b-q8_0')
    if not args.download:
        print('Offline plan. Use --download after checking available disk space.')
        return
    dest = ROOT / 'models/32b-q8_0'
    dest.mkdir(parents=True, exist_ok=True)
    remaining = sum(item['bytes'] for item in parts if not (dest / item['name']).exists())
    if shutil.disk_usage(dest).free < remaining + 2 * 2**30:
        raise SystemExit('Insufficient free disk for the remaining shards and a 2 GiB margin.')
    base = f"https://huggingface.co/{profile['repository']}/resolve/{profile['revision']}"
    for item in parts:
        path = dest / item['name']
        fetch(f'{base}/{path.name}', path, pin=item)
        with path.open('rb') as stream:
            if stream.read(4) != b'GGUF':
                raise SystemExit(f'Invalid GGUF signature in {path.name}')
    runs = ROOT / 'runs'
    runs.mkdir(exist_ok=True)
    (runs / 'large-model-manifest.json').write_text(json.dumps({
        'repository': profile['repository'], 'revision': profile['revision'], 'records': parts
    }, indent=2) + '\n')
    print('Verified all shards. The placement test is a separate command.')


if __name__ == '__main__':
    main()
