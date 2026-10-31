"""Resolve the publisher once; all later runs use the saved commit and local bytes."""
import argparse,json,re
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
from common import save_json
p=argparse.ArgumentParser();p.add_argument('--repo',default='sentence-transformers/all-MiniLM-L6-v2');p.add_argument('--out',default='embeddings/models/minilm');a=p.parse_args()
out=Path(a.out);manifest=out/'book-source.json'
if manifest.exists():
    source=json.loads(manifest.read_text());assert source['repo']==a.repo
else:
    source=dict(repo=a.repo,revision=HfApi().model_info(a.repo).sha)
assert re.fullmatch('[0-9a-f]{40}',source['revision'])
snapshot_download(repo_id=source['repo'],revision=source['revision'],local_dir=out,
    allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/*'],ignore_patterns=['onnx/*','openvino/*'])
save_json(manifest,source)
print('model',source['repo']);print('revision',source['revision']);print('saved',a.out)
