"""Create an empty-run teaching workspace without changing the source project."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args()
dest=a.destination.expanduser().resolve()
if dest.exists():p.error('destination already exists; choose a new directory')
if dest==SOURCE or SOURCE in dest.parents:p.error('place the new lab beside the extracted companion, not inside it')
selection=json.loads((SOURCE/'reader/workspace-files.json').read_text())['files']
# Check every source before creating the destination; no model, environment or execution-archive copies.
for row in selection:
 f=SOURCE/row['path']
 if not f.is_file() or f.is_symlink():p.error('missing or unsafe source: '+row['path'])
 if hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:p.error('source changed: '+row['path'])
dest.mkdir(parents=True)
for row in selection:
 f=dest/row['path'];f.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/row['path'],f)
shutil.copy2(SOURCE/'reader/workspace-files.json',dest/'reader/workspace-files.json')
(dest/'runs').mkdir(exist_ok=True)
(dest/'reader/workspace-origin.json').write_text(json.dumps({'source':str(SOURCE),'files':selection,'scope':'new workspace; no experiment executed'},indent=2)+'\n')
print(json.dumps({'created':str(dest),'copied_files':len(selection),'next':'Read reader/START_HERE.txt','gpu_executed':False},indent=2))
