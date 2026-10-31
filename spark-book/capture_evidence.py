"""Prepare four bounded evidence fields from this laboratory's own files."""
import argparse
import datetime
import json
import pathlib
from spark_lab import save

root = pathlib.Path(__file__).resolve().parent


def read_owned(relative, limit=6000):
    path = (root / relative).resolve()
    if not path.is_relative_to(root / 'runs'):
        raise ValueError('Evidence path is outside the laboratory runs directory')
    if not path.is_file():
        return 'UNAVAILABLE: ' + relative
    stamp = datetime.datetime.fromtimestamp(
        path.stat().st_mtime, datetime.timezone.utc).isoformat()
    text = path.read_text(encoding='utf-8', errors='replace')
    if len(text) > limit:
        text = '[EARLIER CONTENT OMITTED FROM SNAPSHOT]\n' + text[-limit:]
    return f'SOURCE {relative}\nMODIFIED {stamp}\n{text}'


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inventory', default='runs/inventory.txt')
parser.add_argument('--request', default='runs/first.json')
parser.add_argument('--out', help='write a new snapshot inside runs/')
args = parser.parse_args()
out = (root / (args.out or 'runs/evidence.json')).resolve()
if not out.is_relative_to(root / 'runs'):
    parser.error('output must be inside the lab runs directory')
state_path = root / 'runs/linux-server-state.json' 
if state_path.exists():
    state = json.loads(state_path.read_text())
    directory = state['directory']
    runtime = read_owned(directory + '/stderr.txt')
    limits = read_owned(directory + '/command.sh')
else:
    runtime = 'UNAVAILABLE: no saved server state'
    limits = 'UNAVAILABLE: no saved launch command'
snapshot = {
    'inventory': read_owned(args.inventory),
    'runtime': runtime,
    'request': read_owned(args.request),
    'limits': limits
}
if args.out:
    try:
        with out.open('x', encoding='utf-8') as dest:
            json.dump(snapshot, dest, indent=2, ensure_ascii=False)
            dest.write('\n')
    except FileExistsError:
        parser.error('output already exists; choose a new snapshot')
else:
    save(out, snapshot)
print('snapshot=' + str(out.relative_to(root)))
print('keys=inventory,runtime,request,limits')
print('Snapshot is bounded; original files remain unchanged.')
