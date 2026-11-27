"""Prepare four bounded evidence fields from this laboratory's own files."""
import argparse
import datetime
import json
import pathlib
import re
import shlex
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


def launch_evidence(directory):
    folder = (root / directory).resolve()
    if not folder.is_relative_to(root/'runs'):
        raise ValueError('Server evidence must be under runs/')
    # Verbose logs can echo model prompts and replies. Accept startup events
    # only from the native logger, before its first listen event.
    messages = []
    pattern = re.compile(r'^(?:[0-9.]+ [A-Z] )?(?:load_tensors:\s+(offloaded \d+/\d+ layers to GPU)|srv\s+server_run_i:\s+(model loaded)|srv\s+server_on_li:\s+(listening on http://127\.0\.0\.1:\d+))$')
    stderr = folder/'stderr.txt'
    if stderr.is_file():
        with stderr.open(errors='replace') as log:
            for line in log:
                match = pattern.fullmatch(line.strip())
                if match:
                    messages.append(next(x for x in match.groups() if x is not None))
                    if match[3]:break
    argv = shlex.split((folder/'command.sh').read_text()) if (folder/'command.sh').is_file() else []
    flags = [flag+' '+argv[argv.index(flag)+1] for flag in ('-c','-np','-ngl','--port')
             if flag in argv and argv.index(flag)+1<len(argv)]
    return ('\n'.join(messages) or 'UNAVAILABLE: no recognized startup event',
            'Launch settings: '+' '.join(flags) if flags else 'UNAVAILABLE: launch settings')


def exchange(relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root / 'runs'):
        raise ValueError('Request receipt must be inside runs/')
    if not path.is_file():
        return 'UNAVAILABLE: no request receipt'
    receipt = json.loads(path.read_text())
    status = receipt.get('status')
    if type(status) is not int:
        raise ValueError('Request receipt has no numeric HTTP status')
    # Only transport metadata can serve as a measurement of this exchange.
    fields = [f'HTTP status {status}']
    body = receipt.get('body')
    if isinstance(body, dict):
        for key in ('prompt_tokens','completion_tokens'):
            number = (body.get('usage') or {}).get(key)
            if type(number) is int:
                fields.append(f'{key}={number}')
    stamp = datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat()
    return '\n'.join(['SOURCE '+str(path.relative_to(root)), 'MODIFIED '+stamp, *fields])


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
    runtime, limits = launch_evidence(directory)
else:
    runtime = 'UNAVAILABLE: no saved server state'
    limits = 'UNAVAILABLE: no saved launch command'
snapshot = {
    'inventory': read_owned(args.inventory),
    'runtime': (f'SOURCE {directory}/stderr.txt\n' if state_path.exists() else '')+runtime,
    'request': exchange(args.request),
    'limits': (f'SOURCE {directory}/command.sh\n' if state_path.exists() else '')+limits
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
