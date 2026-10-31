"""Bounded WSL2 telemetry and host-command phases. No driver changes."""
import argparse, csv, io, json, math, os, platform, shutil, signal
import subprocess, threading, time
from datetime import datetime, timezone
from pathlib import Path

FIELDS = {
    'utilization.gpu': ('%', 0, 100),
    'utilization.memory': ('%', 0, 100),
    'memory.used': ('MiB', 0, None),
    'memory.free': ('MiB', 0, None),
    'memory.total': ('MiB', 0, None),
    'temperature.gpu': ('C', -40, 200),
    'power.draw': ('W', 0, None),
    'power.limit': ('W', 0, None),
    'clocks.current.sm': ('MHz', 0, None),
    'clocks.current.memory': ('MHz', 0, None),
}

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

def utc():
    return datetime.now(timezone.utc).isoformat()

def value(field, raw):
    text = raw.strip()
    if text.lower() in {'n/a', '[n/a]', 'not supported', '[not supported]', ''}:
        return {'value': None, 'status': 'unavailable'}
    try:
        n = float(text)
        _, lo, hi = FIELDS[field]
        if not math.isfinite(n) or n < lo or (hi is not None and n > hi):
            raise ValueError('out of range')
        return {'value': n, 'status': 'ok'}
    except ValueError:
        return {'value': None, 'status': 'parse_error'}

def parse(fields, raw):
    rows = list(csv.reader(io.StringIO(raw.strip()), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(fields):
        return {f: {'value': None, 'status': 'shape_error'} for f in fields}
    return {f: value(f, x) for f, x in zip(fields, rows[0])}

def call(argv, timeout=5):
    start = time.monotonic_ns()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        result = {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except subprocess.TimeoutExpired as e:
        def txt(x): return x.decode(errors='replace') if isinstance(x, bytes) else x or ''
        result = {'returncode': None, 'stdout': txt(e.stdout), 'stderr': txt(e.stderr), 'error': 'timeout'}
    except OSError as e:
        result = {'returncode': None, 'stdout': '', 'stderr': str(e), 'error': 'launch_error'}
    return dict(result, argv=argv, start_ns=start, end_ns=time.monotonic_ns())

def query(smi, gpu, fields):
    r = call([smi, '-i', gpu, '--query-gpu='+','.join(fields), '--format=csv,noheader,nounits'])
    r['fields'] = fields
    r['parsed'] = (parse(fields, r['stdout']) if r['returncode'] == 0 else
                   {f: {'value': None, 'status': 'query_error'} for f in fields})
    return r

def guest_memory(path='/proc/meminfo'):
    wanted = {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'}
    result = {k: None for k in wanted}
    try:
        for line in Path(path).read_text().splitlines():
            key, rest = line.split(':', 1)
            if key in wanted:
                parts = rest.split()
                if len(parts) == 2 and parts[1] == 'kB':
                    result[key] = int(parts[0]) / 1024
    except (OSError, ValueError):
        pass
    return result

def validate_plan(obj):
    if not isinstance(obj, list) or not 1 <= len(obj) <= 32:
        raise ValueError('plan must have 1..32 phases')
    for p in obj:
        if not isinstance(p, dict) or not isinstance(p.get('label'), str) or not p['label']:
            raise ValueError('each phase requires a label')
        argv = p.get('argv')
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
            raise ValueError('argv must be a nonempty string list; no shell interpolation')
        t = p.get('timeout', 300)
        if not isinstance(t, (int, float)) or not math.isfinite(t) or not 0 < t <= 3600:
            raise ValueError('phase timeout must be in (0, 3600] seconds')
    return obj

def stop_group(p):
    try: os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: p.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try: os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        p.wait()

def run_phase(phase, out, index, origin):
    argv = [x.replace('{run}', str(out.resolve())) for x in phase['argv']]
    event = {'index': index, 'label': phase['label'], 'argv': argv,
             'scope': 'host command including startup and exit',
             'start_s': (time.monotonic_ns()-origin)/1e9, 'utc_start': utc()}
    p = None
    with (out/f'phase-{index:02d}.stdout.txt').open('w') as stdout, (out/f'phase-{index:02d}.stderr.txt').open('w') as stderr:
        try:
            p = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
            event['returncode'] = p.wait(timeout=phase.get('timeout', 300))
        except subprocess.TimeoutExpired:
            stop_group(p); event.update(returncode=124, error='timeout')
        except KeyboardInterrupt:
            if p: stop_group(p)
            event.update(returncode=130, error='interrupted')
        except OSError as e:
            stderr.write(str(e)+'\n'); event.update(returncode=127, error='launch_error')
    event['end_s'] = (time.monotonic_ns()-origin)/1e9
    return event

def select_provider(windows_host=False, smi_path=None):
    if windows_host:
        smi = smi_path or shutil.which('nvidia-smi.exe')
        if not smi or not str(smi).lower().endswith('.exe'):
            raise ValueError('Windows host observations require nvidia-smi.exe; pass its WSL path with --smi')
        return str(smi), 'Windows host through WSL interoperability'
    return str(smi_path or shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'), 'WSL guest driver interface'

def capture(out, plan, interval=1, gpu='0', no_gpu=False, without_sampling=False,
            windows_host=False, smi_path=None):
    smi, provider_scope = select_provider(windows_host, smi_path)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    raw = (out/'raw-queries.jsonl').open('w')
    def log(r):
        raw.write(json.dumps(r, allow_nan=False)+'\n'); raw.flush()
    probes = {}
    meta = {'schema': 1, 'evidence': 'MEASURED local host session; GPU values only where status=ok',
            'platform': platform.platform(), 'machine': platform.machine(), 'cwd': os.getcwd(),
            'python': platform.python_version(), 'gpu_selector': gpu, 'smi': smi,
            'interval_s': interval, 'no_gpu': no_gpu, 'without_sampling': without_sampling,
            'gpu_provider_scope': provider_scope,
            'fields': {k: {'unit': v[0]} for k, v in FIELDS.items()},
            'plan': plan, 'source_sha256': __import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()}
    active = []
    if not no_gpu and not without_sampling:
        for argv in [[smi, '--version'], [smi, '--help-query-gpu'],
                     [smi, '-i', gpu, '--query-gpu=uuid,name,driver_version', '--format=csv,noheader']]:
            r = call(argv); log(r)
        for f in FIELDS:
            r = query(smi, gpu, [f]); log(r); probes[f] = r['parsed'][f]
            # N/A is a valid response shape and may become available under load.
            if r['returncode'] == 0 and probes[f]['status'] in {'ok','unavailable'}: active.append(f)
    meta['probe'] = probes; meta['active_fields'] = active
    origin = time.monotonic_ns(); meta.update(origin_ns=origin, utc_origin=utc())
    dump(out/'metadata.json', meta)
    samples = []; stop = threading.Event(); errors = []
    def sample_loop():
        due = time.monotonic()
        try:
            with (out/'samples.jsonl').open('w') as target:
                while not stop.is_set():
                    begin = time.monotonic_ns()
                    readings = {f: {'value': None, 'status': 'disabled' if no_gpu else probes.get(f, {}).get('status', 'unavailable')} for f in FIELDS}
                    if active:
                        r = query(smi, gpu, active); log(r); readings.update(r['parsed'])
                    end = time.monotonic_ns()
                    row = {'seq': len(samples), 'utc': utc(), 'start_s': (begin-origin)/1e9,
                           'end_s': (end-origin)/1e9, 't_s': ((begin+end)/2-origin)/1e9,
                           'query_ms': (end-begin)/1e6, 'gpu': readings,
                           'guest_MiB': guest_memory()}
                    samples.append(row); target.write(json.dumps(row, allow_nan=False)+'\n'); target.flush()
                    now = time.monotonic()
                    due += interval
                    # Skip missed deadlines instead of emitting a burst of catch-up samples.
                    if due < now: due += (math.floor((now-due)/interval)+1)*interval
                    stop.wait(max(0, due-time.monotonic()))
        except Exception as e:
            errors.append(repr(e)); stop.set()
    thread = threading.Thread(target=sample_loop, daemon=True)
    if not without_sampling: thread.start()
    events = []
    try:
        for i, phase in enumerate(plan):
            e = run_phase(phase, out, i, origin); events.append(e); dump(out/'events.json', events)
            if e['returncode'] or errors: break
    finally:
        stop.set()
        if thread.is_alive(): thread.join()
        raw.close()
    dump(out/'session.json', {'samples': len(samples), 'completed_phases': len(events),
                             'planned_phases': len(plan), 'collector_errors': errors,
                             'success': len(events)==len(plan) and all(e['returncode']==0 for e in events) and not errors})
    print('saved='+str(out))
    print(f'phases={len(events)}/{len(plan)} samples={len(samples)}')
    print('gpu_values='+('disabled' if no_gpu or without_sampling else 'inspect field status and raw queries'))
    return 0 if len(events)==len(plan) and all(e['returncode']==0 for e in events) and not errors else 1

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--out', required=True)
    p.add_argument('--interval', type=float, default=1); p.add_argument('--gpu', default='0')
    p.add_argument('--no-gpu', action='store_true', help='CPU control: collect guest observations only')
    p.add_argument('--without-sampling', action='store_true', help='same phase wrapper without GPU/guest collector')
    p.add_argument('--windows-host', action='store_true', help='query the Windows nvidia-smi.exe from WSL; workloads remain in Linux')
    p.add_argument('--smi', help='explicit query executable path, retained in metadata')
    a=p.parse_args()
    if not math.isfinite(a.interval) or not .1 <= a.interval <= 60: p.error('interval must be 0.1..60 seconds')
    plan=validate_plan(json.loads(Path(a.plan).read_text()))
    raise SystemExit(capture(a.out, plan, a.interval, a.gpu, a.no_gpu, a.without_sampling,
                            a.windows_host, a.smi))
