#!/usr/bin/env python3
"""Capture Exercise 251 stdout, stderr, command identity and exit in a new directory."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allocate-gib', type=int, default=0)
    p.add_argument('--host-note', default='')
    p.add_argument('--timeout-seconds', type=int, default=180)
    a = p.parse_args()
    if not 0 <= a.allocate_gib <= 18:
        p.error('--allocate-gib must be from 0 through 18')
    if a.allocate_gib and not a.host_note.strip():
        p.error('--host-note must record actual host observations for allocation')
    if not 10 <= a.timeout_seconds <= 300:
        p.error('--timeout-seconds must be from 10 through 300')
    source = Path(__file__).with_name('probe.py').resolve()
    argv = [sys.executable, '-B', str(source), '--allocate-gib', str(a.allocate_gib),
            '--host-note', a.host_note]
    a.out.mkdir(parents=True, exist_ok=False)
    metadata = dict(started_utc=utc(), cwd=os.getcwd(), argv=argv,
                    command_shell=shlex.join(argv), timeout_seconds=a.timeout_seconds,
                    sources={x.name: hashlib.sha256(x.read_bytes()).hexdigest()
                             for x in (source, Path(__file__))})
    (a.out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    timed_out = False
    interrupted = False
    start = time.perf_counter()
    with (a.out/'stdout.txt').open('wb') as out, (a.out/'stderr.txt').open('wb') as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=True)
        metadata['pid'] = process.pid
        try:
            process.wait(timeout=a.timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            timed_out = isinstance(exc, subprocess.TimeoutExpired)
            interrupted = not timed_out
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    metadata.update(ended_utc=utc(), child_exit_code=process.returncode,
                    elapsed_seconds=time.perf_counter()-start,
                    timed_out=timed_out, interrupted=interrupted)
    code = 124 if timed_out else 130 if interrupted else process.returncode
    if code < 0:
        code = 128 - code
    metadata['wrapper_exit_code'] = code
    metadata['streams'] = {name: {'bytes': (a.out/name).stat().st_size,
                                   'sha256': hashlib.sha256((a.out/name).read_bytes()).hexdigest()}
                           for name in ('stdout.txt', 'stderr.txt')}
    (a.out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'output_directory': str(a.out), 'wrapper_exit_code': code,
                      'child_exit_code': process.returncode, 'timed_out': timed_out}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
