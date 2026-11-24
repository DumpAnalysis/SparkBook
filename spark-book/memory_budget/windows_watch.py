"""Read the Windows memory counters from WSL and keep a timestamped record."""
import datetime as dt
import json
import math
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time

GIB = 1024**3
HERE = Path(__file__).resolve().parent


class GuardStop(RuntimeError):
    pass


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def guest_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise GuardStop('WSL MemAvailable unavailable')


def validate(row, floor, now):
    try:
        available, committed, limit = (row[k] for k in ('available_bytes', 'committed_bytes', 'commit_limit_bytes'))
        if any(type(v) is not int or v < 0 for v in (available, committed, limit)):
            raise ValueError('invalid byte count')
        elapsed = float(row['query_seconds'])
        stamp = dt.datetime.fromisoformat(row['utc'].replace('Z','+00:00'))
        if stamp.tzinfo is None:
            raise ValueError('missing UTC offset')
        observed = stamp.timestamp()
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise GuardStop('Windows sample was incomplete or malformed') from exc
    if not math.isfinite(elapsed) or not 0 <= elapsed <= 5:
        raise GuardStop('Windows counter query exceeded five seconds or returned invalid timing')
    if not -1 <= now-observed <= 8:
        raise GuardStop('Windows sample was stale or had a future timestamp')
    if available < floor:
        raise GuardStop('Windows available physical memory below floor')
    if limit - committed < floor:
        raise GuardStop('Windows commit headroom below floor')


def stop_owned(process, grace=3):
    """End the process group created for this child, even if its leader exited."""
    if process is None:
        return
    def send(sig):
        try: os.killpg(process.pid, sig)
        except ProcessLookupError: return False
        return True
    if send(signal.SIGTERM):
        deadline=time.monotonic()+grace
        while time.monotonic()<deadline:
            process.poll()
            if not send(0): break
            time.sleep(.05)
        else:
            send(signal.SIGKILL)
    process.wait(timeout=5)


class WindowsWatch:
    def __init__(self, out, floor):
        self.out, self.floor = out, floor
        self.count = 0
        self.latest = None
        self.rows = queue.SimpleQueue()
        self.proc = None

    def __enter__(self):
        self.started = self.received = time.monotonic()
        self.stop_file = self.out/'stop-windows-monitor'
        stop_path = subprocess.check_output(['wslpath','-w',str(self.stop_file)],text=True,timeout=5).strip()
        exe = shutil.which('powershell.exe') or '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
        cmd = "$stopFile='" + stop_path.replace("'","''") + "'\n" + (HERE/'windows_watch.ps1').read_text()
        with (self.out/'host-stderr.txt').open('xb') as errors:
            self.proc = subprocess.Popen([exe,'-NoProfile','-NonInteractive','-Command',cmd],
                                         stdout=subprocess.PIPE,stderr=errors,start_new_session=True)
        self.reader = threading.Thread(target=self._read,daemon=True)
        self.reader.start()
        return self

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.rows.put((time.monotonic(),line))
        finally:
            self.proc.stdout.close()

    def check(self):
        with (self.out/'host.jsonl').open('ab') as output:
            while True:
                try: arrival, raw = self.rows.get_nowait()
                except queue.Empty: break
                output.write(raw)
                try: row=json.loads(raw.decode('utf-8-sig'))
                except (UnicodeError, ValueError) as exc: raise GuardStop('Unreadable Windows sample') from exc
                validate(row,self.floor,time.time())
                self.latest, self.received = row, arrival
                self.count += 1
        if self.proc.poll() is not None:
            raise GuardStop('Windows monitor exited before workload completion')
        if time.monotonic()-self.received>8:
            raise GuardStop('No fresh Windows sample for eight seconds')
        if self.latest is not None:
            validate(self.latest,self.floor,time.time())
        if guest_available()<2*GIB:
            raise GuardStop('WSL available RAM below 2 GiB')

    def baseline(self):
        while self.count<3:
            self.check()
            if self.count<3: time.sleep(.1)
        return self.latest

    def __exit__(self,*unused):
        try:
            self.stop_file.touch()
            try: self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired: pass
        finally:
            stop_owned(self.proc)
            self.reader.join(timeout=2)
