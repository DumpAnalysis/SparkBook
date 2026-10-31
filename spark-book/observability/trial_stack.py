"""Own a fresh local monitoring stack for one exercise and retain its output."""
from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError

from client import query

HERE = Path(__file__).resolve().parent


def require_free_ports():
    for port in (3000, 9090, 9400, 8089):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('127.0.0.1', port))
            except OSError as exc:
                raise RuntimeError(f'Port {port} is occupied. Stop its owner before the comparison.') from exc


def series(expr):
    return query(expr)['data']['result']


def wait_for(owner, condition, description, timeout=100):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if owner.poll() is not None:
            raise RuntimeError('Monitoring stack exited. Inspect stack-output.txt.')
        try:
            if condition():
                return
        except (OSError, URLError):
            pass
        time.sleep(1)
    raise RuntimeError(f'Timed out waiting for {description}')


@contextmanager
def running_stack(tools, dest, options):
    require_free_ports()
    with (dest/'stack-output.txt').open('x') as log:
        args = [sys.executable, str(HERE/'stack.py'), '--tools', str(tools.resolve()),
                '--run-dir', str(dest/'stack'), *options]
        owner = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield owner
        finally:
            if owner.poll() is None:
                owner.terminate()
            owner.wait(timeout=100)
