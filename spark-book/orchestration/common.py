"""Shared contracts for the WSL2 orchestration laboratory."""
import contextlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch
from urllib.parse import urlsplit

# No inherited tracing destination is used by the local teaching programs.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
os.environ['LANGSMITH_OTEL_ENABLED'] = 'false'
os.environ['PYDANTIC_AI_NO_BANNER'] = '1'

EVIDENCE = {
    'gpu_memory': {'source': 'SIMULATED', 'total_gib': 30, 'used_gib': 28},
    'request': {'source': 'SIMULATED', 'context_tokens': 4096, 'workers': 1},
}

def save(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.write('\n')

def endpoint(value):
    u = urlsplit(value)
    if (u.scheme != 'http' or u.hostname != '127.0.0.1' or
        u.username is not None or u.password is not None or
        u.path.rstrip('/') != '/v1' or u.query or u.fragment or
        u.port is None or not 1024 <= u.port <= 65535):
        raise ValueError('use http://127.0.0.1:PORT/v1, PORT 1024..65535')
    return value.rstrip('/')

@contextlib.contextmanager
def no_network():
    attempts = []
    def denied(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError('offline laboratory forbids network connections')
    with patch.object(socket.socket, 'connect', denied), patch.object(socket.socket, 'connect_ex', denied):
        yield
    if attempts:
        raise RuntimeError(f'{len(attempts)} attempted connections in offline control')

def read_evidence(key):
    if key not in EVIDENCE:
        raise ValueError('evidence key is not allowed')
    return json.dumps(EVIDENCE[key], sort_keys=True)
