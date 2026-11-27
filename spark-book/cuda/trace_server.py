"""Run the private loopback inference owner as a child of Nsight Systems."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if (root / 'runs/linux-server-state.json').exists():
        parser.error('Stop the current lab server first with bash serve.sh stop')
    server, = (root / 'linux-bin').rglob('llama-server')
    model, = (root / 'models/q4_k_m').glob('*00001-of-*.gguf')
    libs = sorted({str(p.parent) for p in (root / 'linux-bin').rglob('*.so*')})
    env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(libs + [os.environ.get('LD_LIBRARY_PATH', '')]).rstrip(':'))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    alias = f'trace-{os.getpid()}-{port}'
    base = f'http://127.0.0.1:{port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cmd = [str(server), '-m', str(model), '--alias', alias, '--host', '127.0.0.1',
           '--port', str(port), '-ngl', '99', '-c', '4096']
    args.out.mkdir(parents=True)
    with (args.out / 'server.log').open('w') as log, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT) as child:
        try:
            deadline = time.monotonic() + 90
            while child.poll() is None and time.monotonic() < deadline:
                try:
                    with opener.open(base + '/v1/models', timeout=2) as response:
                        if alias in {x['id'] for x in json.load(response)['data']}:
                            break
                except urllib.error.URLError:
                    pass
                time.sleep(.25)
            else:
                raise RuntimeError('Profiled server did not publish its model alias. See server.log.')
            body = {'model': alias, 'messages': [{'role': 'user', 'content': 'Reply with READY.'}],
                    'temperature': 0, 'max_tokens': 16}
            request = urllib.request.Request(base + '/v1/chat/completions', json.dumps(body).encode(),
                                             {'Content-Type': 'application/json'})
            start = time.time_ns()
            with opener.open(request, timeout=60) as response:
                result = json.load(response)
                end = time.time_ns()
                if not result['choices'][0]['message']['content'] or result['model'] != alias:
                    raise RuntimeError('Missing completion or unexpected model identity')
                evidence = dict(pid=child.pid, command=cmd, url=request.full_url, http=response.status,
                                start_utc_ns=start, end_utc_ns=end, request=body, response=result)
                (args.out / 'request.json').write_text(json.dumps(evidence, indent=2) + '\n')
                print(f'pid={child.pid} HTTP={response.status} model={alias}', flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()


if __name__ == '__main__':
    main()
