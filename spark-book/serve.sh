#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p runs
action=${1:-start}
state=runs/linux-server-state.json
if [[ "$action" == stop ]]; then
  python3 - "$state" <<'PY'
import json, os, pathlib, signal, sys, time
p = pathlib.Path(sys.argv[1])
s = json.loads(p.read_text())
proc = pathlib.Path('/proc') / str(s['pid'])
if proc.exists():
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    if stat[19] != s['start_ticks']:
        raise SystemExit('PID identity changed; stop cancelled')
    if stat[0] != 'Z':
        if os.readlink(proc / 'exe') != s['executable']:
            raise SystemExit('Executable identity changed; stop cancelled')
        os.kill(s['pid'], signal.SIGTERM)
        for _ in range(100):
            if not proc.exists():
                break
            try:
                current = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                break
            if current[0] == 'Z':
                break
            time.sleep(.1)
        else:
            raise SystemExit('Server has not stopped; inspect its state')
p.unlink()
print('server=stopped')
PY
  exit
fi
[[ "$action" == start ]] || { echo 'Use start or stop'; exit 2; }
[[ ! -e "$state" ]] || { echo 'Stop the recorded server first'; exit 2; }
ctx=${2:-4096}
slots=${3:-1}
layers=${4:-99}
device=${5:-gpu}
quant=${6:-q4_k_m}
modeldir=${7:-models/$quant}
[[ "$ctx" =~ ^[0-9]+$ && "$slots" =~ ^[0-9]+$ &&
   "$layers" =~ ^[0-9]+$ ]] || exit 2
[[ "$device" == gpu || "$device" == cpu ]] || exit 2
[[ "$quant" == q4_k_m || "$quant" == q8_0 ]] || exit 2
python3 - <<'PY'
import socket
with socket.socket() as s:
    try:
        s.bind(('127.0.0.1', 8080))
    except OSError:
        raise SystemExit('Port 8080 is occupied; inspect its owner')
PY
mapfile -t binaries < <(find linux-bin -type f -name llama-server)
[[ ${#binaries[@]} == 1 ]] || { echo 'Expected one server binary'; exit 2; }
exe=$(realpath "${binaries[0]}")
chmod u+x "$exe"
mapfile -t shards < <(find "$modeldir" -maxdepth 1 -name '*-00001-of-*.gguf')
[[ ${#shards[@]} == 1 ]] || { echo 'Expected first model shard'; exit 2; }
model=$(realpath "${shards[0]}")
# The distribution can put CUDA libraries in nested directories.
mapfile -t libdirs < <(find linux-bin -type f -name '*.so*' -printf '%h\n' | sort -u)
libpath=$(IFS=:; echo "${libdirs[*]}")
export LD_LIBRARY_PATH="$libpath${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
run=$(mktemp -d runs/server-XXXXXX)
args=(-m "$model" --alias local-model --host 127.0.0.1
      --port 8080 -c "$ctx" -np "$slots" --fit off --no-context-shift
      -b 512 -ub 128 -ctk f16 -ctv f16)
if [[ "$device" == cpu ]]; then
  args+=(--device none -ngl 0 --no-kv-offload)
else
  args+=(-ngl "$layers")
fi
printf '%q ' "$exe" "${args[@]}" > "$run/command.sh"
printf '\n' >> "$run/command.sh"
nohup "$exe" "${args[@]}" > "$run/stdout.txt" 2> "$run/stderr.txt" &
server_pid=$!
python3 - "$server_pid" "$exe" "$run" "$state" <<'PY'
import json, pathlib, sys, time, urllib.request
pid, exe, run, dest = sys.argv[1:]
proc = pathlib.Path('/proc') / pid
try:
    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
except FileNotFoundError:
    raise SystemExit('Server exited; inspect ' + run)
pathlib.Path(dest).write_text(json.dumps({
    'pid': int(pid), 'executable': exe, 'start_ticks': stat[19],
    'directory': run}, indent=2))
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(120):
    try:
        with opener.open('http://127.0.0.1:8080/health', timeout=2) as r:
            if json.load(r).get('status') == 'ok':
                print('server=ready endpoint=http://127.0.0.1:8080')
                print('Complete process output retained in runs/server-*.')
                break
    except Exception:
        if not proc.exists():
            raise SystemExit('Server exited; inspect ' + run)
    time.sleep(1)
else:
    raise SystemExit('Startup timed out; inspect ' + run)
PY
