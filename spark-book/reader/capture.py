"""Capture an explicitly chosen foreground command to a new directory."""
import argparse,datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout',type=float,default=300);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd:p.error('supply a command after --')
if a.timeout<=0:p.error('timeout must be positive')
root=Path.cwd().resolve();out=a.out.expanduser().resolve()
if root not in out.parents or out.relative_to(root).parts[0]!='runs':p.error('output must be a new runs/ directory in the current lab')
if out.exists():p.error('output already exists; choose a new name')
out.mkdir(parents=True);start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic();timed_out=False;interrupted=False;code=127
with (out/'stdout.txt').open('wb') as stdout,(out/'stderr.txt').open('wb') as stderr:
 try:
  child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
  try:code=child.wait(timeout=a.timeout)
  except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
   timed_out=isinstance(exc,subprocess.TimeoutExpired);interrupted=not timed_out
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=124 if timed_out else 130
 except OSError as exc:stderr.write((str(exc)+'\n').encode())
record=dict(origin='NEW READER EXECUTION',argv=cmd,cwd=str(root),started_utc=start,elapsed_seconds=time.monotonic()-t,exit_code=code,timed_out=timed_out,interrupted=interrupted,scope='command capture; exit code alone does not establish exercise success')
(out/'metadata.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2));raise SystemExit(code if code>=0 else 128-code)
