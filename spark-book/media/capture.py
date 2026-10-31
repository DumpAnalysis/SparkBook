"""Capture a target command verbatim, preserving failures, without a shell pipe."""
import argparse,json,subprocess,sys,time,shlex
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('supply a command after --')
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'transcript.txt').open('w') as log:
        header='$ '+shlex.join(command)+'\n';print(header,end='');log.write(header);log.flush()
        start=time.time_ns()
        try:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:print(line,end='');log.write(line);log.flush()
            result=child.wait()
        except OSError as e:
            result=127;log.write(f'{type(e).__name__}: {e}\n');print(e)
        log.write(f'[exit {result}]\n')
    (a.out/'command.json').write_text(json.dumps(dict(argv=command,start_ns=start,end_ns=time.time_ns(),exit=result),indent=2)+'\n')
    raise SystemExit(result)

if __name__=='__main__':main()
