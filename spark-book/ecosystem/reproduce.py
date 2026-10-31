"""Execute all CPU controls and retain complete stdout/stderr plus exit status."""
import argparse,subprocess,sys,shlex,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent.parent
    out=a.out.resolve();rel=lambda n:str((out/n).relative_to(root)) if (out/n).is_relative_to(root) else str(out/n)
    commands=[('01-fixture',['fixture']),('02-analytics',['analytics','--src',rel('fixture')]),
      ('03-onnx',['onnx','--src',rel('fixture')]),('04-ort',['ort','--src',rel('fixture'),'--models',rel('onnx')]),
      ('05-quantize',['quantize','--src',rel('fixture')]),('06-data',['data']),('07-lora',['lora']),('08-inventory',['inventory'])]
    statuses=[]
    for name,arguments in commands:
        command=[sys.executable,'ecosystem/lab.py',*arguments,'--out',rel(arguments[0])]
        result=subprocess.run(command,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        displayed=['python',*command[1:]]
        transcript='$ '+shlex.join(displayed)+'\n'+result.stdout+f'[exit {result.returncode}]\n'
        (out/(name+'.txt')).write_text(transcript);print(transcript,end='')
        statuses.append(dict(name=name,exit=result.returncode))
        if result.returncode:raise SystemExit(result.returncode)
    (out/'commands.json').write_text(json.dumps(statuses,indent=2)+'\n')

if __name__=='__main__':main()
