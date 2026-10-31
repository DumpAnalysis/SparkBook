"""Capture full driver stdout/stderr; stop on failures without hiding evidence."""
import argparse,platform,subprocess,sys
from importlib.metadata import version
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parent.parent;out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    commands=[]
    for i,case in enumerate(['contract','lifecycle','load','sse','qualify'],1):
        commands.append((f'{i:02}-{case}',['serving/lab.py',case,'--out',str(out/f'{i:02}-{case}')]))
    commands += [('06-repository',['serving/build_repository.py','--out',str(out/'repository')]),
      ('07-backend',['serving/backend_control.py','--repository',str(out/'repository'),'--out',str(out/'07-backend')]),
      ('08-onnx-repository',['serving/build_repository.py','--onnx','--out',str(out/'onnx-repository')]),
      ('09-onnx',['serving/onnx_control.py','--repository',str(out/'onnx-repository'),'--out',str(out/'09-onnx')]),
      # Historical tag reproduces the retained EXECUTED fixture transcript.
      # New deployment exercises use book-fixture; do not rewrite old evidence.
      ('10-manifest',['serving/deploy.py','fixture','--image','local/spark-serving:book-v13','--out',str(out/'fixture.yaml')])]
    blocks=[]
    for name,args in commands:
        r=subprocess.run([sys.executable,*args],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        block='$ python '+' '.join(args)+'\n'+r.stdout+f'[exit {r.returncode}]\n'
        (out/(name+'.txt')).write_text(block);blocks.append(block);print(block,end='',flush=True)
        (out/'transcripts.txt').write_text('\n'.join(blocks))
        if r.returncode:raise SystemExit(r.returncode)
    (out/'environment.txt').write_text(f'Python {platform.python_version()}\n{platform.platform()}\nmachine={platform.machine()}\n'+''.join(f'{n}={version(n)}\n' for n in ['numpy','PyYAML','onnx','protobuf','ml_dtypes']))
if __name__=='__main__':main()
