"""Write argv-based plans; run from the spark-book directory."""
import argparse,sys
from pathlib import Path
from monitor import dump
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('kind',choices=['cpu','idle','inference','embedding','training'])
p.add_argument('--out',required=True);p.add_argument('--result',default='{run}/workload')
p.add_argument('--batch',type=int,default=8);a=p.parse_args()
if Path(a.out).exists():p.error('plan exists; use a new filename')
if a.batch<1:p.error('batch must be positive')
python=sys.executable
idle=lambda label,seconds:{'label':label,'argv':[python,'-c',f'import time; time.sleep({seconds})'],'timeout':seconds+10}
plan=[idle('idle',5)]
if a.kind=='cpu':plan=[idle('baseline',1),{'label':'cpu work','argv':[python,'-c','print(sum(i*i for i in range(2000000)))'],'timeout':30},idle('after',1)]
elif a.kind=='idle':plan=[idle('idle',10)]
elif a.kind=='inference':
    plan += [{'label':'requests','argv':[python,'spark_lab.py','bench','prompt.txt','--runs','5','--workers','1','--tokens','128','--out',a.result+'.json'],'timeout':600},idle('after',5)]
elif a.kind=='embedding':
    plan += [{'label':'load and encode','argv':[python,'embeddings/encode.py','--model','embeddings/models/minilm','--device','cuda','--batch',str(a.batch),'--out',a.result],'timeout':600},idle('after',5)]
else:
    plan += [{'label':'load and train','argv':[python,'embeddings/finetune.py','--model','embeddings/models/minilm','--device','cuda','--batch',str(a.batch),'--epochs','3','--out',a.result],'timeout':1200},idle('after',5)]
Path(a.out).parent.mkdir(parents=True,exist_ok=True);dump(a.out,plan)
print('plan='+a.out);print('phases='+str(len(plan)))
