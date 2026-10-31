"""Generate local Kubernetes manifests; no cluster calls or implicit apply."""
import argparse,json,re
from pathlib import Path
import yaml

def documents(backend,image,model_dir=None,first_shard=None,gpu=False):
    if backend!='fixture' and not re.search(r'@sha256:[0-9a-f]{64}$',image):
        raise ValueError('external backend image must be pinned by sha256 digest')
    if backend in ['llama','vllm','triton'] and (not model_dir or not Path(model_dir).is_absolute()):
        raise ValueError('absolute model/repository directory required')
    if backend=='llama' and (not first_shard or Path(first_shard).name!=first_shard):
        raise ValueError('first-shard must be a basename')
    gpu=gpu or backend in ['llama','vllm']
    name='spark-'+backend;port=8000 if backend in ['vllm','triton'] else 8080
    ready='/readyz';live='/livez';args=[];command=None;volumes=[];mounts=[];env=[]
    if backend=='llama':
        ready='/health';live=None
        args=['-m','/models/'+first_shard,'--alias','local-model','--host','0.0.0.0','--port','8080',
              '-c','4096','-np','1','-ngl','99','--fit','off','--no-context-shift','--metrics']
    if backend=='vllm':
        ready='/health';live=None
        args=['/models','--served-model-name','local-model','--host','0.0.0.0','--port','8000',
              '--max-model-len','2048','--max-num-seqs','2','--tensor-parallel-size','1',
              '--kv-cache-memory-bytes','1073741824','--enforce-eager','--dtype','auto']
        env=[{'name':'HOME','value':'/tmp'},{'name':'XDG_CACHE_HOME','value':'/tmp/cache'},
             {'name':'TRITON_CACHE_DIR','value':'/tmp/triton'},{'name':'HF_HOME','value':'/tmp/hf'},{'name':'VLLM_CACHE_ROOT','value':'/tmp/vllm'}]
    if backend=='triton':
        ready='/v2/health/ready';live='/v2/health/live'
        command=['tritonserver'];args=['--model-repository=/models','--model-control-mode=none','--strict-readiness=true']
    if model_dir:
        volumes.append({'name':'models','hostPath':{'path':model_dir,'type':'Directory'}})
        mounts.append({'name':'models','mountPath':'/models','readOnly':True})
    volumes += [{'name':'tmp','emptyDir':{'sizeLimit':'1Gi'}}]
    mounts += [{'name':'tmp','mountPath':'/tmp'}]
    if backend in ['vllm','triton']:
        volumes.append({'name':'shm','emptyDir':{'medium':'Memory','sizeLimit':'1Gi'}})
        mounts.append({'name':'shm','mountPath':'/dev/shm'})
    limits={'cpu':'1','memory':'512Mi'} if backend=='fixture' else {'cpu':'4','memory':'20Gi'}
    requests={'cpu':'100m','memory':'128Mi'} if backend=='fixture' else {'cpu':'1','memory':'2Gi'}
    if gpu:limits['nvidia.com/gpu']=1
    c={'name':'server','image':image,'imagePullPolicy':'Never' if backend=='fixture' else 'IfNotPresent',
       'ports':[{'name':'http','containerPort':port}],
       'resources':{'requests':requests,'limits':limits},
       'securityContext':{'allowPrivilegeEscalation':False,'capabilities':{'drop':['ALL']},'readOnlyRootFilesystem':True},
       'startupProbe':{'httpGet':{'path':ready,'port':'http'},'periodSeconds':5,'timeoutSeconds':2,'failureThreshold':120 if gpu else 24},
       'readinessProbe':{'httpGet':{'path':ready,'port':'http'},'periodSeconds':2,'timeoutSeconds':2,'failureThreshold':1},
       'volumeMounts':mounts,'env':env}
    if args:c['args']=args
    if command:c['command']=command
    if live:c['livenessProbe']={'httpGet':{'path':live,'port':'http'},'periodSeconds':10,'timeoutSeconds':2,'failureThreshold':3}
    spec={'terminationGracePeriodSeconds':30,'automountServiceAccountToken':False,
          'nodeSelector':{'kubernetes.io/arch':'arm64'},
          'securityContext':{'runAsNonRoot':True,'runAsUser':10001,'runAsGroup':10001,'fsGroup':10001,
                             'seccompProfile':{'type':'RuntimeDefault'}},'containers':[c],'volumes':volumes}
    if gpu:spec['runtimeClassName']='nvidia'
    strategy={'type':'Recreate'} if gpu else {'type':'RollingUpdate','rollingUpdate':{'maxSurge':1,'maxUnavailable':0}}
    return [{'apiVersion':'v1','kind':'Namespace','metadata':{'name':'spark-lab'}},
      {'apiVersion':'apps/v1','kind':'Deployment','metadata':{'name':name,'namespace':'spark-lab'},
       'spec':{'replicas':1,'revisionHistoryLimit':3,'progressDeadlineSeconds':900,
       'selector':{'matchLabels':{'app':name}},'strategy':strategy,
       'template':{'metadata':{'labels':{'app':name}},'spec':spec}}},
      {'apiVersion':'v1','kind':'Service','metadata':{'name':name,'namespace':'spark-lab'},
       'spec':{'type':'ClusterIP','selector':{'app':name},'ports':[{'name':'http','port':port,'targetPort':'http'}]}}]

def main():
    p=argparse.ArgumentParser();p.add_argument('backend',choices=['fixture','llama','vllm','triton'])
    p.add_argument('--image',required=True);p.add_argument('--model-dir');p.add_argument('--first-shard')
    p.add_argument('--gpu',action='store_true');p.add_argument('--out',required=True);a=p.parse_args()
    docs=documents(a.backend,a.image,a.model_dir,a.first_shard,a.gpu)
    out=Path(a.out)
    with out.open('x') as f:yaml.safe_dump_all(docs,f,sort_keys=False)
    print(f'GENERATED {out} objects={len(docs)}; not applied or API-validated')
if __name__=='__main__':main()
