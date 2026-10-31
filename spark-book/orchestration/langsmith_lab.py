"""Local SDK traces/evaluation; hosted trace submission is explicit and synthetic."""
import argparse
import json
import os
from uuid import UUID, uuid5, NAMESPACE_URL
from common import EVIDENCE, no_network, save
from langsmith import traceable, tracing_context, Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example

def offline_client():
    # Disabling upload alone does not disable the SDK client's /info discovery.
    # Supply local metadata and disable background batching as well.
    return Client(api_url='http://127.0.0.1:1',api_key='offline-placeholder',
                  auto_batch_tracing=False,info={'version':'offline'},tracing_mode='langsmith')

def control(out):
    completed=[]
    @traceable(name='read_fixture',run_type='tool')
    def read_fixture(): return EVIDENCE['gpu_memory']
    @traceable(name='diagnostic_control',process_inputs=lambda x:{'case_id':x['case_id'],'redacted':True})
    def inspect(case_id,prompt,fail=False):
        read_fixture()
        if fail: raise ValueError('injected diagnostic failure')
        return {'category':'memory','evidence':'SIMULATED'}
    with no_network(),tracing_context(enabled='local',project_name='wsl-offline-lab',client=offline_client()):
        inspect('case-001','PRIVATE-TEACHING-MARKER',langsmith_extra={'on_end':completed.append})
        try:inspect('case-002','PRIVATE-TEACHING-MARKER',True,langsmith_extra={'on_end':completed.append})
        except ValueError:pass
        else:raise AssertionError('fault did not propagate')
    # Child runs are excluded by the SDK's default serialization: preserve them explicitly.
    def record(r):
        row=r.model_dump(mode='json')
        row['child_runs']=[record(c) for c in r.child_runs]
        return row
    rows=[record(r) for r in completed]
    assert len(rows)==2 and all(len(r['child_runs'])==1 for r in rows)
    assert rows[1]['error'] and not rows[0]['error']
    assert all(r['child_runs'][0]['parent_run_id']==r['id'] for r in rows)
    assert 'PRIVATE-TEACHING-MARKER' not in json.dumps(rows)
    save(out,{'evidence':'CPU SDK TRACE; SIMULATED INPUT; NOT UPLOADED','runs':rows})
    print('PASS 2 root traces, 2 child spans, parent IDs preserved')
    print('PASS injected error retained; input marker redacted')
    print('PASS local tracing: zero network connections; no LangSmith UI qualification')

def eval_control(out):
    # Predictions are scripted fixtures, not outputs of a learned classifier.
    labels=['memory','input','memory','unknown','input']
    predictions=['memory','input','unknown','unknown','input']
    examples=[Example(id=uuid5(NAMESPACE_URL,f'wsl-lab/{i}'),inputs={'case':i},outputs={'category':y}) for i,y in enumerate(labels)]
    def predict(inputs):return {'category':predictions[inputs['case']]}
    def correct(run,example):return {'key':'category_correct','score':int(run.outputs['category']==example.outputs['category'])}
    with no_network(),tracing_context(enabled=False):
        results=evaluate(predict,data=examples,evaluators=[correct],max_concurrency=0,
                         client=offline_client(),upload_results=False,experiment_prefix='SIMULATED-offline',disable_evaluator_tracing=True)
        rows=list(results)
    scores=[r['evaluation_results']['results'][0].score for r in rows]
    assert len(scores)==5 and sum(scores)==4
    save(out,{'evidence':'SCRIPTED PREDICTIONS; SDK EVALUATOR','scores':scores,'accuracy':sum(scores)/len(scores),'uploaded':False})
    print('PASS 5 examples; 4 correct; accuracy=0.8 (SCRIPTED control)')
    print('PASS upload_results=False; zero network connections')

def plan(out):
    value={'source':'SYNTHETIC CPU ONLY','project':'wsl-book-synthetic',
           'endpoint':'https://api.smith.langchain.com','requires':'LANGSMITH_API_KEY',
           'network_call_executed':False,'command':'langsmith_lab.py service --upload-synthetic --out NEW_FILE'}
    save(out,value);print(json.dumps(value,indent=2))

def service(out,allowed):
    if not allowed:raise ValueError('service mode requires --upload-synthetic')
    key=os.environ.get('LANGSMITH_API_KEY')
    if not key:raise ValueError('LANGSMITH_API_KEY is required; never put it in a transcript')
    # This explicit command uploads only a fixed synthetic fixture to the named endpoint.
    client=Client(api_url='https://api.smith.langchain.com',api_key=key,auto_batch_tracing=False)
    from langsmith.run_trees import RunTree
    run=RunTree(name='wsl-book-synthetic',run_type='chain',project_name='wsl-book-synthetic',
                inputs={'case':'SIMULATED-001'},ls_client=client)
    run.end(outputs={'outcome':'synthetic control; no GPU'});run.post()
    found=client.read_run(run.id)
    save(out,{'run_id':str(found.id),'project':'wsl-book-synthetic','source':'SYNTHETIC CPU','readback':True})
    print('Service accepted synthetic run and readback succeeded; inspect UI separately.')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['control','evaluate','plan','service'])
    p.add_argument('--out',required=True);p.add_argument('--upload-synthetic',action='store_true');a=p.parse_args()
    if a.mode=='control':control(a.out)
    elif a.mode=='evaluate':eval_control(a.out)
    elif a.mode=='plan':plan(a.out)
    else:service(a.out,a.upload_synthetic)

if __name__=='__main__':main()
