"""TypeSafe wire contract and calibration controls. No local Jev weights supplied."""
import argparse
import copy
import json
import math
import os
import time
from urllib.request import Request,build_opener,HTTPRedirectHandler,ProxyHandler
from common import no_network,save

MODEL='jev-1.14.0'
QUESTIONS={
 'route':{'type':'choice','instructions':'Which diagnostic category is supported by the evidence?',
          'criteria':{'memory':'Allocation pressure or out-of-memory evidence','input':'Input supply is too slow','unknown':'Evidence does not establish either cause'}},
 'severity':{'type':'score','instructions':'How much does this incident block the teaching workload?',
             'criteria':['Workload completes normally','Workload is degraded but completes','Workload cannot complete']},
 'has_oom':{'type':'noul','instructions':'Does the supplied evidence explicitly report an out-of-memory error?'}
}

def request():
    return {'model':MODEL,'state':'SYNTHETIC teaching incident: allocation failed with out-of-memory; the request did not complete. No real logs.', 'questions':QUESTIONS}

def finite(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError('non-finite or non-numeric value')
    return x
def probability(x):
    if not 0<=finite(x)<=1:raise ValueError('probability outside [0,1]')
    return x
def distribution(x,keys):
    if set(x)!=set(keys):raise ValueError('probability labels differ from requested criteria')
    for v in x.values():probability(v)
    if abs(sum(x.values())-1)>1e-3:raise ValueError('probabilities do not sum to one')

def validate(r):
    if r.get('model')!=MODEL:raise ValueError('unexpected model identity')
    if set(r.get('answers',{}))!=set(QUESTIONS):raise ValueError('answer identities differ from questions')
    a=r['answers'];c=a['route'];s=a['severity'];n=a['has_oom']
    if [c.get('type'),s.get('type'),n.get('type')]!=['choice','score','noul']:raise ValueError('answer type mismatch')
    distribution(c['probabilities'],QUESTIONS['route']['criteria'])
    if c['choice'] not in c['probabilities'] or c['probabilities'][c['choice']]<max(c['probabilities'].values())-1e-6:raise ValueError('choice is not a highest-probability option')
    probability(c['confidence']);probability(s['confidence']);probability(n['noul'])
    keys=[str(i) for i in range(len(QUESTIONS['severity']['criteria']))]
    distribution(s['probabilities'],keys)
    if s['legend']!={str(i):t for i,t in enumerate(QUESTIONS['severity']['criteria'])}:raise ValueError('rubric legend changed')
    if abs(finite(s['score'])-sum(int(k)*v for k,v in s['probabilities'].items()))>1e-3:raise ValueError('score differs from expected rubric level')
    for k in ['input_tokens','output_tokens']:
        if type(r['usage'].get(k)) is not int or r['usage'][k]<0:raise ValueError('invalid usage count')
    return {'route':c['choice'],'top_probability':max(c['probabilities'].values()),'confidence':c['confidence'],'score':s['score'],'noul':n['noul']}

def fixture():
    return {'model':MODEL,'answers':{
      'route':{'type':'choice','choice':'memory','probabilities':{'memory':.8,'input':.1,'unknown':.1},'confidence':.7},
      'severity':{'type':'score','score':1.8,'confidence':.7,'legend':{str(i):t for i,t in enumerate(QUESTIONS['severity']['criteria'])},'probabilities':{'0':0.,'1':.2,'2':.8}},
      'has_oom':{'type':'noul','noul':.95}},'usage':{'input_tokens':0,'output_tokens':0}}

def control(out):
    with no_network():
        f=fixture();summary=validate(f)
        faults=[]
        for name in ['missing-answer','wrong-model','invalid-probability','wrong-label','bad-sum','score-mismatch']:
            bad=copy.deepcopy(f)
            if name=='missing-answer':del bad['answers']['has_oom']
            if name=='wrong-model':bad['model']='jev-latest'
            if name=='invalid-probability':bad['answers']['route']['probabilities']['memory']=float('nan')
            if name=='wrong-label':bad['answers']['route']['choice']='shell'
            if name=='bad-sum':bad['answers']['route']['probabilities']['memory']=.4
            if name=='score-mismatch':bad['answers']['severity']['score']=.2
            try:validate(bad)
            except ValueError:faults.append(name)
            else:raise AssertionError(name+' accepted')
        save(out,{'source':'FABRICATED RESPONSE; NO JEV EXECUTION','request':request(),'response':f,'summary':summary,'rejected_faults':faults})
        print('PASS Choice, Score and Noul response contracts (FABRICATED)')
        print('PASS 6 faults rejected:',', '.join(faults))
        print('Choice top probability=0.8; confidence=0.7; these are different quantities')
        print('PASS no network call, no Jev model, no GPU inference')

def calibration(out):
    # Six invented binary labels/probabilities test evaluator arithmetic only.
    rows=[('a',.9,1),('b',.8,0),('c',.7,1),('d',.4,0),('e',.2,0),('f',.1,1)]
    brier=sum((p-y)**2 for _,p,y in rows)/len(rows)
    table=[]
    for threshold in [.5,.7,.9]:
        selected=[(p,y) for _,p,y in rows if max(p,1-p)>=threshold]
        errors=sum((p>=.5)!=bool(y) for p,y in selected)
        table.append({'threshold':threshold,'selected':len(selected),'coverage':len(selected)/len(rows),'risk':errors/len(selected) if selected else None})
    assert abs(brier-1.75/6)<1e-12
    save(out,{'source':'SYNTHETIC CALIBRATION CONTROL','rows':rows,'brier':brier,'risk_coverage':table})
    print(f'SYNTHETIC binary Brier score={brier:.6f}')
    for r in table:print(f"threshold={r['threshold']:.1f} selected={r['selected']}/6 coverage={r['coverage']:.3f} risk={r['risk']:.3f}")
    print('PASS high confidence still contains wrong decisions; no model accuracy measured')

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):raise ValueError('redirect refused')

def live(out,allowed):
    if not allowed:raise ValueError('hosted mode requires --send-synthetic')
    key=os.environ.get('TYPESAFE_API_KEY')
    if not key:raise ValueError('TYPESAFE_API_KEY required; never print it')
    opener=build_opener(ProxyHandler({}),NoRedirect())
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    started=time.perf_counter()
    # Only this fixed synthetic input is eligible for submission. No retries.
    req=Request('https://api.typesafe.ai/v1/systemone',data=json.dumps(request()).encode(),headers=headers,method='POST')
    with opener.open(req,timeout=30) as response:
        raw=response.read(1048577)
        if len(raw)>1048576:raise ValueError('response exceeds 1 MiB budget')
    r=json.loads(raw)
    # Retain unexpected results before validation; secrets/headers are never written.
    save(out,{'source':'HOSTED JEV WITH SYNTHETIC INPUT','request':request(),'response':r,'client_seconds':time.perf_counter()-started})
    print(json.dumps(validate(r),indent=2));print('Hosted result retained; client latency is not local GPU kernel time.')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['control','calibration','plan','hosted']);p.add_argument('--out',required=True);p.add_argument('--send-synthetic',action='store_true');a=p.parse_args()
    if a.mode=='control':control(a.out)
    elif a.mode=='calibration':calibration(a.out)
    elif a.mode=='plan':
        save(a.out,request());print(json.dumps(request(),indent=2));print('PLAN ONLY: hosted request not sent')
    else:live(a.out,a.send_synthetic)

if __name__=='__main__':main()
