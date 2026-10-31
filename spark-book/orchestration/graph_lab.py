"""SQLite checkpoint, interruption, restart and bounded model-node study."""
import argparse
import json
from pathlib import Path
from typing_extensions import TypedDict
from common import no_network, save
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite import SqliteSaver

class State(TypedDict, total=False):
    case_id: str
    approved: bool
    verdict: str

def review(state):
    # This code executes again on resume. No external side effect precedes interrupt.
    answer=interrupt({'case_id':state['case_id'], 'proposal':'Inspect simulated memory evidence'})
    if type(answer) is not bool: raise ValueError('approval must be a JSON boolean')
    return {'approved':answer}

def build(saver, model=None, output=None):
    def finish(state):
        if not state['approved']: return {'verdict':'denied'}
        if model is None: return {'verdict':'SIMULATED inspection approved; no GPU action'}
        from langchain_lab import run
        messages=run(model,output)
        return {'verdict':str(messages[-1].content)}
    b=StateGraph(State);b.add_node('review',review);b.add_node('finish',finish)
    b.add_edge(START,'review');b.add_edge('review','finish');b.add_edge('finish',END)
    return b.compile(checkpointer=saver)

def step(db,thread,action,approve,case,model=None,output=None):
    config={'configurable':{'thread_id':thread}, 'recursion_limit':8}
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph=build(saver,model,output)
        snapshot=graph.get_state(config)
        if action=='start':
            if snapshot.values: raise ValueError('thread already exists; choose a fresh thread_id')
            result=graph.invoke({'case_id':case},config)
            assert result.get('__interrupt__')
            return {'thread':thread,'status':'paused','case_id':result['case_id'],'next':list(graph.get_state(config).next)}
        if action=='resume':
            if not snapshot.values or not snapshot.next: raise ValueError('no pending checkpoint for this thread')
            result=graph.invoke(Command(resume=approve),config)
            return {'thread':thread,'status':'complete','approved':result['approved'],'verdict':result['verdict']}
        return {'thread':thread,'values':snapshot.values,'next':list(snapshot.next)}

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','resume','status','faults'])
    p.add_argument('--db',required=True);p.add_argument('--thread',default='case-001');p.add_argument('--case',default='SIMULATED-001')
    p.add_argument('--approve',choices=['yes','no'],default='no');p.add_argument('--out',required=True)
    p.add_argument('--local-model');p.add_argument('--base-url',default='http://127.0.0.1:8080/v1');a=p.parse_args()
    Path(a.db).parent.mkdir(parents=True,exist_ok=True)
    if a.action=='faults':
        with no_network():
            for name,thread in [('duplicate-resume',a.thread),('wrong-thread',a.thread+'-absent')]:
                try:step(a.db,thread,'resume',True,a.case)
                except ValueError as e: print('PASS',name,str(e))
                else:raise AssertionError(name+' unexpectedly succeeded')
            state=step(a.db,a.thread,'status',False,a.case)
            save(a.out,state);print('PASS completed checkpoint unchanged')
        return
    model=None
    if a.local_model:
        from langchain_lab import local_model
        model=local_model(a.base_url,a.local_model)
    try:
        if model:result=step(a.db,a.thread,a.action,a.approve=='yes',a.case,model,a.out+'.agent.json')
        else:
            with no_network():result=step(a.db,a.thread,a.action,a.approve=='yes',a.case)
        save(a.out,result);print(json.dumps(result,indent=2))
    finally:
        if model:
            from langchain_lab import close_model
            close_model(model)

if __name__=='__main__':main()
