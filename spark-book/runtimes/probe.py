"""One bounded target request, retaining wire bytes even on HTTP/parse failure."""
import argparse,json,time,sys
from common import exchange,ndjson,sse,vectors,fresh,write_json,ContractError

def main(a):
    out=fresh(a.out);result={'scope':'actual endpoint request','backend':a.backend,'base':a.base,'model':a.model,'status':'failed'}
    try:
        ollama=a.backend=='ollama'
        path='/api/tags' if ollama else '/v1/models'
        code,body=exchange(a.base,path);(out/'inventory.json').write_bytes(body)
        if code!=200:raise ContractError('inventory HTTP '+str(code))
        if ollama:
            entry=next((x for x in json.loads(body)['models'] if x.get('name')==a.model or x.get('model')==a.model),None)
            if not entry:raise ContractError('exact model name absent from inventory')
            if not a.digest or entry.get('digest')!=a.digest:raise ContractError('provide matching recorded --digest')
            result['model_digest']=a.digest
            code,body=exchange(a.base,'/api/show',{'model':a.model});(out/'show.json').write_bytes(body)
            if code!=200:raise ContractError('show HTTP '+str(code))
        if a.embedding:
            texts=['CUDA allocation failed','The input loader is slow','Device memory is exhausted']
            payload={'model':a.model,'input':texts}
            if ollama:payload.update({'truncate':False,'keep_alive':0})
            path='/api/embed' if ollama else '/v1/embeddings'
        elif ollama:
            payload={'model':a.model,'prompt':'Name the evidence needed to diagnose CUDA out of memory. Use two short sentences.','stream':True,'keep_alive':a.keep_alive,'options':{'num_ctx':a.context,'num_predict':128,'temperature':0,'seed':42}}
            path='/api/generate'
        else:
            payload={'model':a.model,'messages':[{'role':'user','content':'Name the evidence needed to diagnose CUDA out of memory. Use two short sentences.'}],'stream':True,'max_tokens':128,'temperature':0,'seed':42}
            path='/v1/chat/completions'
        write_json(out/'request.json',payload);t=time.monotonic();code,raw=exchange(a.base,path,payload);elapsed=time.monotonic()-t
        (out/'response.raw').write_bytes(raw);result.update({'http_status':code,'client_total_seconds':elapsed,'raw_response_bytes':len(raw)})
        if code!=200:raise ContractError('HTTP '+str(code))
        if a.embedding:
            obj=json.loads(raw)
            if ollama:rows=obj['embeddings']
            else:
                data=obj['data']
                if sorted(x['index'] for x in data)!=list(range(len(texts))):raise ContractError('embedding index mismatch')
                rows=[x['embedding'] for x in sorted(data,key=lambda x:x['index'])]
            vectors(rows,len(texts));write_json(out/'vectors.json',{'texts':texts,'vectors':rows,'model':a.model})
            result.update({'rows':len(rows),'dimensions':len(rows[0]),'status':'complete'})
        else:
            events=ndjson(raw) if ollama else sse(raw)
            if ollama:
                final=events[-1];answer=''.join(x.get('response','') for x in events)
                result['server_final']=final
                reason=final.get('done_reason','unknown')
            else:
                answer=''.join(c.get('delta',{}).get('content','') or '' for e in events for c in e.get('choices',[]))
                reason=next((c['finish_reason'] for e in reversed(events) for c in e.get('choices',[]) if c.get('finish_reason')), 'unknown')
                result['resolved_models']=sorted(set(e['model'] for e in events if 'model' in e))
            (out/'answer.txt').write_text(answer)
            result.update({'finish_reason':reason,'status':'complete' if reason=='stop' else 'bounded_or_unknown_finish'})
        if ollama:
            code,raw=exchange(a.base,'/api/ps');(out/'residency.json').write_bytes(raw)
            code,raw=exchange(a.base,'/api/tags');(out/'inventory-after.json').write_bytes(raw)
            entry=next((x for x in json.loads(raw)['models'] if x.get('name')==a.model or x.get('model')==a.model),None)
            if code!=200 or not entry or entry.get('digest')!=a.digest:raise ContractError('model identity changed during request')
        print(json.dumps(result,indent=2));return 0
    except Exception as exc:
        result.update({'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
        print(json.dumps(result,indent=2));return 1
    finally:write_json(out/'result.json',result)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['ollama','openai'],required=True);p.add_argument('--base',required=True);p.add_argument('--model',required=True);p.add_argument('--digest');p.add_argument('--embedding',action='store_true');p.add_argument('--context',type=int,default=2048);p.add_argument('--keep-alive',default='0');p.add_argument('--out',required=True);a=p.parse_args()
    if not 128<=a.context<=8192:p.error('context must be 128..8192 for this lab')
    sys.exit(main(a))
