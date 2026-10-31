"""Exercise actual adapters against a loopback HTTP fixture, never a real LLM."""
import argparse
import asyncio
import json
import os
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import patch
from common import endpoint,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    requests=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.path!='/v1/chat/completions':self.send_error(404);return
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<65536:self.send_error(413);return
            body=json.loads(self.rfile.read(n));requests.append(body)
            names=[t['function']['name'] for t in body.get('tools',[])]
            if 'inspect_evidence' in names:
                if any(m.get('role')=='tool' for m in body['messages']):
                    message={'role':'assistant','content':'HTTP FIXTURE: inspect memory evidence.'};reason='stop'
                else:
                    message={'role':'assistant','content':None,'tool_calls':[{'id':'fixture-tool','type':'function','function':{'name':'inspect_evidence','arguments':'{"key":"gpu_memory"}'}}]};reason='tool_calls'
            elif names:
                message={'role':'assistant','content':None,'tool_calls':[{'id':'fixture-output','type':'function','function':{'name':names[0],'arguments':'{"category":"memory","evidence":"gpu_memory","confidence":0.8}'}}]};reason='tool_calls'
            else:self.send_error(400,'expected a tool schema');return
            result={'id':'fixture-completion','object':'chat.completion','created':0,'model':'fixture-model','choices':[{'index':0,'message':message,'finish_reason':reason}],'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
            raw=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}/v1'
    try:
        with patch.dict(os.environ,{'HTTP_PROXY':'http://127.0.0.1:1','HTTPS_PROXY':'http://127.0.0.1:1','NO_PROXY':''}):
            from langchain_lab import local_model,run,close_model
            model=local_model(base,'fixture-model')
            try:messages=run(model,a.out+'.langchain.json',evidence='CPU HTTP FIXTURE; NO MODEL')
            finally:close_model(model)
            from pydantic_lab import local
            asyncio.run(local(SimpleNamespace(base_url=base,model='fixture-model',out=a.out+'.pydantic.json'),evidence='CPU HTTP FIXTURE; NO MODEL'))
        assert len(requests)==3 and len(messages)==4
        for bad in ['https://example.com/v1','http://127.0.0.1:8080/v1?x=1','http://user@127.0.0.1:8080/v1']:
            try:endpoint(bad)
            except ValueError:pass
            else:raise AssertionError('bad endpoint accepted')
        save(a.out,{'evidence':'CPU HTTP FIXTURE, NOT MODEL INFERENCE','requests':requests,'requests_count':3,'proxy_bypass':True,'endpoint_guards':3})
        print('PASS actual ChatOpenAI and PydanticAI adapters: 3 loopback HTTP requests')
        print('PASS invalid inherited proxy ignored; 3 endpoint forms rejected')
        print('No model weights loaded; token counts are fixture constants, not measurements')
    finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':main()
