"""Typed output, validation retries, request budgets and a local endpoint."""
import argparse
import asyncio
import time
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from common import endpoint, no_network, save
from pydantic_ai import Agent, models, capture_run_messages
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, ModelMessagesTypeAdapter
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

class Finding(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    category: Literal['memory','input','unknown']
    evidence: Literal['gpu_memory','request']
    confidence: float=Field(ge=0,le=1)

def scripted(always_bad=False):
    calls=[]
    def respond(messages,info):
        calls.append(len(messages))
        data={'category':'memory','evidence':'gpu_memory','confidence':1.2 if always_bad or len(calls)==1 else .8}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,data,tool_call_id=f'fixture-{len(calls)}')])
    return FunctionModel(respond),calls

async def local(a,evidence='LOCAL MODEL WITH SIMULATED INPUT'):
    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider
    async with httpx.AsyncClient(trust_env=False,follow_redirects=False,timeout=30) as http:
        client=AsyncOpenAI(base_url=endpoint(a.base_url),api_key='local-unused',max_retries=0,http_client=http)
        model=OpenAIChatModel(a.model,provider=OpenAICompatibleProvider('wsl-lab',openai_client=client))
        agent=Agent(model,output_type=Finding,retries=1,model_settings={'temperature':0,'max_tokens':256})
        start_ns=time.time_ns();start=time.perf_counter()
        r=await agent.run('SIMULATED evidence: gpu_memory reports 28 GiB used of 30 GiB. Choose a category and cite gpu_memory. Confidence is a self-rating.',usage_limits=UsageLimits(request_limit=3))
        save(a.out,{'evidence':evidence,'start_epoch_ns':start_ns,'end_epoch_ns':time.time_ns(),'client_seconds':time.perf_counter()-start,'output':r.output.model_dump(),'usage':str(r.usage),'messages':r.all_messages_json().decode()})
        print(r.output.model_dump_json());print('Structured validity checked; factual correctness and calibration require separate evaluation.')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['control','budget','local']);p.add_argument('--out',required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8080/v1');p.add_argument('--model',default='local-model');a=p.parse_args()
    if a.mode=='local':asyncio.run(local(a));return
    models.ALLOW_MODEL_REQUESTS=False
    with no_network():
        model,calls=scripted(a.mode=='budget');agent=Agent(model,output_type=Finding,retries=4)
        if a.mode=='control':
            r=agent.run_sync('SIMULATED control',usage_limits=UsageLimits(request_limit=3))
            assert len(calls)==2 and r.output.confidence==.8
            save(a.out,{'evidence':'SCRIPTED FunctionModel','output':r.output.model_dump(),'model_requests':len(calls),'messages':r.all_messages_json().decode()})
            print('PASS invalid confidence=1.2 rejected; second response accepted')
            print('PASS model_requests=2; validation retries=1')
            print(r.output.model_dump_json())
        else:
            try:
                with capture_run_messages() as history:
                    agent.run_sync('SIMULATED control',usage_limits=UsageLimits(request_limit=2))
            except UsageLimitExceeded:
                assert len(calls)==2
                save(a.out,{'evidence':'SCRIPTED FunctionModel','model_requests':2,'outcome':'UsageLimitExceeded','messages':ModelMessagesTypeAdapter.dump_json(history).decode()})
                print('PASS persistent invalid output stopped before request 3')
            else:raise AssertionError('request limit failed')
    print('PASS offline control; no LLM or GPU executed')

if __name__=='__main__':main()
