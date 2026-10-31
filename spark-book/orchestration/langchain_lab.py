"""Real LangChain agent loop; scripted model is explicitly a test double."""
import argparse
import json
import time
from typing import Literal
from common import EVIDENCE, endpoint, no_network, read_evidence, save
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage, messages_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

@tool
def inspect_evidence(key: Literal['gpu_memory', 'request']) -> str:
    """Read one immutable SIMULATED teaching evidence record; no shell access."""
    return read_evidence(key)

class ScriptedChat(BaseChatModel):
    fault: str = 'none'
    @property
    def _llm_type(self):
        return 'SCRIPTED-CONTROL-NOT-A-MODEL'
    def bind_tools(self, tools, **kwargs):
        return self
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage) and self.fault != 'loop':
            message = AIMessage(content='SCRIPTED: inspect memory evidence before changing context.')
        else:
            key = 'secret_file' if self.fault == 'invalid-key' else 'gpu_memory'
            message = AIMessage(content='', tool_calls=[{
                'name': 'inspect_evidence', 'args': {'key': key},
                'id': f'fixture-call-{len(messages)}', 'type': 'tool_call'}])
        return ChatResult(generations=[ChatGeneration(message=message)])

def run(model, out, evidence=None):
    agent = create_agent(model, tools=[inspect_evidence],
                         system_prompt='Read gpu_memory before answering. Evidence is SIMULATED. Do not invent observations.')
    start_ns=time.time_ns();start=time.perf_counter()
    result = agent.invoke({'messages': [{'role': 'user', 'content': 'Inspect gpu_memory, then state the next diagnostic step.'}]},
                          {'recursion_limit': 6})
    save(out, {'evidence': evidence or ('SCRIPTED' if isinstance(model, ScriptedChat) else 'LOCAL MODEL WITH SIMULATED EVIDENCE'),
               'start_epoch_ns':start_ns,'end_epoch_ns':time.time_ns(),'client_seconds':time.perf_counter()-start,
               'messages': messages_to_dict(result['messages'])})
    if not isinstance(model,ScriptedChat):
        if not any(isinstance(x,ToolMessage) and x.status=='success' and x.content==read_evidence('gpu_memory') for x in result['messages']):
            raise RuntimeError('local adapter did not complete the required gpu_memory tool call; response retained')
    return result['messages']

def local_model(base_url, model):
    import httpx
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, base_url=endpoint(base_url), api_key='local-unused',
        temperature=0, max_tokens=256, timeout=30, max_retries=0,
        use_responses_api=False, http_client=httpx.Client(trust_env=False, follow_redirects=False),
        http_async_client=httpx.AsyncClient(trust_env=False,follow_redirects=False))

def close_model(model):
    import asyncio
    model.http_client.close()
    asyncio.run(model.http_async_client.aclose())

def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['control','faults','local'])
    p.add_argument('--out', required=True); p.add_argument('--base-url', default='http://127.0.0.1:8080/v1')
    p.add_argument('--model', default='local-model'); a=p.parse_args()
    if a.mode=='local':
        m=local_model(a.base_url,a.model)
        try:
            messages=run(m,a.out)
            print('LOCAL model response retained; inspect tool messages and server CUDA logs.')
            print('Tool responses:',sum(isinstance(x,ToolMessage) for x in messages))
        finally: close_model(m)
        return
    with no_network():
        if a.mode=='control':
            messages=run(ScriptedChat(),a.out)
            assert [x.type for x in messages]==['human','ai','tool','ai']
            assert json.loads(messages[2].content)==EVIDENCE['gpu_memory']
            print('PASS real create_agent loop: human -> ai -> tool -> ai')
            print('PASS tool returned SIMULATED gpu_memory; no model weights loaded')
        else:
            messages=run(ScriptedChat(fault='invalid-key'),a.out)
            assert messages[2].status=='error' and 'secret_file' in messages[2].content
            print('PASS invalid evidence key became a tool validation error')
            try: run(ScriptedChat(fault='loop'),str(a.out)+'.loop.json')
            except GraphRecursionError: print('PASS endless tool loop stopped by recursion_limit=6')
            else: raise AssertionError('loop did not stop')
    print('PASS offline control: zero network connections')

if __name__=='__main__': main()
