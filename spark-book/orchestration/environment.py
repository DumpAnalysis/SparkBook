"""Record independent orchestration package and platform identities."""
import argparse
import importlib.metadata as md
import platform
import sys
from common import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    names=['langchain','langchain-core','langchain-openai','langgraph','langgraph-checkpoint-sqlite','langsmith','pydantic-ai-slim','pydantic','httpx','openai']
    v={n:md.version(n) for n in names}
    result={'python':platform.python_version(),'machine':platform.machine(),'platform':sys.platform,'packages':v,'scope':'package identity only; no GPU qualification'}
    save(a.out,result)
    print('Python',result['python'],'platform',result['platform'],'machine',result['machine'])
    for k,value in v.items():print(k+'=='+value)
    print(result['scope'])

if __name__=='__main__':main()
