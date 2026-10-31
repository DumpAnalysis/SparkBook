"""Qualify the interpreter, architecture and actual CUDA execution separately."""
import argparse,json,platform,importlib.metadata
import torch
def main():
    p=argparse.ArgumentParser();p.add_argument('--require-cuda',action='store_true');a=p.parse_args()
    packages={}
    for n in ['torch','torchvision','transformers','diffusers','accelerate','huggingface-hub','numpy','scipy','Pillow','safetensors']:
        try:packages[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:packages[n]='not installed'
    r=dict(machine=platform.machine(),python=platform.python_version(),packages=packages,cuda_available=torch.cuda.is_available(),torch_cuda=torch.version.cuda)
    if torch.cuda.is_available():
        x=torch.ones(32,32,device='cuda');y=x@x;torch.cuda.synchronize()
        r.update(gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),compiled_arches=torch.cuda.get_arch_list(),cuda_memory=list(torch.cuda.mem_get_info()),matmul_correct=bool(torch.all(y==32)))
        if not r['matmul_correct']:raise AssertionError('GPU matmul check failed')
    print(json.dumps(r,sort_keys=True))
    if a.require_cuda and not r['cuda_available']:raise SystemExit('CUDA required but unavailable')
if __name__=='__main__':main()
