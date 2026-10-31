"""Target CUDA arithmetic qualification. No model weights and no CPU fallback."""
import json,platform,sys
import torch
result={'architecture':platform.machine(),'torch':torch.__version__,'framework_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available()}
if not result['cuda_available']:
    result['status']='UNAVAILABLE';print(json.dumps(result,sort_keys=True));raise SystemExit(2)
x=torch.arange(4096,device='cuda',dtype=torch.float32);y=x+x
error=float((y.cpu()-2*torch.arange(4096,dtype=torch.float32)).abs().max())
torch.cuda.synchronize()
result.update(status='PASS' if error==0 else 'FAIL',device=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),max_abs_error=error,allocated_bytes=torch.cuda.memory_allocated(),free_total_bytes=torch.cuda.mem_get_info())
print(json.dumps(result,sort_keys=True));raise SystemExit(0 if error==0 else 1)
