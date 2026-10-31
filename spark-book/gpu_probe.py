"""Small CUDA probe for an already installed, compatible PyTorch build."""
import gc
import platform
import sys
print("machine=" + platform.machine())
try:
    import torch
except ImportError:
    print("STOP: PyTorch is not installed in this interpreter.")
    sys.exit(2)
print("torch=" + torch.__version__)
print("compiled_cuda=" + str(torch.version.cuda))
print("cuda_available=" + str(torch.cuda.is_available()))
if not torch.cuda.is_available():
    sys.exit(2)
print("device=" + torch.cuda.get_device_name(0))
print("capability=" + str(torch.cuda.get_device_capability(0)))
print("compiled_arches=" + str(torch.cuda.get_arch_list()))
try:
    x = torch.ones((1024, 1024), device="cuda", dtype=torch.float32)
    y = x @ x
    torch.cuda.synchronize()
    print("matmul_value=" + str(y[0, 0].item()))
    print("allocated_mib=" + str(torch.cuda.memory_allocated() / 2**20))
    print("reserved_mib=" + str(torch.cuda.memory_reserved() / 2**20))
    del x, y
    gc.collect()
    torch.cuda.empty_cache()
    print("after_release_allocated_mib=" +
          str(torch.cuda.memory_allocated() / 2**20))
except RuntimeError as exc:
    print("CUDA_PROBE_FAILED: " + str(exc))
    sys.exit(2)
