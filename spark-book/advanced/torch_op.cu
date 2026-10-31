#include <algorithm>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
// BEGIN muladd
// Scalar c is a configuration value; the autograd formula has no dc output.
template<class T> __global__ void muladd_kernel(
    const T* a,const T* b,T* out,int64_t n,T c){
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
      i<n;i+=int64_t(blockDim.x)*gridDim.x)out[i]=a[i]*b[i]+c;
}
at::Tensor muladd_cuda(const at::Tensor& a,const at::Tensor& b,double c){
  TORCH_CHECK(a.is_cuda() && b.is_cuda(),"CUDA tensors required");
  TORCH_CHECK(a.device()==b.device(),"same device required");
  TORCH_CHECK(a.scalar_type()==b.scalar_type(),"same dtype required");
  TORCH_CHECK(a.scalar_type()==at::kFloat || a.scalar_type()==at::kDouble,
              "float32 or float64 required");
  TORCH_CHECK(a.layout()==at::kStrided && b.layout()==at::kStrided,
              "strided layout required");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),"contiguous required");
  TORCH_CHECK(a.sizes()==b.sizes(),"equal shapes required");
  c10::cuda::CUDAGuard guard(a.device());
  auto out=at::empty_like(a);
  int64_t n=a.numel();
  if(n){
    int blocks=int(std::min<int64_t>((n+255)/256,1024));
    auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(),"spark4_muladd",[&]{
      muladd_kernel<scalar_t><<<blocks,256,0,stream>>>(
        a.data_ptr<scalar_t>(),b.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),n,scalar_t(c));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
TORCH_LIBRARY(spark4,m){m.def("muladd(Tensor a, Tensor b, float c) -> Tensor");}
TORCH_LIBRARY_IMPL(spark4,CUDA,m){m.impl("muladd",&muladd_cuda);}
// END muladd
