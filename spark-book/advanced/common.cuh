#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
inline void cuda_check(cudaError_t status,const char* call) {
  if(status!=cudaSuccess)
    throw std::runtime_error(std::string(call)+": "+cudaGetErrorString(status));
}
#define CK(call) do { cuda_check((call),#call); } while(0)
template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { CK(cudaMalloc((void**)&p,n*sizeof(T))); }
  ~Device(){ if(p) cudaFree(p); }
  Device(const Device&)=delete;
};
struct Stream {
  cudaStream_t s;
  Stream(){CK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}
  ~Stream(){cudaStreamDestroy(s);}
};
struct Event {
  cudaEvent_t e;
  Event(){CK(cudaEventCreate(&e));}
  ~Event(){cudaEventDestroy(e);}
};
inline void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
inline void check(const std::vector<float>& a,const std::vector<float>& b,
                  float atol=1e-5f){
  require(a.size()==b.size(),"size mismatch");
  for(size_t i=0;i<a.size();++i)
    require(std::isfinite(a[i]) && std::fabs(a[i]-b[i])<=atol,
            "numerical mismatch");
}
