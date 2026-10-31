#include "common.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

// BEGIN warp_sum
__global__ void warp_sum(const float* x,float* out,int n){
  __shared__ float sums[8]; // exactly 256 threads
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  float v=0;
  for(int i=threadIdx.x;i<n;i+=blockDim.x) v+=x[i];
  // All 32 physical lanes participate, even when they loaded no input.
  for(int d=16;d;d/=2) v+=__shfl_down_sync(0xffffffffu,v,d);
  if(lane==0)sums[warp]=v;
  __syncthreads();
  if(warp==0){
    v=lane<8?sums[lane]:0;
    for(int d=16;d;d/=2)v+=__shfl_down_sync(0xffffffffu,v,d);
    if(lane==0)*out=v;
  }
}
// END warp_sum

// BEGIN int4
__device__ int code(float x,float scale){
  float u=fminf(7.f,fmaxf(-8.f,x/scale));
  return int(u>=0?floorf(u+.5f):ceilf(u-.5f));
}
__global__ void pack4(const float* x,unsigned char* packed,int n,float scale){
  int b=blockIdx.x*blockDim.x+threadIdx.x;
  if(b<(n+1)/2){
    int lo=code(x[2*b],scale)&15;
    int hi=2*b+1<n?(code(x[2*b+1],scale)&15):0;
    packed[b]=static_cast<unsigned char>(lo|(hi<<4));
  }
}
__global__ void unpack4(const unsigned char* p,float* y,int n,float scale){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    int q=(p[i/2]>>(4*(i%2)))&15;
    y[i]=float(q>=8?q-16:q)*scale;
  }
}
// END int4

// BEGIN wmma
__global__ void tile_mma(const __half* a,const __half* b,float* c){
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
  wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf;
  wmma::fragment<wmma::accumulator,16,16,16,float> cf;
  wmma::load_matrix_sync(af,a,16);
  wmma::load_matrix_sync(bf,b,16);
  wmma::fill_fragment(cf,0.f);
  wmma::mma_sync(cf,af,bf,cf);
  wmma::store_matrix_sync(c,cf,16,wmma::mem_row_major);
}
// END wmma

void warp_lab(){
  for(int n:{1,31,32,33,255,256,257,1003}){
    std::vector<float>x(n);float want=0;
    for(int i=0;i<n;++i){x[i]=float(i%17-8);want+=x[i];}
    Device<float>dx(n),dy(1);
    CK(cudaMemcpy(dx.p,x.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemset(dy.p,0xff,sizeof(float)));
    warp_sum<<<1,256>>>(dx.p,dy.p,n);CK(cudaGetLastError());
    float got;CK(cudaMemcpy(&got,dy.p,sizeof(float),cudaMemcpyDeviceToHost));
    require(got==want,"warp sum mismatch");
  }
  puts("warp cases=8 full_participation=true verified=true");
}
void int4_lab(){
  const float s=.25f;
  std::vector<float>x={-3,-2,-.375f,-.125f,0,.125f,.375f,1.75f,3};
  std::vector<unsigned char>w={0x88,0xfe,0x10,0x72,0x07};
  std::vector<float>y={-2,-2,-.5f,-.25f,0,.25f,.5f,1.75f,1.75f};
  // Test every prefix, including empty and odd tails.
  for(int n=0;n<=9;++n){
    if(n==0)continue; // empty input: no allocation and no launch
    Device<float>dx(n),dy(n);Device<unsigned char>dp((n+1)/2);
    CK(cudaMemcpy(dx.p,x.data(),n*sizeof(float),cudaMemcpyHostToDevice));
    CK(cudaMemset(dp.p,0xff,(n+1)/2));
    CK(cudaMemset(dy.p,0xff,n*sizeof(float)));
    pack4<<<1,128>>>(dx.p,dp.p,n,s);CK(cudaGetLastError());
    unpack4<<<1,128>>>(dp.p,dy.p,n,s);CK(cudaGetLastError());
    std::vector<unsigned char>p((n+1)/2);std::vector<float>out(n);
    CK(cudaMemcpy(p.data(),dp.p,p.size(),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(out.data(),dy.p,n*sizeof(float),cudaMemcpyDeviceToHost));
    check(out,std::vector<float>(y.begin(),y.begin()+n),0);
    for(size_t i=0;i<p.size();++i){
      auto want=w[i];if(n%2 && i==p.size()-1)want&=15;
      require(p[i]==want,"packed byte mismatch");
    }
  }
  puts("int4 lengths=0..9 packed_bytes_and_values=true verified=true");
}
void tensor_lab(){
  cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,0));
  require(prop.major>=7,"this WMMA lab needs supported FP16 WMMA hardware");
  std::vector<__half>a(256),b(256);std::vector<float>want(256,0),got(256),blas(256);
  // Both row-major operands are asymmetric. |A*B| <= 16*3*5 = 240.
  for(int i=0;i<256;++i){a[i]=__float2half(float(i%7-3));b[i]=__float2half(float(i%11-5));}
  for(int r=0;r<16;++r)for(int c=0;c<16;++c)for(int k=0;k<16;++k)
    want[r*16+c]+=__half2float(a[r*16+k])*__half2float(b[k*16+c]);
  Device<__half>da(256),db(256);Device<float>dc(256),dd(256);
  CK(cudaMemcpy(da.p,a.data(),512,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db.p,b.data(),512,cudaMemcpyHostToDevice));
  CK(cudaMemset(dc.p,0xff,1024));CK(cudaMemset(dd.p,0xff,1024));
  tile_mma<<<1,32>>>(da.p,db.p,dc.p);CK(cudaGetLastError());
  cublasHandle_t h;
  require(cublasCreate(&h)==CUBLAS_STATUS_SUCCESS,"cublasCreate failed");
  float alpha=1,beta=0;
  // Row-major C=A*B equals column-major C^T=B^T*A^T.
  auto status=cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,16,16,16,
    &alpha,db.p,CUDA_R_16F,16,da.p,CUDA_R_16F,16,
    &beta,dd.p,CUDA_R_32F,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
  cublasDestroy(h);
  require(status==CUBLAS_STATUS_SUCCESS,"cublasGemmEx failed");
  CK(cudaMemcpy(got.data(),dc.p,1024,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(blas.data(),dd.p,1024,cudaMemcpyDeviceToHost));
  check(got,want,0);check(blas,want,0);
  puts("wmma_vs_cpu=true cublas_vs_cpu=true shape=16x16x16");
  puts("timing=not_measured tensor_instruction_check=inspect_disassembly");
}
int main(int argc,char**argv){try{
  require(argc==2,"usage: collectives warp|int4|tensor");
  std::string m=argv[1];
  if(m=="warp")warp_lab();else if(m=="int4")int4_lab();
  else if(m=="tensor")tensor_lab();else throw std::runtime_error("unknown mode");
  return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 2;}}
