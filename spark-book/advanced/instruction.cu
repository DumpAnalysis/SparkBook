#include "common.cuh"
__global__ void inspect_kernel(const float* x,float* y){
  int i=threadIdx.x;
  float v=x[i];
  y[i]=v*2.f+1.f; // place a source breakpoint here after the entry stop
}
int main(){try{
  std::vector<float>x(32),y(32),want(32);
  for(int i=0;i<32;++i){x[i]=float(i);want[i]=2.f*i+1;}
  Device<float>dx(32),dy(32);
  CK(cudaMemcpy(dx.p,x.data(),128,cudaMemcpyHostToDevice));
  inspect_kernel<<<1,32>>>(dx.p,dy.p);CK(cudaGetLastError());
  CK(cudaMemcpy(y.data(),dy.p,128,cudaMemcpyDeviceToHost));check(y,want,0);
  puts("instruction result_first=1 result_last=63 verified=true");return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 2;}}
