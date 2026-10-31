#include "common.cuh"
#include <chrono>
// BEGIN transform
__global__ void transform(const float*x,float*y,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n)y[i]=2*x[i]+1;
}
// END transform
struct Host {
  float* p;
  explicit Host(int n){CK(cudaMallocHost((void**)&p,n*sizeof(float)));}
  ~Host(){cudaFreeHost(p);}
};
void buffers(){
  constexpr int n=4096,chunks=6;
  Host h0(n),h1(n),o0(n),o1(n);
  float* in[]={h0.p,h1.p};float* out[]={o0.p,o1.p};
  Device<float>d0(n),d1(n),y0(n),y1(n);
  float* dx[]={d0.p,d1.p};float* dy[]={y0.p,y1.p};
  Stream s0,s1;cudaStream_t streams[]={s0.s,s1.s};
  Event e0,e1;cudaEvent_t done[]={e0.e,e1.e};
  int tag[2]={-1,-1};
  auto verify=[&](int slot){
    if(tag[slot]<0)return;
    CK(cudaEventSynchronize(done[slot]));
    for(int i=0;i<n;++i)require(out[slot][i]==2*(tag[slot]+i%7)+1,
                               "buffer was reused before completion");
  };
  for(int chunk=0;chunk<chunks;++chunk){
    int slot=chunk%2;verify(slot); // ownership returns to CPU here
    for(int i=0;i<n;++i)in[slot][i]=float(chunk+i%7);
    CK(cudaMemcpyAsync(dx[slot],in[slot],n*4,cudaMemcpyHostToDevice,streams[slot]));
    transform<<<n/256,256,0,streams[slot]>>>(dx[slot],dy[slot],n);
    CK(cudaGetLastError());
    CK(cudaMemcpyAsync(out[slot],dy[slot],n*4,cudaMemcpyDeviceToHost,streams[slot]));
    CK(cudaEventRecord(done[slot],streams[slot]));tag[slot]=chunk;
  }
  verify(0);verify(1);
  puts("double_buffer chunks=6 verified=true overlap=inspect_timeline");
}
void dependency(bool fixed){
  Device<float>x(256),y(256);Stream produce,consume;Event ready;
  CK(cudaMemset(x.p,0,1024));CK(cudaDeviceSynchronize());
  if(!fixed){
    // Force the legal but wrong order. No undefined data race is needed.
    transform<<<1,256,0,consume.s>>>(x.p,y.p,256);CK(cudaGetLastError());
    CK(cudaStreamSynchronize(consume.s));
  }
  CK(cudaMemsetAsync(x.p,0,1024,produce.s));
  // transform in-place is elementwise, giving x=1 for this controlled input.
  transform<<<1,256,0,produce.s>>>(x.p,x.p,256);CK(cudaGetLastError());
  if(fixed){
    CK(cudaEventRecord(ready.e,produce.s));
    CK(cudaStreamWaitEvent(consume.s,ready.e,0));
    transform<<<1,256,0,consume.s>>>(x.p,y.p,256);CK(cudaGetLastError());
  }
  CK(cudaStreamSynchronize(produce.s));CK(cudaStreamSynchronize(consume.s));
  std::vector<float>out(256);CK(cudaMemcpy(out.data(),y.p,1024,cudaMemcpyDeviceToHost));
  check(out,std::vector<float>(256,fixed?3.f:1.f),0);
  printf("dependency fixed=%s observed=%.0f expected=3\n",fixed?"true":"false",out[0]);
  puts(fixed?"repair=PASS":"controlled_stale_read=EXPOSED");
}
void graphs(){
  constexpr int n=4096,repeats=100;
  Device<float>x(n),y(n),z(n);Stream s;Event start,stop;
  CK(cudaMemsetAsync(x.p,0,n*4,s.s));
  auto work=[&]{
    transform<<<n/256,256,0,s.s>>>(x.p,y.p,n);CK(cudaGetLastError());
    transform<<<n/256,256,0,s.s>>>(y.p,z.p,n);CK(cudaGetLastError());
  };
  for(int i=0;i<5;++i)work();CK(cudaStreamSynchronize(s.s));
  cudaGraph_t graph;cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(s.s,cudaStreamCaptureModeGlobal));work();
  CK(cudaStreamEndCapture(s.s,&graph));
  CK(cudaGraphInstantiate(&exec,graph,0));
  CK(cudaGraphLaunch(exec,s.s));CK(cudaStreamSynchronize(s.s));
  std::vector<float>out(n);
  for(int mode=0;mode<2;++mode){
    CK(cudaMemsetAsync(z.p,0xff,n*4,s.s));
    CK(cudaEventRecord(start.e,s.s));
    auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<repeats;++i){if(mode)CK(cudaGraphLaunch(exec,s.s));else work();}
    auto submitted=std::chrono::steady_clock::now();
    CK(cudaEventRecord(stop.e,s.s));CK(cudaEventSynchronize(stop.e));
    float ms;CK(cudaEventElapsedTime(&ms,start.e,stop.e));
    CK(cudaMemcpy(out.data(),z.p,n*4,cudaMemcpyDeviceToHost));
    check(out,std::vector<float>(n,3),0);
    double us=std::chrono::duration<double,std::micro>(submitted-begin).count();
    printf("mode=%s repeats=%d device_interval_ms=%.6f host_submit_us=%.3f verified=true\n",
           mode?"graph":"ordinary",repeats,ms,us);
  }
  CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));
}
int main(int argc,char**argv){try{
  require(argc==2,"usage: async-lab buffers|bad|fixed|graphs");
  std::string m=argv[1];
  if(m=="buffers")buffers();else if(m=="bad")dependency(false);
  else if(m=="fixed")dependency(true);else if(m=="graphs")graphs();
  else throw std::runtime_error("unknown mode");return 0;
}catch(const std::exception&e){fprintf(stderr,"ERROR %s\n",e.what());return 2;}}
