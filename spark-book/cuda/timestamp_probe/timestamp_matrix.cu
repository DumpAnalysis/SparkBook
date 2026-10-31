#include <cuda_runtime.h>
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <chrono>
#include <atomic>
static std::atomic<unsigned> kernels{0},valids{0},invalids{0};static clockid_t selected=CLOCK_MONOTONIC_RAW;
static int cp(CUptiResult r,const char *s){const char *msg="unknown";cuptiGetResultString(r,&msg);printf("API %s code=%d %s\n",s,(int)r,msg);return r;}
static uint64_t CUPTIAPI timestamp(){timespec t{};clock_gettime(selected,&t);return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;}
static void CUPTIAPI requested(uint8_t **b,size_t *s,size_t *n){*s=1024*1024;*n=0;void *p=nullptr;posix_memalign(&p,8,*s);*b=(uint8_t*)p;}
static void CUPTIAPI completed(CUcontext ctx,uint32_t stream,uint8_t *b,size_t size,size_t used){
 CUpti_Activity *r=nullptr;CUptiResult rc;
 while((rc=cuptiActivityGetNextRecord(b,used,&r))==CUPTI_SUCCESS){
  if(r->kind==CUPTI_ACTIVITY_KIND_KERNEL||r->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL){auto *k=(CUpti_ActivityKernel13*)r;kernels++;bool ok=k->start>0&&k->end>k->start;if(ok)valids++;else invalids++;printf("KERNEL name=%s start=%llu end=%llu correlation=%u valid=%d\n",k->name,(unsigned long long)k->start,(unsigned long long)k->end,k->correlationId,ok);}
 }
 if(rc!=CUPTI_ERROR_MAX_LIMIT_REACHED)cp(rc,"getNextRecord");size_t lost=0;cp(cuptiActivityGetNumDroppedRecords(ctx,stream,&lost),"droppedRecords");printf("BUFFER dropped=%zu used=%zu\n",lost,used);free(b);
}
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s %s\n",#x,cudaGetErrorString(e));return 3;}}while(0)
__global__ void odd(int *p){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<4096)p[i]=2*i+1;}
__global__ void spin(unsigned long long *out){unsigned long long a=clock64(),b=a;while(b-a<1000000ULL)b=clock64();*out=b-a;}
int main(int argc,char **argv){
 const char *mode=argc>1?argv[1]:"baseline";printf("MODE %s\n",mode);bool serial=strstr(mode,"serial"),warm=strstr(mode,"warm"),forced=strstr(mode,"forced");int hes=strstr(mode,"hes")?1:0;
 if(warm){CK(cudaFree(nullptr));}
 if(strstr(mode,"clock-")){selected=strstr(mode,"realtime")?CLOCK_REALTIME:strstr(mode,"mono-only")?CLOCK_MONOTONIC:CLOCK_MONOTONIC_RAW;cp(cuptiActivityRegisterTimestampCallback(timestamp),"timestampCallback");}
 if(hes)cp(cuptiActivityEnableHWTrace(1),"enableHWTrace");
 if(strstr(mode,"latency"))cp(cuptiActivityEnableLatencyTimestamps(1),"latencyTimestamps");
 uint32_t version=0;cp(cuptiGetVersion(&version),"version");printf("CUPTI_VERSION %u kernel_struct_size=%zu\n",version,sizeof(CUpti_ActivityKernel13));
 cp(cuptiActivityRegisterCallbacks(requested,completed),"registerCallbacks");cp(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME),"enableRuntime");int er=cp(cuptiActivityEnable(serial?CUPTI_ACTIVITY_KIND_KERNEL:CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),"enableKernel");
 int *p;unsigned long long *ticks;CK(cudaMalloc(&p,4096*sizeof(int)));CK(cudaMalloc(&ticks,sizeof(*ticks)));cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));CK(cudaEventRecord(a));odd<<<16,256>>>(p);spin<<<1,1>>>(ticks);CK(cudaGetLastError());CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));CK(cudaDeviceSynchronize());float ms=0;CK(cudaEventElapsedTime(&ms,a,b));
 int h[4096];unsigned long long gpuTicks=0;CK(cudaMemcpy(h,p,sizeof(h),cudaMemcpyDeviceToHost));CK(cudaMemcpy(&gpuTicks,ticks,sizeof(gpuTicks),cudaMemcpyDeviceToHost));long long sum=0;for(int i=0;i<4096;i++){if(h[i]!=2*i+1)return 4;sum+=h[i];}
 int support=-1;cp(cuptiDeviceSupported(0,&support),"deviceSupported");printf("DEVICE_SUPPORTED %d\n",support);
 if(strstr(mode,"delay"))std::this_thread::sleep_for(std::chrono::milliseconds(500));
 cp(cuptiActivityFlushAll(forced?1:0),"flushBeforeFree");CK(cudaFree(p));CK(cudaFree(ticks));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaDeviceSynchronize());cp(cuptiActivityFlushAll(0),"flushAfterFree");
 printf("RESULT numerical=PASSED checksum=%lld CUDA_event_ms=%.9g GPU_clock_ticks=%llu kernels=%u valid=%u invalid=%u\n",sum,ms,gpuTicks,kernels.load(),valids.load(),invalids.load());
 return er==0&&valids>=2&&invalids==0?0:2;
}
