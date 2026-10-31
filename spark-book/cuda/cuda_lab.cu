// Small owned-process experiments. Fault modes are deliberately incorrect.
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define LAB_NVTX 1
#else
#define LAB_NVTX 0
inline int nvtxRangePushA(const char*) { return 0; }
inline int nvtxRangePop() { return 0; }
#endif

static void checked(cudaError_t code, const char* call, int line) {
    if (code != cudaSuccess) {
        std::fprintf(stderr,"line=%d call=%s error=%s\n",line,call,
                     cudaGetErrorString(code));
        std::exit(2);
    }
}
#define CUDA_OK(call) checked((call),#call,__LINE__)

__global__ void transform(std::uint32_t* values, std::size_t count) {
    std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) values[i]=2*values[i]+1;
}
__global__ void bad_write(std::uint32_t* values, std::size_t count) {
    if(threadIdx.x==0) values[count]=99; // Deliberate one-past-end write.
}
__global__ void trap_kernel() {
    if(threadIdx.x==0) asm volatile("trap;");
}
__global__ void shared_race(std::uint32_t* values) {
    __shared__ volatile std::uint32_t cell;
    cell=threadIdx.x; // Deliberate write-after-write hazard.
    __syncthreads();
    if(threadIdx.x==0) values[0]=cell;
}
__global__ void uninitialized_read(std::uint32_t* out,
                                   const std::uint32_t* temporary) {
    out[threadIdx.x]=temporary[threadIdx.x]; // Deliberate unreadied input.
}
__global__ void correct_barrier(std::uint32_t* out) {
    __shared__ std::uint32_t cells[32];
    cells[threadIdx.x]=threadIdx.x;
    __syncthreads();
    out[threadIdx.x]=cells[31-threadIdx.x];
}
// The breakpoint is reached after device synchronization, while d is alive.
extern "C" __attribute__((noinline))
void host_checkpoint(std::uint32_t* device_values, std::size_t count) {
    asm volatile("" : : "r"(device_values), "r"(count) : "memory");
}
static void snapshot_file(const std::filesystem::path& dir,
                          const std::vector<std::uint32_t>& values) {
    if(std::filesystem::exists(dir)) {
        std::fprintf(stderr,"Use a new snapshot directory\n");std::exit(3);
    }
    std::filesystem::create_directories(dir);
    std::ofstream raw(dir/"values.u32le",std::ios::binary);
    for(auto v:values) {
        unsigned char bytes[4]={static_cast<unsigned char>(v),
            static_cast<unsigned char>(v>>8),
            static_cast<unsigned char>(v>>16),
            static_cast<unsigned char>(v>>24)};
        raw.write(reinterpret_cast<const char*>(bytes),4);
    }
    raw.close();
    if(!raw) { std::fprintf(stderr,"Snapshot write failed\n");std::exit(3); }
    std::ofstream meta(dir/"metadata.json");
    meta << "{\n  \"evidence\": \"CUDA device-to-host copy\",\n"
         << "  \"dtype\": \"uint32\", \"byte_order\": \"little\",\n"
         << "  \"count\": " << values.size() << ",\n"
         << "  \"stage\": \"after transform and synchronization\",\n"
         << "  \"file\": \"values.u32le\"\n}\n";
    meta.close();
    if(!meta) { std::fprintf(stderr,"Metadata write failed\n");std::exit(3); }
}
#include "memory_probe.cuh"
int main(int argc,char** argv) {
    const std::string mode=argc>1?argv[1]:"clean";
    if(mode=="probe") {
        cudaDeviceProp prop{};int driver=0,runtime=0;
        CUDA_OK(cudaGetDeviceProperties(&prop,0));
        CUDA_OK(cudaDriverGetVersion(&driver));
        CUDA_OK(cudaRuntimeGetVersion(&runtime));
        std::printf("{\"major\":%d,\"minor\":%d,\"driver\":%d,"
                    "\"runtime\":%d,\"nvtx\":%s,",prop.major,prop.minor,
                    driver,runtime,LAB_NVTX?"true":"false");
        print_memory_capabilities();std::puts("}");return 0;
    }
    const bool trace=mode=="trace" || mode=="work";
    if(mode!="clean" && mode!="snapshot" && mode!="bad-write" &&
       mode!="trap" && mode!="race" && mode!="uninit" &&
       mode!="sync" && !trace) {
        std::fprintf(stderr,"Unknown mode\n");return 3;
    }
    if(mode=="snapshot" && argc!=3) {
        std::fprintf(stderr,"snapshot needs a new directory\n");return 3;
    }
    const std::size_t count=trace?(1u<<20):256;
    const std::size_t bytes=count*sizeof(std::uint32_t);
    std::uint32_t *d=nullptr,*host=nullptr;
    CUDA_OK(cudaMalloc(&d,bytes));
    CUDA_OK(cudaMallocHost(&host,bytes));
    for(std::size_t i=0;i<count;++i) host[i]=static_cast<std::uint32_t>(i);
    CUDA_OK(cudaMemcpy(d,host,bytes,cudaMemcpyHostToDevice));
    if(trace) {
        cudaStream_t stream;CUDA_OK(cudaStreamCreate(&stream));
        transform<<<(count+255)/256,256,0,stream>>>(d,count);
        CUDA_OK(cudaGetLastError());CUDA_OK(cudaStreamSynchronize(stream));
        if(mode=="trace") CUDA_OK(cudaProfilerStart());
        nvtxRangePushA("measured_iterations");
        for(int step=0;step<4;++step) {
            for(std::size_t i=0;i<count;++i)host[i]=static_cast<std::uint32_t>(i);
            const std::string label="iteration_"+std::to_string(step);
            nvtxRangePushA(label.c_str());
            CUDA_OK(cudaMemcpyAsync(d,host,bytes,cudaMemcpyHostToDevice,stream));
            transform<<<(count+255)/256,256,0,stream>>>(d,count);
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaMemcpyAsync(host,d,bytes,cudaMemcpyDeviceToHost,stream));
            CUDA_OK(cudaStreamSynchronize(stream));
            nvtxRangePop();
        }
        nvtxRangePop();
        if(mode=="trace") CUDA_OK(cudaProfilerStop());
        CUDA_OK(cudaStreamDestroy(stream));
        std::printf("iterations=4\nnvtx=%s\n",LAB_NVTX?"enabled":"unavailable");
    } else {
        if(mode=="bad-write") bad_write<<<1,32>>>(d,count);
        else if(mode=="trap") trap_kernel<<<1,32>>>();
        else if(mode=="race") shared_race<<<1,32>>>(d);
        else if(mode=="sync") correct_barrier<<<1,32>>>(d);
        else if(mode=="uninit") {
            std::uint32_t* temporary=nullptr;
            CUDA_OK(cudaMalloc(&temporary,32*sizeof(std::uint32_t)));
            uninitialized_read<<<1,32>>>(d,temporary);
            CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaFree(temporary));
        } else transform<<<1,256>>>(d,count);
        CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());
        if(mode=="snapshot") host_checkpoint(d,count);
        CUDA_OK(cudaMemcpy(host,d,bytes,cudaMemcpyDeviceToHost));
    }
    if(mode=="clean" || mode=="snapshot" || trace) {
        std::uint64_t sum=0;
        for(std::size_t i=0;i<count;++i) {
            if(host[i]!=2*i+1) { std::fprintf(stderr,"Verification failed\n");return 4; }
            sum+=host[i];
        }
        std::printf("values=%zu\nchecksum=%llu\nverified=true\n",count,
                    static_cast<unsigned long long>(sum));
        if(mode=="snapshot") {
            snapshot_file(argv[2],std::vector<std::uint32_t>(host,host+count));
            std::printf("snapshot=%s\n",argv[2]);
        }
    } else std::printf("mode=%s completed; inspect diagnostic-tool output\n",mode.c_str());
    CUDA_OK(cudaFree(d));CUDA_OK(cudaFreeHost(host));return 0;
}
