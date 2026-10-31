// Small bounded correctness and timing harness. No model weights are needed.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef LAB_STARTER
#include "kernels_starter.cuh"
#else
#include "kernels.cuh"
#endif

static void check(cudaError_t e,const char* call,int line) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(call)+" line="+
        std::to_string(line)+" "+cudaGetErrorString(e));
}
#define CK(call) check((call),#call,__LINE__)
template<class T> struct Buffer {
    T* p=nullptr; std::size_t n;
    explicit Buffer(std::size_t count):n(count) { CK(cudaMalloc(&p,n*sizeof(T))); }
    ~Buffer() { if(p) cudaFree(p); } // Best-effort cleanup, also during an error.
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    void put(const std::vector<T>& v) {
        if(v.size()!=n) throw std::runtime_error("Host/device size mismatch");
        CK(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> v(n);
        CK(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v;
    }
    void clear() { CK(cudaMemset(p,0,n*sizeof(T))); }
    void poison() { CK(cudaMemset(p,0xff,n*sizeof(T))); } // NaN for float.
};
static void completed() { CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); }
static int integer(const char* text,int maximum) {
    std::string s(text); std::size_t used=0;
    const long long v=std::stoll(s,&used);
    if(used!=s.size() || v<1 || v>maximum)
        throw std::runtime_error("Integer outside the bounded lab range");
    return static_cast<int>(v);
}
static void args(int got,int wanted) {
    if(got!=wanted) throw std::runtime_error("Wrong argument count; see README.txt");
}
static int block_size(const char* text) {
    const int b=integer(text,512);
    if(b!=128 && b!=256 && b!=512)
        throw std::runtime_error("Use block size 128, 256, or 512");
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop,0));
    if(b>prop.maxThreadsPerBlock) throw std::runtime_error("Unsupported block size");
    return b;
}
static void verify(const std::vector<float>& got,const std::vector<double>& expected,
                   double atol=0.0,double rtol=0.0) {
    if(got.size()!=expected.size()) throw std::runtime_error("Output size mismatch");
    for(std::size_t i=0;i<got.size();++i)
        if(!std::isfinite(got[i]) ||
           std::abs(double(got[i])-expected[i])>atol+rtol*std::abs(expected[i]))
            throw std::runtime_error("Verification failed at index "+std::to_string(i));
}
static std::vector<float> signal(std::size_t n,int modulus,int offset,float scale) {
    std::vector<float> v(n);
    for(std::size_t i=0;i<n;++i) v[i]=(int(i%modulus)-offset)*scale;
    return v;
}
static void vectors(const std::string& mode,int n,int block,int grid) {
    auto a=signal(n,17,8,0.125f), b=signal(n,13,6,0.25f);
    std::vector<double> expected(n);
    for(int i=0;i<n;++i) {
        expected[i]=double(a[i])+b[i];
        if(mode=="fused") expected[i]=std::max(expected[i],0.0);
    }
    Buffer<float> da(n),db(n),out(n);da.put(a);db.put(b);out.poison();
    if(mode=="add") add_once<<<(n+block-1)/block,block>>>(da.p,db.p,out.p,n);
    else if(mode=="stride") add_stride<<<grid,block>>>(da.p,db.p,out.p,n);
    else fused_relu<<<grid,block>>>(da.p,db.p,out.p,n);
    completed();verify(out.get(),expected);
    std::printf("mode=%s n=%d block=%d verified=true\n",mode.c_str(),n,block);
}
static void transpose(const std::string& version,int rows,int cols) {
    auto input=signal(std::size_t(rows)*cols,257,128,0.25f);
    std::vector<double> expected(input.size());
    for(int r=0;r<rows;++r) for(int c=0;c<cols;++c)
        expected[c*rows+r]=input[r*cols+c];
    Buffer<float> in(input.size()),out(input.size());in.put(input);out.poison();
    if(version=="naive")
        transpose_naive<<<dim3((cols+31)/32,(rows+7)/8),dim3(32,8)>>>(in.p,out.p,rows,cols);
    else if(version=="tile")
        transpose_tile<<<dim3((cols+31)/32,(rows+31)/32),dim3(32,8)>>>(in.p,out.p,rows,cols);
    else throw std::runtime_error("Choose naive or tile");
    completed();verify(out.get(),expected);
    std::printf("transpose=%s rows=%d cols=%d verified=true\n",version.c_str(),rows,cols);
}
static void reduce(int n,int block) {
    auto input=signal(n,17,8,1.0f);double expected=0;
    for(float x:input) expected+=x;
    const int grid=std::min(128,(n+block-1)/block);
    Buffer<float> in(n),partial(grid),out(1);in.put(input);partial.poison();out.poison();
    reduce_blocks<<<grid,block,block*sizeof(float)>>>(in.p,partial.p,n);
    CK(cudaGetLastError());
    reduce_blocks<<<1,block,block*sizeof(float)>>>(partial.p,out.p,grid);
    completed();auto result=out.get();verify(result,{expected});
    std::printf("reduce n=%d block=%d sum=%.0f verified=true\n",n,block,result[0]);
}
static void histogram(int n) {
    std::vector<unsigned> input(n),expected(16,0);
    for(int i=0;i<n;++i) { input[i]=i%16; ++expected[input[i]]; }
    Buffer<unsigned> in(n),out(16);in.put(input);out.clear();
    histogram16<<<std::min(128,(n+255)/256),256>>>(in.p,out.p,n);completed();
    auto result=out.get();
    if(result!=expected) throw std::runtime_error("Histogram verification failed");
    std::printf("histogram n=%d bins=",n);
    for(int i=0;i<16;++i)std::printf("%s%u",i?",":"",result[i]);
    std::printf(" verified=true\n");
}
static void matmul(const std::string& version,int m,int k,int n) {
    auto a=signal(m*k,7,3,1.0f),b=signal(k*n,5,2,1.0f);
    std::vector<double> expected(m*n,0.0);
    for(int r=0;r<m;++r)for(int c=0;c<n;++c)for(int q=0;q<k;++q)
        expected[r*n+c]+=double(a[r*k+q])*b[q*n+c];
    Buffer<float> da(a.size()),db(b.size()),out(expected.size());
    da.put(a);db.put(b);out.poison();dim3 grid((n+15)/16,(m+15)/16),block(16,16);
    if(version=="naive") matmul_naive<<<grid,block>>>(da.p,db.p,out.p,m,k,n);
    else if(version=="tile") matmul_tile<<<grid,block>>>(da.p,db.p,out.p,m,k,n);
    else throw std::runtime_error("Choose naive or tile");
    completed();verify(out.get(),expected,1e-5,1e-5);
    std::printf("matmul=%s m=%d k=%d n=%d verified=true\n",version.c_str(),m,k,n);
}
static void rows(const std::string& mode,int count,int width) {
    auto input=signal(count*width,17,8,0.125f);
    // Include an all-zero row for RMS and large finite logits for softmax.
    if(mode=="rms") std::fill(input.begin(),input.begin()+width,0.0f);
    else for(float& x:input)x+=1000.0f;
    auto gamma=signal(width,7,0,0.0625f);for(float& g:gamma)g+=1.0f;
    std::vector<double> expected(input.size());
    for(int r=0;r<count;++r) {
        if(mode=="rms") {
            double sum=0;for(int c=0;c<width;++c)sum+=double(input[r*width+c])*input[r*width+c];
            const double scale=1/std::sqrt(sum/width+double(1e-5f));
            for(int c=0;c<width;++c)expected[r*width+c]=input[r*width+c]*scale*gamma[c];
        } else {
            const auto begin=input.begin()+r*width;
            const double maximum=*std::max_element(begin,begin+width);
            double sum=0;for(int c=0;c<width;++c)sum+=std::exp(input[r*width+c]-maximum);
            for(int c=0;c<width;++c)expected[r*width+c]=std::exp(input[r*width+c]-maximum)/sum;
        }
    }
    Buffer<float> in(input.size()),scale(width),out(input.size());
    in.put(input);scale.put(gamma);out.poison();
    if(mode=="rms") rms_rows<<<count,256,256*sizeof(float)>>>(in.p,scale.p,out.p,width,1e-5f);
    else softmax_rows<<<count,256,256*sizeof(float)>>>(in.p,out.p,width);
    completed();auto result=out.get();verify(result,expected,1e-7,3e-5);
    if(mode=="softmax")for(int r=0;r<count;++r) {
        double sum=0;for(int c=0;c<width;++c) { const float v=result[r*width+c];
            if(v<0 || v>1)throw std::runtime_error("Probability outside [0,1]");sum+=v; }
        if(std::abs(sum-1.0)>3e-5)throw std::runtime_error("Softmax row sum failed");
    }
    std::printf("%s rows=%d width=%d verified=true\n",mode.c_str(),count,width);
}
static void benchmark(int n,int block,int repeats,const std::string& filename) {
    if(std::filesystem::exists(filename)) throw std::runtime_error("Choose a new CSV path");
    auto a=signal(n,17,8,0.125f),b=signal(n,13,6,0.25f);
    std::vector<double> expected(n);
    for(int i=0;i<n;++i)expected[i]=std::max(double(a[i])+b[i],0.0);
    Buffer<float> da(n),db(n),out(n);da.put(a);db.put(b);out.poison();
    const int grid=std::min(128,(n+block-1)/block);
    std::ofstream csv(filename);if(!csv)throw std::runtime_error("Cannot create CSV");
    csv<<"mode,n,block,repeats,total_ms,mean_ms,logical_bytes,effective_GB_s\n";
    auto launch=[&](bool fused) {
        if(fused) fused_relu<<<grid,block>>>(da.p,db.p,out.p,n);
        else {
            add_stride<<<grid,block>>>(da.p,db.p,out.p,n);CK(cudaGetLastError());
            relu_inplace<<<grid,block>>>(out.p,n);
        }
        CK(cudaGetLastError());
    };
    cudaEvent_t start,stop;CK(cudaEventCreate(&start));CK(cudaEventCreate(&stop));
    for(bool fused:{false,true}) {
        out.poison(); // A previous variant's correct output must not hide missing writes.
        for(int i=0;i<5;++i)launch(fused);
        CK(cudaDeviceSynchronize());
        verify(out.get(),expected); // Reject an incorrect warm-up before timing.
        CK(cudaEventRecord(start));
        for(int i=0;i<repeats;++i)launch(fused);
        CK(cudaEventRecord(stop));CK(cudaEventSynchronize(stop));
        float ms=0;CK(cudaEventElapsedTime(&ms,start,stop));
        if(!(ms>0) || !std::isfinite(ms))throw std::runtime_error("Timing unresolved");
        verify(out.get(),expected);
        const double mean=double(ms)/repeats,bytes=double(n)*(fused?12:20);
        csv<<std::setprecision(10)<<(fused?"fused":"split")<<','<<n<<','<<block<<','
           <<repeats<<','<<ms<<','<<mean<<','<<bytes<<','<<bytes/(mean*1e6)<<'\n';
    }
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(stop));csv.close();
    if(!csv)throw std::runtime_error("CSV write failed");
    std::printf("benchmark n=%d block=%d repeats=%d verified=true\n",n,block,repeats);
    std::printf("saved=%s\n",filename.c_str());
}
int main(int argc,char** argv) {
    try {
        if(argc<2)throw std::runtime_error("See kernels/README.txt for commands");
        std::string mode=argv[1];
        if(mode=="add" || mode=="stride" || mode=="fused") {
            args(argc,mode=="stride"?5:4);int n=integer(argv[2],2097152),b=block_size(argv[3]);
            vectors(mode,n,b,mode=="stride"?integer(argv[4],128):std::min(128,(n+b-1)/b));
        } else if(mode=="transpose") {
            args(argc,5);transpose(argv[2],integer(argv[3],2048),integer(argv[4],2048));
        } else if(mode=="reduce") {
            args(argc,4);reduce(integer(argv[2],2097152),block_size(argv[3]));
        } else if(mode=="hist") {
            args(argc,3);histogram(integer(argv[2],2097152));
        } else if(mode=="matmul") {
            args(argc,6);matmul(argv[2],integer(argv[3],128),integer(argv[4],128),integer(argv[5],128));
        } else if(mode=="rms" || mode=="softmax") {
            args(argc,4);rows(mode,integer(argv[2],16),integer(argv[3],4096));
        } else if(mode=="bench") {
            args(argc,6);benchmark(integer(argv[2],2097152),block_size(argv[3]),integer(argv[4],1000),argv[5]);
        } else throw std::runtime_error("Unknown mode");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"ERROR: %s\n",e.what());return 2; }
}
