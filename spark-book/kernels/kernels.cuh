#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

// Each kernel has an explicit launch contract in the book and host driver.
// BEGIN add_once
__global__ void add_once(const float* a, const float* b,
                         float* out, std::size_t n) {
    const std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) out[i]=a[i]+b[i];
}
// END add_once

// BEGIN add_stride
__global__ void add_stride(const float* a, const float* b,
                           float* out, std::size_t n) {
    const std::size_t first=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const std::size_t step=std::size_t(gridDim.x)*blockDim.x;
    for(std::size_t i=first;i<n;i+=step) out[i]=a[i]+b[i];
}
// END add_stride

// BEGIN transpose_naive
__global__ void transpose_naive(const float* in, float* out,
                                int rows, int cols) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    const int r=blockIdx.y*blockDim.y+threadIdx.y;
    if(r<rows && c<cols) out[c*rows+r]=in[r*cols+c];
}
// END transpose_naive

// BEGIN transpose_tile
// Launch contract: block=(32,8), grid=(ceil(cols/32),ceil(rows/32)).
__global__ void transpose_tile(const float* in, float* out,
                               int rows, int cols) {
    __shared__ float tile[32][33];
    const int x=threadIdx.x, y=threadIdx.y;
    int c=blockIdx.x*32+x, r=blockIdx.y*32+y;
    for(int j=0;j<32;j+=8)
        if(r+j<rows && c<cols) tile[y+j][x]=in[(r+j)*cols+c];
    __syncthreads();
    c=blockIdx.y*32+x; r=blockIdx.x*32+y;
    for(int j=0;j<32;j+=8)
        if(r+j<cols && c<rows) out[(r+j)*rows+c]=tile[x][y+j];
}
// END transpose_tile

// BEGIN reduce_blocks
// Launch contract: power-of-two block size; blockDim.x*sizeof(float) shared bytes.
__global__ void reduce_blocks(const float* in, float* partial,
                              std::size_t n) {
    extern __shared__ float sums[];
    const unsigned t=threadIdx.x;
    const std::size_t first=std::size_t(blockIdx.x)*blockDim.x+t;
    const std::size_t step=std::size_t(gridDim.x)*blockDim.x;
    float value=0.0f;
    for(std::size_t i=first;i<n;i+=step) value+=in[i];
    sums[t]=value;
    __syncthreads();
    for(unsigned stride=blockDim.x/2;stride;stride/=2) {
        if(t<stride) sums[t]+=sums[t+stride];
        __syncthreads();
    }
    if(t==0) partial[blockIdx.x]=sums[0];
}
// END reduce_blocks

// BEGIN histogram16
// Input values are in [0,15]; output is zeroed by the host before each call.
__global__ void histogram16(const unsigned* in, unsigned* out,
                            std::size_t n) {
    __shared__ unsigned bins[16];
    const unsigned t=threadIdx.x;
    if(t<16) bins[t]=0;
    __syncthreads();
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+t;i<n;
        i+=std::size_t(gridDim.x)*blockDim.x) atomicAdd(&bins[in[i]],1u);
    __syncthreads();
    if(t<16) atomicAdd(&out[t],bins[t]);
}
// END histogram16

// BEGIN matmul_naive
__global__ void matmul_naive(const float* a, const float* b,
                             float* out, int m, int k, int n) {
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    const int r=blockIdx.y*blockDim.y+threadIdx.y;
    if(r<m && c<n) {
        float sum=0;
        for(int q=0;q<k;++q) sum+=a[r*k+q]*b[q*n+c];
        out[r*n+c]=sum;
    }
}
// END matmul_naive

// BEGIN matmul_tile
// Launch contract: block=(16,16), grid=(ceil(n/16),ceil(m/16)).
__global__ void matmul_tile(const float* a, const float* b,
                            float* out, int m, int k, int n) {
    __shared__ float left[16][16], right[16][16];
    const int x=threadIdx.x, y=threadIdx.y;
    const int r=blockIdx.y*16+y, c=blockIdx.x*16+x;
    float sum=0;
    for(int base=0;base<k;base+=16) {
        left[y][x]=(r<m && base+x<k)?a[r*k+base+x]:0.0f;
        right[y][x]=(base+y<k && c<n)?b[(base+y)*n+c]:0.0f;
        __syncthreads();
        for(int q=0;q<16;++q) sum+=left[y][q]*right[q][x];
        __syncthreads();
    }
    if(r<m && c<n) out[r*n+c]=sum;
}
// END matmul_tile

// Supplied control kernel; not a starter assignment.
__global__ void relu_inplace(float* data, std::size_t n) {
    const std::size_t first=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    for(std::size_t i=first;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
        data[i]=fmaxf(data[i],0.0f);
}

// BEGIN fused_relu
__global__ void fused_relu(const float* a, const float* b,
                           float* out, std::size_t n) {
    const std::size_t first=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    for(std::size_t i=first;i<n;i+=std::size_t(gridDim.x)*blockDim.x)
        out[i]=fmaxf(a[i]+b[i],0.0f);
}
// END fused_relu

// BEGIN rms_rows
// One block per row, power-of-two block size, one float per thread in shared memory.
__global__ void rms_rows(const float* in, const float* gamma,
                         float* out, int width, float epsilon) {
    extern __shared__ float scratch[];
    const int row=blockIdx.x, t=threadIdx.x;
    float square_sum=0;
    for(int c=t;c<width;c+=blockDim.x) {
        const float x=in[row*width+c]; square_sum+=x*x;
    }
    scratch[t]=square_sum;
    __syncthreads();
    for(unsigned s=blockDim.x/2;s;s/=2) {
        if(t<s) scratch[t]+=scratch[t+s];
        __syncthreads();
    }
    const float scale=1.0f/sqrtf(scratch[0]/width+epsilon);
    for(int c=t;c<width;c+=blockDim.x)
        out[row*width+c]=in[row*width+c]*scale*gamma[c];
}
// END rms_rows

// BEGIN softmax_rows
// Same launch contract as rms_rows; finite inputs and nonempty rows only.
__global__ void softmax_rows(const float* in, float* out, int width) {
    extern __shared__ float scratch[];
    const int row=blockIdx.x, t=threadIdx.x;
    float maximum=-INFINITY;
    for(int c=t;c<width;c+=blockDim.x)
        maximum=fmaxf(maximum,in[row*width+c]);
    scratch[t]=maximum;
    __syncthreads();
    for(unsigned s=blockDim.x/2;s;s/=2) {
        if(t<s) scratch[t]=fmaxf(scratch[t],scratch[t+s]);
        __syncthreads();
    }
    maximum=scratch[0];
    __syncthreads(); // Every thread must read the maximum before scratch is reused.
    float sum=0;
    for(int c=t;c<width;c+=blockDim.x)
        sum+=expf(in[row*width+c]-maximum);
    scratch[t]=sum;
    __syncthreads();
    for(unsigned s=blockDim.x/2;s;s/=2) {
        if(t<s) scratch[t]+=scratch[t+s];
        __syncthreads();
    }
    const float denominator=scratch[0];
    for(int c=t;c<width;c+=blockDim.x)
        out[row*width+c]=expf(in[row*width+c]-maximum)/denominator;
}
// END softmax_rows
