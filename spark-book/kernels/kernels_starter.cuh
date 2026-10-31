#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>

// Each kernel has an explicit launch contract in the book and host driver.
// BEGIN add_once
__global__ void add_once(const float* a, const float* b,
                         float* out, std::size_t n) {
    // TODO Exercise 45: implement this kernel.
}
// END add_once

// BEGIN add_stride
__global__ void add_stride(const float* a, const float* b,
                           float* out, std::size_t n) {
    // TODO Exercise 46: implement this kernel.
}
// END add_stride

// BEGIN transpose_naive
__global__ void transpose_naive(const float* in, float* out,
                                int rows, int cols) {
    // TODO Exercise 47: implement this kernel.
}
// END transpose_naive

// BEGIN transpose_tile
// Launch contract: block=(32,8), grid=(ceil(cols/32),ceil(rows/32)).
__global__ void transpose_tile(const float* in, float* out,
                               int rows, int cols) {
    // TODO Exercise 48: implement this kernel.
}
// END transpose_tile

// BEGIN reduce_blocks
// Launch contract: power-of-two block size; blockDim.x*sizeof(float) shared bytes.
__global__ void reduce_blocks(const float* in, float* partial,
                              std::size_t n) {
    // TODO Exercise 49: implement this kernel.
}
// END reduce_blocks

// BEGIN histogram16
// Input values are in [0,15]; output is zeroed by the host before each call.
__global__ void histogram16(const unsigned* in, unsigned* out,
                            std::size_t n) {
    // TODO Exercise 50: implement this kernel.
}
// END histogram16

// BEGIN matmul_naive
__global__ void matmul_naive(const float* a, const float* b,
                             float* out, int m, int k, int n) {
    // TODO Exercise 51: implement this kernel.
}
// END matmul_naive

// BEGIN matmul_tile
// Launch contract: block=(16,16), grid=(ceil(n/16),ceil(m/16)).
__global__ void matmul_tile(const float* a, const float* b,
                            float* out, int m, int k, int n) {
    // TODO Exercise 52: implement this kernel.
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
    // TODO Exercise 53: implement this kernel.
}
// END fused_relu

// BEGIN rms_rows
// One block per row, power-of-two block size, one float per thread in shared memory.
__global__ void rms_rows(const float* in, const float* gamma,
                         float* out, int width, float epsilon) {
    // TODO Exercise 55: implement this kernel.
}
// END rms_rows

// BEGIN softmax_rows
// Same launch contract as rms_rows; finite inputs and nonempty rows only.
__global__ void softmax_rows(const float* in, float* out, int width) {
    // TODO Exercise 56: implement this kernel.
}
// END softmax_rows
