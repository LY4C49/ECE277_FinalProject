#include <cstdio>
#include <math.h>


__global__ void comp_kernel(const float* a, const float* b, float* c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;


    float result = a[idy * n + idx] * b[threadIdx.y * 8 + threadIdx.x];

    c[idy * n + idx] = result;

}


void comp_launcher(const float* a, const float* b, float* c, int n){
    int block_size = ceil(n / 8);
    dim3 blockSize(block_size,block_size);
    dim3 threadPerBlock(8,8);
    comp_kernel<<<blockSize,threadPerBlock>>>(a,b,c,n);


}