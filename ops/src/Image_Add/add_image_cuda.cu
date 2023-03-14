#include <cstdio>
#include <math.h>

__global__ void image_add_kernel(const float* a, const float* b, float* c,int n){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    c[idy * n + idx] = a[idy * n + idx] + b[idy * n + idx];
}


void image_add_launcher(const float* a, const float* b, float* c, int n){
    int block_size = ceil((n - 1) / 8);
    dim3 blockSize(block_size,block_size);
    dim3 threadPerBlock(8,8);
    image_add_kernel<<<blockSize,threadPerBlock>>>(a,b,c,n);


}