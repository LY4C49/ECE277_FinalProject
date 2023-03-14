#include <cstdio>
#include <math.h>


__global__ void dct2_kernel(const float* a, const float* b, float* c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int initial_x = idx - idx % 8;
    int relative_x = threadIdx.x;

    float result = 0.0;
    for (int i = 0;i < 8;i++){
        result += a[idy * n + (initial_x + i)] * b[i * 8 + relative_x];
    }

    c[idy * n + idx] = result;

}


void dct2_launcher(const float* a, const float* b, float* c, int n){
    //dim3 blockSize(DIVUP(n, THREADS_PER_BLOCK));
    //dim3 threadSize(THREADS_PER_BLOCK);
    //two_sum_kernel<<<blockSize, threadSize>>>(a, b, c, n);
    int block_size = ceil(n / 8);
    dim3 blockSize(block_size,block_size);
    dim3 threadPerBlock(8,8);
    dct2_kernel<<<blockSize,threadPerBlock>>>(a,b,c,n);


}