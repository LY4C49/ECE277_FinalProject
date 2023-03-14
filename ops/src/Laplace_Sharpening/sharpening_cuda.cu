#include <cstdio>
#include <math.h>
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)


__global__ void sharpening_kernel(const float* a, const float* b, float* c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    float result = 0.0;
    for (int i = 0;i < 3;i++){
        for (int j = 0;j < 3;j++){
            int index_x = 8 * blockIdx.x + threadIdx.x + j;
            int index_y = 8 * blockIdx.y + threadIdx.y + i;
            float temp = b[j * 3 + i] * a[index_y * n + index_x];
            result += temp;
        }
    }
    c[idy * (n - 2) + idx] = result;
    //c[idx][idy] = result;
}


void sharpening_launcher(const float* a, const float* b, float* c, int n){
    //dim3 blockSize(DIVUP(n, THREADS_PER_BLOCK));
    //dim3 threadSize(THREADS_PER_BLOCK);
    //two_sum_kernel<<<blockSize, threadSize>>>(a, b, c, n);
    int block_size = ceil((n - 1) / 8);
    dim3 blockSize(block_size,block_size);
    dim3 threadPerBlock(8,8);
    sharpening_kernel<<<blockSize,threadPerBlock>>>(a,b,c,n);


}