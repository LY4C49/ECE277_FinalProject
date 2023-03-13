#include <cstdio>
#include <math.h>
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define DIVUP(m, n) ((m + n - 1) / n)


__global__ void sharpening_kernel(const float* a, const float* b, float* c, int n){
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //if (idx < n){
    //    c[idx] = a[idx] + b[idx];
    //}
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    float result = 0.0;
    for (int i = 0;i < 3;i++){
        for (int j = 0;j < 3;j++){
            int index_x = 8 * blockIdx.x + threadIdx.x + i;
            int index_y = 8 * blockIdx.y + threadIdx.y + j;
            //temp = b[i][j] * a[index_x][index_y];
            float temp = b[i * 3 + j] * a[index_x * n + index_y];
            result += temp;
        }
    }
    c[idx * (n - 2) + idy] = result;
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