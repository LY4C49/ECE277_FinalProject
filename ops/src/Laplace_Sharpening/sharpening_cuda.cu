#include <cstdio>
#include <math.h>

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
}


void sharpening_launcher(const float* a, const float* b, float* c, int n){
    int block_size = ceil((n - 1) / 8);
    dim3 blockSize(block_size,block_size);
    dim3 threadPerBlock(8,8);
    sharpening_kernel<<<blockSize,threadPerBlock>>>(a,b,c,n);
}