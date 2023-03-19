#include <cstdio>
#include <math.h>

# define M_PI  3.14159265358979323846  /* pi */

__global__ void generate_dct_matrix(int n){
    __shared__ float dct_array[64];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;


    if (idy == 0){
        dct_array[idy * 8 + idx] = 1.0 / sqrt(8.0);
    }
    else{
        dct_array[idy * 8 + idx] = sqrt((float)(2 / 8)) * cos((M_PI * (2.0 * idx + 1.0) * idy) / (2.0 * 8.0));
    }
    __syncthreads();
}

__global__ void dct2_kernel(const float* a, const float* b, float* c, int n){
/*
    __shared__ float dct_array_2[8][8];
    int x = threadIdx.x;
    int y = threadIdx.y;
    if (y == 0){
        dct_array_2[x][y] = 1.0 / sqrt(8.0);
        //dct_array_2[y * 8 + x] = 1.0;
        //dct_array[idy * 8 + idx] = sqrt((float)(1 / 2));
    }
    else{
        dct_array_2[x][y] = 0.5 * cos((M_PI * (2.0 * x + 1.0) * y) / (2.0 * 8.0));
        //printf("sqrt %f", sqrt((float)(2 / 8)));
        //printf("cos %f ---- %f.\n", cos((M_PI * (2.0 * y + 1.0) * x) / (2.0 * 8.0)));
        //printf("Thread %f failed % f.\n", dct_array[y * 8 + x],a[y*8 + x]);
        //dct_array[idy * 8 + idx] = 2.2;
    }
    __syncthreads();
*/
    __shared__ float dct_array_2[64];
    dct_array_2[threadIdx.y * 8 + threadIdx.x] = b[threadIdx.y * 8 + threadIdx.x];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int initial_x = idx - idx % 8;
    int relative_x = threadIdx.x;

    float result = 0.0;
    for (int i = 0;i < 8;i++){
        result += a[idy * n + (initial_x + i)] * dct_array_2[i * 8 + relative_x];
    }

    c[idy * n + idx] = result;

}


void dct2_launcher(const float* a, const float* b, float* c, int n){
    int block_size = ceil(n / 8);
    dim3 blockSize(block_size,block_size);
    dim3 threadPerBlock(8,8);
    dct2_kernel<<<blockSize,threadPerBlock>>>(a,b,c,n);


}