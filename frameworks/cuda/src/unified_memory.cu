/*
 * Unified memory defines a managed memory space that has a common memory addressing space,
 * enabling both CPU and GPU to access it as though it is part of their memory space.
 * 
 * This is done by:
 * - replacing __device__ with __managed__
 * - replacing cudaMalloc with cudaMallocManaged
 * 
 * However, unified memory is slightly slower than global memory.
 */
#include "utils.h"
#define n 12
#define k 10

__managed__
int deviceArr2[n + 1];

__global__
void increment(int *arr, int num) {
    int threadId = 1 + threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    printf("Thread %d has started incrementing.\n", threadId);
    for (int i = 0; i < num; ++i) atomicAdd(&arr[0], threadId), atomicAdd(&deviceArr2[0], threadId);
    printf("Thread %d is done incrementing idx 0.\n", threadId);
    for (int i = 0; i < num; ++i) atomicAdd(&arr[threadId], threadId), atomicAdd(&deviceArr2[threadId], threadId);
    printf("Thread %d has completed incrementing.\n", threadId);
}

int main(int argc, char **argv) {
    int size = (n + 1) * sizeof(int);
    int *deviceArr;

    cudaMallocManaged((void **)&deviceArr, size);             // allocate space on unified memory
    for (int i = 0; i <= n; ++i) deviceArr[i] = 0;
    for (int i = 0; i <= n; ++i) deviceArr2[i] = 0;

    increment<<<1, n>>>(deviceArr, k);                        // launch kernel
    cudaDeviceSynchronize();                                  // blocks until devices have completed all preceding requested tasks
    for (int i = 0; i <= n; ++i) printf("%d ", deviceArr[i]); printf("\n"); // memcpy not needed
    for (int i = 0; i <= n; ++i) printf("%d ", deviceArr2[i]); printf("\n");

    checkCudaError();
    cudaFree(deviceArr);                                      // free device memory

    return 0;
}
