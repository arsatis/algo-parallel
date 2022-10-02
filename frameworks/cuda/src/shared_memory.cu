/*
 * Shared memory is shared within the same thread block, and is accessible to the device faster
 * than global memory. This is done by using the __shared__ specifier.
 */
#include <vector>
#include "utils.h"

#define n 12
#define k 10

__global__
void increment() {
    __shared__ int deviceArr[n + 1];

    int threadId = 1 + threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    printf("Thread %d has started incrementing.\n", threadId);
    for (int i = 0; i < k; ++i) atomicAdd(&deviceArr[0], threadId);
    printf("Thread %d is done incrementing idx 0.\n", threadId);
    for (int i = 0; i < k; ++i) atomicAdd(&deviceArr[threadId], threadId);
    printf("Thread %d has completed incrementing.\n", threadId);
    
    if (threadId == 1) printf("%d ", deviceArr[0]);
    printf("%d ", deviceArr[threadId]);
    if (threadId == 1) printf("\n");
}

int main(int argc, char **argv) {
    increment<<<1, n>>>(); 
    cudaDeviceSynchronize();
    checkCudaError();
    
    return 0;
}
