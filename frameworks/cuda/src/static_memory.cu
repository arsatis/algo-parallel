/*
 * We can statically allocate device memory by using the __device__ specifier.
 * However, these arrays would need to have their sizes explicitly specified during compile time.
 */
#include "utils.h"
#define n 12
#define k 10

__device__
int deviceArr[n + 1];

__global__
void increment(int num) {
    int threadId = 1 + threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    printf("Thread %d has started incrementing.\n", threadId);
    for (int i = 0; i < num; ++i) atomicAdd(&deviceArr[0], threadId);
    printf("Thread %d is done incrementing idx 0.\n", threadId);
    for (int i = 0; i < num; ++i) atomicAdd(&deviceArr[threadId], threadId);
    printf("Thread %d has completed incrementing.\n", threadId);
}

int main(int argc, char **argv) {
    int arr[n + 1];
    for (int i = 0; i <= n; ++i) arr[i] = 0;

    cudaMemcpyToSymbol(deviceArr, arr, sizeof(deviceArr));   // copy contents of arr to allocated space
    increment<<<1, n>>>(k);                                  // launch kernel
    cudaDeviceSynchronize();                                 // blocks until devices have completed all preceding requested tasks

    for (int i : arr) printf("%d ", i); printf("\n");        // contents of arr before incrementing
    cudaMemcpyFromSymbol(arr, deviceArr, sizeof(deviceArr)); // copy contents of deviceArr to arr
    for (int i : arr) printf("%d ", i); printf("\n");        // contents of arr after incrementing

    checkCudaError();

    return 0;
}
