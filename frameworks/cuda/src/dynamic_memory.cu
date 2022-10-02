/*
 * We can dynamically allocate device memory in a similar manner to memory allocation in C, using:
 * - cudaMalloc(memAddr, size);
 * - cudaFree(memAddr);
 * 
 * We can also copy from host to device/device to host using:
 * - cudaMemcpy(fromAddr, toAddr, size, typeOfTransfer);
 * 
 * We can also get the host to wait for CUDA threads to finish executing using:
 * - cudaDeviceSynchronize();
 */
#include "utils.h"
#define n 12
#define k 10

__global__
void increment(int *arr, int num) {
    int threadId = 1 + threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    printf("Thread %d has started incrementing.\n", threadId);
    for (int i = 0; i < num; ++i) atomicAdd(&arr[0], threadId);
    printf("Thread %d is done incrementing idx 0.\n", threadId);
    for (int i = 0; i < num; ++i) atomicAdd(&arr[threadId], threadId);
    printf("Thread %d has completed incrementing.\n", threadId);
}

int main(int argc, char **argv) {
    int size = (n + 1) * sizeof(int);
    int arr[n + 1];
    for (int i = 0; i <= n; ++i) arr[i] = 0;

    int *deviceArr;
    cudaMalloc((void **)&deviceArr, size);                    // allocate space on device memory
    cudaMemcpy(deviceArr, arr, size, cudaMemcpyHostToDevice); // copy contents of arr to allocated space

    increment<<<1, n>>>(deviceArr, k);                        // launch kernel
    cudaDeviceSynchronize();                                  // blocks until devices have completed all preceding requested tasks

    for (int i : arr) printf("%d ", i); printf("\n");         // contents of arr before incrementing
    cudaMemcpy(arr, deviceArr, size, cudaMemcpyDeviceToHost); // copy contents of deviceArr to arr
    for (int i : arr) printf("%d ", i); printf("\n");         // contents of arr after incrementing

    checkCudaError();
    cudaFree(deviceArr);                                      // free device memory

    return 0;
}
