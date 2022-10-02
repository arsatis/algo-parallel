/*
 * CUDA supports atomic memory accesses (e.g., atomicAdd) and synchronization constructs
 * such as barriers.
 */
#include "utils.h"

__device__ __managed__ int volatile done;
__device__ __managed__ int counter;

__global__
void no_sync() {
    while (threadIdx.x >> 5 != 0 && done == 0); // block till first warp passes atomicAdd
    atomicAdd(&counter, 1);
    if (threadIdx.x == 0) done = 1;

    if (threadIdx.x == 0) printf("Counter value (no sync): %d\n", counter);
}

__global__
void with_sync() {
    while (threadIdx.x >> 5 != 0 && done == 0); // block till first warp passes atomicAdd
    atomicAdd(&counter, 1);
    if (threadIdx.x == 0) done = 1;

    __syncthreads(); // threads wait until all threads in the same block have reached this point

    if (threadIdx.x == 0) printf("Counter value (with sync): %d\n", counter);
}

int main(int argc, char **argv) {
    done = 0;
    counter = 0;
    no_sync<<<1, 1024>>>();
    cudaDeviceSynchronize();
    checkCudaError();

    done = 0;
    counter = 0;
    with_sync<<<1, 1024>>>();
    cudaDeviceSynchronize();
    checkCudaError();

    return 0;
}
