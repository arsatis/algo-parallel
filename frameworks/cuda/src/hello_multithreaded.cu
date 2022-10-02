/*
 * An invocation of a CUDA kernel function launches a new grid.
 * Each grid is a collection of thread blocks (can be 1D, 2D, or 3D).
 * Each block is a collection of threads (can be 1D, 2D, or 3D).
 * 
 * Notice that threads within the same block will execute in sequence, (only true for block sizes <=32)
 * but blocks within the same grid do not execute in sequence.
 */
#include "utils.h"

__global__
void helloGPU() {
    int threadId = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
    int blockId = blockIdx.x + gridDim.x * blockIdx.y + gridDim.x * gridDim.y * blockIdx.z;
    printf("Thread %d in block %d says: hello world!\n", threadId, blockId);
}

int main(int argc, char **argv) {
    helloGPU<<<1, 3>>>(); // 1 block, 3 threads
    helloGPU<<<3, 1>>>(); // 3 blocks, 1 thread

    dim3 dimensions(2, 2, 2);
    helloGPU<<<1, dimensions>>>(); // 1 block, 2 * 2 * 2 threads
    helloGPU<<<dimensions, 1>>>(); // 2 * 2 * 2 blocks, 1 thread
    helloGPU<<<dimensions, dimensions>>>(); // 2 * 2 * 2 blocks, 2 * 2 * 2 threads

    checkCudaError();
    return 0;
}
