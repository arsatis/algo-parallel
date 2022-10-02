/*
 * Function execution space specifiers: specifies where the function will be executed (i.e., CPU or GPU).
 * 
 * Compile using: nvcc hello.cu -o hello
 */
#include <stdio.h>

__device__ // executed on the device, only callable from the device
void byeGPU() {
    printf("GPU says: goodbye world!\n");
}

__global__ // executed on the device, callable from both the device and host
void helloGPU() {
    printf("GPU says: hello world!\n");
    byeGPU();
}

__host__ // executed on the host, only callable from the host
void helloCPU() {
    printf("CPU says: hello world!\n");
}

// function with no specifiers is implicitly deemed as __host__
int main(int argc, char **argv) {

    helloGPU<<<1, 1>>>();
    helloCPU();
    // byeGPU<<<1, 10>>>(); // Error: __device__ function call cannot be configured
    // byeGPU();            // Error: __device__ function cannot be called from a __host__ function
    printf("CPU says: goodbye world!\n");

    // was there any error?
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error));   
    }

    return 0;
}
