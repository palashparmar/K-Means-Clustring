#include <cuda.h>
#include <cuda_runtime.h>

//some helper function for printing error related to calling cuda functions and runtime kernal errors

void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}


void gpuErrchk(cudaError_t ans) {
    gpuAssert((ans), __FILE__, __LINE__);
}

inline void checkCuda(cudaError_t e)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "Runtime Error %d: %s\n", e, cudaGetErrorString(e));
    }
}


inline void checkLastCudaError()
{
    checkCuda(cudaGetLastError());
}
