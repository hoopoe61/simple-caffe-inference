#ifndef CAFFE_UTIL_DEVICE_ALTERNATE_H_
#define CAFFE_UTIL_DEVICE_ALTERNATE_H_

#ifdef CPU_ONLY // CPU-only Caffe.

#else // Normal GPU + CPU Caffe.

#include <cuda.h>
#include <cuda_runtime.h>

#endif

//
// CUDA宏定义
//

// CUDA: various checks for different function calls.
// 检测cuda调用函数是否成功
#define CUDA_CHECK(condition)                                             \
    /* Code block avoids redefinition of cudaError_t error */             \
    do                                                                    \
    {                                                                     \
        cudaError_t error = condition;                                    \
        CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
    } while (0)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)

namespace caffe
{
    // CUDA: use 512 threads per block
    // 每个block开512个线程
    const int CAFFE_CUDA_NUM_THREADS = 512;

    // CUDA: number of blocks for threads.
    // 计算block数量
    inline int CAFFE_GET_BLOCKS(const int N)
    {
        return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
    }
} // namespace caffe
#endif //CAFFE_UTIL_DEVICE_ALTERNATE_H_