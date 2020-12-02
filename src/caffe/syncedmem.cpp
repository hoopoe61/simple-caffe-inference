#include "caffe/syncedmem.hpp"

namespace caffe
{
    SyncedMemory::SyncedMemory(size_t size)
        : cpu_ptr_(nullptr), gpu_ptr_(nullptr), size_(size), head_(UNINITIALIZED),
          own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false)
    {
    }

    SyncedMemory::~SyncedMemory()
    {
        if (cpu_ptr_ && own_cpu_data_)
        {
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        if (gpu_ptr_ && own_gpu_data_)
        {
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
    }

    inline void SyncedMemory::to_cpu()
    {
        //将数据同步到CPU上
        switch (head_)
        {
        case UNINITIALIZED:
        {
            CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_); //分配空间
            caffe_memset(size_, 0, cpu_ptr_);                         //全部置0
            head_ = HEAD_AT_CPU;                                      //切换状态
            own_cpu_data_ = true;
            break;
        }
        case HEAD_AT_GPU:
        {
            // 如果head指向GPU,则首先判断CPU的数据是否为空,如果为空,则应该分配空间;
            // 同时,将GPU的数据拷贝到CPU上
            if (cpu_ptr_ == nullptr)
            {
                CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
                own_cpu_data_ = true;
            }
            caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_); //本质是调用cudaMemcpy()
            head_ = SYNCED;
            break;
        }
        case HEAD_AT_CPU:
        case SYNCED:
            break;
        }
    }

    inline void SyncedMemory::to_gpu()
    {
        //将数据同步到GPU上
        switch (head_)
        {
        case UNINITIALIZED:
        {
            CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
            caffe_gpu_memset(size_, 0, gpu_ptr_);
            head_ = HEAD_AT_GPU;
            own_gpu_data_ = true;
            break;
        }
        case HEAD_AT_CPU:
        {
            if (gpu_ptr_ == NULL)
            {
                CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                own_gpu_data_ = true;
            }
            caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
            head_ = SYNCED;
            break;
        }
        case HEAD_AT_GPU:
        case SYNCED:
            break;
        }
    }

    const void *SyncedMemory::cpu_data()
    {
        to_cpu();
        return (const void *)cpu_ptr_;
    }

    const void *SyncedMemory::gpu_data()
    {
        to_gpu();
        return (const void *)gpu_ptr_;
    }

    void SyncedMemory::set_cpu_data(void *data)
    {
        CHECK(data);
        if (own_cpu_data_)
        {
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        cpu_ptr_ = data;
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = false; //TODO(dengshunge) 为什么会为false呢?
    }

    void SyncedMemory::set_gpu_data(void *data)
    {
        CHECK(data);
        if (own_gpu_data_)
        {
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
        gpu_ptr_ = data;
        head_ = HEAD_AT_GPU;
        own_gpu_data_ = false;
    }

    void *SyncedMemory::mutable_cpu_data()
    {
        to_cpu();
        head_ = HEAD_AT_CPU;
        return cpu_ptr_;
    }

    void *SyncedMemory::mutable_gpu_data()
    {
        to_gpu();
        head_ = HEAD_AT_GPU;
        return gpu_ptr_;
    }

} // namespace caffe