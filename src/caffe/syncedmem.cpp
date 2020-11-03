#include "caffe/syncedmem.hpp"

namespace caffe
{
    SyncedMemory::SyncedMemory(size_t size)
        : cpu_ptr_(nullptr), size_(size), head_(UNINITIALIZED), own_cpu_data_(false),
          cpu_malloc_use_cuda_(false)
    {
    }

    SyncedMemory::~SyncedMemory()
    {
        if (cpu_ptr_ && own_cpu_data_)
        {
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
    }

    inline void SyncedMemory::to_cpu()
    {
        switch (head_)
        {
        case UNINITIALIZED:
        {
            CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_); //分配空间
            caffe_memset(size_, 0, cpu_ptr_);                          //全部置0
            head_ = HEAD_AT_CPU;                                      //切换状态
            own_cpu_data_ = true;
            break;
        }
        case HEAD_AT_GPU:
        {
            LOG(FATAL) << "Cannot use GPU in CPU-only Caffe: check mode.";
            break;
        }
        case HEAD_AT_CPU:
        case SYNCED:
            break;
        }
    }

    const void *SyncedMemory::cpu_data()
    {
        to_cpu();
        return (const void *)cpu_ptr_;
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
        own_cpu_data_ = false;
    }

    void *SyncedMemory::mutable_cpu_data()
    {
        to_cpu();
        head_ = HEAD_AT_CPU;
        return cpu_ptr_;
    }

} // namespace caffe