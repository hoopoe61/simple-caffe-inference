#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

/**
* 用于同步CPU与GPU的数据,保证两者数据的一致性
* 利用有限状态机的原理,其中包含了4个状态:
* 1.UNINITIALIZED(未初始化);
* 2.HEAD_AT_CPU(数据位于CPU上)
* 3.HEAD_AT_GPU(数据位于GPU上)
* 4.SYNCED(CPU与GPU上的数据一致,处于同步)
**/

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
    inline void CaffeMallocHost(void **ptr, size_t size, bool *use_cuda)
    {
        if (Caffe::mode() == Caffe::GPU)
        {
            // 调用cudaMallocHost()为主机分配内存,分配的内存可以让host和device共同访问,类似于cudaMallocManaged()
            // 优点是cudaMallocHost()会自动自用某些函数,如cudaMemcpy()
            // 缺点是大量使用cudaMallocHost()会减少系统的可用内存.
            CUDA_CHECK(cudaMallocHost(ptr, size)); //TODO(dengshunge) 可以尝试用cudaMallocManage来进行管理
            *use_cuda = true;
        }
        else
        {
            *ptr = malloc(size); //分配空间
            *use_cuda = false;
            CHECK(*ptr) << "host allocation of size " << size << " failed"; //判断是否分配成功
        }
    }

    inline void CaffeFreeHost(void *ptr, bool use_cuda)
    {
        if (use_cuda)
        {
            CUDA_CHECK(cudaFreeHost(ptr));
        }
        else
        {
            free(ptr);
        }
    }

    class SyncedMemory
    {
    public:
        explicit SyncedMemory(size_t size = 0);
        ~SyncedMemory();
        const void *cpu_data();        //返回cpu中的数据,只能读
        void set_cpu_data(void *data); //设置cpu中的数据
        void *mutable_cpu_data();      //返回cpu中的数据,可以写
        const void *gpu_data();
        void set_gpu_data(void *data);
        void *mutable_gpu_data();
        enum SyncedHead
        {
            UNINITIALIZED,
            HEAD_AT_CPU,
            HEAD_AT_GPU,
            SYNCED
        };
        size_t size() const { return size_; }

    private:
        void to_cpu();
        void to_gpu();

        void *cpu_ptr_;
        void *gpu_ptr_;
        size_t size_;
        SyncedHead head_;
        bool own_cpu_data_;
        bool cpu_malloc_use_cuda_;
        bool own_gpu_data_;

        DISABLE_COPY_AND_ASSIGN(SyncedMemory);
    }; //class SyncedMemory

} //namespace caffe

#endif //CAFFE_SYNCEDMEM_HPP_