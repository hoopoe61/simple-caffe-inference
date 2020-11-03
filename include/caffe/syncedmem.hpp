#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

/**
* 用于同步CPU与GPU的数据
* 由于此版本是CPU版本,并没有和GPU数据进行同步,因此,此函数的优势没有体现出来
* 只是进行空间的分配与回收
**/

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{
    inline void CaffeMallocHost(void **ptr, size_t size, bool *use_cuda)
    {
        *ptr = malloc(size); //分配空间
        *use_cuda = false;
        CHECK(*ptr) << "host allocation of size " << size << " failed"; //判断是否分配成功
    }

    inline void CaffeFreeHost(void *ptr, bool use_cuda)
    {
        free(ptr);
    }

    class SyncedMemory
    {
    public:
        explicit SyncedMemory(size_t size = 0);
        ~SyncedMemory();
        const void *cpu_data();        //返回cpu中的数据,只能读
        void set_cpu_data(void *data); //设置cpu中的数据
        void *mutable_cpu_data();      //返回cpu中的数据,可以写
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

        void *cpu_ptr_;
        size_t size_;
        SyncedHead head_;
        bool own_cpu_data_;
        bool cpu_malloc_use_cuda_;
    }; //class SyncedMemory

} //namespace caffe

#endif //CAFFE_SYNCEDMEM_HPP_