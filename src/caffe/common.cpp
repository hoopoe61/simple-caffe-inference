#include "caffe/common.hpp"

namespace caffe
{
  // 确保每个进程有不同的值
  static boost::thread_specific_ptr<Caffe> thread_instance_;

  Caffe &Caffe::Get()
  {
    if (!thread_instance_.get())
    {
      //如果该handle为空,返回新的
      thread_instance_.reset(new Caffe());
    }
    return *(thread_instance_.get());
  }

#ifdef CPU_ONLY // CPU-only Caffe.
  Caffe::Caffe()
      : mode_(Caffe::CPU), solver_rank_(0)
  {
  }

  Caffe::~Caffe()
  {
  }

#else // Normal GPU + CPU Caffe.
  Caffe::Caffe()
      : mode_(Caffe::CPU), solver_rank_(0), cublas_handle_(NULL)
  {
    // Try to create a cublas handler, and report an error if failed (but we will
    // keep the program running as one might just want to run CPU code).
    if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS)
    {
      LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
    }
  }

  Caffe::~Caffe()
  {
    if (cublas_handle_)
      CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  }

  const char *cublasGetErrorString(cublasStatus_t error)
  {
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    }
    return "Unknown cublas status";
  }

#endif

} // namespace caffe