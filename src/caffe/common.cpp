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
      thread_instance_.reset(new Caffe()); //TODO 需要记得释放
    }
    return *(thread_instance_.get());
  }

  Caffe::Caffe() : mode_(Caffe::CPU)
  {
  }

  Caffe::~Caffe()
  {
  }

} // namespace caffe